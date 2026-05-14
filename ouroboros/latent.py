"""Coconut/DGAC latent execution utilities.

This module owns generic latent-forward mechanics: preparing model runtime
handles, building latent context, running latent passes, injecting latent states,
computing CE, and decoding from latent context. DGAC policy stays in
``ouroboros.dgac``.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ouroboros.model import (
    _amp_dtype,
    _autocast_ctx,
    _extract_last_hidden_state,
    _get_backbone,
    _get_embed_tokens,
    _get_lm_head,
)


@dataclass(frozen=True)
class LatentRuntime:
    """Resolved model handles needed by latent execution."""

    model: Any
    device: torch.device
    amp_dtype: torch.dtype
    backbone: Any
    embed_tokens: Any
    lm_head: Any

    def autocast(self):
        return _autocast_ctx(self.device, self.amp_dtype)


@dataclass(frozen=True)
class DecodeResult:
    """Token-by-token decode result from an existing latent context."""

    token_ids: List[int]
    text: str


def prepare_latent_runtime(
    model,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
) -> LatentRuntime:
    """Resolve and cache the model handles needed for latent execution."""
    resolved_amp_dtype = _amp_dtype(device) if amp_dtype is None else amp_dtype
    return LatentRuntime(
        model=model,
        device=device,
        amp_dtype=resolved_amp_dtype,
        backbone=_get_backbone(model),
        embed_tokens=_get_embed_tokens(model),
        lm_head=_get_lm_head(model),
    )


def compute_ce_mean_by_row(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1)
    losses = F.cross_entropy(shift_logits, shift_labels, reduction="none", ignore_index=-100)
    losses = losses.view(labels.size(0), -1)
    valid = labels[:, 1:] != -100
    counts = valid.sum(dim=1)
    ce_by_row = losses.sum(dim=1) / counts.clamp_min(1).to(dtype=losses.dtype)
    return ce_by_row.to(dtype=torch.float32), counts


def compute_ce_sum_and_count(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1)
    valid = shift_labels != -100
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return logits.new_zeros((), dtype=torch.float32), 0
    ce_sum = F.cross_entropy(shift_logits[valid], shift_labels[valid], reduction="sum")
    return ce_sum, n_valid


def compute_ce_from_hidden(
    *,
    runtime: LatentRuntime,
    hidden: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Compute CE by projecting only supervised next-token positions."""
    shift_hidden = hidden[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    valid_by_row = valid.sum(dim=1)
    n_valid = int(valid_by_row.sum().item())
    if n_valid == 0:
        zeros = hidden.new_zeros((labels.size(0),), dtype=torch.float32)
        return hidden.new_zeros((), dtype=torch.float32), 0, zeros, valid_by_row

    selected_hidden = shift_hidden[valid]
    selected_labels = shift_labels[valid]
    with runtime.autocast():
        selected_logits = runtime.lm_head(selected_hidden).float()
    losses = F.cross_entropy(selected_logits, selected_labels, reduction="none")
    ce_sum = losses.sum()

    row_ids = torch.arange(labels.size(0), device=labels.device).unsqueeze(1).expand_as(valid)
    selected_rows = row_ids[valid]
    row_sums = losses.new_zeros((labels.size(0),), dtype=losses.dtype)
    row_sums.index_add_(0, selected_rows, losses)
    ce_by_row = row_sums / valid_by_row.clamp_min(1).to(dtype=losses.dtype)
    return ce_sum, n_valid, ce_by_row.to(dtype=torch.float32), valid_by_row


def build_question_context(
    all_embeds: torch.Tensor,
    q_lens: torch.Tensor,
    pad_embed: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(all_embeds.size(0))
    if q_lens.numel() == 0:
        empty_ctx = all_embeds.new_empty((batch_size, 0, all_embeds.size(-1)))
        empty_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=all_embeds.device)
        return empty_ctx, empty_mask

    max_q_len = int(q_lens.max().item())
    if max_q_len <= 0:
        empty_ctx = all_embeds.new_empty((batch_size, 0, all_embeds.size(-1)))
        empty_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=all_embeds.device)
        return empty_ctx, empty_mask

    ctx = all_embeds[:, :max_q_len, :].clone()
    positions = torch.arange(max_q_len, device=all_embeds.device).unsqueeze(0)
    ctx_mask = positions < q_lens.unsqueeze(1)
    pad_value = pad_embed.to(device=all_embeds.device, dtype=ctx.dtype).view(1, 1, -1)
    ctx = torch.where(ctx_mask.unsqueeze(-1), ctx, pad_value)
    return ctx, ctx_mask


def run_latent_passes(
    runtime: LatentRuntime,
    ctx: torch.Tensor,
    ctx_mask: torch.Tensor,
    n_latent: int | torch.Tensor | Sequence[int],
    halt_gate: Optional[Any],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """Run fixed-depth or HaltGate-shortened latent passes from question context."""
    device = runtime.device

    if isinstance(n_latent, int):
        target_steps = torch.full((ctx.size(0),), int(n_latent), device=device, dtype=torch.long)
        scalar_input = True
    elif isinstance(n_latent, torch.Tensor):
        target_steps = n_latent.to(device=device, dtype=torch.long).view(-1)
        scalar_input = False
    else:
        target_steps = torch.tensor(list(n_latent), device=device, dtype=torch.long)
        scalar_input = False

    if (
        (
            bool(getattr(args, "latent_cache", False))
            or bool(getattr(args, "mac_mps_latent_cache", False))
        )
        and halt_gate is None
    ):
        cached_result = _try_run_latent_passes_with_cache(
            runtime=runtime,
            ctx=ctx,
            ctx_mask=ctx_mask,
            target_steps=target_steps,
            scalar_input=scalar_input,
        )
        if cached_result is not None:
            return cached_result

    batch_size = int(ctx.size(0))
    actual_k = torch.zeros(batch_size, dtype=torch.long, device=device)
    prev_hidden = ctx.new_zeros((batch_size, ctx.size(-1)))
    halted = torch.zeros(batch_size, dtype=torch.bool, device=device)

    max_steps = int(target_steps.max().item()) if target_steps.numel() > 0 else 0
    for latent_step in range(max_steps):
        active_indices = ((target_steps > latent_step) & (~halted)).nonzero(as_tuple=False).flatten()
        if active_indices.numel() == 0:
            break

        prefix_lens = ctx_mask[active_indices].sum(dim=1).to(dtype=torch.long)
        max_prefix_len = int(prefix_lens.max().item())
        prefix_embeds = ctx[active_indices, :max_prefix_len, :].clone()
        prefix_positions = torch.arange(max_prefix_len, device=device).unsqueeze(0)
        prefix_mask = prefix_positions < prefix_lens.unsqueeze(1)
        if max_prefix_len > 0:
            prefix_embeds = torch.where(
                prefix_mask.unsqueeze(-1),
                prefix_embeds,
                prefix_embeds.new_zeros((1, 1, prefix_embeds.size(-1))),
            )

        with runtime.autocast():
            outputs = runtime.backbone(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_mask,
                use_cache=False,
            )
        hidden = _extract_last_hidden_state(outputs, f"latent pass step={latent_step}")
        last_positions = prefix_lens - 1
        h_step = hidden[torch.arange(active_indices.numel(), device=device), last_positions, :]

        append_mask = torch.ones(active_indices.numel(), dtype=torch.bool, device=device)
        if halt_gate is not None:
            has_prev = actual_k[active_indices] > 0
            if bool(has_prev.any().item()):
                halt_probs = halt_gate(
                    h_step[has_prev].to(dtype=torch.float32),
                    prev_hidden[active_indices[has_prev]].to(dtype=torch.float32),
                )
                halt_now = halt_probs > args.halt_threshold
                if bool(halt_now.any().item()):
                    blocked_local = has_prev.nonzero(as_tuple=False).flatten()[halt_now]
                    append_mask[blocked_local] = False
                    halted[active_indices[blocked_local]] = True

        next_col = ctx.new_zeros((batch_size, 1, ctx.size(-1)))
        next_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        if bool(append_mask.any().item()):
            append_indices = active_indices[append_mask]
            append_hidden = h_step[append_mask].to(dtype=ctx.dtype)
            next_col[append_indices, 0, :] = append_hidden
            next_mask[append_indices, 0] = True
            actual_k[append_indices] += 1
            prev_hidden[append_indices] = append_hidden

        ctx = torch.cat([ctx, next_col], dim=1)
        ctx_mask = torch.cat([ctx_mask, next_mask], dim=1)

    if ctx_mask.numel() > 0 and ctx_mask.size(1) > 0:
        active_cols = ctx_mask.any(dim=0)
        if bool(active_cols.any().item()):
            last_active_col = int(active_cols.nonzero(as_tuple=False).max().item()) + 1
            ctx = ctx[:, :last_active_col, :]
            ctx_mask = ctx_mask[:, :last_active_col]
        else:
            ctx = ctx[:, :0, :]
            ctx_mask = ctx_mask[:, :0]

    return_k: Any = int(actual_k[0].item()) if scalar_input and batch_size == 1 else actual_k
    return ctx, ctx_mask, return_k


def _clone_cache_for_autograd(cache: Any) -> Any:
    """Clone cache tensors so cache updates do not mutate saved autograd values."""
    cloned = copy.copy(cache)
    if not hasattr(cache, "__dict__"):
        return cloned
    cloned.__dict__ = {}
    for key, value in cache.__dict__.items():
        if isinstance(value, torch.Tensor):
            cloned.__dict__[key] = value.clone()
        elif isinstance(value, list):
            cloned.__dict__[key] = [
                _clone_cache_for_autograd(item)
                if hasattr(item, "__dict__")
                else (item.clone() if isinstance(item, torch.Tensor) else item)
                for item in value
            ]
        else:
            cloned.__dict__[key] = value
    return cloned


def _gradient_checkpointing_is_enabled(model: Any) -> bool:
    enabled = getattr(model, "is_gradient_checkpointing", False)
    return bool(enabled() if callable(enabled) else enabled)


def _set_gradient_checkpointing(model: Any, enabled: bool) -> None:
    method_name = "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
    method = getattr(model, method_name, None)
    if callable(method):
        method()
    if enabled:
        input_grad_method = getattr(model, "enable_input_require_grads", None)
        if callable(input_grad_method):
            input_grad_method()


def _try_run_latent_passes_with_cache(
    *,
    runtime: LatentRuntime,
    ctx: torch.Tensor,
    ctx_mask: torch.Tensor,
    target_steps: torch.Tensor,
    scalar_input: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, Any]]:
    """Cache-backed fixed-depth latent pass for accelerator fallback runs."""
    batch_size = int(ctx.size(0))
    max_steps = int(target_steps.max().item()) if target_steps.numel() > 0 else 0
    if batch_size < 1 or max_steps <= 0:
        actual_k = torch.zeros(batch_size, dtype=torch.long, device=runtime.device)
        return_k: Any = int(actual_k[0].item()) if scalar_input and batch_size == 1 else actual_k
        return ctx, ctx_mask, return_k

    checkpointing_was_enabled = _gradient_checkpointing_is_enabled(runtime.model)
    if checkpointing_was_enabled:
        _set_gradient_checkpointing(runtime.model, False)

    actual_k = torch.zeros(batch_size, dtype=torch.long, device=runtime.device)
    cache = None
    try:
        initial_embeds = torch.where(
            ctx_mask.unsqueeze(-1),
            ctx,
            ctx.new_zeros((1, 1, ctx.size(-1))),
        )
        with runtime.autocast():
            outputs = runtime.backbone(inputs_embeds=initial_embeds, attention_mask=ctx_mask, use_cache=True)
        cache = getattr(outputs, "past_key_values", None)
        if cache is None:
            return None
        hidden = _extract_last_hidden_state(outputs, "cached latent pass step=0")
        prefix_lens = ctx_mask.sum(dim=1).to(dtype=torch.long).clamp_min(1)
        h_step = hidden[torch.arange(batch_size, device=runtime.device), prefix_lens - 1, :]

        for latent_step in range(max_steps):
            active_mask = target_steps > latent_step
            if not bool(active_mask.any().item()):
                break
            next_col = ctx.new_zeros((batch_size, 1, ctx.size(-1)))
            next_col[active_mask, 0, :] = h_step[active_mask].to(dtype=ctx.dtype)
            next_mask = active_mask.view(batch_size, 1)
            ctx = torch.cat([ctx, next_col], dim=1)
            ctx_mask = torch.cat([ctx_mask, next_mask], dim=1)
            actual_k[active_mask] += 1

            if latent_step + 1 >= max_steps or not bool((target_steps > latent_step + 1).any().item()):
                break

            step_cache = _clone_cache_for_autograd(cache)
            with runtime.autocast():
                outputs = runtime.backbone(
                    inputs_embeds=next_col,
                    attention_mask=ctx_mask,
                    past_key_values=step_cache,
                    use_cache=True,
                )
            cache = getattr(outputs, "past_key_values", None)
            if cache is None:
                return None
            h_step = _extract_last_hidden_state(outputs, f"cached latent pass step={latent_step + 1}")[:, -1, :]
    finally:
        if checkpointing_was_enabled:
            _set_gradient_checkpointing(runtime.model, True)

    return_k: Any = int(actual_k[0].item()) if scalar_input and batch_size == 1 else actual_k
    return ctx, ctx_mask, return_k


def collect_latent_hidden_sequences(
    latent_ctx: torch.Tensor,
    max_q_len: int,
    actual_n_latents: torch.Tensor,
) -> List[List[torch.Tensor]]:
    hidden_sequences: List[List[torch.Tensor]] = [[] for _ in range(int(latent_ctx.size(0)))]
    max_steps = int(actual_n_latents.max().item()) if actual_n_latents.numel() > 0 else 0
    for latent_step in range(max_steps):
        active_indices = (actual_n_latents > latent_step).nonzero(as_tuple=False).flatten()
        if active_indices.numel() == 0:
            break
        step_hidden = latent_ctx[active_indices, max_q_len + latent_step, :]
        for local_idx, sample_idx in enumerate(active_indices.tolist()):
            hidden_sequences[sample_idx].append(step_hidden[local_idx : local_idx + 1])
    return hidden_sequences


def forward_latent_batch(
    *,
    runtime: LatentRuntime,
    batch: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
    n_latents: Optional[torch.Tensor] = None,
    include_hidden_sequences: bool = False,
) -> Dict[str, Any]:
    """Run a full Coconut latent batch forward and return CE/latent artifacts."""
    device = runtime.device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    q_lens = batch["q_lens"].to(device)
    actual_target_latents = batch["n_latents"].to(device) if n_latents is None else n_latents.to(device)
    pad_id = batch["pad_id"].to(device)

    with runtime.autocast():
        all_embeds = runtime.embed_tokens(input_ids)
        pad_embed = runtime.embed_tokens(pad_id.view(1)).squeeze(0)

    q_ctx, q_ctx_mask = build_question_context(all_embeds, q_lens, pad_embed)
    latent_ctx, _, actual_k = run_latent_passes(
        runtime=runtime,
        ctx=q_ctx,
        ctx_mask=q_ctx_mask,
        n_latent=actual_target_latents,
        halt_gate=None,
        args=args,
    )
    actual_n_latents = actual_k if isinstance(actual_k, torch.Tensor) else torch.full_like(actual_target_latents, int(actual_k))

    patched = all_embeds.clone()
    max_q_len = int(q_ctx.size(1))
    max_n_latent = int(actual_n_latents.max().item()) if actual_n_latents.numel() > 0 else 0
    for latent_step in range(max_n_latent):
        active_indices = (actual_n_latents > latent_step).nonzero(as_tuple=False).flatten()
        if active_indices.numel() == 0:
            break
        inject_pos = q_lens[active_indices] + latent_step
        valid_inject = inject_pos < patched.size(1)
        if not bool(torch.all(valid_inject).item()):
            active_indices = active_indices[valid_inject]
            inject_pos = inject_pos[valid_inject]
            if active_indices.numel() == 0:
                continue
        h_step = latent_ctx[active_indices, max_q_len + latent_step, :]
        patched_next = patched.clone()
        patched_next[active_indices, inject_pos, :] = h_step.to(dtype=patched_next.dtype)
        patched = patched_next

    with runtime.autocast():
        outputs = runtime.backbone(inputs_embeds=patched, attention_mask=attention_mask, use_cache=False)
        hidden = _extract_last_hidden_state(outputs, "coconut batched full forward")

    ce_sum, n_valid, ce_by_row, valid_by_row = compute_ce_from_hidden(
        runtime=runtime,
        hidden=hidden,
        labels=labels,
    )
    ce_value = float(ce_sum.item() / max(n_valid, 1))
    hidden_sequences = (
        collect_latent_hidden_sequences(latent_ctx, max_q_len, actual_n_latents)
        if include_hidden_sequences
        else None
    )
    return {
        "ce_sum": ce_sum,
        "n_valid": n_valid,
        "ce": ce_value,
        "ce_by_row": ce_by_row,
        "valid_by_row": valid_by_row,
        "actual_n_latents": actual_n_latents,
        "hidden_sequences": hidden_sequences,
        "latent_ctx": latent_ctx,
        "max_q_len": max_q_len,
    }


def decode_from_latent_context(
    *,
    runtime: LatentRuntime,
    ctx: torch.Tensor,
    ctx_mask: torch.Tensor,
    tokenizer,
    args: argparse.Namespace,
    context: str,
) -> DecodeResult:
    """Greedily decode tokens from an existing latent context."""
    generated: List[int] = []
    eos_id = tokenizer.eos_token_id
    for _ in range(args.gen_max_tokens):
        if ctx.size(1) > args.max_seq_len:
            ctx = ctx[:, -args.max_seq_len :, :]
            ctx_mask = ctx_mask[:, -args.max_seq_len :]
        with runtime.autocast():
            outputs = runtime.backbone(inputs_embeds=ctx, attention_mask=ctx_mask, use_cache=False)
            hidden = _extract_last_hidden_state(outputs, context)
            logits = runtime.lm_head(hidden)
        next_id = int(logits[:, -1, :].argmax(-1).item())
        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)
        next_embed = runtime.embed_tokens(torch.tensor([[next_id]], device=runtime.device))
        ctx = torch.cat([ctx, next_embed], dim=1)
        ctx_mask = torch.cat(
            [ctx_mask, torch.ones((1, 1), dtype=torch.bool, device=runtime.device)],
            dim=1,
        )

    return DecodeResult(token_ids=generated, text=tokenizer.decode(generated, skip_special_tokens=True))
