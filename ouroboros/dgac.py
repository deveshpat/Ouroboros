"""DGAC halt gate and Coconut latent-forward utilities."""

from __future__ import annotations

import re as _re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ouroboros.model import (
    _amp_dtype,
    _autocast_ctx,
    _extract_last_hidden_state,
    _get_backbone,
    _get_embed_tokens,
    _get_lm_head,
)

_LAST_NUM = _re.compile(r"[\d,]+(?:\.\d+)?")

class HaltGate(nn.Module):
    """
    Halt gate for DGAC. Zero-initialized: outputs ~0.5 at start of Phase 3.4.
    Input: h_curr [B, D] + h_prev [B, D] at question-end position.
    Output: halt_prob [B].
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h_curr: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(torch.cat([h_curr, h_prev], dim=-1))).squeeze(-1)


def compute_dgac_lambda1(step: int, warmup: int, ramp: int, lmax: float) -> float:
    """lambda_1=0 for warmup steps, then linearly ramps to lmax over ramp steps."""
    if step < warmup:
        return 0.0
    return lmax * min((step - warmup) / max(ramp, 1), 1.0)


def build_dgac_halt_probe_depths(stage_k: int, probe_steps: str | Sequence[int] | None) -> List[int]:
    """Return sorted unique DGAC CE-probe depths, always including full stage_k."""
    max_depth = max(int(stage_k), 0)
    if max_depth <= 0:
        return []

    raw_steps: Iterable[Any]
    if probe_steps is None:
        raw_steps = (1, 2, 4, max_depth)
    elif isinstance(probe_steps, str):
        raw_steps = [part.strip() for part in probe_steps.split(",") if part.strip()]
    else:
        raw_steps = probe_steps

    depths: List[int] = []
    for raw in raw_steps:
        if isinstance(raw, str):
            token = raw.strip().lower().replace("-", "_")
            if token in {"stage", "stage_k", "max", "k", "full", "full_k"}:
                depth = max_depth
            else:
                depth = int(token)
        else:
            depth = int(raw)
        if depth <= 0:
            continue
        depths.append(min(depth, max_depth))

    depths.append(max_depth)
    return sorted(set(depths))


def construct_dgac_halt_targets(
    *,
    ce_by_probe_depth: Mapping[int, torch.Tensor],
    full_ce: torch.Tensor,
    full_depths: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Choose the smallest CE-preserving halt depth for each sample.

    Probe depths may be larger than a sample's available latent depth; those
    probes are interpreted as the sample's full depth for that row.
    """
    if not ce_by_probe_depth:
        return full_depths.to(dtype=torch.long).clone()

    device = full_depths.device
    target_depths = full_depths.to(device=device, dtype=torch.long).clone()
    full_ce = full_ce.to(device=device, dtype=torch.float32).view(-1)
    full_depths = full_depths.to(device=device, dtype=torch.long).view(-1)
    tol = float(tolerance)

    sorted_probe_depths = sorted(int(depth) for depth in ce_by_probe_depth)
    batch_size = int(full_depths.numel())
    for row in range(batch_size):
        row_full_depth = max(int(full_depths[row].item()), 0)
        if row_full_depth <= 0:
            target_depths[row] = 0
            continue
        seen_depths: set[int] = set()
        allowed_ce = float(full_ce[row].item()) + tol
        for probe_depth in sorted_probe_depths:
            candidate_depth = max(0, min(int(probe_depth), row_full_depth))
            if candidate_depth <= 0 or candidate_depth in seen_depths:
                continue
            seen_depths.add(candidate_depth)
            probe_ce = ce_by_probe_depth[probe_depth].to(device=device, dtype=torch.float32).view(-1)
            if float(probe_ce[row].item()) <= allowed_ce:
                target_depths[row] = candidate_depth
                break
    return target_depths


def build_halt_supervision_labels(
    target_depths: torch.Tensor,
    *,
    max_depth: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build continue/halt labels for available HaltGate decisions.

    Decision depth `d` means: after `d` latent states exist, halt before
    appending the next one. A full-depth target has no positive halt label
    because there is no extra latent proposal after the configured maximum.
    """
    target_depths = target_depths.to(dtype=torch.long).view(-1)
    batch_size = int(target_depths.numel())
    n_decisions = max(int(max_depth) - 1, 0)
    labels = torch.zeros((batch_size, n_decisions), dtype=torch.float32, device=target_depths.device)
    mask = torch.zeros_like(labels, dtype=torch.bool)
    if n_decisions == 0:
        return labels, mask

    decision_depths = torch.arange(1, int(max_depth), dtype=torch.long, device=target_depths.device)
    for row in range(batch_size):
        target = max(int(target_depths[row].item()), 0)
        if target <= 0:
            continue
        active = decision_depths <= min(target, int(max_depth) - 1)
        mask[row, active] = True
        halt_column = (decision_depths == target).nonzero(as_tuple=False).flatten()
        if halt_column.numel() > 0:
            labels[row, halt_column[0]] = 1.0
    return labels, mask


def _compute_ce_mean_by_row(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1)
    losses = F.cross_entropy(shift_logits, shift_labels, reduction="none", ignore_index=-100)
    losses = losses.view(labels.size(0), -1)
    valid = labels[:, 1:] != -100
    counts = valid.sum(dim=1)
    ce_by_row = losses.sum(dim=1) / counts.clamp_min(1).to(dtype=losses.dtype)
    return ce_by_row.to(dtype=torch.float32), counts


def _compute_supervised_halt_loss(
    *,
    hidden_sequences: List[List[torch.Tensor]],
    target_depths: torch.Tensor,
    halt_gate: HaltGate,
) -> Optional[Dict[str, Any]]:
    bce_terms: List[torch.Tensor] = []
    target_terms: List[torch.Tensor] = []
    supervised_decisions = 0

    for row, hidden_at_q_end in enumerate(hidden_sequences):
        if len(hidden_at_q_end) < 2:
            continue
        target_depth = int(target_depths[row].item())
        if target_depth <= 0:
            continue

        for idx in range(1, len(hidden_at_q_end)):
            if idx > target_depth:
                break
            h_curr = hidden_at_q_end[idx].to(dtype=torch.float32)
            h_prev = hidden_at_q_end[idx - 1].to(dtype=torch.float32)
            halt_prob = halt_gate(h_curr, h_prev).clamp(1e-6, 1.0 - 1e-6)
            target = torch.ones_like(halt_prob) if idx == target_depth else torch.zeros_like(halt_prob)
            bce_terms.append(F.binary_cross_entropy(halt_prob, target))
            supervised_decisions += int(halt_prob.numel())

        target_terms.append(torch.tensor(float(target_depth), device=target_depths.device))

    if not bce_terms:
        return None

    return {
        "halt_loss": torch.stack(bce_terms).mean(),
        "halt_target_mean": torch.stack(target_terms).mean(),
        "halt_supervised_count": float(supervised_decisions),
    }


def _compute_ce_sum_and_count(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1)
    valid = shift_labels != -100
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return logits.new_zeros((), dtype=torch.float32), 0
    ce_sum = F.cross_entropy(shift_logits[valid], shift_labels[valid], reduction="sum")
    return ce_sum, n_valid


def _build_question_context(
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


def _run_latent_passes(
    model,
    ctx: torch.Tensor,
    ctx_mask: torch.Tensor,
    n_latent,
    halt_gate: Optional[HaltGate],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    backbone = _get_backbone(model)

    if isinstance(n_latent, int):
        target_steps = torch.full((ctx.size(0),), int(n_latent), device=device, dtype=torch.long)
        scalar_input = True
    elif isinstance(n_latent, torch.Tensor):
        target_steps = n_latent.to(device=device, dtype=torch.long).view(-1)
        scalar_input = False
    else:
        target_steps = torch.tensor(list(n_latent), device=device, dtype=torch.long)
        scalar_input = False

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

        with _autocast_ctx(device, amp_dtype):
            outputs = backbone(
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


def _collect_latent_hidden_sequences(
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


def _compute_batched_halt_metrics(
    hidden_sequences: List[List[torch.Tensor]],
    actual_n_latents: torch.Tensor,
    halt_gate: HaltGate,
    device: torch.device,
    args: argparse.Namespace,
    step_in_phase: int,
) -> Optional[Dict[str, Any]]:
    lam1 = compute_dgac_lambda1(
        step_in_phase,
        args.dgac_warmup_steps,
        args.dgac_ramp_steps,
        args.dgac_lambda_ponder_max,
    )
    one = torch.ones(1, device=device, dtype=torch.float32)
    ponder_terms: List[torch.Tensor] = []
    diversity_terms: List[torch.Tensor] = []
    halt_terms: List[torch.Tensor] = []

    for row, hidden_at_q_end in enumerate(hidden_sequences):
        if len(hidden_at_q_end) < 2:
            continue

        ponder = torch.zeros(1, device=device, dtype=torch.float32)
        div_loss = torch.zeros(1, device=device, dtype=torch.float32)
        remainder = one.clone()
        halt_steps = torch.zeros(1, device=device, dtype=torch.float32)

        for idx in range(1, len(hidden_at_q_end)):
            h_curr = hidden_at_q_end[idx].to(dtype=torch.float32)
            h_prev = hidden_at_q_end[idx - 1].to(dtype=torch.float32)
            halt_prob = halt_gate(h_curr, h_prev)
            ponder = ponder + remainder
            if idx < len(hidden_at_q_end) - 1:
                remainder = remainder * (1.0 - halt_prob)
            div_loss = div_loss + F.relu(F.cosine_similarity(h_curr, h_prev, dim=-1) - args.dgac_tau)
            with torch.no_grad():
                halted = (halt_prob > args.halt_threshold) & (halt_steps == 0)
                halt_steps = torch.where(halted, torch.full_like(halt_steps, float(idx)), halt_steps)

        halt_steps = torch.where(
            halt_steps == 0,
            torch.full_like(halt_steps, float(int(actual_n_latents[row].item()))),
            halt_steps,
        )
        ponder_terms.append(ponder.mean())
        diversity_terms.append(div_loss.mean())
        halt_terms.append(halt_steps.mean())

    if not diversity_terms:
        return None

    return {
        "ponder": torch.stack(ponder_terms).mean(),
        "diversity": torch.stack(diversity_terms).mean(),
        "halt_step_mean": torch.stack(halt_terms).mean(),
        "lambda1": lam1,
    }


def _forward_batched_latent(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    q_lens: torch.Tensor,
    n_latents: torch.Tensor,
    pad_id: torch.Tensor,
    device: torch.device,
    halt_gate: Optional[HaltGate],
    args: argparse.Namespace,
    step_in_phase: int,
    amp_dtype: torch.dtype,
) -> Dict[str, Any]:
    backbone = _get_backbone(model)
    embed_fn = _get_embed_tokens(model)
    lm_head_fn = _get_lm_head(model)

    with _autocast_ctx(device, amp_dtype):
        all_embeds = embed_fn(input_ids)
        pad_embed = embed_fn(pad_id.view(1)).squeeze(0)

    q_ctx, q_ctx_mask = _build_question_context(all_embeds, q_lens, pad_embed)
    latent_ctx, _, actual_k = _run_latent_passes(
        model=model,
        ctx=q_ctx,
        ctx_mask=q_ctx_mask,
        n_latent=n_latents,
        halt_gate=None,
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )
    actual_n_latents = actual_k if isinstance(actual_k, torch.Tensor) else torch.full_like(n_latents, int(actual_k))

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

    with _autocast_ctx(device, amp_dtype):
        outputs = backbone(inputs_embeds=patched, attention_mask=attention_mask, use_cache=False)
        hidden = _extract_last_hidden_state(outputs, "coconut batched full forward")
        logits = lm_head_fn(hidden).float()

    ce_sum, n_valid = _compute_ce_sum_and_count(logits, labels)
    ce_by_row, valid_by_row = _compute_ce_mean_by_row(logits, labels)
    ce_value = float(ce_sum.item() / max(n_valid, 1))
    result: Dict[str, Any] = {
        "ce_sum": ce_sum,
        "n_valid": n_valid,
        "ce": ce_value,
        "ce_by_row": ce_by_row,
        "valid_by_row": valid_by_row,
        "actual_n_latents": actual_n_latents,
        "hidden_sequences": None,
        "ponder": None,
        "diversity": None,
        "halt_step_mean": None,
        "lambda1": 0.0,
    }

    if halt_gate is None:
        return result

    hidden_sequences = _collect_latent_hidden_sequences(latent_ctx, max_q_len, actual_n_latents)
    result["hidden_sequences"] = hidden_sequences
    halt_metrics = _compute_batched_halt_metrics(
        hidden_sequences=hidden_sequences,
        actual_n_latents=actual_n_latents,
        halt_gate=halt_gate,
        device=device,
        args=args,
        step_in_phase=step_in_phase,
    )
    if halt_metrics is None:
        return result

    result.update(halt_metrics)
    return result


def coconut_forward(
    model,
    batch: Dict[str, torch.Tensor],
    stage_k: int,
    device: torch.device,
    halt_gate: Optional[HaltGate],
    args: argparse.Namespace,
    step_in_phase: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    q_lens = batch["q_lens"].to(device)
    n_latents = batch["n_latents"].to(device)
    pad_id = batch["pad_id"].to(device)
    amp_dtype = _amp_dtype(device)

    result = _forward_batched_latent(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        q_lens=q_lens,
        n_latents=n_latents,
        pad_id=pad_id,
        device=device,
        halt_gate=halt_gate,
        args=args,
        step_in_phase=step_in_phase,
        amp_dtype=amp_dtype,
    )

    if result["n_valid"] == 0:
        zero = torch.zeros((), device=device, requires_grad=True)
        return zero, {"ce": 0.0}

    ce = result["ce_sum"] / result["n_valid"]
    total_loss = ce
    metrics: Dict[str, float] = {"ce": float(ce.item())}

    if halt_gate is not None:
        halt_supervision_weight = float(getattr(args, "dgac_halt_supervision_weight", 0.0))
        hidden_sequences = result.get("hidden_sequences")
        actual_n_latents = result.get("actual_n_latents")
        if halt_supervision_weight > 0.0 and hidden_sequences is not None and actual_n_latents is not None:
            probe_depths = build_dgac_halt_probe_depths(
                stage_k=stage_k,
                probe_steps=getattr(args, "dgac_halt_probe_steps", None),
            )
            ce_by_probe_depth: Dict[int, torch.Tensor] = {}
            with torch.no_grad():
                for probe_depth in probe_depths:
                    if int(probe_depth) >= int(stage_k):
                        continue
                    probe_n_latents = torch.minimum(
                        torch.full_like(n_latents, int(probe_depth)),
                        n_latents,
                    )
                    probe_result = _forward_batched_latent(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        q_lens=q_lens,
                        n_latents=probe_n_latents,
                        pad_id=pad_id,
                        device=device,
                        halt_gate=None,
                        args=args,
                        step_in_phase=step_in_phase,
                        amp_dtype=amp_dtype,
                    )
                    ce_by_probe_depth[int(probe_depth)] = probe_result["ce_by_row"].detach()
            target_depths = construct_dgac_halt_targets(
                ce_by_probe_depth=ce_by_probe_depth,
                full_ce=result["ce_by_row"].detach(),
                full_depths=actual_n_latents.detach(),
                tolerance=float(getattr(args, "dgac_halt_ce_tolerance", 0.02)),
            )
            supervised = _compute_supervised_halt_loss(
                hidden_sequences=hidden_sequences,
                target_depths=target_depths,
                halt_gate=halt_gate,
            )
            if supervised is not None:
                total_loss = total_loss + halt_supervision_weight * supervised["halt_loss"]
                halt_loss_value = float(supervised["halt_loss"].item())
                halt_target_mean = float(supervised["halt_target_mean"].item())
                metrics.update(
                    {
                        "dgac_halt_loss": halt_loss_value,
                        "dgac_halt_target_mean": halt_target_mean,
                        "dgac_halt_supervised_count": float(supervised["halt_supervised_count"]),
                        "halt_loss": halt_loss_value,
                    }
                )

    if halt_gate is not None and result["diversity"] is not None:
        total_loss = total_loss + result["lambda1"] * result["ponder"] + args.dgac_lambda_diversity * result["diversity"]
        metrics.update(
            {
                "ponder": float(result["ponder"].item()),
                "diversity": float(result["diversity"].item()),
                "halt_step_mean": float(result["halt_step_mean"].item()),
                "lambda1": float(result["lambda1"]),
                "dgac_ponder": float(result["ponder"].item()),
                "dgac_diversity": float(result["diversity"].item()),
                "dgac_halt_step_mean": float(result["halt_step_mean"].item()),
                "dgac_lambda1": float(result["lambda1"]),
            }
        )

    return total_loss, metrics


def normalize_pred(text: str) -> str:
    boxed = _re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed.group(1).strip().replace(",", "")
    numeric = _re.search(r"(?:answer is|=)\s*\**\s*([\d,\.\-]+)", text, _re.IGNORECASE)
    if numeric:
        return numeric.group(1).strip().replace(",", "")
    nums = _LAST_NUM.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    stripped = text.strip()
    if not stripped:
        return ""
    last_line = stripped.splitlines()[-1].strip()
    last_line = _re.sub(r"^(?:final answer|answer)\s*[:\-]\s*", "", last_line, flags=_re.IGNORECASE)
    return last_line.strip(" .,:;!*")