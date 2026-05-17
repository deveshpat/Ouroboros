"""Evaluation orchestration."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch

from ouroboros.coconut.data import build_sample_at_stage, collate_stage_k
from ouroboros.coconut.dgac import HaltGate, coconut_forward, normalize_pred
from ouroboros.coconut.latent import (
    build_question_context,
    decode_from_latent_context,
    prepare_latent_runtime,
    run_latent_passes,
)
from ouroboros.models import (
    _ddp_sum,
    _is_main_process,
    _maybe_apply_chat_template,
    _maybe_empty_cuda_cache,
    _rank,
    _world_size,
    barrier,
)


def _eval_progress_every(args: argparse.Namespace) -> int:
    """Return the validation progress cadence. Zero disables progress logs."""
    return max(int(getattr(args, "eval_progress_every", 25) or 0), 0)


def _emit_progress(message: str) -> None:
    """Print rank-0 progress immediately so Kaggle logs do not look stalled."""
    if _is_main_process():
        print(message, flush=True)


def _maybe_emit_progress(*, label: str, processed: int, total: int, every: int) -> None:
    if every <= 0 or total <= 0:
        return
    processed = min(max(int(processed), 0), int(total))
    if processed <= 0:
        return
    if processed == total or processed % every == 0:
        pct = 100.0 * processed / max(int(total), 1)
        _emit_progress(f"  [{label}] {processed}/{total} ({pct:.1f}%)")


@torch.no_grad()
def evaluate_stage(
    model,
    val_samples: List[Dict[str, Any]],
    tokenizer,
    lat_token_id: int,
    stage_k: int,
    device: torch.device,
    args: argparse.Namespace,
    halt_gate: Optional[HaltGate] = None,
) -> Tuple[float, float]:
    """
    Runs on ALL DDP ranks. Each rank processes its interleaved shard of val_samples,
    then all-reduces CE and accuracy counts.
    """
    _maybe_empty_cuda_cache()
    model.eval()
    if halt_gate is not None:
        halt_gate.eval()

    rank = _rank()
    world_size = _world_size()
    pad_id = tokenizer.pad_token_id or 0
    runtime = prepare_latent_runtime(model, device)
    batch_size = max(int(args.val_batch_size), 1)

    local_val_samples = val_samples[rank::world_size]
    progress_every = _eval_progress_every(args)
    _emit_progress(
        f"  [eval] validation CE start: rank0_shard={len(local_val_samples)} "
        f"global_samples={len(val_samples)} batch_size={batch_size} stage={stage_k}"
    )

    ce_numer = 0.0
    ce_denom = 0

    for start in range(0, len(local_val_samples), batch_size):
        batch_raw = local_val_samples[start : start + batch_size]
        built = [
            build_sample_at_stage(tokenizer, sample, stage_k, lat_token_id, args.max_seq_len)
            for sample in batch_raw
        ]
        built = [sample for sample in built if sample is not None]
        if not built:
            continue
        batch = collate_stage_k(built, pad_id)
        loss, _ = coconut_forward(
            model,
            batch,
            stage_k,
            device,
            halt_gate=None,
            args=args,
            step_in_phase=0,
        )
        valid_tokens = int((batch["labels"][:, 1:].contiguous().view(-1) != -100).sum().item())
        ce_numer += float(loss.item()) * valid_tokens
        ce_denom += valid_tokens
        _maybe_emit_progress(
            label="eval CE rank0",
            processed=min(start + batch_size, len(local_val_samples)),
            total=len(local_val_samples),
            every=progress_every,
        )

    ce_numer, ce_denom = _ddp_sum([ce_numer, ce_denom], device)
    ce_denom = int(round(ce_denom))

    acc_samples = val_samples
    local_acc_samples = acc_samples[rank::world_size]
    _emit_progress(
        f"  [eval] accuracy decode start: rank0_shard={len(local_acc_samples)} "
        f"global_samples={len(acc_samples)}"
    )

    n_correct = 0
    n_total = 0

    for acc_index, sample in enumerate(local_acc_samples, start=1):
        built = build_sample_at_stage(tokenizer, sample, stage_k, lat_token_id, args.max_seq_len)
        if built is None:
            continue
        q_len = int(built["q_len"])
        n_latent = int(built["n_latent"])
        q_ids = built["full_ids"][:q_len]
        q_tensor = q_ids.unsqueeze(0).to(device)
        ctx = runtime.embed_tokens(q_tensor)
        ctx_mask = torch.ones((1, ctx.size(1)), dtype=torch.bool, device=device)
        ctx, ctx_mask, _ = run_latent_passes(
            runtime=runtime,
            ctx=ctx,
            ctx_mask=ctx_mask,
            n_latent=n_latent,
            halt_gate=halt_gate,
            args=args,
        )
        decoded = decode_from_latent_context(
            runtime=runtime,
            ctx=ctx,
            ctx_mask=ctx_mask,
            tokenizer=tokenizer,
            args=args,
            context="eval decode",
        )

        pred = normalize_pred(decoded.text).strip()
        gold = str(sample.get("answer_norm", "")).strip()
        if pred and gold and (pred == gold or pred.lower() == gold.lower()):
            n_correct += 1
        n_total += 1
        _maybe_emit_progress(
            label="eval acc rank0",
            processed=acc_index,
            total=len(local_acc_samples),
            every=max(1, min(progress_every, 10)) if progress_every > 0 else 0,
        )

    n_correct, n_total = _ddp_sum([n_correct, n_total], device)
    n_correct = int(round(n_correct))
    n_total = int(round(n_total))

    model.train()
    if halt_gate is not None:
        halt_gate.train()

    return ce_numer / max(ce_denom, 1), n_correct / max(n_total, 1)


def run_eval_only(
    *,
    model,
    tokenizer,
    halt_gate: Optional[HaltGate],
    val_samples: List[Dict[str, Any]],
    lat_token_id: int,
    stage_k: int,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    wandb_run=None,
) -> Dict[str, float]:
    """Run a validation preflight and exit without optimizer steps."""

    _emit_progress(f"\n  [eval-only] starting validation stage={stage_k} samples={len(val_samples)}")
    val_ce, val_acc = evaluate_stage(
        model=model,
        val_samples=val_samples,
        tokenizer=tokenizer,
        lat_token_id=lat_token_id,
        stage_k=stage_k,
        device=device,
        args=args,
        halt_gate=halt_gate,
    )

    metrics: Dict[str, float] = {"val_ce": float(val_ce), "val_acc": float(val_acc)}
    if _is_main_process():
        print(
            f"\n  [eval-only] stage={stage_k} val_ce={val_ce:.4f} val_acc={val_acc:.4f}"
        )
        if wandb_run is not None:
            import wandb
            wandb.log(
                {"eval_only/ce": val_ce, "eval_only/acc": val_acc, "eval_only/stage": stage_k},
                step=step,
            )

    barrier()
    return metrics
