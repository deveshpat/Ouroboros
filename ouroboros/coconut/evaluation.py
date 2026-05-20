"""Evaluation orchestration."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch

from ouroboros.coconut.data import build_sample_at_stage, collate_stage_k
from ouroboros.coconut.dgac import HaltGate
from ouroboros.coconut.latent import forward_latent_batch, prepare_latent_runtime
from ouroboros.models import (
    _ddp_sum,
    _is_main_process,
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


def _ddp_min_max(local_min: float, local_max: float, count: int, device: torch.device) -> tuple[float, float]:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized() and _world_size() > 1):
        if count <= 0:
            return 0.0, 0.0
        return float(local_min), float(local_max)
    min_value = local_min if count > 0 else float("inf")
    max_value = local_max if count > 0 else float("-inf")
    tensor = torch.tensor([min_value, max_value], device=device, dtype=torch.float64)
    torch.distributed.all_reduce(tensor[0:1], op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(tensor[1:2], op=torch.distributed.ReduceOp.MAX)
    reduced_min = float(tensor[0].item())
    reduced_max = float(tensor[1].item())
    if reduced_min == float("inf"):
        reduced_min = 0.0
    if reduced_max == float("-inf"):
        reduced_max = 0.0
    return reduced_min, reduced_max


@torch.no_grad()
def evaluate_stage_health_metrics(
    model,
    val_samples: List[Dict[str, Any]],
    tokenizer,
    lat_token_id: int,
    stage_k: int,
    device: torch.device,
    args: argparse.Namespace,
    halt_gate: Optional[HaltGate] = None,
) -> Dict[str, Any]:
    """
    Runs teacher-forced validation on ALL DDP ranks and returns clearly scoped
    training-health metrics. If ``halt_gate`` is present, latent passes use it.
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
    halt_gate_used = halt_gate is not None

    local_val_samples = val_samples[rank::world_size]
    progress_every = _eval_progress_every(args)
    _emit_progress(
        f"  [eval] teacher-forced CE/token-acc start: rank0_shard={len(local_val_samples)} "
        f"global_samples={len(val_samples)} batch_size={batch_size} stage={stage_k} "
        f"halt_gate_used={halt_gate_used}"
    )

    ce_numer = 0.0
    ce_denom = 0
    token_correct = 0
    token_total = 0
    actual_latents_sum = 0.0
    actual_latents_count = 0
    actual_latents_min = float("inf")
    actual_latents_max = float("-inf")

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
        result = forward_latent_batch(
            runtime=runtime,
            batch=batch,
            args=args,
            halt_gate=halt_gate,
            include_hidden_sequences=halt_gate is not None,
            include_token_accuracy=True,
        )
        valid_tokens = int(result["n_valid"])
        if valid_tokens <= 0:
            continue
        ce_numer += float(result["ce_sum"].item())
        ce_denom += valid_tokens
        token_correct += int(result.get("token_correct") or 0)
        token_total += valid_tokens
        actual = result.get("actual_n_latents")
        if isinstance(actual, torch.Tensor) and actual.numel() > 0:
            actual_f = actual.detach().to(dtype=torch.float32)
            actual_latents_sum += float(actual_f.sum().item())
            actual_latents_count += int(actual_f.numel())
            actual_latents_min = min(actual_latents_min, float(actual_f.min().item()))
            actual_latents_max = max(actual_latents_max, float(actual_f.max().item()))
        _maybe_emit_progress(
            label="eval CE rank0",
            processed=min(start + batch_size, len(local_val_samples)),
            total=len(local_val_samples),
            every=progress_every,
        )

    reduced = _ddp_sum(
        [ce_numer, ce_denom, token_correct, token_total, actual_latents_sum, actual_latents_count],
        device,
    )
    ce_numer, ce_denom, token_correct, token_total, actual_latents_sum, actual_latents_count = reduced
    ce_denom = int(round(ce_denom))
    token_correct = int(round(token_correct))
    token_total = int(round(token_total))
    actual_latents_count = int(round(actual_latents_count))
    actual_latents_min, actual_latents_max = _ddp_min_max(
        actual_latents_min,
        actual_latents_max,
        actual_latents_count,
        device,
    )

    model.train()
    if halt_gate is not None:
        halt_gate.train()

    ce = float(ce_numer / max(ce_denom, 1))
    token_acc = float(token_correct / max(token_total, 1))
    actual_latents_mean = float(actual_latents_sum / max(actual_latents_count, 1))
    metrics = {
        "health_metrics": {
            "teacher_forced": {
                "ce": ce,
                "token_acc": token_acc,
                "halt_gate_used": halt_gate_used,
                "actual_latents_mean": actual_latents_mean,
                "actual_latents_min": float(actual_latents_min),
                "actual_latents_max": float(actual_latents_max),
            }
        }
    }
    if _is_main_process():
        tf = metrics["health_metrics"]["teacher_forced"]
        _emit_progress(
            "  [eval] teacher-forced health: "
            f"ce={tf['ce']:.4f} token_acc={tf['token_acc']:.4f} "
            f"halt_gate_used={tf['halt_gate_used']} "
            f"actual_latents_mean={tf['actual_latents_mean']:.2f} "
            f"range=[{tf['actual_latents_min']:.0f}, {tf['actual_latents_max']:.0f}]"
        )
    return metrics


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
    """Backward-compatible teacher-forced validation API."""
    metrics = evaluate_stage_health_metrics(
        model=model,
        val_samples=val_samples,
        tokenizer=tokenizer,
        lat_token_id=lat_token_id,
        stage_k=stage_k,
        device=device,
        args=args,
        halt_gate=halt_gate,
    )
    tf = metrics["health_metrics"]["teacher_forced"]
    return float(tf["ce"]), float(tf["token_acc"])


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
    health = evaluate_stage_health_metrics(
        model=model,
        val_samples=val_samples,
        tokenizer=tokenizer,
        lat_token_id=lat_token_id,
        stage_k=stage_k,
        device=device,
        args=args,
        halt_gate=halt_gate,
    )
    tf = health["health_metrics"]["teacher_forced"]
    val_ce = float(tf["ce"])
    val_acc = float(tf["token_acc"])

    metrics: Dict[str, float] = {
        "val_ce": val_ce,
        "val_acc": val_acc,
        "val_token_acc": val_acc,
        "health_metrics.teacher_forced.ce": val_ce,
        "health_metrics.teacher_forced.token_acc": val_acc,
        "health_metrics.teacher_forced.halt_gate_used": float(bool(tf["halt_gate_used"])),
        "health_metrics.teacher_forced.actual_latents_mean": float(tf["actual_latents_mean"]),
        "health_metrics.teacher_forced.actual_latents_min": float(tf["actual_latents_min"]),
        "health_metrics.teacher_forced.actual_latents_max": float(tf["actual_latents_max"]),
    }
    if _is_main_process():
        print(
            f"\n  [eval-only] stage={stage_k} val_ce={val_ce:.4f} val_token_acc={val_acc:.4f} "
            f"halt_gate_used={tf['halt_gate_used']} actual_latents_mean={tf['actual_latents_mean']:.2f}"
        )
        if wandb_run is not None:
            import wandb
            wandb.log(
                {
                    "eval_only/ce": val_ce,
                    "eval_only/acc": val_acc,
                    "eval_only/token_acc": val_acc,
                    "eval_only/stage": stage_k,
                    "health_metrics/teacher_forced/ce": val_ce,
                    "health_metrics/teacher_forced/token_acc": val_acc,
                    "health_metrics/teacher_forced/halt_gate_used": bool(tf["halt_gate_used"]),
                    "health_metrics/teacher_forced/actual_latents_mean": tf["actual_latents_mean"],
                    "health_metrics/teacher_forced/actual_latents_min": tf["actual_latents_min"],
                    "health_metrics/teacher_forced/actual_latents_max": tf["actual_latents_max"],
                },
                step=step,
            )

    barrier()
    return metrics
