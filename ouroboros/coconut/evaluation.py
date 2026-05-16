"""Evaluation, generation, DGAC diagnostics, and eval-only orchestration."""

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

GEN_PROMPTS = [
    "What is 15 + 27?",
    "Write a Python function that returns the factorial of n.",
    "What is the capital of Japan?",
    "Explain what a neural network is in simple terms.",
    "Solve for x: 3x + 6 = 21.",
]


def _eval_progress_every(args: argparse.Namespace) -> int:
    """Return the validation progress cadence. Zero disables progress logs."""
    return max(int(getattr(args, "eval_progress_every", 25) or 0), 0)


def _diagnostic_batch_size(args: argparse.Namespace) -> int:
    """Return DGAC diagnostic microbatch size independent of normal validation."""
    value = getattr(args, "dgac_diagnostics_batch_size", None)
    if value is None:
        value = getattr(args, "val_batch_size", 1)
    return max(int(value), 1)


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

    acc_samples = val_samples[:50]
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


@torch.no_grad()
def run_generation_callback(
    model,
    tokenizer,
    halt_gate: Optional[HaltGate],
    stage_k: int,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    wandb_run=None,
) -> float:
    _maybe_empty_cuda_cache()
    model.eval()
    if halt_gate is not None:
        halt_gate.eval()
    print(f"\n  -- Generation @ step {step} stage={stage_k} --")
    runtime = prepare_latent_runtime(model, device)
    mean_uwr = 0.0

    for prompt in GEN_PROMPTS:
        prefix = _maybe_apply_chat_template(tokenizer, prompt)
        q_ids = tokenizer.encode(prefix, add_special_tokens=False)
        q_tensor = torch.tensor(q_ids, device=device).unsqueeze(0)
        ctx = runtime.embed_tokens(q_tensor)
        ctx_mask = torch.ones((1, ctx.size(1)), dtype=torch.bool, device=device)
        ctx, ctx_mask, actual_k = run_latent_passes(
            runtime=runtime,
            ctx=ctx,
            ctx_mask=ctx_mask,
            n_latent=stage_k,
            halt_gate=halt_gate,
            args=args,
        )

        decoded = decode_from_latent_context(
            runtime=runtime,
            ctx=ctx,
            ctx_mask=ctx_mask,
            tokenizer=tokenizer,
            args=args,
            context="generation decode",
        )
        text = decoded.text
        words = text.split()
        uwr = len(set(words)) / max(len(words), 1)
        mean_uwr += uwr
        display = text[:200].replace("\n", " ")
        print(f"  Q: {prompt}")
        print(f"  A: {display}  [k_actual={int(actual_k)} uwr={uwr:.3f}]")

    mean_uwr /= max(len(GEN_PROMPTS), 1)
    print(f"  Mean UWR: {mean_uwr:.3f}\n")
    if wandb_run is not None:
        import wandb
        wandb.log({"gen/mean_uwr": mean_uwr, "gen/stage": stage_k}, step=step)
    model.train()
    if halt_gate is not None:
        halt_gate.train()
    return mean_uwr


@torch.inference_mode()
def _evaluate_ce_for_sample_stage_pairs(
    *,
    model,
    tokenizer,
    sample_stage_pairs: List[Tuple[Dict[str, Any], int]],
    lat_token_id: int,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    """Evaluate CE for already-sharded samples with per-sample latent counts."""
    _maybe_empty_cuda_cache()
    model.eval()
    pad_id = tokenizer.pad_token_id or 0
    batch_size = _diagnostic_batch_size(args)
    progress_every = _eval_progress_every(args)
    _emit_progress(
        f"  [DGAC diagnostic] CE start: rank0_pairs={len(sample_stage_pairs)} batch_size={batch_size}"
    )

    ce_numer = 0.0
    ce_denom = 0

    for start in range(0, len(sample_stage_pairs), batch_size):
        batch_pairs = sample_stage_pairs[start : start + batch_size]
        built = []
        max_stage_for_batch = 0
        for sample, sample_stage in batch_pairs:
            stage_i = max(0, int(sample_stage))
            item = build_sample_at_stage(tokenizer, sample, stage_i, lat_token_id, args.max_seq_len)
            if item is None:
                continue
            built.append(item)
            max_stage_for_batch = max(max_stage_for_batch, int(item["n_latent"]))
        if not built:
            continue
        batch = collate_stage_k(built, pad_id)
        loss, _ = coconut_forward(
            model=model,
            batch=batch,
            stage_k=max_stage_for_batch,
            device=device,
            halt_gate=None,
            args=args,
            step_in_phase=0,
        )
        valid_tokens = int((batch["labels"][:, 1:].contiguous().view(-1) != -100).sum().item())
        ce_numer += float(loss.item()) * valid_tokens
        ce_denom += valid_tokens
        _maybe_emit_progress(
            label="DGAC diagnostic CE rank0",
            processed=min(start + batch_size, len(sample_stage_pairs)),
            total=len(sample_stage_pairs),
            every=progress_every,
        )

    ce_numer, ce_denom = _ddp_sum([ce_numer, ce_denom], device)
    ce_denom = int(round(ce_denom))
    model.train()
    return ce_numer / max(ce_denom, 1)


@torch.no_grad()
def _collect_local_halt_gate_stage_plan(
    *,
    model,
    tokenizer,
    halt_gate: HaltGate,
    val_samples: List[Dict[str, Any]],
    lat_token_id: int,
    stage_k: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[Tuple[Dict[str, Any], int]], List[int]]:
    """Return this rank's validation samples paired with HaltGate-selected k_actual."""
    _maybe_empty_cuda_cache()
    model.eval()
    halt_gate.eval()

    rank = _rank()
    world_size = _world_size()
    local_val_samples = val_samples[rank::world_size]
    batch_size = _diagnostic_batch_size(args)
    progress_every = _eval_progress_every(args)
    _emit_progress(
        f"  [DGAC diagnostic] HaltGate plan start: rank0_shard={len(local_val_samples)} "
        f"global_samples={len(val_samples)} batch_size={batch_size} stage={stage_k}"
    )
    pad_id = tokenizer.pad_token_id or 0
    runtime = prepare_latent_runtime(model, device)

    local_pairs: List[Tuple[Dict[str, Any], int]] = []
    local_ks: List[int] = []

    for start in range(0, len(local_val_samples), batch_size):
        batch_raw = local_val_samples[start : start + batch_size]
        built_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for sample in batch_raw:
            item = build_sample_at_stage(tokenizer, sample, stage_k, lat_token_id, args.max_seq_len)
            if item is not None:
                built_pairs.append((sample, item))
        if not built_pairs:
            continue

        batch = collate_stage_k([item for _, item in built_pairs], pad_id)
        input_ids = batch["input_ids"].to(device)
        q_lens = batch["q_lens"].to(device)

        with runtime.autocast():
            all_embeds = runtime.embed_tokens(input_ids)
        max_q_len = int(q_lens.max().item()) if q_lens.numel() > 0 else 0
        if max_q_len <= 0:
            actual_values = [0] * len(built_pairs)
        else:
            zero_pad_embed = all_embeds.new_zeros((all_embeds.size(-1),))
            ctx, ctx_mask = build_question_context(all_embeds, q_lens, zero_pad_embed)
            _, _, actual_k = run_latent_passes(
                runtime=runtime,
                ctx=ctx,
                ctx_mask=ctx_mask,
                n_latent=stage_k,
                halt_gate=halt_gate,
                args=args,
            )
            if isinstance(actual_k, torch.Tensor):
                actual_values = [int(v) for v in actual_k.detach().cpu().tolist()]
            else:
                actual_values = [int(actual_k)] * len(built_pairs)

        for (sample, _), actual in zip(built_pairs, actual_values):
            clipped = max(0, min(int(stage_k), int(actual)))
            local_pairs.append((sample, clipped))
            local_ks.append(clipped)
        _maybe_emit_progress(
            label="DGAC HaltGate plan rank0",
            processed=min(start + batch_size, len(local_val_samples)),
            total=len(local_val_samples),
            every=progress_every,
        )

    model.train()
    halt_gate.train()
    return local_pairs, local_ks


def _percentile_from_histogram(counts: List[float], percentile: float) -> int:
    total = sum(counts)
    if total <= 0:
        return 0
    threshold = total * float(percentile)
    cumulative = 0.0
    for idx, count in enumerate(counts):
        cumulative += count
        if cumulative >= threshold:
            return idx
    return len(counts) - 1


def _summarize_halt_histogram(local_ks: List[int], stage_k: int, device: torch.device) -> Dict[str, Any]:
    max_bucket = max(int(stage_k), 0)
    local_counts = [0.0 for _ in range(max_bucket + 1)]
    for value in local_ks:
        bucket = max(0, min(max_bucket, int(value)))
        local_counts[bucket] += 1.0

    counts = _ddp_sum(local_counts, device)
    total = float(sum(counts))
    weighted = sum(float(idx) * float(count) for idx, count in enumerate(counts))
    k_mean = weighted / max(total, 1.0)
    pct_at_1 = counts[1] / total if total > 0 and len(counts) > 1 else 0.0
    pct_2_4 = sum(counts[2 : min(4, max_bucket) + 1]) / total if total > 0 and max_bucket >= 2 else 0.0
    pct_5_plus = sum(counts[5:]) / total if total > 0 and max_bucket >= 5 else 0.0

    return {
        "samples": int(round(total)),
        "histogram": {idx: int(round(count)) for idx, count in enumerate(counts)},
        "k_mean": float(k_mean),
        "k_p50": float(_percentile_from_histogram(counts, 0.50)),
        "k_p90": float(_percentile_from_histogram(counts, 0.90)),
        "pct_at_1": float(pct_at_1),
        "pct_2_4": float(pct_2_4),
        "pct_5_plus": float(pct_5_plus),
    }


@torch.inference_mode()
def run_dgac_diagnostics(
    *,
    model,
    tokenizer,
    halt_gate: HaltGate,
    val_samples: List[Dict[str, Any]],
    lat_token_id: int,
    stage_k: int,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    wandb_run=None,
    val_ce_forced_kmax: Optional[float] = None,
) -> Dict[str, float]:
    """Log HaltGate adaptivity diagnostics without mutating checkpoints or state."""
    if halt_gate is None:
        raise ValueError("--dgac_diagnostics requires --use_halt_gate")
    if stage_k <= 0:
        raise ValueError("--dgac_diagnostics requires a positive stage")

    local_gate_pairs, local_ks = _collect_local_halt_gate_stage_plan(
        model=model,
        tokenizer=tokenizer,
        halt_gate=halt_gate,
        val_samples=val_samples,
        lat_token_id=lat_token_id,
        stage_k=stage_k,
        device=device,
        args=args,
    )
    summary = _summarize_halt_histogram(local_ks, stage_k, device)

    rank = _rank()
    world_size = _world_size()
    local_val_samples = val_samples[rank::world_size]
    local_forced_k1_pairs = [(sample, 1) for sample in local_val_samples]
    forced_k1_ce = _evaluate_ce_for_sample_stage_pairs(
        model=model,
        tokenizer=tokenizer,
        sample_stage_pairs=local_forced_k1_pairs,
        lat_token_id=lat_token_id,
        device=device,
        args=args,
    )
    gated_ce = _evaluate_ce_for_sample_stage_pairs(
        model=model,
        tokenizer=tokenizer,
        sample_stage_pairs=local_gate_pairs,
        lat_token_id=lat_token_id,
        device=device,
        args=args,
    )
    forced_kmax_ce = float(val_ce_forced_kmax) if val_ce_forced_kmax is not None else _evaluate_ce_for_sample_stage_pairs(
        model=model,
        tokenizer=tokenizer,
        sample_stage_pairs=[(sample, stage_k) for sample in local_val_samples],
        lat_token_id=lat_token_id,
        device=device,
        args=args,
    )

    metrics: Dict[str, float] = {
        "dgac_diag/val_ce_forced_k1": float(forced_k1_ce),
        "dgac_diag/val_ce_gated": float(gated_ce),
        f"dgac_diag/val_ce_forced_k{stage_k}": float(forced_kmax_ce),
        "dgac_diag/k_mean": float(summary["k_mean"]),
        "dgac_diag/k_p50": float(summary["k_p50"]),
        "dgac_diag/k_p90": float(summary["k_p90"]),
        "dgac_diag/pct_at_1": float(summary["pct_at_1"]),
        "dgac_diag/pct_2_4": float(summary["pct_2_4"]),
        "dgac_diag/pct_5_plus": float(summary["pct_5_plus"]),
        "dgac_diag/samples": float(summary["samples"]),
    }
    for k_value, count in summary["histogram"].items():
        metrics[f"dgac_diag/hist_k{k_value}"] = float(count)

    if _is_main_process():
        hist_text = " ".join(
            f"k{k_value}={summary['histogram'].get(k_value, 0)}"
            for k_value in range(1, stage_k + 1)
        )
        print(
            f"\n  [DGAC diagnostic] stage={stage_k} samples={summary['samples']}"
        )
        print(f"  [DGAC diagnostic] k_actual histogram: {hist_text}")
        print(
            "  [DGAC diagnostic] "
            f"k_mean={summary['k_mean']:.3f} "
            f"p50={summary['k_p50']:.0f} "
            f"p90={summary['k_p90']:.0f} "
            f"pct_at_1={summary['pct_at_1']:.3f} "
            f"pct_2_4={summary['pct_2_4']:.3f} "
            f"pct_5_plus={summary['pct_5_plus']:.3f}"
        )
        print(
            "  [DGAC diagnostic] "
            f"val_ce_forced_k1={forced_k1_ce:.4f} "
            f"val_ce_gated={gated_ce:.4f} "
            f"val_ce_forced_k{stage_k}={forced_kmax_ce:.4f}\n"
        )
        if wandb_run is not None:
            import wandb
            wandb.log(metrics, step=step)

    return metrics


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
    run_generation: bool = True,
) -> Dict[str, float]:
    """Run a validation/generation preflight and exit without optimizer steps."""
    diagnostics_requested = bool(getattr(args, "dgac_diagnostics", False))
    diagnostics_only = bool(getattr(args, "dgac_diagnostics_only", False))
    if diagnostics_only:
        if not diagnostics_requested:
            raise ValueError("--dgac_diagnostics_only requires --dgac_diagnostics")
        if halt_gate is None:
            raise ValueError("--dgac_diagnostics_only requires --use_halt_gate")
        known_kmax_ce = getattr(args, "dgac_diagnostics_forced_kmax_ce", None)
        _emit_progress(
            f"\n  [eval-only] DGAC diagnostics-only start: stage={stage_k} "
            f"samples={len(val_samples)} known_forced_kmax_ce={known_kmax_ce}"
        )
        metrics = run_dgac_diagnostics(
            model=model,
            tokenizer=tokenizer,
            halt_gate=halt_gate,
            val_samples=val_samples,
            lat_token_id=lat_token_id,
            stage_k=stage_k,
            device=device,
            args=args,
            step=step,
            wandb_run=wandb_run,
            val_ce_forced_kmax=(float(known_kmax_ce) if known_kmax_ce is not None else None),
        )
        barrier()
        return metrics

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
        if run_generation:
            mean_uwr = run_generation_callback(
                model=model,
                tokenizer=tokenizer,
                halt_gate=halt_gate,
                stage_k=stage_k,
                device=device,
                args=args,
                step=step,
                wandb_run=wandb_run,
            )
            metrics["gen_mean_uwr"] = float(mean_uwr)

    barrier()
    if diagnostics_requested:
        if halt_gate is None:
            raise ValueError("--dgac_diagnostics requires --use_halt_gate")
        metrics.update(
            run_dgac_diagnostics(
                model=model,
                tokenizer=tokenizer,
                halt_gate=halt_gate,
                val_samples=val_samples,
                lat_token_id=lat_token_id,
                stage_k=stage_k,
                device=device,
                args=args,
                step=step,
                wandb_run=wandb_run,
                val_ce_forced_kmax=float(val_ce),
            )
        )

    barrier()
    return metrics
