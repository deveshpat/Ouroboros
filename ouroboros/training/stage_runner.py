"""Stage-loop execution for training sessions."""

from __future__ import annotations

import argparse
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from ouroboros.data import build_sample_at_stage, collate_stage_k
from ouroboros.dgac import HaltGate, coconut_forward
from ouroboros.hub import _read_training_state
from ouroboros.model import (
    _rank,
    _world_size,
    all_reduce_gradients,
    barrier,
    broadcast_bool,
    get_trainable_parameters,
)
from ouroboros.training.checkpointing import (
    load_checkpoint,
    prune_epoch_checkpoints,
    save_checkpoint,
)
from ouroboros.training.evaluation import evaluate_stage, run_generation_callback

_SCRIPT_START = time.perf_counter()

def _stage_grad_clip_norm(args: argparse.Namespace, stage_k: int) -> float:
    clip_norm = float(args.max_grad_norm)
    if stage_k >= 2:
        return min(clip_norm, 0.3)
    return clip_norm


def build_optimizer_and_scheduler(
    model: nn.Module,
    halt_gate: Optional[nn.Module],
    args: argparse.Namespace,
    total_steps: int,
) -> Tuple[AdamW, LambdaLR]:
    trainable = get_trainable_parameters(model, halt_gate)
    decay    = [p for p in trainable if p.ndim >= 2]
    no_decay = [p for p in trainable if p.ndim < 2]
    optimizer = AdamW(
        [
            {"params": decay,    "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return optimizer, LambdaLR(optimizer, lr_lambda)


def make_timeout_checker(
    args: argparse.Namespace,
    rank: int,
    session_start: Optional[float] = None,
):
    if session_start is None:
        session_start = _SCRIPT_START
    timeout_limit_s  = args.session_timeout_hours * 3600.0
    timeout_buffer_s = args.graceful_exit_buffer_minutes * 60.0
    triggered = [False]

    def check() -> bool:
        if triggered[0]:
            return True
        if rank != 0:
            return False
        elapsed = time.perf_counter() - session_start
        if elapsed + timeout_buffer_s >= timeout_limit_s:
            remaining = (timeout_limit_s - elapsed) / 60.0
            print(
                f"\n  [timeout] {elapsed / 3600:.2f}h elapsed - "
                f"{remaining:.1f} min remaining "
                f"(< {args.graceful_exit_buffer_minutes:.0f} min buffer)."
            )
            triggered[0] = True
        return triggered[0]

    return check


def _best_state_for_stage(stage_dir: Path) -> Tuple[float, float, Optional[Path]]:
    best_dir = stage_dir / "best"
    if not (best_dir / "training_state.pt").exists():
        return -1.0, float("inf"), None
    state = _read_training_state(best_dir, map_location="cpu")
    val_acc = float(state.get("val_acc", -1.0) if state.get("val_acc") is not None else -1.0)
    val_ce  = float(state.get("val_ce", float("inf")) if state.get("val_ce") is not None else float("inf"))
    return val_acc, val_ce, best_dir


def _optimizer_step_sample_count(step_idx: int, batch_size: int, grad_accum: int, dataset_size: int) -> int:
    step_start = step_idx * grad_accum * batch_size
    step_end = min((step_idx + 1) * grad_accum * batch_size, dataset_size)
    return max(step_end - step_start, 0)


def run_training_stages(
    *,
    model,
    tokenizer,
    halt_gate: Optional[HaltGate],
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    lat_token_id: int,
    pad_id: int,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    session_start: float,
    wandb_run,
    stages: List[int],
    curriculum_max_stage: int,
    resume_path: Optional[Path] = None,
    resume_same_stage: bool = False,
    resume_stage: int = 0,
    resume_epoch: int = 0,
    resume_step_in_epoch: int = -1,
    global_step: int = 0,
    step_in_phase: int = 0,
    load_best_between_stages: bool = True,
    run_generation_at_stage_end: bool = True,
    run_epoch_end_val: bool = True,
) -> Dict[str, Any]:
    if not train_samples:
        raise ValueError("No training samples available for this training plan.")

    rank = _rank()
    world_size = _world_size()
    distributed = world_size > 1
    is_main = rank == 0
    local_bs = args.batch_size // world_size if distributed else args.batch_size
    check_timeout = make_timeout_checker(args, rank, session_start=session_start)

    timeout_triggered = False
    val_budget_triggered = False
    max_train_steps_triggered = False
    samples_seen_total = 0
    max_train_steps = getattr(args, "max_train_steps", None)
    max_train_steps = int(max_train_steps) if max_train_steps is not None else None
    if max_train_steps is not None and max_train_steps <= 0:
        raise ValueError("--max_train_steps must be positive when provided")
    train_steps_this_run = 0

    for stage_k in stages:
        if args.use_halt_gate:
            step_in_phase = step_in_phase if resume_same_stage and stage_k == resume_stage else 0

        n_epochs = (args.stage_0_epochs or args.epochs_per_stage) if stage_k == 0 else args.epochs_per_stage
        steps_per_epoch = max(
            1,
            math.ceil(len(train_samples) / max(args.batch_size * args.grad_accum, 1)),
        )
        total_stage_steps = max(1, n_epochs * steps_per_epoch)
        optimizer, scheduler = build_optimizer_and_scheduler(model, halt_gate, args, total_stage_steps)
        trainable_params = get_trainable_parameters(model, halt_gate)

        stage_start_epoch = 0
        stage_start_step_in_epoch = -1
        if resume_same_stage and stage_k == resume_stage and resume_path is not None:
            state = load_checkpoint(
                resume_path,
                model,
                halt_gate,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                verbose=is_main,
            )
            global_step = int(state.get("step", global_step))
            stage_start_epoch = int(state.get("epoch", resume_epoch))
            stage_start_step_in_epoch = int(state.get("step_in_epoch", resume_step_in_epoch))
            if args.use_halt_gate:
                step_in_phase = int(state.get("step_in_phase", step_in_phase))
        else:
            resume_same_stage = False

        stage_dir = output_dir / f"stage_{stage_k}"
        best_val_acc, best_val_ce, best_ckpt = _best_state_for_stage(stage_dir)

        if is_main:
            print()
            print("=" * 64)
            if args.use_halt_gate and getattr(args, "dgac_round_label", None):
                label = str(args.dgac_round_label)
                print(f"  {label}  ({stage_k} latent pass(es))")
            else:
                label = "(CoT warmup)" if stage_k == 0 else f"{stage_k} latent pass(es)"
                extra = "  + DGAC" if args.use_halt_gate else ""
                print(f"  Stage {stage_k}/{curriculum_max_stage}  {label}{extra}")
            print(f"  Epochs: {n_epochs}  Steps/epoch: {steps_per_epoch}  Total: {total_stage_steps}")
            if stage_start_epoch > 0 or stage_start_step_in_epoch >= 0:
                print(
                    f"  Resuming stage from epoch={stage_start_epoch} "
                    f"step_in_epoch={stage_start_step_in_epoch} global_step={global_step}"
                )
            print("=" * 64)

        timeout_triggered = False
        stage_val_budget_triggered = False
        val_budget_exhausted = False
        for epoch in range(stage_start_epoch, n_epochs):
            rng = random.Random(args.seed + stage_k * 100_003 + epoch)
            perm = list(range(len(train_samples)))
            rng.shuffle(perm)

            model.train()
            if halt_gate is not None:
                halt_gate.train()
            optimizer.zero_grad(set_to_none=True)
            val_budget_exhausted = False

            start_step = stage_start_step_in_epoch + 1 if epoch == stage_start_epoch else 0
            remaining_steps = max(steps_per_epoch - start_step, 0)
            pbar = (
                tqdm(total=remaining_steps, desc=f"S{stage_k}E{epoch}", dynamic_ncols=True)
                if is_main else None
            )

            for step_idx in range(start_step, steps_per_epoch):
                timeout_triggered = broadcast_bool(check_timeout() or timeout_triggered, device)
                if timeout_triggered:
                    if is_main:
                        print(f"  [timeout] saving emergency checkpoint at step {global_step} ...")
                        save_checkpoint(
                            output_dir=output_dir,
                            step=global_step,
                            epoch=epoch,
                            step_in_epoch=step_idx - 1,
                            step_in_phase=step_in_phase,
                            stage_k=stage_k,
                            model=model,
                            halt_gate=halt_gate,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            args=args,
                            val_ce=None,
                            val_acc=best_val_acc if best_val_acc >= 0 else None,
                        )
                    barrier()
                    break

                step_metrics_accum: Dict[str, float] = defaultdict(float)
                micro_count = 0
                did_backward = False
                step_samples_seen = _optimizer_step_sample_count(
                    step_idx,
                    args.batch_size,
                    args.grad_accum,
                    len(train_samples),
                )

                for micro in range(args.grad_accum):
                    global_micro_base = (step_idx * args.grad_accum + micro) * args.batch_size
                    rank_base = global_micro_base + rank * local_bs
                    batch_indices = [
                        perm[(rank_base + offset) % len(train_samples)]
                        for offset in range(local_bs)
                    ]
                    batch_raw = [train_samples[idx] for idx in batch_indices]
                    built = [
                        build_sample_at_stage(
                            tokenizer, sample, stage_k, lat_token_id, args.max_seq_len,
                        )
                        for sample in batch_raw
                    ]
                    built = [s for s in built if s is not None]
                    if not built:
                        continue
                    batch = collate_stage_k(built, pad_id)
                    loss, metrics = coconut_forward(
                        model=model,
                        batch=batch,
                        stage_k=stage_k,
                        device=device,
                        halt_gate=halt_gate if args.use_halt_gate else None,
                        args=args,
                        step_in_phase=step_in_phase,
                    )
                    (loss / args.grad_accum).backward()
                    did_backward = True
                    micro_count += 1
                    for k, v in metrics.items():
                        step_metrics_accum[k] += float(v)

                samples_seen_total += step_samples_seen

                if not did_backward:
                    optimizer.zero_grad(set_to_none=True)
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(skip="1")
                    continue

                if distributed:
                    all_reduce_gradients(trainable_params, world_size)

                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        _stage_grad_clip_norm(args, stage_k),
                    )
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                train_steps_this_run += 1
                if args.use_halt_gate:
                    step_in_phase += 1

                mean_metrics = {k: v / max(micro_count, 1) for k, v in step_metrics_accum.items()}
                mean_ce = mean_metrics.get("ce", 0.0)

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(ce=f"{mean_ce:.3f}", gn=f"{grad_norm:.3f}")

                if is_main and global_step % args.log_every == 0:
                    log_payload = {
                        "train/ce": mean_ce,
                        "train/grad_norm": grad_norm,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/stage": stage_k,
                        **{f"train/{k}": v for k, v in mean_metrics.items()},
                    }
                    tqdm.write(
                        f"  step={global_step:6d} s={stage_k} ep={epoch} "
                        f"ce={mean_ce:.4f} gn={grad_norm:.4f}"
                    )
                    if wandb_run is not None:
                        import wandb
                        wandb.log(log_payload, step=global_step)

                if max_train_steps is not None and train_steps_this_run >= max_train_steps:
                    max_train_steps_triggered = True
                    if is_main:
                        tqdm.write(
                            f"  [canary] reached --max_train_steps={max_train_steps}; "
                            "saving checkpoint and exiting before val/gen."
                        )
                        save_checkpoint(
                            output_dir=output_dir,
                            step=global_step,
                            epoch=epoch,
                            step_in_epoch=step_idx,
                            step_in_phase=step_in_phase,
                            stage_k=stage_k,
                            model=model,
                            halt_gate=halt_gate,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            args=args,
                            val_ce=None,
                            val_acc=None,
                            tag="canary",
                        )
                    barrier()
                    break

            if pbar is not None:
                pbar.close()

            stage_start_step_in_epoch = -1

            if timeout_triggered or max_train_steps_triggered:
                break

            should_budget_guard = run_epoch_end_val or run_generation_at_stage_end
            if is_main and should_budget_guard:
                elapsed = time.perf_counter() - session_start
                remaining_min = (args.session_timeout_hours * 3600 - elapsed) / 60.0
                val_budget_exhausted = check_timeout() or (remaining_min < args.val_skip_buffer_minutes)
                if val_budget_exhausted:
                    tqdm.write(
                        f"  [timeout] Skipping val/gen at epoch {epoch} - "
                        f"{remaining_min:.0f}min remaining "
                        f"(< {args.val_skip_buffer_minutes:.0f}min val budget)."
                    )
                    save_checkpoint(
                        output_dir=output_dir,
                        step=global_step,
                        epoch=epoch,
                        step_in_epoch=steps_per_epoch - 1,
                        step_in_phase=step_in_phase,
                        stage_k=stage_k,
                        model=model,
                        halt_gate=halt_gate,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        val_ce=None,
                        val_acc=None,
                    )

            val_budget_exhausted = broadcast_bool(val_budget_exhausted, device) if should_budget_guard else False
            if val_budget_exhausted:
                stage_val_budget_triggered = True
                barrier()
                break

            if is_main:
                save_checkpoint(
                    output_dir=output_dir,
                    step=global_step,
                    epoch=epoch,
                    step_in_epoch=steps_per_epoch - 1,
                    step_in_phase=step_in_phase,
                    stage_k=stage_k,
                    model=model,
                    halt_gate=halt_gate,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    val_ce=None,
                    val_acc=None,
                )
                if not run_epoch_end_val:
                    prune_epoch_checkpoints(stage_dir, args.keep_checkpoints_per_stage)

            if not run_epoch_end_val:
                barrier()
                continue

            val_ce, val_acc = evaluate_stage(
                model=model,
                val_samples=val_samples,
                tokenizer=tokenizer,
                lat_token_id=lat_token_id,
                stage_k=stage_k,
                device=device,
                args=args,
                halt_gate=halt_gate if args.use_halt_gate else None,
            )

            if is_main:
                tqdm.write(
                    f"  [val] s={stage_k} ep={epoch} "
                    f"val_ce={val_ce:.4f} val_acc={val_acc:.4f}"
                )
                if wandb_run is not None:
                    import wandb
                    wandb.log(
                        {"val/ce": val_ce, "val/acc": val_acc, "val/stage": stage_k},
                        step=global_step,
                    )

                save_checkpoint(
                    output_dir=output_dir,
                    step=global_step,
                    epoch=epoch,
                    step_in_epoch=steps_per_epoch - 1,
                    step_in_phase=step_in_phase,
                    stage_k=stage_k,
                    model=model,
                    halt_gate=halt_gate,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    val_ce=val_ce,
                    val_acc=val_acc,
                )
                prune_epoch_checkpoints(stage_dir, args.keep_checkpoints_per_stage)

                is_better = (val_acc > best_val_acc) or (
                    math.isclose(val_acc, best_val_acc) and val_ce < best_val_ce
                )
                if is_better:
                    best_val_acc = val_acc
                    best_val_ce = val_ce
                    best_ckpt = save_checkpoint(
                        output_dir=output_dir,
                        step=global_step,
                        epoch=epoch,
                        step_in_epoch=steps_per_epoch - 1,
                        step_in_phase=step_in_phase,
                        stage_k=stage_k,
                        model=model,
                        halt_gate=halt_gate,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        val_ce=val_ce,
                        val_acc=val_acc,
                        tag="best",
                    )
                    tqdm.write(f"  [best] stage={stage_k} new best acc={best_val_acc:.4f}")

            barrier()

        if timeout_triggered or stage_val_budget_triggered or max_train_steps_triggered:
            val_budget_triggered = val_budget_triggered or stage_val_budget_triggered
            break

        if is_main and run_generation_at_stage_end:
            if not (check_timeout() or val_budget_exhausted):
                run_generation_callback(
                    model=model,
                    tokenizer=tokenizer,
                    halt_gate=halt_gate if args.use_halt_gate else None,
                    stage_k=stage_k,
                    device=device,
                    args=args,
                    step=global_step,
                    wandb_run=wandb_run,
                )
            else:
                tqdm.write("  [timeout] Skipping gen callback - insufficient time.")

        barrier()

        if load_best_between_stages and best_ckpt is not None and not args.use_halt_gate:
            if is_main:
                print(
                    f"  [stage] Stage {stage_k} done. Best acc={best_val_acc:.4f}. "
                    "Loading best ckpt before advancing."
                )
            best_dir = output_dir / f"stage_{stage_k}" / "best"
            load_checkpoint(
                best_dir,
                model=model,
                halt_gate=halt_gate,
                optimizer=None,
                scheduler=None,
                device=device,
                verbose=is_main,
            )

        barrier()
        resume_same_stage = False

    return {
        "global_step": global_step,
        "step_in_phase": step_in_phase,
        "timeout_triggered": timeout_triggered,
        "val_budget_triggered": val_budget_triggered,
        "max_train_steps_triggered": max_train_steps_triggered,
        "samples_seen": int(samples_seen_total),
        "stages": list(stages),
    }
