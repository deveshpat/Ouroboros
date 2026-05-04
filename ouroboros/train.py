"""Training loop, checkpointing, evaluation, and CLI orchestration."""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

_SCRIPT_START = time.perf_counter()

from ouroboros.bootstrap import (
    _require_valid_diloco_worker_id,
    _resolve_github_token_common,
    _wandb_credentials_available,
)
from ouroboros.data import build_sample_at_stage, collate_stage_k, get_max_stage, load_canonical_dataset
from ouroboros.dgac import HaltGate, coconut_forward, normalize_pred, _run_latent_passes
from ouroboros.diloco.worker import (
    _diloco_reset_triggered_at,
    diloco_download_anchor,
    diloco_push_signal,
    diloco_read_round_state,
    run_diloco_worker,
)
from ouroboros.hub import (
    _hub_download_checkpoint,
    _hub_upload_checkpoint,
    _list_hub_stage_checkpoints,
    _parse_stage_dir_name,
    _read_training_state,
    _resolve_hf_token,
)
from ouroboros.model import (
    _amp_dtype,
    _autocast_ctx,
    _ddp_sum,
    _distributed_is_initialized,
    _extract_last_hidden_state,
    _get_backbone,
    _get_embed_tokens,
    _get_lm_head,
    _is_main_process,
    _local_rank,
    _maybe_apply_chat_template,
    _maybe_empty_cuda_cache,
    _rank,
    _wandb_config,
    _world_size,
    all_reduce_gradients,
    barrier,
    broadcast_bool,
    broadcast_parameters,
    get_trainable_parameters,
    load_model_and_tokenizer,
    set_seed,
)

GEN_PROMPTS = [
    "What is 15 + 27?",
    "Write a Python function that returns the factorial of n.",
    "What is the capital of Japan?",
    "Explain what a neural network is in simple terms.",
    "Solve for x: 3x + 6 = 21.",
]


def _stage_grad_clip_norm(args: argparse.Namespace, stage_k: int) -> float:
    clip_norm = float(args.max_grad_norm)
    if stage_k >= 2:
        return min(clip_norm, 0.3)
    return clip_norm


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
    embed_fn = _get_embed_tokens(model)
    lm_head_fn = _get_lm_head(model)
    backbone = _get_backbone(model)
    amp_dtype = _amp_dtype(device)
    batch_size = max(int(args.val_batch_size), 1)

    local_val_samples = val_samples[rank::world_size]

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

    ce_numer, ce_denom = _ddp_sum([ce_numer, ce_denom], device)
    ce_denom = int(round(ce_denom))

    acc_samples = val_samples[:50]
    local_acc_samples = acc_samples[rank::world_size]

    n_correct = 0
    n_total = 0

    for sample in local_acc_samples:
        built = build_sample_at_stage(tokenizer, sample, stage_k, lat_token_id, args.max_seq_len)
        if built is None:
            continue
        q_len = int(built["q_len"])
        n_latent = int(built["n_latent"])
        q_ids = built["full_ids"][:q_len]
        q_tensor = q_ids.unsqueeze(0).to(device)
        ctx = embed_fn(q_tensor)
        ctx_mask = torch.ones((1, ctx.size(1)), dtype=torch.bool, device=device)
        ctx, ctx_mask, _ = _run_latent_passes(
            model=model,
            ctx=ctx,
            ctx_mask=ctx_mask,
            n_latent=n_latent,
            halt_gate=halt_gate,
            args=args,
            device=device,
            amp_dtype=amp_dtype,
        )

        generated: List[int] = []
        eos_id = tokenizer.eos_token_id
        for _ in range(args.gen_max_tokens):
            if ctx.size(1) > args.max_seq_len:
                ctx = ctx[:, -args.max_seq_len :, :]
                ctx_mask = ctx_mask[:, -args.max_seq_len :]
            with _autocast_ctx(device, amp_dtype):
                outputs = backbone(inputs_embeds=ctx, attention_mask=ctx_mask, use_cache=False)
                hidden = _extract_last_hidden_state(outputs, "eval decode")
                logits = lm_head_fn(hidden)
            next_id = int(logits[:, -1, :].argmax(-1).item())
            if eos_id is not None and next_id == eos_id:
                break
            generated.append(next_id)
            next_embed = embed_fn(torch.tensor([[next_id]], device=device))
            ctx = torch.cat([ctx, next_embed], dim=1)
            ctx_mask = torch.cat(
                [ctx_mask, torch.ones((1, 1), dtype=torch.bool, device=device)],
                dim=1,
            )

        pred = normalize_pred(tokenizer.decode(generated, skip_special_tokens=True)).strip()
        gold = str(sample.get("answer_norm", "")).strip()
        if pred and gold and (pred == gold or pred.lower() == gold.lower()):
            n_correct += 1
        n_total += 1

    n_correct, n_total = _ddp_sum([n_correct, n_total], device)
    n_correct = int(round(n_correct))
    n_total = int(round(n_total))

    model.train()
    if halt_gate is not None:
        halt_gate.train()

    return ce_numer / max(ce_denom, 1), n_correct / max(n_total, 1)


def save_checkpoint(
    output_dir: Path,
    step: int,
    epoch: int,
    step_in_epoch: int,
    step_in_phase: int,
    stage_k: int,
    model,
    halt_gate: Optional[HaltGate],
    optimizer: Optional[AdamW],
    scheduler: Optional[LambdaLR],
    args: argparse.Namespace,
    val_ce: Optional[float],
    val_acc: Optional[float],
    tag: str = "",
) -> Path:
    stage_dir = output_dir / f"stage_{stage_k}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    name = "best" if tag == "best" else f"checkpoint-{step:07d}"
    ckpt = stage_dir / name
    tmp = stage_dir / f"{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(tmp / "adapter_model"))
    if halt_gate is not None:
        torch.save(halt_gate.state_dict(), tmp / "halt_gate.pt")
    torch.save(
        {
            "stage_k": stage_k,
            "step": step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "step_in_phase": step_in_phase,
            "val_ce": val_ce,
            "val_acc": val_acc,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "use_halt_gate": args.use_halt_gate,
            "model_id": args.model_id,
        },
        tmp / "training_state.pt",
    )

    if ckpt.exists():
        shutil.rmtree(ckpt, ignore_errors=True)
    tmp.replace(ckpt)
    label = "best" if tag == "best" else "saved"
    if _is_main_process():
        print(f"  [ckpt] {label} -> {ckpt}  acc={val_acc}  ce={val_ce}")

    hf_token  = getattr(args, "_resolved_hf_token", None)
    push      = getattr(args, "push_to_hub", False)
    repo_id   = getattr(args, "hf_repo_id", "WeirdRunner/Ouroboros")
    subdir    = getattr(args, "hf_stage_subdir", "runs/stage3")
    if push and hf_token and _is_main_process():
      stage_remote_prefix = f"{subdir.strip('/')}/stage_{stage_k}"
      _hub_upload_checkpoint(ckpt, repo_id, hf_token, remote_prefix=stage_remote_prefix)
    return ckpt


def load_checkpoint(
    ckpt_dir: Path,
    model,
    halt_gate: Optional[HaltGate],
    optimizer: Optional[AdamW],
    scheduler: Optional[LambdaLR],
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, Any]:
    state = _read_training_state(ckpt_dir, map_location=device)
    adapter_dir = ckpt_dir / "adapter_model"
    if adapter_dir.exists():
        from peft import set_peft_model_state_dict

        loaded = False
        for fname in ["adapter_model.safetensors", "adapter_model.bin"]:
            fpath = adapter_dir / fname
            if not fpath.exists():
                continue
            if fname.endswith(".safetensors"):
                from safetensors.torch import load_file
                weights = load_file(str(fpath))
            else:
                weights = torch.load(fpath, map_location=device)
            set_peft_model_state_dict(model, weights)
            loaded = True
            break
        if not loaded:
            raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")

    halt_gate_path = ckpt_dir / "halt_gate.pt"
    if halt_gate is not None and halt_gate_path.exists():
        halt_gate.load_state_dict(torch.load(halt_gate_path, map_location=device))

    if optimizer is not None and state.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception as exc:
            if verbose:
                print(f"  [resume] optimizer state mismatch ({exc}); resetting optimizer.")
    if scheduler is not None and state.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(state["scheduler"])
        except Exception as exc:
            if verbose:
                print(f"  [resume] scheduler state mismatch ({exc}); resetting scheduler.")

    if verbose:
        print(
            f"  [resume] step={int(state.get('step', 0))} "
            f"epoch={int(state.get('epoch', 0))} "
            f"stage_k={int(state.get('stage_k', 0))} "
            f"val_acc={state.get('val_acc')}"
        )
    return state


def prune_epoch_checkpoints(stage_dir: Path, keep: int) -> None:
    keep = max(int(keep), 0)
    checkpoints = sorted(
        [p for p in stage_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    for old in checkpoints[:-keep] if keep > 0 else checkpoints:
        shutil.rmtree(old, ignore_errors=True)


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


def find_latest_resume_checkpoint(
    output_dir: Path,
    hf_token: Optional[str] = None,
    hf_repo_id: str = "WeirdRunner/Ouroboros",
    hf_stage_subdir: str = "runs/stage3",
) -> Optional[Path]:
    best_path: Optional[Path] = None
    best_key:  Optional[Tuple[int, int, int, int]] = None

    if output_dir.exists():
        for stage_dir in output_dir.iterdir():
            stage_k = _parse_stage_dir_name(stage_dir.name)
            if stage_k is None or not stage_dir.is_dir():
                continue
            for ckpt in stage_dir.iterdir():
                if not ckpt.is_dir() or not ckpt.name.startswith("checkpoint-"):
                    continue
                state_path = ckpt / "training_state.pt"
                if not state_path.exists():
                    continue
                try:
                    state = _read_training_state(ckpt, map_location="cpu")
                except Exception:
                    continue
                key = (
                    stage_k,
                    int(state.get("epoch", -1)),
                    int(state.get("step_in_epoch", -1)),
                    int(state.get("step", -1)),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_path = ckpt

    if best_path is not None:
        return best_path

    if not hf_token:
        return None

    if _is_main_process():
        print("  [resume] No local checkpoints found. Scanning Hub...")

    hub_candidates = _list_hub_stage_checkpoints(hf_repo_id, hf_token, hf_stage_subdir)
    hub_candidates = [(k, s, n) for k, s, n in hub_candidates if not n.endswith("/best")]
    hub_resume_dir = output_dir / ".hub_resume"

    for stage_k, step, rel_name in hub_candidates:
        ckpt_name = rel_name.split("/")[-1]
        if _is_main_process():
            print(f"  [hub] downloading {rel_name} ...")
        downloaded = _hub_download_checkpoint(
            ckpt_name=ckpt_name,
            local_dir=hub_resume_dir,
            hf_repo_id=hf_repo_id,
            hf_token=hf_token,
            remote_prefix=f"{hf_stage_subdir}/stage_{stage_k}",
        )
        if downloaded is not None and (downloaded / "training_state.pt").exists():
            if _is_main_process():
                print(f"  [hub] using {rel_name} as resume checkpoint")
            return downloaded

    return None


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
    embed_fn = _get_embed_tokens(model)
    lm_head_fn = _get_lm_head(model)
    backbone = _get_backbone(model)
    amp_dtype = _amp_dtype(device)
    mean_uwr = 0.0

    for prompt in GEN_PROMPTS:
        prefix = _maybe_apply_chat_template(tokenizer, prompt)
        q_ids = tokenizer.encode(prefix, add_special_tokens=False)
        q_tensor = torch.tensor(q_ids, device=device).unsqueeze(0)
        ctx = embed_fn(q_tensor)
        ctx_mask = torch.ones((1, ctx.size(1)), dtype=torch.bool, device=device)
        ctx, ctx_mask, actual_k = _run_latent_passes(
            model=model,
            ctx=ctx,
            ctx_mask=ctx_mask,
            n_latent=stage_k,
            halt_gate=halt_gate,
            args=args,
            device=device,
            amp_dtype=amp_dtype,
        )

        generated: List[int] = []
        eos_id = tokenizer.eos_token_id
        for _ in range(args.gen_max_tokens):
            if ctx.size(1) > args.max_seq_len:
                ctx = ctx[:, -args.max_seq_len :, :]
                ctx_mask = ctx_mask[:, -args.max_seq_len :]
            with _autocast_ctx(device, amp_dtype):
                outputs = backbone(inputs_embeds=ctx, attention_mask=ctx_mask, use_cache=False)
                hidden = _extract_last_hidden_state(outputs, "generation decode")
                logits = lm_head_fn(hidden)
            next_id = int(logits[:, -1, :].argmax(-1).item())
            if eos_id is not None and next_id == eos_id:
                break
            generated.append(next_id)
            next_embed = embed_fn(torch.tensor([[next_id]], device=device))
            ctx = torch.cat([ctx, next_embed], dim=1)
            ctx_mask = torch.cat(
                [ctx_mask, torch.ones((1, 1), dtype=torch.bool, device=device)],
                dim=1,
            )

        text = tokenizer.decode(generated, skip_special_tokens=True)
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


def _best_state_for_stage(stage_dir: Path) -> Tuple[float, float, Optional[Path]]:
    best_dir = stage_dir / "best"
    if not (best_dir / "training_state.pt").exists():
        return -1.0, float("inf"), None
    state = _read_training_state(best_dir, map_location="cpu")
    val_acc = float(state.get("val_acc", -1.0) if state.get("val_acc") is not None else -1.0)
    val_ce  = float(state.get("val_ce", float("inf")) if state.get("val_ce") is not None else float("inf"))
    return val_acc, val_ce, best_dir


def startup_hub_sync_and_prune(
    output_dir: Path,
    resume_path: Optional[Path],
    hf_token: str,
    hf_repo_id: str,
    hf_stage_subdir: str,
) -> None:
    """
    Called once at session start (rank 0 only, before training).
    1. Upload every local checkpoint that exists to Hub (best + numbered).
    2. Delete all local numbered checkpoints EXCEPT the one we are resuming from.
       Always preserve best/ dirs.

    This prevents Kaggle disk overflow across sessions. Upload failures are
    logged, but local pruning still follows the keep policy so stale numbered
    checkpoints do not accumulate across sessions.
    """
    if not _is_main_process():
        return

    all_ckpts: List[Tuple[Path, bool]] = []  # (path, is_resume)
    if output_dir.exists():
        for stage_dir in sorted(output_dir.iterdir()):
            if _parse_stage_dir_name(stage_dir.name) is None or not stage_dir.is_dir():
                continue
            for ckpt in sorted(stage_dir.iterdir()):
                if not ckpt.is_dir():
                    continue
                if not (ckpt / "training_state.pt").exists():
                    continue
                is_resume = resume_path is not None and ckpt.resolve() == resume_path.resolve()
                all_ckpts.append((ckpt, is_resume))

    if not all_ckpts:
        print("  [startup] No local checkpoints found; nothing to sync/prune.")
        return

    print(f"  [startup] Found {len(all_ckpts)} local checkpoint(s). Uploading to Hub before pruning...")
    for ckpt, is_resume in all_ckpts:
        stage_dir_name = ckpt.parent.name
        remote_prefix = f"{hf_stage_subdir.strip('/')}/{stage_dir_name}"
        ok = _hub_upload_checkpoint(ckpt, hf_repo_id, hf_token, remote_prefix=remote_prefix)
        resume_marker = "  (resume)" if is_resume else ""
        status = "✓" if ok else "✗ (upload failed)"
        print(f"  [startup]   {stage_dir_name}/{ckpt.name}{resume_marker}  {status}")

    pruned = 0
    for ckpt, is_resume in all_ckpts:
        if is_resume:
            continue
        shutil.rmtree(ckpt, ignore_errors=True)
        print(f"  [startup]   pruned {ckpt.parent.name}/{ckpt.name}")
        pruned += 1

    print(f"  [startup] Sync+prune complete. Pruned {pruned} checkpoint(s) locally.")


def _distributed_resume_marker(output_dir: Path) -> Path:
    return output_dir / ".resolved_resume_path.txt"


def _resolve_resume_checkpoint_for_all_ranks(
    *,
    output_dir: Path,
    requested_resume: Optional[Path],
    hf_token: Optional[str],
    hf_repo_id: str,
    hf_stage_subdir: str,
    distributed: bool,
    is_main: bool,
) -> Optional[Path]:
    marker_path = _distributed_resume_marker(output_dir)
    if is_main and marker_path.exists():
        marker_path.unlink(missing_ok=True)

    resolved: Optional[Path] = requested_resume
    if distributed:
        if is_main and resolved is None:
            resolved = find_latest_resume_checkpoint(
                output_dir,
                hf_token=hf_token,
                hf_repo_id=hf_repo_id,
                hf_stage_subdir=hf_stage_subdir,
            )
            if resolved is not None:
                print(f"  [resume] discovered latest checkpoint: {resolved}")
        if is_main:
            marker_path.write_text(
                str(resolved.resolve()) if resolved is not None else "",
                encoding="utf-8",
            )
        barrier()
        if not is_main:
            raw = marker_path.read_text(encoding="utf-8").strip() if marker_path.exists() else ""
            resolved = Path(raw) if raw else None
        barrier()
        return resolved

    if resolved is None:
        resolved = find_latest_resume_checkpoint(
            output_dir,
            hf_token=hf_token,
            hf_repo_id=hf_repo_id,
            hf_stage_subdir=hf_stage_subdir,
        )
        if resolved is not None and is_main:
            print(f"  [resume] discovered latest checkpoint: {resolved}")
    return resolved


def _cleanup_distributed_resume_artifacts(
    output_dir: Path,
    hub_resume_dir: Path,
    distributed: bool,
    is_main: bool,
) -> None:
    if distributed:
        barrier()
    if is_main:
        marker_path = _distributed_resume_marker(output_dir)
        marker_path.unlink(missing_ok=True)
        if hub_resume_dir.exists():
            shutil.rmtree(hub_resume_dir, ignore_errors=True)
    if distributed:
        barrier()


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
    samples_seen_total = 0

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

            if pbar is not None:
                pbar.close()

            stage_start_step_in_epoch = -1

            if timeout_triggered:
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

        if timeout_triggered or stage_val_budget_triggered:
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
        "samples_seen": int(samples_seen_total),
        "stages": list(stages),
    }


def run_cli(args: argparse.Namespace, *, script_start: float) -> None:
    if getattr(args, "resume_from_diloco_anchor", False) and not args.use_halt_gate:
        raise ValueError(
            "--resume_from_diloco_anchor requires --use_halt_gate. "
            "This flag is only valid for Phase 3.4 DGAC training."
        )
    if args.diloco_mode:
        args.diloco_worker_id = _require_valid_diloco_worker_id(args.diloco_worker_id)
        os.environ["DILOCO_WORKER_ID"] = args.diloco_worker_id
        os.environ.setdefault("OUROBOROS_DILOCO_WORKER_ID", args.diloco_worker_id)
        os.environ.setdefault("WORKER_ID", args.diloco_worker_id)

    hf_token = _resolve_hf_token(getattr(args, "hf_token", None))
    args._resolved_hf_token = hf_token
    if getattr(args, "resume_from_diloco_anchor", False) and not hf_token:
        raise ValueError(
            "--resume_from_diloco_anchor requires an HF token. "
            "Provide --hf_token, set HF_TOKEN, or define a Kaggle secret."
        )
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
    if args.diloco_mode and not hf_token:
        raise ValueError(
            "HF token required for DiLoCo mode. Provide --hf_token, set HF_TOKEN / "
            "HUGGINGFACE_HUB_TOKEN, or define a Kaggle/Colab secret named HF_TOKEN."
        )

    github_token = _resolve_github_token_common()
    args._resolved_github_token = github_token
    if github_token:
        os.environ["GITHUB_TOKEN"] = github_token
        os.environ.setdefault("GH_TOKEN", github_token)

    set_seed(args.seed)

    rank = _rank()
    world_size = _world_size()
    local_rank = _local_rank()
    distributed = world_size > 1
    is_main = rank == 0

    if args.diloco_mode and args.use_halt_gate:
        raise ValueError(
            "--diloco_mode and --use_halt_gate should not be combined. "
            "DiLoCo syncs LoRA adapters only; DGAC should stay on the sequential path."
        )

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        init_kwargs = dict(
            backend=backend,
            init_method="env://",
            timeout=timedelta(hours=4),
        )
        if torch.cuda.is_available():
            try:
                torch.distributed.init_process_group(
                    **init_kwargs,
                    device_id=torch.device("cuda", local_rank),
                )
            except TypeError:
                torch.distributed.init_process_group(**init_kwargs)
        else:
            torch.distributed.init_process_group(**init_kwargs)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if distributed and args.batch_size % world_size != 0:
        raise ValueError(
            f"--batch_size ({args.batch_size}) must be divisible by WORLD_SIZE ({world_size})"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_start = script_start

    if args.wandb_mode == "online" and not _wandb_credentials_available():
        if is_main:
            print(
                "[warn] --wandb_mode=online requested but no W&B credentials were "
                "found; falling back to disabled."
            )
        args.wandb_mode = "disabled"

    # ── DiLoCo GPU architecture guard ────────────────────────────────────────
    # Safety net: even with --accelerator NvidiaTeslaT4 in the push command,
    # Kaggle may silently fall back to P100 (sm60) if T4 quota is exhausted.
    # P100 → bootstrap attempts 30+ min source compile → kernel cancellation.
    # Fast-fail: detect wrong GPU, reset triggered_at=0 (coordinator
    # immediately re-dispatches via existing triggered_at=0 branch), exit.
    if args.diloco_mode and device.type == "cuda" and args.diloco_worker_id:
        _cc = torch.cuda.get_device_capability(device)
        if _cc < (7, 5):  # sm75 = T4 (minimum for cached Mamba wheels on Hub)
            _gpu_name = torch.cuda.get_device_name(device)
            if is_main:
                print(
                    f"\n  [diloco] GPU MISMATCH: {_gpu_name} sm{_cc[0]}{_cc[1]} detected "
                    f"(requires sm75+/T4 for cached Mamba wheels).\n"
                    f"  [diloco] Resetting triggered_at=0 for immediate coordinator re-dispatch."
                )
                _diloco_reset_triggered_at(
                    hf_token or "",
                    getattr(args, "diloco_state_repo", "WeirdRunner/Ouroboros"),
                )
                _round_state_quick = diloco_read_round_state(
                    hf_token or "",
                    getattr(args, "diloco_state_repo", "WeirdRunner/Ouroboros"),
                )
                _github_token = _resolve_github_token_common()
                if _github_token and getattr(args, "diloco_signal_repo", ""):
                    diloco_push_signal(
                        args.diloco_worker_id,
                        int(_round_state_quick.get("stage_k", 0)),
                        int(_round_state_quick.get("round_n", 0)),
                        _github_token,
                        args.diloco_signal_repo,
                    )
            if distributed:
                barrier()
            if distributed and _distributed_is_initialized():
                torch.distributed.destroy_process_group()
            sys.exit(0)
    # ─────────────────────────────────────────────────────────────────────────

    if getattr(args, "push_to_hub", False) and not hf_token:
        if _is_main_process():
            print("[warn] --push_to_hub set but no HF token found; Hub sync disabled.")
        args.push_to_hub = False

    wandb_run = None
    if is_main and args.wandb_mode != "disabled" and not args.diloco_mode:
        # Sequential curriculum path: init wandb normally.
        # DiLoCo path defers init to run_diloco_worker() where stage_k/round_n are known.
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config=_wandb_config(args),
            )
        except ImportError:
            print("[warn] wandb not installed")

    try:
        train_samples, val_samples, stats = load_canonical_dataset(Path(args.data_dir), args.max_samples)
        if not train_samples:
            raise ValueError("No training samples were loaded. Check --data_dir / --max_samples.")
        curriculum_max_stage = get_max_stage(args, stats)

        model, tokenizer, d_model, lat_token_id = load_model_and_tokenizer(args, device)
        pad_id = tokenizer.pad_token_id or 0
        if is_main:
            tokenizer_dir = output_dir / "tokenizer"
            tokenizer.save_pretrained(tokenizer_dir)

        halt_gate: Optional[HaltGate] = None
        if args.use_halt_gate:
            halt_gate = HaltGate(d_model).to(device=device, dtype=torch.float32)
            if is_main:
                n_params = sum(p.numel() for p in halt_gate.parameters())
                print(f"  DGAC HaltGate: d_model={d_model}  params={n_params}")

        if args.diloco_mode:
            run_diloco_worker(
                model=model,
                tokenizer=tokenizer,
                halt_gate=halt_gate,
                train_samples=train_samples,
                val_samples=val_samples,
                curriculum_max_stage=curriculum_max_stage,
                lat_token_id=lat_token_id,
                pad_id=pad_id,
                args=args,
                device=device,
                output_dir=output_dir,
                session_start=session_start,
                wandb_run=wandb_run,
                hf_token=hf_token or "",
            )
            return

        # ── DiLoCo anchor load for DGAC ──────────────────────────────────────────────
        # When --use_halt_gate + --resume_from_diloco_anchor are set together, load the
        # DiLoCo stage-10 aggregate from Hub as the base LoRA weights, then run
        # training directly and return. The outer try/finally handles all cleanup
        # (destroy_process_group, wandb.finish). Do NOT call them here.
        if getattr(args, "resume_from_diloco_anchor", False):
            if not args.use_halt_gate:
                raise ValueError(
                    "--resume_from_diloco_anchor requires --use_halt_gate. "
                    "This flag is only valid for Phase 3.4 DGAC training."
                )
            if not hf_token:
                raise ValueError(
                    "--resume_from_diloco_anchor requires an HF token. "
                    "Provide --hf_token, set HF_TOKEN, or define a Kaggle secret."
                )
            anchor_repo = getattr(args, "diloco_state_repo", "WeirdRunner/Ouroboros")
            if is_main:
                print(
                    f"\n  [DGAC] Loading DiLoCo anchor from {anchor_repo}/diloco_state/anchor "
                    "as base weights for Phase 3.4 DGAC training."
                )
            diloco_download_anchor(
                model,
                hf_token,
                anchor_repo,
                "diloco_state/anchor",
                device,
            )
            if is_main:
                print(
                    "  [DGAC] Anchor loaded. HaltGate at zero-init. "
                    "gate_stage will default to curriculum_max_stage. Optimizer starts fresh."
                )
            if distributed:
                broadcast_parameters(get_trainable_parameters(model, halt_gate), src=0)
                barrier()
            run_training_stages(
                model=model,
                tokenizer=tokenizer,
                halt_gate=halt_gate,
                train_samples=train_samples,
                val_samples=val_samples,
                lat_token_id=lat_token_id,
                pad_id=pad_id,
                args=args,
                device=device,
                output_dir=output_dir,
                session_start=session_start,
                wandb_run=wandb_run,
                stages=[curriculum_max_stage],
                curriculum_max_stage=curriculum_max_stage,
                resume_path=None,
                resume_same_stage=False,
                resume_stage=curriculum_max_stage,
                resume_epoch=0,
                resume_step_in_epoch=-1,
                global_step=0,
                step_in_phase=0,
                load_best_between_stages=False,
                run_generation_at_stage_end=bool(args.gen_every_stage),
                run_epoch_end_val=True,
            )
            return  # finally in main() handles destroy_process_group and wandb.finish
        # ─────────────────────────────────────────────────────────────────────────────
        requested_resume_path: Optional[Path] = Path(args.resume_from) if args.resume_from else None
        hub_resume_dir = output_dir / ".hub_resume"
        resume_path = _resolve_resume_checkpoint_for_all_ranks(
            output_dir=output_dir,
            requested_resume=requested_resume_path,
            hf_token=hf_token,
            hf_repo_id=getattr(args, "hf_repo_id", "WeirdRunner/Ouroboros"),
            hf_stage_subdir=getattr(args, "hf_stage_subdir", "runs/stage3"),
            distributed=distributed,
            is_main=is_main,
        )

        if resume_path is not None and not (resume_path / "training_state.pt").exists():
            if is_main:
                print(f"  [warn] resume checkpoint not found: {resume_path}")
            resume_path = None
            if distributed and is_main:
                _distributed_resume_marker(output_dir).write_text("", encoding="utf-8")
        if distributed:
            barrier()
            if not is_main and resume_path is None:
                raw = _distributed_resume_marker(output_dir).read_text(encoding="utf-8").strip() if _distributed_resume_marker(output_dir).exists() else ""
                resume_path = Path(raw) if raw else None
            barrier()

        if hf_token and getattr(args, "push_to_hub", False) and is_main:
            startup_hub_sync_and_prune(
                output_dir=output_dir,
                resume_path=resume_path,
                hf_token=hf_token,
                hf_repo_id=getattr(args, "hf_repo_id", "WeirdRunner/Ouroboros"),
                hf_stage_subdir=getattr(args, "hf_stage_subdir", "runs/stage3"),
            )
        barrier()

        resume_state: Optional[Dict[str, Any]] = None
        resume_same_stage = False
        resume_stage = 0
        resume_epoch = 0
        resume_step_in_epoch = -1
        global_step = 0
        step_in_phase = 0

        if resume_path is not None:
            resume_state = load_checkpoint(
                resume_path,
                model,
                halt_gate,
                optimizer=None,
                scheduler=None,
                device=device,
                verbose=is_main,
            )

            resume_stage = int(resume_state.get("stage_k", 0))
            global_step = int(resume_state.get("step", 0))
            if args.use_halt_gate:
                resume_same_stage = bool(resume_state.get("use_halt_gate", False) and resume_path.name != "best")
                if resume_same_stage:
                    resume_epoch = int(resume_state.get("epoch", 0))
                    resume_step_in_epoch = int(resume_state.get("step_in_epoch", -1))
                    step_in_phase = int(resume_state.get("step_in_phase", 0))
            else:
                resume_same_stage = resume_path.name != "best"
                if resume_same_stage:
                    resume_epoch = int(resume_state.get("epoch", 0))
                    resume_step_in_epoch = int(resume_state.get("step_in_epoch", -1))

        if args.use_halt_gate:
            gate_stage = resume_stage if resume_state is not None else curriculum_max_stage
            stages = [gate_stage]
            if resume_state is None and is_main:
                print(
                    "  [warn] --use_halt_gate without --resume_from: "
                    "training DGAC from current weights at Stage K."
                )
        else:
            if resume_state is not None and resume_path is not None and resume_path.name == "best":
                start_stage = resume_stage + 1
            else:
                start_stage = resume_stage if resume_state is not None else 0
            stages = list(range(start_stage, curriculum_max_stage + 1))

        if distributed:
            broadcast_parameters(get_trainable_parameters(model, halt_gate), src=0)

        if is_main and not stages:
            print("  No stages left to run. Nothing to do.")
        if not stages:
            _cleanup_distributed_resume_artifacts(output_dir, hub_resume_dir, distributed, is_main)
            return

        result = run_training_stages(
            model=model,
            tokenizer=tokenizer,
            halt_gate=halt_gate,
            train_samples=train_samples,
            val_samples=val_samples,
            lat_token_id=lat_token_id,
            pad_id=pad_id,
            args=args,
            device=device,
            output_dir=output_dir,
            session_start=session_start,
            wandb_run=wandb_run,
            stages=stages,
            curriculum_max_stage=curriculum_max_stage,
            resume_path=resume_path,
            resume_same_stage=resume_same_stage,
            resume_stage=resume_stage,
            resume_epoch=resume_epoch,
            resume_step_in_epoch=resume_step_in_epoch,
            global_step=global_step,
            step_in_phase=step_in_phase,
            load_best_between_stages=not args.use_halt_gate,
            run_generation_at_stage_end=bool(args.gen_every_stage),
            run_epoch_end_val=True,
        )

        _cleanup_distributed_resume_artifacts(output_dir, hub_resume_dir, distributed, is_main)

        if is_main:
            print("\n" + "=" * 64)
            if result["timeout_triggered"] or result["val_budget_triggered"]:
                if result["val_budget_triggered"] and not result["timeout_triggered"]:
                    print(
                        "  [timeout] Remaining session time fell below "
                        f"--val_skip_buffer_minutes ({args.val_skip_buffer_minutes:.0f} min) - checkpoint saved."
                    )
                else:
                    print("  [timeout] Session budget exhausted - checkpoint saved.")
                print("  Re-run the same command with the same --output_dir to auto-resume.")
            else:
                print(f"  Curriculum complete. Stages: {stages}  Global steps: {result['global_step']}")
                if not args.use_halt_gate:
                    best_k_dir = output_dir / f"stage_{curriculum_max_stage}" / "best"
                    print(
                        "  Phase 3.4 (DGAC):\n"
                        f"    python jamba_coconut_finetune.py --use_halt_gate "
                        f"--resume_from {best_k_dir} "
                        f"--output_dir {args.output_dir}_dgac [...]"
                    )
            print("=" * 64)

    finally:
        if distributed and _distributed_is_initialized():
            torch.distributed.destroy_process_group()
        if wandb_run is not None:
            import wandb
            wandb.finish()