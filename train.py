"""
train.py
========
The training core: a curriculum Trainer over model.Ouroboros, the DGAC loss
math layered on top of CE, the Kaggle-hardening callbacks, and the session
driver that runs the sequential stage loop.

Inherits, doesn't reimplement: CurriculumTrainer extends transformers.Trainer
(optimizer/scheduler/grad-accum/DDP/checkpointing/wandb all inherited — only
compute_loss, _save, evaluate, and the optimizer/scheduler factories are
overridden); OuroborosTrainingArguments extends TrainingArguments with the few
project-specific fields the callbacks read. The DGAC loss is the one piece of
real, non-inheritable math, so it lives here as private methods on the trainer.

Stdlib-only at module top so `python train.py --help` works without torch; the
heavy imports happen inside main() after bootstrap. Every Trainer-API touchpoint
is treated as version-sensitive and guarded defensively — the installed
transformers is not pinned, so no behavior is asserted from a specific version.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

# Stdlib-only at module top so `python train.py --help` works without torch.
# torch / transformers / peft are imported inside main() and the post-bootstrap
# helpers below.


# ── CLI (stdlib-only; torch-free for --help) ──────────────────────────────────

MODEL_ID = "ai21labs/AI21-Jamba-Reasoning-3B"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ouroboros curriculum + DGAC training (Jamba-Reasoning-3B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model / LoRA
    p.add_argument("--model_id", default=MODEL_ID)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--use_4bit", action="store_true", help="QLoRA 4-bit NF4 (CUDA+bitsandbytes).")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # Dataset / curriculum
    p.add_argument("--data_dir", default="data/coconut_v1")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_stage", type=int, default=None, help="Override K. None = n_steps_median from stats.json.")
    p.add_argument("--epochs_per_stage", type=int, default=3)
    p.add_argument("--stage_0_epochs", type=int, default=None)
    p.add_argument("--stochastic_depth", action="store_true",
                   help="P1a: sample n_latent ~ Uniform(1, stage_k) per sample.")
    # DGAC / HaltGate
    p.add_argument("--use_halt_gate", action="store_true")
    p.add_argument("--halt_threshold", type=float, default=0.9)
    p.add_argument("--dgac_halt_supervision_weight", type=float, default=0.1)
    p.add_argument("--dgac_halt_ce_tolerance", type=float, default=0.02)
    p.add_argument("--dgac_halt_probe_steps", default="1,2,4,stage_k")
    p.add_argument("--dgac_lambda_ponder_max", type=float, default=0.01)
    p.add_argument("--dgac_lambda_diversity", type=float, default=0.1)
    p.add_argument("--dgac_tau", type=float, default=0.9)
    p.add_argument("--dgac_warmup_steps", type=int, default=200)
    p.add_argument("--dgac_ramp_steps", type=int, default=300)
    p.add_argument("--dgac_lambda_ponder_kl", type=float, default=0.0,
                   help="P1c: PonderNet KL weight. 0 = ACT ponder (default); >0 replaces it.")
    p.add_argument("--dgac_pondernet_prior_mean", type=float, default=2.0,
                   help="P1c: geometric-prior mean halt steps.")
    # Training (map to TrainingArguments)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Base grad clip. Stages k>=2 are additionally capped at 0.3.")
    p.add_argument("--grad_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_batch_size", type=int, default=1)
    # Kaggle session management
    p.add_argument("--session_timeout_hours", type=float, default=11.0)
    p.add_argument("--graceful_exit_buffer_minutes", type=float, default=20.0)
    p.add_argument("--val_skip_buffer_minutes", type=float, default=60.0)
    # Hub checkpoint sync
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hf_token", default=None)
    p.add_argument("--hf_repo_id", default="WeirdRunner/Ouroboros")
    p.add_argument("--hf_stage_subdir", default="runs/stage3")
    p.add_argument("--keep_checkpoints_per_stage", type=int, default=2)
    # Resume / modes
    p.add_argument("--resume_from", default=None)
    p.add_argument("--resume_from_anchor", action="store_true",
                   help="Load the Hub anchor as base LoRA weights for DGAC. Requires --use_halt_gate.")
    p.add_argument("--resume_anchor_repo_id", default="WeirdRunner/Ouroboros")
    p.add_argument("--resume_anchor_subdir", default="diloco_state/anchor")
    p.add_argument("--latent_cache", action="store_true",
                   help="P0: cache latent prefixes at inference. Off during training regardless.")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--output_dir", default="runs/stage3")
    # wandb
    p.add_argument("--wandb_project", default="ouroboros-stage3-jamba")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    return p.parse_args(argv)


def resolve_hf_token(cli_value: Optional[str]) -> Optional[str]:
    """One-liner HF token resolution: CLI flag, else HF_TOKEN/HUGGINGFACE_HUB_TOKEN."""
    val = (cli_value or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    return val or None


# ── TrainingArguments subclass ────────────────────────────────────────────────

def _build_training_arguments(args: argparse.Namespace, stage_k: int, world_size: int):
    """Map CLI args + per-stage knobs to an OuroborosTrainingArguments instance.

    Lives here (not at module top) because it imports transformers; called only
    after bootstrap from inside the session driver, once per stage.
    """
    import torch
    from train_args import OuroborosTrainingArguments  # local module, see below

    per_device_bs = max(args.batch_size // max(world_size, 1), 1)
    n_epochs = (args.stage_0_epochs or args.epochs_per_stage) if stage_k == 0 else args.epochs_per_stage
    max_grad_norm = min(args.max_grad_norm, 0.3) if stage_k >= 2 else args.max_grad_norm
    amp_dtype = _amp_dtype_for_args(args)

    return OuroborosTrainingArguments(
        output_dir=str(Path(args.output_dir) / f"stage_{stage_k}"),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=max_grad_norm,
        max_steps=args.max_train_steps if args.max_train_steps else -1,
        seed=args.seed + stage_k * 100003,
        gradient_checkpointing=args.grad_checkpoint,
        logging_steps=args.log_every,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=args.keep_checkpoints_per_stage,
        save_only_model=False,
        eval_strategy="no" if (args.max_train_steps and not args.eval_only) else "epoch",
        metric_for_best_model="eval_token_acc",
        greater_is_better=True,
        load_best_model_at_end=not args.use_halt_gate,
        remove_unused_columns=False,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_ratio},
        bf16=amp_dtype == torch.bfloat16,
        fp16=amp_dtype == torch.float16,
        # HaltGate is data-dependently used: it's invoked only on batches with
        # a >=2-latent row (model.py gates the call on actual_k>0 at step>=1),
        # which can't be statically guaranteed for any stage — esp. max_stage==1
        # or small micro-batches where every row resolves to n_latent==1. So
        # under halt_gate the gate param may be unused on a given batch and DDP
        # needs find_unused_parameters=True. (stage_k==0 never runs under
        # halt_gate anyway — _plan_stages collapses to [curriculum_max_stage].)
        ddp_find_unused_parameters=bool(args.use_halt_gate),
        report_to=["wandb"] if args.wandb_mode != "disabled" else [],
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hf_repo_id,
        hub_token=args.hf_token,
        hub_private_repo=True,
        # project-specific fields the callbacks read:
        session_timeout_hours=args.session_timeout_hours,
        graceful_exit_buffer_minutes=args.graceful_exit_buffer_minutes,
        val_skip_buffer_minutes=args.val_skip_buffer_minutes,
        stage_k=stage_k,
        use_halt_gate=args.use_halt_gate,
    )


def _amp_dtype_for_args(args: argparse.Namespace):
    """Pick the autocast dtype from device capability; mirrors model._resolve_dtype."""
    import torch
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.get_device_capability(0) >= (8, 0) else torch.float16
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def _build_dgac_config(args: argparse.Namespace):
    """None when --use_halt_gate is off (DGAC disabled)."""
    from data import DGACConfig
    if not args.use_halt_gate:
        return None
    return DGACConfig(
        halt_supervision_weight=args.dgac_halt_supervision_weight,
        halt_ce_tolerance=args.dgac_halt_ce_tolerance,
        halt_probe_steps=args.dgac_halt_probe_steps,
        lambda_ponder_max=args.dgac_lambda_ponder_max,
        lambda_diversity=args.dgac_lambda_diversity,
        tau=args.dgac_tau,
        warmup_steps=args.dgac_warmup_steps,
        ramp_steps=args.dgac_ramp_steps,
        lambda_ponder_kl=args.dgac_lambda_ponder_kl,
        pondernet_prior_mean=args.dgac_pondernet_prior_mean,
    )


# ── Session driver ────────────────────────────────────────────────────────────

def run_training_session(args: argparse.Namespace, *, script_start: float) -> None:
    """The post-CLI entry point: device, data, model, resume, stage loop."""
    import torch
    from model import Ouroboros
    from data import CoconutDataset, load_canonical_dataset, get_max_stage
    from trainer import CurriculumTrainer
    from callbacks import SessionTimeoutCallback, ValBudgetGuardCallback, CheckpointSidecarCallback

    hf_token = resolve_hf_token(getattr(args, "hf_token", None))
    args._resolved_hf_token = hf_token
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    if args.resume_from_anchor and not args.use_halt_gate:
        raise ValueError("--resume_from_anchor requires --use_halt_gate.")
    if args.resume_from_anchor and not hf_token:
        raise ValueError("--resume_from_anchor requires an HF token.")

    torch.manual_seed(args.seed)

    # Accelerate owns DDP — launched via torchrun, no manual init_process_group.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = os.environ.get("RANK", "0") == "0"

    device = _select_device()
    _fail_fast_on_unsupported_cuda(device)

    if world_size > 1 and args.batch_size % world_size != 0:
        raise ValueError(f"--batch_size ({args.batch_size}) must be divisible by WORLD_SIZE ({world_size}).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb_mode == "online" and not (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY")):
        if is_main:
            print("[warn] --wandb_mode=online but no W&B key; falling back to disabled.")
        args.wandb_mode = "disabled"
    if getattr(args, "push_to_hub", False) and not hf_token:
        if is_main:
            print("[warn] --push_to_hub set but no HF token; Hub sync disabled.")
        args.push_to_hub = False

    wandb_run = _maybe_init_wandb(args, is_main)

    try:
        train_samples, val_samples, stats = load_canonical_dataset(Path(args.data_dir), args.max_samples)
        if not train_samples:
            raise ValueError("No training samples loaded. Check --data_dir / --max_samples.")
        curriculum_max_stage = get_max_stage(args.max_stage, stats)

        # Tokenizer: load once, add <|lat|>, save alongside checkpoints.
        tokenizer = _load_tokenizer(args.model_id, is_main)
        lat_token_id = tokenizer.convert_tokens_to_ids("<|lat|>")
        pad_id = tokenizer.pad_token_id or 0
        if is_main:
            (output_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(output_dir / "tokenizer")

        # The model is a real Ouroboros (in-place LoRA, fresh HaltGate) — NOT a
        # PeftModel wrap — so Ouroboros.forward runs the latent passes.
        model = Ouroboros.for_training(
            base_model_id=args.model_id, tokenizer=tokenizer,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            use_halt_gate=args.use_halt_gate, halt_threshold=args.halt_threshold,
            device=device, torch_dtype="auto",
        )
        model.config.use_latent_cache = bool(args.latent_cache)
        if is_main:
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Ouroboros for_training: trainable params={n_train}  d_model={model.config.hidden_size}")

        if args.resume_from_anchor:
            _load_hub_anchor(model, args, hf_token or "", device, is_main)

        # Decide which stages to run.
        resume_path = _resolve_resume(args, output_dir, hf_token, world_size, is_main) if not args.resume_from_anchor else None
        stages = _plan_stages(args, resume_path, curriculum_max_stage)

        if args.eval_only:
            eval_stage = stages[0] if stages else curriculum_max_stage
            _run_eval_only(model, tokenizer, val_samples, lat_token_id, pad_id, eval_stage, device, args, wandb_run)
            return

        if is_main and not stages:
            print("  No stages left to run. Nothing to do.")
        if not stages:
            return

        result = _run_stage_loop(
            model=model, tokenizer=tokenizer, train_samples=train_samples, val_samples=val_samples,
            lat_token_id=lat_token_id, pad_id=pad_id, args=args, device=device,
            output_dir=output_dir, session_start=script_start, wandb_run=wandb_run,
            stages=stages, curriculum_max_stage=curriculum_max_stage,
            resume_path=resume_path, world_size=world_size, is_main=is_main,
        )

        if is_main:
            print("\n" + "=" * 64)
            if result["timeout_triggered"] or result["val_budget_triggered"]:
                print("  [timeout] Session budget exhausted - checkpoint saved. "
                      "Re-run with the same --output_dir to auto-resume.")
            else:
                print(f"  Curriculum complete. Stages: {stages}  Global steps: {result['global_step']}")
            print("=" * 64)
    finally:
        if wandb_run is not None:
            import wandb
            wandb.finish()


def _select_device() -> torch.device:
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fail_fast_on_unsupported_cuda(device: torch.device) -> None:
    import torch
    if device.type != "cuda":
        return
    cc = torch.cuda.get_device_capability(device)
    if cc < (7, 5):
        raise SystemExit(
            f"Unsupported CUDA device: {torch.cuda.get_device_name(device)} sm{cc[0]}{cc[1]}. "
            "Use a T4-or-newer GPU."
        )


def _maybe_init_wandb(args: argparse.Namespace, is_main: bool):
    if not is_main or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
        return wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name,
            mode=args.wandb_mode, config={k: ("***" if k in ("hf_token", "_resolved_hf_token") and v else v)
                                          for k, v in vars(args).items()},
        )
    except ImportError:
        print("[warn] wandb not installed")
        return None


def _load_tokenizer(model_id: str, is_main: bool):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if "<|lat|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|lat|>"]})
    if is_main:
        print(f"  <|lat|> token id: {tokenizer.convert_tokens_to_ids('<|lat|>')}  vocab: {len(tokenizer)}")
    return tokenizer


def _load_hub_anchor(model, args, token: str, device, is_main: bool) -> None:
    """Load the Hub LoRA adapter + halt_gate.pt as base weights for DGAC."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import set_peft_model_state_dict

    repo, sub = args.resume_anchor_repo_id, args.resume_anchor_subdir.strip().strip("/")
    cache_dir = Path("/tmp/ouroboros_anchor")
    snapshot_download(repo_id=repo, token=token, local_dir=str(cache_dir),
                      allow_patterns=[f"{sub}/*"] if sub else None)
    anchor_dir = cache_dir / sub if sub else cache_dir
    if not anchor_dir.exists():
        raise FileNotFoundError(f"Hub anchor folder not found: {anchor_dir}")

    adapter_dir = anchor_dir / "adapter_model" if (anchor_dir / "adapter_model").exists() else anchor_dir
    for fname, loader in (("adapter_model.safetensors", _load_safetensors),
                          ("adapter_model.bin", lambda p: torch.load(p, map_location="cpu"))):
        path = adapter_dir / fname
        if path.exists():
            set_peft_model_state_dict(model, loader(path))
            break
    else:
        raise FileNotFoundError(f"No PEFT adapter weights in {adapter_dir}")

    if model.halt_gate is not None:
        gate_path = anchor_dir / "halt_gate.pt"
        if gate_path.exists():
            model.halt_gate.load_state_dict(torch.load(gate_path, map_location=device))
            model.halt_gate.eval()
        elif is_main:
            print(f"  [warn] no halt_gate.pt in anchor {anchor_dir}; gate stays fresh-init.")
    if is_main:
        print(f"  [DGAC] anchor loaded from {anchor_dir}; optimizer starts fresh.")


def _load_safetensors(path: Path):
    from safetensors.torch import load_file
    return load_file(str(path))


def _resolve_resume(args, output_dir: Path, hf_token, world_size: int, is_main: bool):
    """Local-first latest-checkpoint scan, with a cross-rank marker protocol."""
    import torch
    from checkpointing import find_latest_resume_checkpoint, _distributed_resume_marker
    requested = Path(args.resume_from) if args.resume_from else None
    marker = _distributed_resume_marker(output_dir)
    if is_main and marker.exists():
        marker.unlink(missing_ok=True)

    resolved = requested
    if world_size > 1:
        if is_main and resolved is None:
            resolved = find_latest_resume_checkpoint(output_dir, hf_token, args.hf_repo_id, args.hf_stage_subdir)
            if resolved:
                print(f"  [resume] discovered latest checkpoint: {resolved}")
        if is_main:
            marker.write_text(str(resolved.resolve()) if resolved else "", encoding="utf-8")
        torch.distributed.barrier()
        if not is_main:
            raw = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
            resolved = Path(raw) if raw else None
        torch.distributed.barrier()
    elif resolved is None:
        resolved = find_latest_resume_checkpoint(output_dir, hf_token, args.hf_repo_id, args.hf_stage_subdir)
        if resolved and is_main:
            print(f"  [resume] discovered latest checkpoint: {resolved}")

    if resolved is not None and not (resolved / "trainer_state.json").exists():
        if is_main:
            print(f"  [warn] resume checkpoint not found/invalid: {resolved}")
        resolved = None
    return resolved


def _plan_stages(args, resume_path, curriculum_max_stage: int) -> list[int]:
    if args.use_halt_gate:
        # DGAC runs a single stage at the curriculum max (or the resumed stage).
        return [curriculum_max_stage]
    if resume_path is not None:
        from checkpointing import read_stage_from_sidecar
        stage = read_stage_from_sidecar(resume_path)
        return list(range(stage, curriculum_max_stage + 1)) if stage is not None else [curriculum_max_stage]
    return list(range(0, curriculum_max_stage + 1))


def _run_stage_loop(*, model, tokenizer, train_samples, val_samples, lat_token_id, pad_id,
                    args, device, output_dir, session_start, wandb_run, stages,
                    curriculum_max_stage, resume_path, world_size, is_main) -> dict[str, Any]:
    import torch
    from data import CoconutDataset
    from trainer import CurriculumTrainer
    from callbacks import SessionTimeoutCallback, ValBudgetGuardCallback, CheckpointSidecarCallback
    from functools import partial

    dgac_cfg = _build_dgac_config(args)
    global_step = 0
    timeout_triggered = False
    val_budget_triggered = False

    for i, stage_k in enumerate(stages):
        train_ds = CoconutDataset(
            train_samples, tokenizer, lat_token_id, stage_k, args.max_seq_len,
            stochastic_depth=args.stochastic_depth, seed=args.seed + stage_k * 100003,
        )
        val_ds = CoconutDataset(
            val_samples, tokenizer, lat_token_id, stage_k, args.max_seq_len,
            stochastic_depth=False, seed=args.seed + stage_k * 100003 + 1,
        ) if val_samples else None

        training_args = _build_training_arguments(args, stage_k, world_size)
        training_args.session_start = session_start

        trainer = CurriculumTrainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=val_ds,
            data_collator=partial(CoconutDataset.collate, pad_id=pad_id),
            processing_class=tokenizer,
            dgac=dgac_cfg, tokenizer=tokenizer, lat_token_id=lat_token_id, pad_id=pad_id,
            callbacks=[
                SessionTimeoutCallback(),
                ValBudgetGuardCallback(),
                CheckpointSidecarCallback(args=args),
            ],
        )
        trainer._current_stage_k = stage_k
        if args.use_halt_gate:
            trainer._dgac_start_step = trainer.state.global_step if i == 0 and resume_path else 0

        if is_main:
            label = "(CoT warmup)" if stage_k == 0 else f"{stage_k} latent pass(es)"
            extra = "  + DGAC" if args.use_halt_gate else ""
            print(f"\n{'='*64}\n  Stage {stage_k}/{curriculum_max_stage}  {label}{extra}\n{'='*64}")

        resume_for_this_stage = resume_path if (i == 0 and resume_path is not None) else None
        trainer.train(resume_from_checkpoint=resume_for_this_stage)
        global_step = trainer.state.global_step

        if getattr(trainer.control, "should_training_stop", False):
            timeout_triggered = True
            break

        # Load best before advancing (non-DGAC only) — prevents drift across stages.
        if not args.use_halt_gate and stage_k != stages[-1]:
            best_ckpt = _find_best_checkpoint(output_dir / f"stage_{stage_k}")
            if best_ckpt is not None and is_main:
                print(f"  [stage] loading best ckpt {best_ckpt} before advancing.")
                from checkpointing import load_adapter_into_model
                load_adapter_into_model(model, best_ckpt, device)
            if world_size > 1:
                torch.distributed.barrier()
                for p in model.parameters():
                    if p.requires_grad:
                        torch.distributed.broadcast(p.data, src=0)

    return {"global_step": global_step, "timeout_triggered": timeout_triggered,
            "val_budget_triggered": val_budget_triggered, "stages": list(stages)}


def _find_best_checkpoint(stage_dir: Path) -> Optional[Path]:
    if not stage_dir.exists():
        return None
    # Trainer tracks best in trainer_state.json's best_model_checkpoint; fall back
    # to the highest-numbered checkpoint dir.
    state_path = stage_dir / "trainer_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            best = state.get("best_model_checkpoint")
            if best:
                p = Path(best)
                if p.exists():
                    return p
        except Exception:
            pass
    ckpts = sorted([p for p in stage_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
                   key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
    return ckpts[-1] if ckpts else None


def _run_eval_only(model, tokenizer, val_samples, lat_token_id, pad_id, stage_k, device, args, wandb_run):
    import torch
    from data import CoconutDataset
    from trainer import CurriculumTrainer
    from functools import partial

    val_ds = CoconutDataset(val_samples, tokenizer, lat_token_id, stage_k, args.max_seq_len, seed=args.seed)
    training_args = _build_training_arguments(args, stage_k, 1)
    trainer = CurriculumTrainer(
        model=model, args=training_args, eval_dataset=val_ds,
        data_collator=partial(CoconutDataset.collate, pad_id=pad_id),
        processing_class=tokenizer, dgac=_build_dgac_config(args),
        tokenizer=tokenizer, lat_token_id=lat_token_id, pad_id=pad_id,
    )
    metrics = trainer.evaluate()
    print(f"\n  [eval-only] stage={stage_k} {metrics}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    # Defer bootstrap + heavy imports until after argparse so --help is torch-free.
    from bootstrap import OuroborosBootstrap
    OuroborosBootstrap().ensure_environment()
    run_training_session(args, script_start=time.perf_counter())


if __name__ == "__main__":  # pragma: no cover
    main()
