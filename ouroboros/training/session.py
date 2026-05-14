"""Post-CLI training-session orchestration."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ouroboros.bootstrap import (
    _require_valid_diloco_worker_id,
    _resolve_github_token_common,
    _wandb_credentials_available,
)
from ouroboros.data import get_max_stage, load_canonical_dataset
from ouroboros.dgac import HaltGate
from ouroboros.hub import _resolve_hf_token
from ouroboros.model import (
    _distributed_is_initialized,
    _is_main_process,
    _local_rank,
    _rank,
    _wandb_config,
    _world_size,
    barrier,
    broadcast_parameters,
    get_trainable_parameters,
    load_model_and_tokenizer,
    set_seed,
)
from ouroboros.training.checkpointing import (
    _cleanup_distributed_resume_artifacts,
    _distributed_resume_marker,
    _resolve_resume_checkpoint_for_all_ranks,
    load_checkpoint,
    startup_hub_sync_and_prune,
)
from ouroboros.training.evaluation import run_eval_only
from ouroboros.training.stage_runner import run_training_stages
from ouroboros.training_plan import plan_training_session
from ouroboros.wandb_runtime import wandb_init_kwargs


def _select_training_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_training_session(args: argparse.Namespace, *, script_start: float) -> None:
    session_plan = plan_training_session(args)
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

    if args.diloco_mode and args.use_halt_gate and not getattr(args, "resume_from_diloco_anchor", False):
        raise ValueError(
            "--diloco_mode + --use_halt_gate is only supported for DGAC DiLoCo "
            "workers launched with --resume_from_diloco_anchor."
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

    device = _select_training_device(local_rank)

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
        from ouroboros.diloco.worker import (
            _diloco_reset_triggered_at,
            diloco_push_signal,
            diloco_read_round_state,
        )

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
                **wandb_init_kwargs(wandb),
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

        if session_plan.delegates_to_diloco:
            from ouroboros.diloco.worker import run_diloco_worker

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
            from ouroboros.diloco.worker import diloco_download_anchor

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
                halt_gate=halt_gate,
                required=True,
            )
            if is_main:
                print(
                    "  [DGAC] Anchor load complete. If the anchor contains halt_gate.pt, "
                    "HaltGate was restored; otherwise it remains zero-init. "
                    "Optimizer starts fresh unless this is eval-only."
                )
            if distributed:
                broadcast_parameters(get_trainable_parameters(model, halt_gate), src=0)
                barrier()
            if getattr(args, "eval_only", False):
                run_eval_only(
                    model=model,
                    tokenizer=tokenizer,
                    halt_gate=halt_gate,
                    val_samples=val_samples,
                    lat_token_id=lat_token_id,
                    stage_k=curriculum_max_stage,
                    device=device,
                    args=args,
                    step=0,
                    wandb_run=wandb_run,
                    run_generation=bool(args.gen_every_stage),
                )
                return
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

        if getattr(args, "eval_only", False):
            eval_stage = stages[0] if stages else (resume_stage if resume_state is not None else curriculum_max_stage)
            run_eval_only(
                model=model,
                tokenizer=tokenizer,
                halt_gate=halt_gate,
                val_samples=val_samples,
                lat_token_id=lat_token_id,
                stage_k=eval_stage,
                device=device,
                args=args,
                step=global_step,
                wandb_run=wandb_run,
                run_generation=bool(args.gen_every_stage),
            )
            _cleanup_distributed_resume_artifacts(output_dir, hub_resume_dir, distributed, is_main)
            return

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
