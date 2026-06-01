"""Post-CLI training-session orchestration."""

from __future__ import annotations

import argparse
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ouroboros.bootstrap.runtime import _wandb_credentials_available
from ouroboros.coconut.data import get_max_stage, load_canonical_dataset
from ouroboros.coconut.dgac import HaltGate
from ouroboros.models import barrier, load_model_and_tokenizer
from ouroboros.models.loading import (
    _distributed_is_initialized,
    _local_rank,
    _rank,
    _wandb_config,
    _world_size,
    broadcast_parameters,
    get_trainable_parameters,
    set_seed,
)
from ouroboros.utils.runtime_env import is_main_process, resolve_hf_token
from ouroboros.coconut.checkpointing import (
    _cleanup_distributed_resume_artifacts,
    _distributed_resume_marker,
    _resolve_resume_checkpoint_for_all_ranks,
    load_checkpoint,
    startup_hub_sync_and_prune,
)
from ouroboros.coconut.evaluation import run_eval_only
from ouroboros.coconut.stage_runner import run_training_stages
from ouroboros.coconut.training_plan import plan_training_session
from ouroboros.utils.wandb_runtime import wandb_init_kwargs


def _truthy_anchor_resume(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "resume_from_anchor", False))


def _fail_fast_on_unsupported_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        return
    cc = torch.cuda.get_device_capability(device)
    if cc >= (7, 5):
        return
    gpu_name = torch.cuda.get_device_name(device)
    raise SystemExit(
        f"Unsupported CUDA device for this cached Jamba/Mamba runtime: {gpu_name} sm{cc[0]}{cc[1]}. "
        "Use a T4-or-newer GPU or prepare matching cached wheels before launching training."
    )


def _load_hub_anchor_into_model(
    *,
    model,
    halt_gate: Optional[HaltGate],
    repo_id: str,
    subfolder: str,
    token: str,
    device: torch.device,
) -> Path:
    from huggingface_hub import snapshot_download
    from peft import set_peft_model_state_dict

    subfolder = subfolder.strip().strip("/")
    cache_dir = Path("/tmp/ouroboros_anchor")
    allow_patterns = [f"{subfolder}/*"] if subfolder else None
    snapshot_root = Path(
        snapshot_download(
            repo_id=repo_id,
            token=token,
            local_dir=str(cache_dir),
            allow_patterns=allow_patterns,
        )
    )
    anchor_dir = snapshot_root / subfolder if subfolder else snapshot_root
    if not anchor_dir.exists():
        raise FileNotFoundError(f"Hub anchor folder not found after download: {anchor_dir}")

    adapter_dir = anchor_dir / "adapter_model" if (anchor_dir / "adapter_model").exists() else anchor_dir
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        adapter_weights = load_file(str(safetensors_path), device="cpu")
    elif bin_path.exists():
        adapter_weights = torch.load(bin_path, map_location=device)
    else:
        raise FileNotFoundError(f"No PEFT adapter weights found in {adapter_dir}")
    set_peft_model_state_dict(model, adapter_weights)

    gate_path = anchor_dir / "halt_gate.pt"
    if halt_gate is not None:
        if not gate_path.exists():
            raise FileNotFoundError(f"Expected halt_gate.pt in Hub anchor, but none was found at {gate_path}")
        halt_gate.load_state_dict(torch.load(gate_path, map_location=device))
        halt_gate.eval()
    return anchor_dir


def _select_training_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_training_session(args: argparse.Namespace, *, script_start: float) -> None:
    plan_training_session(args)
    resume_from_anchor = _truthy_anchor_resume(args)
    if resume_from_anchor and not args.use_halt_gate:
        raise ValueError(
            "--resume_from_anchor requires --use_halt_gate. "
            "This flag is only valid for Phase 3.4 DGAC training."
        )

    hf_token = resolve_hf_token(getattr(args, "hf_token", None))
    args._resolved_hf_token = hf_token
    if resume_from_anchor and not hf_token:
        raise ValueError(
            "--resume_from_anchor requires an HF token. "
            "Provide --hf_token, set HF_TOKEN, or define a Kaggle secret."
        )
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    set_seed(args.seed)

    rank = _rank()
    world_size = _world_size()
    local_rank = _local_rank()
    distributed = world_size > 1
    is_main = rank == 0

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
    _fail_fast_on_unsupported_cuda(device)

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

    if getattr(args, "push_to_hub", False) and not hf_token:
        if is_main_process():
            print("[warn] --push_to_hub set but no HF token found; Hub sync disabled.")
        args.push_to_hub = False

    wandb_run = None
    if is_main and args.wandb_mode != "disabled":
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

        if resume_from_anchor:
            anchor_repo = getattr(args, "resume_anchor_repo_id", "WeirdRunner/Ouroboros")
            anchor_subdir = getattr(args, "resume_anchor_subdir", "diloco_state/anchor")
            if is_main:
                print(
                    f"\n  [DGAC] Loading Hub anchor from {anchor_repo}/{anchor_subdir} "
                    "as base weights for Phase 3.4 DGAC training."
                )
            if is_main:
                anchor_dir = _load_hub_anchor_into_model(
                    model=model,
                    halt_gate=halt_gate,
                    repo_id=anchor_repo,
                    subfolder=anchor_subdir,
                    token=hf_token or "",
                    device=device,
                )
                print(
                    f"  [DGAC] Anchor load complete from {anchor_dir}. Optimizer starts fresh "
                    "unless this is eval-only."
                )
            if distributed:
                barrier()
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
                run_epoch_end_val=True,
            )
            return  # finally in main() handles destroy_process_group and wandb.finish

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
                        f"    python -m ouroboros.coconut --use_halt_gate "
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
