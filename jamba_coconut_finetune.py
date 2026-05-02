#!/usr/bin/env python3
"""Thin CLI entrypoint for Jamba Coconut-Ouroboros fine-tuning."""

import time

# Wall-clock start for Kaggle timeout accounting. This must be captured before
# ensure_environment() so dependency install and model load time are included.
_SCRIPT_START = time.perf_counter()

import argparse
import os
from typing import Optional

# Set critical env vars BEFORE any torch/nccl import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", str(4 * 3600))
os.environ.setdefault("NCCL_TIMEOUT", str(4 * 3600))

from ouroboros.bootstrap import ensure_environment

ensure_environment()

from ouroboros.diloco.shared import WORKER_IDS, normalize_text
from ouroboros.model import MODEL_ID


def _parse_diloco_worker_id_cli(value: str) -> str:
    worker_id = normalize_text(value, uppercase=True)
    if worker_id is None:
        raise argparse.ArgumentTypeError("DiLoCo worker id cannot be empty")
    return worker_id


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    action = getattr(argparse, "BooleanOptionalAction", None)
    if action is not None:
        parser.add_argument(name, action=action, default=default, help=help_text)
    elif default:
        parser.add_argument(name, action="store_true", default=default, help=help_text)
    else:
        parser.add_argument(name, action="store_false", default=default, help=help_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jamba Reasoning 3B Coconut-Ouroboros fine-tuning", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add("--model_id", default=MODEL_ID); add("--max_seq_len", type=int, default=1024)
    add("--use_4bit", action="store_true", help="QLoRA (4-bit NF4). Requires CUDA + bitsandbytes."); add("--lora_r", type=int, default=32); add("--lora_alpha", type=int, default=64); add("--lora_dropout", type=float, default=0.05)
    add("--data_dir", default="data/coconut_v1"); add("--max_samples", type=int, default=None)
    add("--max_stage", type=int, default=None, help="Override K. None = read n_steps_median from stats.json."); add("--epochs_per_stage", type=int, default=3); add("--stage_0_epochs", type=int, default=None)
    add("--use_halt_gate", action="store_true"); add("--resume_from_diloco_anchor", action="store_true", help="Load DiLoCo anchor weights from diloco_state/anchor/ on Hub as the base model for DGAC (--use_halt_gate) training. Requires --use_halt_gate, --hf_token, and --diloco_state_repo. Bypasses find_latest_resume_checkpoint(); halt_gate starts from zero-init. Use instead of --resume_from when curriculum was trained via DiLoCo mode.")
    add("--halt_threshold", type=float, default=0.5); add("--dgac_lambda_ponder_max", type=float, default=0.01); add("--dgac_lambda_diversity", type=float, default=0.1); add("--dgac_tau", type=float, default=0.9); add("--dgac_warmup_steps", type=int, default=200); add("--dgac_ramp_steps", type=int, default=300)
    add("--batch_size", type=int, default=2); add("--grad_accum", type=int, default=8); add("--lr", type=float, default=1e-4); add("--min_lr_ratio", type=float, default=0.1); add("--warmup_steps", type=int, default=50); add("--weight_decay", type=float, default=0.01)
    add("--max_grad_norm", type=float, default=1.0, help="Base gradient clip norm. Stages k>=2 are additionally capped at 0.3."); _add_bool_arg(parser, "--grad_checkpoint", True, "Enable gradient checkpointing."); add("--seed", type=int, default=42)
    add("--session_timeout_hours", type=float, default=11.0); add("--graceful_exit_buffer_minutes", type=float, default=20.0); add("--val_skip_buffer_minutes", type=float, default=60.0, help="Skip val+gen if remaining session time is below this threshold (minutes). With DDP val on Dual T4 (val_batch_size=2, 50 acc samples), val takes ~37min. Default 60min provides a 23min safety margin.")
    add("--push_to_hub", action="store_true", help="Push checkpoints to HF Hub after each epoch save."); add("--hf_token", default=None, help="HF write token. Falls back to HF_TOKEN env var."); add("--hf_repo_id", default="WeirdRunner/Ouroboros", help="HF model repo to sync checkpoints to."); add("--hf_stage_subdir", default="runs/stage3", help="Remote subdirectory inside the HF repo for Stage 3 checkpoints.")
    add("--diloco_mode", action="store_true", help="Enable DiLoCo parallel training mode."); add("--diloco_worker_id", type=_parse_diloco_worker_id_cli, default=None, choices=list(WORKER_IDS), help="This worker's identity. If omitted in --diloco_mode, falls back to DILOCO_WORKER_ID env / notebook secret.")
    add("--diloco_outer_lr", type=float, default=0.7, help="Outer SGD learning rate for DiLoCo aggregation. Default: 0.7 (DiLoCo paper)."); add("--diloco_min_workers", type=int, default=2, help="Minimum workers needed for coordinator to aggregate (default: 2 of 3)."); add("--diloco_state_repo", default="WeirdRunner/Ouroboros", help="HF Hub repo used as shared state store."); add("--diloco_signal_repo", default="deveshpat/Ouroboros", help="GitHub repo to push coordinator trigger signals to."); add("--diloco_run_val", action="store_true", help="Run val pass before training begins (used by the first worker of a new stage).")
    add("--resume_from", type=str, default=None); add("--output_dir", default="runs/stage3"); add("--keep_checkpoints_per_stage", type=int, default=2)
    add("--log_every", type=int, default=20); add("--val_batch_size", type=int, default=1, help="Batch size for val forward passes. Keep at 1 to avoid OOM."); _add_bool_arg(parser, "--gen_every_stage", True, "Run generation callback at stage end."); add("--gen_max_tokens", type=int, default=200)
    add("--wandb_project", default="ouroboros-stage3-jamba"); add("--wandb_entity", default=None); add("--wandb_run_name", default=None); add("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    return parser.parse_args()


def main() -> None:
    from ouroboros.train import run_cli
    run_cli(parse_args(), script_start=_SCRIPT_START)


if __name__ == "__main__":
    main()
