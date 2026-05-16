"""Bootstrap-safe CLI surface for the Ouroboros training entrypoint.

This module intentionally imports only the Python standard library. It is safe to
import before dependency bootstrap, CUDA probing, or torch/transformers imports.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional, Sequence, Tuple

MODEL_ID = "ai21labs/AI21-Jamba-Reasoning-3B"
_VALID_DILOCO_WORKER_IDS: Tuple[str, ...] = ("A", "B", "C")


def _normalize_optional_text(value: Optional[Any], *, uppercase: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.upper() if uppercase else text


def _parse_diloco_worker_id_cli(value: str) -> str:
    worker_id = _normalize_optional_text(value, uppercase=True)
    if worker_id is None:
        raise argparse.ArgumentTypeError("DiLoCo worker id cannot be empty")
    return worker_id


def bootstrap_free_help_text() -> str:
    """Return stable help text without importing bootstrap, torch, or ML deps."""
    return (
        "usage: -m ouroboros.coconut [-h] [--model_id MODEL_ID] "
        "[--max_seq_len MAX_SEQ_LEN] [--use_4bit] [--data_dir DATA_DIR] "
        "[--max_stage MAX_STAGE] [--epochs_per_stage EPOCHS_PER_STAGE] "
        "[--batch_size BATCH_SIZE] [--grad_accum GRAD_ACCUM] "
        "[--diloco_mode] [--diloco_worker_id {A,B,C}] "
        "[--val_batch_size VAL_BATCH_SIZE] "
        "[--gen_every_stage | --no-gen_every_stage] [--wandb_mode {online,offline,disabled}]\n\n"
        "Jamba Reasoning 3B Coconut-Ouroboros fine-tuning\n\n"
        "Key options preserved from the source monolith:\n"
        "  --model_id MODEL_ID\n"
        "  --max_seq_len MAX_SEQ_LEN\n"
        "  --use_4bit\n"
        "  --data_dir DATA_DIR\n"
        "  --max_stage MAX_STAGE\n"
        "  --epochs_per_stage EPOCHS_PER_STAGE\n"
        "  --stage_0_epochs STAGE_0_EPOCHS\n"
        "  --use_halt_gate\n"
        "  --batch_size BATCH_SIZE\n"
        "  --grad_accum GRAD_ACCUM\n"
        "  --max_grad_norm MAX_GRAD_NORM\n"
        "  --session_timeout_hours SESSION_TIMEOUT_HOURS\n"
        "  --graceful_exit_buffer_minutes GRACEFUL_EXIT_BUFFER_MINUTES\n"
        "  --push_to_hub\n"
        "  --hf_token HF_TOKEN\n"
        "  --hf_repo_id HF_REPO_ID\n"
        "  --diloco_mode\n"
        "  --diloco_worker_id {A,B,C}\n"
        "  --resume_from RESUME_FROM\n"
        "  --resume_from_diloco_anchor\n"
        "  --latent_cache\n"
        "  --mac_mps_latent_cache\n"
        "  --eval_only\n"
        "  --dgac_diagnostics\n"
        "  --dgac_diagnostics_only\n"
        "  --dgac_diagnostics_batch_size DGAC_DIAGNOSTICS_BATCH_SIZE\n"
        "  --dgac_diagnostics_forced_kmax_ce DGAC_DIAGNOSTICS_FORCED_KMAX_CE\n"
        "  --dgac_halt_supervision_weight DGAC_HALT_SUPERVISION_WEIGHT\n"
        "  --dgac_halt_ce_tolerance DGAC_HALT_CE_TOLERANCE\n"
        "  --dgac_halt_probe_steps DGAC_HALT_PROBE_STEPS\n"
        "  --output_dir OUTPUT_DIR\n"
        "  --profile_training_timing\n"
        "  --val_batch_size VAL_BATCH_SIZE\n"
        "  --eval_progress_every EVAL_PROGRESS_EVERY\n"
        "  --gen_every_stage, --no-gen_every_stage\n"
        "  --gen_max_tokens GEN_MAX_TOKENS\n"
        "  --wandb_mode {online,offline,disabled}\n"
    )


def print_bootstrap_free_help_and_exit() -> None:
    """Render CLI help without dependency installs, CUDA probes, or network calls."""
    print(bootstrap_free_help_text(), end="")
    sys.exit(0)

def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    action = getattr(argparse, "BooleanOptionalAction", None)
    if action is not None:
        parser.add_argument(name, action=action, default=default, help=help_text)
    else:
        if default:
            parser.add_argument(name, action="store_true", default=default, help=help_text)
        else:
            parser.add_argument(name, action="store_false", default=default, help=help_text)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jamba Reasoning 3B Coconut-Ouroboros fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--max_seq_len", type=int, default=1024)

    # LoRA / QLoRA
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="QLoRA (4-bit NF4). Requires CUDA + bitsandbytes.",
    )
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Dataset
    parser.add_argument("--data_dir", default="data/coconut_v1")
    parser.add_argument("--max_samples", type=int, default=None)

    # Curriculum
    parser.add_argument(
        "--max_stage",
        type=int,
        default=None,
        help="Override K. None = read n_steps_median from stats.json.",
    )
    parser.add_argument("--epochs_per_stage", type=int, default=3)
    parser.add_argument("--stage_0_epochs", type=int, default=None)

    # DGAC halt gate (Phase 3.4)
    parser.add_argument("--use_halt_gate", action="store_true")
    parser.add_argument("--halt_threshold", type=float, default=0.5)
    parser.add_argument("--dgac_lambda_ponder_max", type=float, default=0.01)
    parser.add_argument("--dgac_lambda_diversity", type=float, default=0.1)
    parser.add_argument("--dgac_tau", type=float, default=0.9)
    parser.add_argument("--dgac_warmup_steps", type=int, default=200)
    parser.add_argument("--dgac_ramp_steps", type=int, default=300)
    parser.add_argument(
        "--dgac_halt_supervision_weight",
        type=float,
        default=0.1,
        help=(
            "Weight for correctness-aware supervised HaltGate loss. Set to 0 to "
            "fall back to ponder/diversity-only DGAC regularization."
        ),
    )
    parser.add_argument(
        "--dgac_halt_ce_tolerance",
        type=float,
        default=0.02,
        help="CE tolerance for choosing the smallest depth that preserves full-k quality.",
    )
    parser.add_argument(
        "--dgac_halt_probe_steps",
        default="1,2,4,stage_k",
        help="Comma-separated CE probe depths for supervised HaltGate targets.",
    )
    parser.add_argument(
        "--dgac_halt_target_mode",
        choices=["ce_within_tolerance"],
        default="ce_within_tolerance",
        help="Strategy for constructing supervised HaltGate targets.",
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=(
            "Stop training after this many optimizer steps in the current run. "
            "Intended for GPU canaries/debug runs; None means run the planned epochs."
        ),
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Base gradient clip norm. Stages k>=2 are additionally capped at 0.3.",
    )
    _add_bool_arg(parser, "--grad_checkpoint", True, "Enable gradient checkpointing.")
    parser.add_argument("--seed", type=int, default=42)

    # Session timeout (MANDATORY for Kaggle)
    parser.add_argument("--session_timeout_hours", type=float, default=11.0)
    parser.add_argument("--graceful_exit_buffer_minutes", type=float, default=20.0)
    parser.add_argument(
        "--val_skip_buffer_minutes",
        type=float,
        default=60.0,
        help=(
            "Skip val+gen if remaining session time is below this threshold (minutes). "
            "With DDP val on Dual T4 (val_batch_size=2, 50 acc samples), val takes ~37min. "
            "Default 60min provides a 23min safety margin."
        ),
    )

    # Hub checkpoint sync
    parser.add_argument("--push_to_hub", action="store_true",
        help="Push checkpoints to HF Hub after each epoch save.")
    parser.add_argument("--hf_token", default=None,
        help="HF write token. Falls back to HF_TOKEN env var.")
    parser.add_argument("--hf_repo_id", default="WeirdRunner/Ouroboros",
        help="HF model repo to sync checkpoints to.")
    parser.add_argument("--hf_stage_subdir", default="runs/stage3",
        help="Remote subdirectory inside the HF repo for Stage 3 checkpoints.")

    # DiLoCo
    parser.add_argument("--diloco_mode", action="store_true",
        help="Enable DiLoCo parallel training mode.")
    parser.add_argument(
        "--diloco_worker_id",
        type=_parse_diloco_worker_id_cli,
        default=None,
        choices=list(_VALID_DILOCO_WORKER_IDS),
        help=(
            "This worker's identity. If omitted in --diloco_mode, falls back to "
            "DILOCO_WORKER_ID env / notebook secret."
        ),
    )
    parser.add_argument("--diloco_outer_lr", type=float, default=0.7,
        help="Outer SGD learning rate for DiLoCo aggregation. Default: 0.7 (DiLoCo paper).")
    parser.add_argument("--diloco_min_workers", type=int, default=2,
        help="Minimum workers needed for coordinator to aggregate (default: 2 of 3).")
    parser.add_argument("--diloco_state_repo", default="WeirdRunner/Ouroboros",
        help="HF Hub repo used as shared state store.")
    parser.add_argument("--diloco_signal_repo", default="deveshpat/Ouroboros",
        help="GitHub repo to push coordinator trigger signals to.")
    parser.add_argument("--diloco_run_val", action="store_true",
        help="Run val pass before training begins (used by the first worker of a new stage).")

    # I/O
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--resume_from_diloco_anchor",
        action="store_true",
        help=(
            "Load diloco_state/anchor from --diloco_state_repo as the base LoRA "
            "weights for Phase 3.4 DGAC training. Requires --use_halt_gate."
        ),
    )
    parser.add_argument(
        "--latent_cache",
        action="store_true",
        help=(
            "Cache fixed-depth Coconut latent prefixes so accelerator runs avoid "
            "recomputing the full prefix at every latent step."
        ),
    )
    parser.add_argument(
        "--mac_mps_latent_cache",
        dest="latent_cache",
        action="store_true",
        help=(
            "Compatibility alias for --latent_cache kept for older local commands."
        ),
    )
    parser.add_argument("--output_dir", default="runs/stage3")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help=(
            "Load the requested checkpoint/DiLoCo anchor, run validation and optional "
            "generation, then exit without optimizer steps or checkpoint writes."
        ),
    )
    parser.add_argument(
        "--dgac_diagnostics",
        action="store_true",
        help=(
            "During eval-only HaltGate runs, log DGAC diagnostics: k_actual "
            "histogram plus forced-k and gated validation CE comparisons."
        ),
    )
    parser.add_argument(
        "--dgac_diagnostics_only",
        action="store_true",
        help=(
            "During eval-only DGAC runs, skip the normal validation/generation preflight "
            "and run only HaltGate diagnostics. Use with "
            "--dgac_diagnostics_forced_kmax_ce to reuse a known forced-kmax CE from a "
            "previous successful eval."
        ),
    )
    parser.add_argument(
        "--dgac_diagnostics_batch_size",
        type=int,
        default=1,
        help=(
            "Microbatch size for DGAC diagnostic HaltGate planning and forced-k CE. "
            "Keep at 1 on 16GB T4 because diagnostics run multiple extra full-depth forwards."
        ),
    )
    parser.add_argument(
        "--dgac_diagnostics_forced_kmax_ce",
        type=float,
        default=None,
        help=(
            "Optional known forced-kmax validation CE to reuse in --dgac_diagnostics_only "
            "instead of recomputing the full-depth validation CE."
        ),
    )
    parser.add_argument("--keep_checkpoints_per_stage", type=int, default=2)

    # Monitoring
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument(
        "--profile_training_timing",
        action="store_true",
        help="Log per-step training timing breakdowns for canary/profiling runs.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=1,
        help="Batch size for val forward passes. Keep at 1 to avoid OOM.",
    )
    parser.add_argument(
        "--eval_progress_every",
        type=int,
        default=25,
        help=(
            "Print rank-0 validation/diagnostic progress every N local samples; "
            "set 0 to disable progress logs."
        ),
    )
    _add_bool_arg(parser, "--gen_every_stage", True, "Run generation callback at stage end.")
    parser.add_argument("--gen_max_tokens", type=int, default=200)

    # wandb
    parser.add_argument("--wandb_project", default="ouroboros-stage3-jamba")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )

    return parser.parse_args(argv)


__all__ = [
    "MODEL_ID",
    "bootstrap_free_help_text",
    "parse_args",
    "print_bootstrap_free_help_and_exit",
]
