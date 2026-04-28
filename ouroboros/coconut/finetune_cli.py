"""Command-line parsing for Coconut fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-wise Coconut/Jamba fine-tuning runtime."
    )

    # Data/model/runtime.
    parser.add_argument("--data_dir", default="data/coconut_stage3")
    parser.add_argument("--model_id", default="ai21labs/AI21-Jamba-Reasoning-3B")
    parser.add_argument("--output_dir", default="runs/stage3_curriculum")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", action="store_false", dest="bf16")

    # Curriculum/training.
    parser.add_argument("--stage_k", type=int, default=None)
    parser.add_argument("--max_stage", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--limit_train_samples", type=int, default=None)

    # Coconut/DGAC switches.
    parser.add_argument("--use_halt_gate", action="store_true")
    parser.add_argument("--ponder_weight", type=float, default=0.01)
    parser.add_argument("--diversity_weight", type=float, default=0.01)

    # Hub/W&B publication.
    parser.add_argument("--hub_repo_id", default=None)
    parser.add_argument("--hub_private", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_name", default=None)

    # DiLoCo worker mode.  These names mirror the legacy script so Kaggle
    # notebooks and GitHub Actions can keep passing the same environment.
    parser.add_argument("--diloco_worker_id", default=None)
    parser.add_argument("--diloco_round", type=int, default=None)
    parser.add_argument("--diloco_stage", type=int, default=None)
    parser.add_argument("--diloco_total_samples", type=int, default=None)
    parser.add_argument("--diloco_total_samples_seen", type=int, default=0)
    parser.add_argument("--diloco_outer_lr", type=float, default=1.0)
    parser.add_argument("--diloco_status_path", default=None)
    parser.add_argument("--diloco_weights_path", default=None)
    parser.add_argument("--diloco_anchor_path", default=None)
    parser.add_argument("--diloco_repo_id", default=None)
    parser.add_argument("--diloco_commit_message", default=None)

    args = parser.parse_args(argv)
    args.data_dir = str(Path(args.data_dir))
    args.output_dir = str(Path(args.output_dir))
    return args


__all__ = ["parse_args"]
