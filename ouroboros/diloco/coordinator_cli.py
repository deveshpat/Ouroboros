"""CLI parsing for the DiLoCo coordinator."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ouroboros.diloco.protocol import WORKER_IDS

DEFAULT_KAGGLE_NOTEBOOK_PATH = Path(__file__).resolve().parents[2] / "kaggle-utils.ipynb"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coordinate DiLoCo worker rounds.")
    parser.add_argument("--hub_repo_id", "--repo_id", dest="hub_repo_id", required=True)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--github_token", "--gh_token", dest="github_token", default=None)
    parser.add_argument("--runtime_repo_url", default=None)
    parser.add_argument("--runtime_repo_ref", default=None)
    parser.add_argument("--runtime_repo_commit", default=None)
    parser.add_argument("--stage", "--stage_k", dest="stage", type=int, default=0)
    parser.add_argument("--round", "--round_n", dest="round", type=int, default=0)
    parser.add_argument("--total_samples", "--total_train_samples", dest="total_samples", type=int, required=True)
    parser.add_argument("--total_samples_seen", type=int, default=0)
    parser.add_argument("--outer_lr", type=float, default=1.0)
    parser.add_argument("--min_ready_workers", type=int, default=len(WORKER_IDS))
    parser.add_argument("--min_shard_samples", type=int, default=32)
    parser.add_argument("--worker_timeout_hours", type=float, default=13.0)
    parser.add_argument("--max_wait_seconds", type=int, default=0)
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--force_worker_ids", default=None)
    parser.add_argument("--skip_trigger", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--kaggle_notebook_path", default=str(DEFAULT_KAGGLE_NOTEBOOK_PATH))
    parser.add_argument("--kaggle_user", "--kaggle_username", dest="kaggle_user", default=None)
    parser.add_argument("--kaggle_key", default=None)
    for worker in WORKER_IDS:
        suffix = worker.lower()
        parser.add_argument(f"--kaggle_username_{suffix}", default=None)
        parser.add_argument(f"--kaggle_key_{suffix}", default=None)
    parser.add_argument("--wandb_key", default=None)
    parser.add_argument("--wandb_project", default="ouroboros-diloco")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    args = parser.parse_args(argv)
    # Keep the historical attribute available for code/tests that still read it.
    setattr(args, "repo_id", args.hub_repo_id)
    setattr(args, "total_train_samples", args.total_samples)
    setattr(args, "stage_k", args.stage)
    setattr(args, "round_n", args.round)
    return args


__all__ = ["DEFAULT_KAGGLE_NOTEBOOK_PATH", "parse_args"]
