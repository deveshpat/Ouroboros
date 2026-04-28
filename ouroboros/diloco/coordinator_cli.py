"""CLI parsing for the DiLoCo coordinator."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ouroboros.diloco.protocol import WORKER_IDS

DEFAULT_KAGGLE_NOTEBOOK_PATH = Path(__file__).resolve().parents[2] / "kaggle-utils.ipynb"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coordinate DiLoCo worker rounds.")
    parser.add_argument("--hub_repo_id", required=True)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--github_token", default=None)
    parser.add_argument("--runtime_repo_url", default=None)
    parser.add_argument("--runtime_repo_ref", default=None)
    parser.add_argument("--runtime_repo_commit", default=None)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--total_samples", type=int, required=True)
    parser.add_argument("--total_samples_seen", type=int, default=0)
    parser.add_argument("--outer_lr", type=float, default=1.0)
    parser.add_argument("--min_ready_workers", type=int, default=len(WORKER_IDS))
    parser.add_argument("--max_wait_seconds", type=int, default=0)
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--force_worker_ids", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--kaggle_notebook_path", default=str(DEFAULT_KAGGLE_NOTEBOOK_PATH))
    parser.add_argument("--kaggle_user", default=None)
    parser.add_argument("--wandb_project", default="ouroboros-diloco")
    parser.add_argument("--wandb_run_name", default=None)
    return parser.parse_args(argv)


__all__ = ["DEFAULT_KAGGLE_NOTEBOOK_PATH", "parse_args"]
