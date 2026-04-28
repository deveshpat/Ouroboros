"""Kaggle worker dispatch for the DiLoCo coordinator."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ouroboros.diloco.kaggle_dispatch import (
    build_kaggle_kernel_metadata,
    build_worker_dispatch_cell,
    stage_local_kaggle_kernel,
    trigger_single_worker,
)
from ouroboros.diloco.protocol import WORKER_IDS, determine_round_mode, reconcile_post_dispatch_state
from ouroboros.diloco.runtime_env import build_worker_runtime_env, encode_runtime_env_payload

WORKER_KAGGLE_SLUGS = {
    "A": "worker-a",
    "B": "worker-b",
    "C": "worker-c",
}


def mode_from_active_workers(active_workers: Sequence[str]) -> str:
    return determine_round_mode(tuple(active_workers))


def trigger_kaggle_workers(args: Any, worker_ids: Sequence[str], round_state: Mapping[str, Any]) -> dict[str, Any]:
    dispatched: dict[str, Any] = {}
    if not worker_ids:
        return dispatched

    notebook_path = Path(getattr(args, "kaggle_notebook_path", "kaggle-utils.ipynb"))
    for worker_id in worker_ids:
        worker = str(worker_id).strip().upper()
        runtime_env = build_worker_runtime_env(args, worker)
        payload = encode_runtime_env_payload(runtime_env)
        metadata = build_kaggle_kernel_metadata(
            getattr(args, "kaggle_user", None),
            WORKER_KAGGLE_SLUGS.get(worker, f"worker-{worker.lower()}"),
        )
        dispatch_cell = build_worker_dispatch_cell(payload)
        staged = stage_local_kaggle_kernel(
            notebook_path=notebook_path,
            metadata=metadata,
            dispatch_cell=dispatch_cell,
            worker_id=worker,
        )
        trigger_single_worker(staged.kernel_dir)
        dispatched[worker] = {
            "worker_id": worker,
            "round": round_state.get("round"),
            "stage": round_state.get("stage"),
            "kernel_dir": str(staged.kernel_dir),
            "dispatched_at": time.time(),
        }
    return dispatched


def reconcile_after_dispatch(round_state: Mapping[str, Any], dispatched_workers: Sequence[str]) -> dict[str, Any]:
    return reconcile_post_dispatch_state(
        round_state=dict(round_state),
        dispatched_worker_ids=tuple(dispatched_workers),
        now=time.time(),
    )


_trigger_kaggle_workers = trigger_kaggle_workers
_mode_from_active_workers = mode_from_active_workers
_reconcile_post_dispatch_state = reconcile_after_dispatch


__all__ = [
    "WORKER_KAGGLE_SLUGS",
    "mode_from_active_workers",
    "reconcile_after_dispatch",
    "trigger_kaggle_workers",
]
