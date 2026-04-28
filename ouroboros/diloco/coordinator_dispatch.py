"""Kaggle worker dispatch for the DiLoCo coordinator."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ouroboros.diloco.kaggle_dispatch import trigger_single_worker
from ouroboros.diloco.protocol import (
    WORKER_IDS,
    mode_from_active_workers as protocol_mode_from_active_workers,
    ordered_unique_worker_ids,
    reconcile_post_dispatch_state,
)
from ouroboros.diloco.runtime_env import build_worker_runtime_env

WORKER_KAGGLE_SLUGS = {
    "A": "worker-a",
    "B": "worker-b",
    "C": "worker-c",
}

SUCCESSFUL_DISPATCH_OUTCOMES = {"success", "manual", "triggered"}


def mode_from_active_workers(active_workers: Sequence[str]) -> str:
    return protocol_mode_from_active_workers(active_workers)


def _worker_attr(args: Any, prefix: str, worker_id: str) -> str | None:
    worker = str(worker_id).strip().lower()
    return getattr(args, f"{prefix}_{worker}", None)


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _kaggle_credentials(args: Any, worker_id: str) -> tuple[str | None, str | None]:
    worker = str(worker_id).strip().upper()
    return (
        _first_text(
            _worker_attr(args, "kaggle_username", worker),
            getattr(args, "kaggle_user", None),
            getattr(args, "kaggle_username", None),
            os.environ.get(f"KAGGLE_USERNAME_{worker}"),
            os.environ.get("KAGGLE_USERNAME"),
        ),
        _first_text(
            _worker_attr(args, "kaggle_key", worker),
            getattr(args, "kaggle_key", None),
            os.environ.get(f"KAGGLE_KEY_{worker}"),
            os.environ.get("KAGGLE_KEY"),
        ),
    )


def _slug_for_worker(args: Any, worker_id: str, username: str | None) -> str:
    worker = str(worker_id).strip().upper()
    configured = _first_text(getattr(args, f"kaggle_slug_{worker.lower()}", None), WORKER_KAGGLE_SLUGS.get(worker))
    if configured is None:
        configured = f"worker-{worker.lower()}"
    if "/" in configured or not username:
        return configured
    return f"{username}/{configured}"


def trigger_kaggle_workers(args: Any, worker_ids: Sequence[str], round_state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Trigger Kaggle workers and return per-worker dispatch outcomes.

    The low-level Kaggle seam owns notebook staging and metadata writing. This
    adapter only resolves worker-specific credentials/env and records outcomes.
    """

    dispatched: dict[str, dict[str, Any]] = {}
    workers = ordered_unique_worker_ids(worker_ids)
    if not workers:
        return dispatched

    notebook_path = Path(getattr(args, "kaggle_notebook_path", "kaggle-utils.ipynb"))
    for worker in workers:
        username, key = _kaggle_credentials(args, worker)
        slug = _slug_for_worker(args, worker, username)
        runtime_env = build_worker_runtime_env(args, worker)
        now = time.time()
        if not username or not key:
            outcome = "missing_credentials"
            ok = False
        else:
            try:
                ok = bool(
                    trigger_single_worker(
                        worker,
                        username,
                        key,
                        slug,
                        notebook_path,
                        injected_env=runtime_env,
                    )
                )
                outcome = "success" if ok else "failed"
            except Exception as exc:  # noqa: BLE001 - dispatch returns structured failure
                ok = False
                outcome = "failed"
                dispatched[worker] = {
                    "worker_id": worker,
                    "round": round_state.get("round", round_state.get("round_n")),
                    "round_n": round_state.get("round_n", round_state.get("round")),
                    "stage": round_state.get("stage", round_state.get("stage_k")),
                    "stage_k": round_state.get("stage_k", round_state.get("stage")),
                    "slug": slug,
                    "outcome": outcome,
                    "ok": ok,
                    "error": str(exc),
                    "dispatched_at": now,
                }
                continue
        dispatched[worker] = {
            "worker_id": worker,
            "round": round_state.get("round", round_state.get("round_n")),
            "round_n": round_state.get("round_n", round_state.get("round")),
            "stage": round_state.get("stage", round_state.get("stage_k")),
            "stage_k": round_state.get("stage_k", round_state.get("stage")),
            "slug": slug,
            "outcome": outcome,
            "ok": ok,
            "dispatched_at": now,
        }
    return dispatched


def _normalize_dispatch_results(dispatched_workers: Sequence[str] | Mapping[str, Any]) -> dict[str, str]:
    if isinstance(dispatched_workers, Mapping):
        results: dict[str, str] = {}
        for worker_id, payload in dispatched_workers.items():
            worker = str(worker_id).strip().upper()
            if isinstance(payload, Mapping):
                outcome = str(payload.get("outcome") or ("success" if payload.get("ok") else "failed"))
            else:
                outcome = str(payload or "success")
            if outcome == "True":
                outcome = "success"
            elif outcome == "False":
                outcome = "failed"
            results[worker] = outcome
        return results
    return {worker: "success" for worker in ordered_unique_worker_ids(dispatched_workers)}


def reconcile_after_dispatch(round_state: Mapping[str, Any], dispatched_workers: Sequence[str] | Mapping[str, Any]) -> dict[str, Any]:
    dispatch_results = _normalize_dispatch_results(dispatched_workers)
    planned_active = ordered_unique_worker_ids(
        round_state.get("triggered_workers"),
        round_state.get("active_workers"),
        dispatch_results.keys(),
    )
    planned_attendance = ordered_unique_worker_ids(round_state.get("attendance_workers"))
    now = time.time()
    reconciled = reconcile_post_dispatch_state(
        state=dict(round_state),
        planned_active_workers=planned_active,
        planned_attendance_workers=planned_attendance,
        dispatch_results=dispatch_results,
        now=now,
    )
    if reconciled is not None:
        return reconciled

    successful = [w for w in planned_active if dispatch_results.get(w) in SUCCESSFUL_DISPATCH_OUTCOMES]
    return {
        **dict(round_state),
        "triggered_workers": successful,
        "active_workers": successful,
        "mode": mode_from_active_workers(successful),
        "triggered_at": now if successful else 0.0,
        "last_updated": now,
        "updated_at": now,
        "dispatch_results": dispatch_results,
    }


_trigger_kaggle_workers = trigger_kaggle_workers
_mode_from_active_workers = mode_from_active_workers
_reconcile_post_dispatch_state = reconcile_after_dispatch


__all__ = [
    "WORKER_KAGGLE_SLUGS",
    "mode_from_active_workers",
    "reconcile_after_dispatch",
    "trigger_kaggle_workers",
]
