"""Pure state-planning helpers for the DiLoCo coordinator."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

WORKER_IDS = ["A", "B", "C"]


def _compute_projected_shards(
    total_samples: int,
    stage_k: int,
    round_n: int,
    seed: int,
    total_samples_seen: int,
    worker_ids: List[str] = None,
) -> Dict[str, int]:
    """
    Deterministically compute each worker's projected shard size for the
    upcoming round. Uses identical partition logic to diloco_get_shard()
    in jamba_coconut_finetune.py.

    Returns dict: {worker_id: projected_shard_size}
    """
    if worker_ids is None:
        worker_ids = WORKER_IDS

    worker_index = {"A": 0, "B": 1, "C": 2}
    remaining = max(total_samples - int(total_samples_seen), 0)

    result: Dict[str, int] = {}
    for wid in worker_ids:
        idx = worker_index.get(wid, 0)
        n_parts = 3  # always 3-way partition for determinism, even if C is inactive
        base = remaining // n_parts
        remainder = remaining % n_parts
        width = base + (1 if idx < remainder else 0)
        result[wid] = max(int(width), 0)

    return result


def _determine_round_mode(
    projected_shards: Dict[str, int],
    credentialed_workers: List[str],
    min_shard_samples: int,
    force_worker_ids: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """
    Determine the coordination mode and which workers to trigger.

    Returns:
        mode: "complete" | "solo" | "diloco"
        active_workers: list of worker IDs to trigger
    """
    if force_worker_ids:
        # Manual override: trigger exactly the specified workers, no threshold check
        active = [w for w in force_worker_ids if w in credentialed_workers]
        if not active:
            return "complete", []
        mode = "solo" if len(active) == 1 else "diloco"
        return mode, active

    active = [
        w for w in credentialed_workers
        if projected_shards.get(w, 0) >= min_shard_samples
    ]

    if not active:
        return "complete", []
    if len(active) == 1:
        return "solo", active
    return "diloco", active


def _ordered_unique_worker_ids(*groups: Optional[List[str]]) -> List[str]:
    """Return worker IDs in first-seen order, filtered to known workers."""
    ordered: List[str] = []
    seen = set()
    for group in groups:
        for worker_id in group or []:
            wid = str(worker_id).upper()
            if wid not in WORKER_IDS or wid in seen:
                continue
            ordered.append(wid)
            seen.add(wid)
    return ordered


def _partition_ready_workers(
    ready_workers: List[Dict],
    *,
    expected_workers: Optional[List[str]],
    attendance_workers: Optional[List[str]],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split ready workers into active-round completions and attendance check-ins.

    Attendance workers are opportunistic: they should be observed and promoted
    when they check in, but they must not block aggregation for the active round.
    If a worker is somehow present in both groups, treat it as active so the
    round still enforces its completion semantics.
    """
    expected_set = set(_ordered_unique_worker_ids(expected_workers))
    attendance_set = set(_ordered_unique_worker_ids(attendance_workers))

    active_ready: List[Dict] = []
    attendance_ready: List[Dict] = []
    for status in ready_workers:
        worker_id = str(status.get("worker_id", "")).upper()
        if worker_id in attendance_set and worker_id not in expected_set:
            attendance_ready.append(status)
        else:
            active_ready.append(status)
    return active_ready, attendance_ready


def _mode_from_active_workers(active_workers: List[str], fallback: str = "complete") -> str:
    if not active_workers:
        return fallback
    return "solo" if len(active_workers) == 1 else "diloco"


def _reconcile_post_dispatch_state(
    *,
    state: Dict[str, Any],
    planned_active_workers: List[str],
    planned_attendance_workers: List[str],
    dispatch_results: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """
    Correct round_state after trigger failures.

    The coordinator must write round_state before dispatch so workers see the new
    round/stage. If one or more Kaggle pushes fail, the just-written state can
    incorrectly wait on workers that were never actually launched. This helper
    demotes failed active workers to attendance, preserves successfully/manual
    dispatched workers, and clears triggered_at when nothing was really sent.
    """
    failed_workers = [w for w, status in dispatch_results.items() if status == "failed"]
    if not failed_workers:
        return None

    dispatched_active = [
        w for w in planned_active_workers if dispatch_results.get(w) in {"success", "manual"}
    ]
    dispatched_attendance = [
        w for w in planned_attendance_workers if dispatch_results.get(w) in {"success", "manual"}
    ]
    failed_active = [w for w in planned_active_workers if dispatch_results.get(w) == "failed"]
    failed_attendance = [w for w in planned_attendance_workers if dispatch_results.get(w) == "failed"]

    corrected_triggered_workers = _ordered_unique_worker_ids(dispatched_active)
    corrected_attendance_workers = _ordered_unique_worker_ids(
        dispatched_attendance,
        failed_active,
        failed_attendance,
    )

    corrected_mode = state.get("mode", "complete")
    if corrected_triggered_workers:
        corrected_mode = _mode_from_active_workers(corrected_triggered_workers, fallback=corrected_mode)
    elif corrected_attendance_workers:
        corrected_mode = "waiting"

    outstanding_dispatches = _ordered_unique_worker_ids(
        corrected_triggered_workers,
        corrected_attendance_workers,
    )
    successful_or_manual = [
        w for w in outstanding_dispatches if dispatch_results.get(w) in {"success", "manual"}
    ]

    corrected_state = {
        **state,
        "mode": corrected_mode,
        "triggered_workers": corrected_triggered_workers,
        "attendance_workers": corrected_attendance_workers,
        "triggered_at": time.time() if successful_or_manual else 0.0,
        "last_updated": time.time(),
        "dispatch_failures": failed_workers,
    }
    print(
        "[coordinator] Reconciled failed dispatches. "
        f"triggered={corrected_triggered_workers} attendance={corrected_attendance_workers}"
    )
    return corrected_state
