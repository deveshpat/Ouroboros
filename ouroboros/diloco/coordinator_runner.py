"""DiLoCo coordinator orchestration.

Keep orchestration here and push protocol, environment, dispatch, and Hub I/O to
seams.  This file should remain readable; large helper growth belongs in the seam
modules next to the behavior they own.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from ouroboros.diloco.aggregation import aggregate_worker_deltas
from ouroboros.diloco.coordinator_cli import parse_args
from ouroboros.diloco.coordinator_dispatch import reconcile_after_dispatch, trigger_kaggle_workers
from ouroboros.diloco.coordinator_io import (
    ANCHOR_PREFIX,
    hub_download_json,
    hub_upload_json,
    load_adapter_weights_cpu,
    save_and_upload_anchor,
    weighted_average_deltas,
)
from ouroboros.diloco.hub_state import worker_status_path
from ouroboros.diloco.protocol import (
    WORKER_IDS,
    compute_projected_shards,
    determine_round_mode,
    ordered_unique_worker_ids,
    partition_ready_workers,
)

ROUND_STATE_PATH = "diloco_state/round_state.json"


def _compute_projected_shards(*, total_samples: int, total_samples_seen: int, worker_ids: Sequence[str]) -> dict[str, int]:
    return compute_projected_shards(
        total_samples=total_samples,
        total_samples_seen=total_samples_seen,
        worker_ids=worker_ids,
    )


def _determine_round_mode(active_workers: Sequence[str]) -> str:
    return determine_round_mode(active_workers)


def _ordered_unique_worker_ids(worker_ids: Sequence[str]) -> tuple[str, ...]:
    return ordered_unique_worker_ids(worker_ids)


def _partition_ready_workers(statuses: Mapping[str, Mapping[str, Any]], *, min_ready_workers: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return partition_ready_workers(statuses, min_ready_workers=min_ready_workers)


def _initial_round_state(args: Any) -> dict[str, Any]:
    projected = _compute_projected_shards(
        total_samples=args.total_samples,
        total_samples_seen=args.total_samples_seen,
        worker_ids=WORKER_IDS,
    )
    return {
        "stage": int(args.stage),
        "round": int(args.round),
        "mode": _determine_round_mode(WORKER_IDS),
        "active_workers": list(WORKER_IDS),
        "projected_shards": projected,
        "total_samples": int(args.total_samples),
        "total_samples_seen": int(args.total_samples_seen),
        "updated_at": time.time(),
    }


def collect_ready_workers(args: Any, round_state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for worker_id in round_state.get("active_workers", WORKER_IDS):
        worker = str(worker_id).strip().upper()
        status_path = worker_status_path(stage=int(round_state["stage"]), round=int(round_state["round"]), worker_id=worker)
        status = hub_download_json(args.hub_repo_id, status_path, token=args.hf_token, default={})
        if status:
            statuses[worker] = dict(status)
    return statuses


def _aggregate_ready_workers(args: Any, round_state: Mapping[str, Any], statuses: Mapping[str, Mapping[str, Any]], ready_workers: Sequence[str]) -> dict[str, Any]:
    weighted: list[tuple[Mapping[str, Any], float]] = []
    for worker in ready_workers:
        status = statuses[worker]
        weights_path = status.get("weights_path") or status.get("adapter_delta_path")
        if not weights_path:
            continue
        local = hub_download_json(args.hub_repo_id, weights_path, token=args.hf_token, default=None)
        weights = load_adapter_weights_cpu(local if isinstance(local, str) else weights_path)
        shard_size = float(status.get("samples", status.get("shard_size", 1)) or 1)
        weighted.append((weights, shard_size))
    if not weighted:
        raise RuntimeError("no ready worker adapter deltas found")

    try:
        anchor = aggregate_worker_deltas(weighted)
    except TypeError:
        anchor = weighted_average_deltas(weighted)

    anchor_path = f"{ANCHOR_PREFIX}/stage_{round_state['stage']}/round_{round_state['round']}/anchor.safetensors"
    save_and_upload_anchor(args.hub_repo_id, anchor_path, anchor, token=args.hf_token)
    return {"anchor_path": anchor_path, "contributing_workers": list(ready_workers)}


def _next_round_state(args: Any, current: Mapping[str, Any], aggregate_result: Mapping[str, Any]) -> dict[str, Any]:
    completed = sum(int(v or 0) for v in current.get("projected_shards", {}).values())
    total_seen = min(int(args.total_samples), int(current.get("total_samples_seen", 0)) + completed)
    next_round = int(current.get("round", 0)) + 1
    stage = int(current.get("stage", args.stage))
    projected = _compute_projected_shards(
        total_samples=int(args.total_samples),
        total_samples_seen=total_seen,
        worker_ids=WORKER_IDS,
    )
    return {
        "stage": stage,
        "round": next_round,
        "mode": _determine_round_mode(WORKER_IDS),
        "active_workers": list(WORKER_IDS),
        "projected_shards": projected,
        "total_samples": int(args.total_samples),
        "total_samples_seen": total_seen,
        "previous_anchor_path": aggregate_result.get("anchor_path"),
        "updated_at": time.time(),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    round_state = hub_download_json(
        args.hub_repo_id,
        ROUND_STATE_PATH,
        token=args.hf_token,
        default=_initial_round_state(args),
    )

    if args.force_worker_ids:
        forced = _ordered_unique_worker_ids(args.force_worker_ids.split(","))
        round_state = dict(round_state)
        round_state["active_workers"] = list(forced)
        round_state["mode"] = _determine_round_mode(forced)

    if args.dry_run:
        print({"round_state": round_state})
        return

    start = time.time()
    while True:
        statuses = collect_ready_workers(args, round_state)
        ready, missing = _partition_ready_workers(statuses, min_ready_workers=int(args.min_ready_workers))
        if ready:
            break
        if args.max_wait_seconds and time.time() - start >= int(args.max_wait_seconds):
            trigger_kaggle_workers(args, missing or round_state.get("active_workers", WORKER_IDS), round_state)
            reconciled = reconcile_after_dispatch(round_state, missing or round_state.get("active_workers", WORKER_IDS))
            hub_upload_json(args.hub_repo_id, ROUND_STATE_PATH, reconciled, token=args.hf_token)
            return
        time.sleep(max(int(args.poll_seconds), 1))

    aggregate_result = _aggregate_ready_workers(args, round_state, statuses, ready)
    next_state = _next_round_state(args, round_state, aggregate_result)
    hub_upload_json(args.hub_repo_id, ROUND_STATE_PATH, next_state, token=args.hf_token)
    dispatched = trigger_kaggle_workers(args, next_state.get("active_workers", WORKER_IDS), next_state)
    if dispatched:
        next_state = reconcile_after_dispatch(next_state, tuple(dispatched))
        hub_upload_json(args.hub_repo_id, ROUND_STATE_PATH, next_state, token=args.hf_token)


__all__ = [
    "ROUND_STATE_PATH",
    "collect_ready_workers",
    "main",
    "weighted_average_deltas",
    "_compute_projected_shards",
    "_determine_round_mode",
    "_ordered_unique_worker_ids",
    "_partition_ready_workers",
]
