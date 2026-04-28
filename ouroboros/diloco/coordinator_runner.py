"""DiLoCo coordinator orchestration.

Keep orchestration here and push protocol, environment, dispatch, and Hub I/O to
seams. This file should remain readable; large helper growth belongs in the seam
modules next to the behavior they own.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ouroboros.diloco.aggregation import weighted_average_deltas as apply_diloco_outer_update
from ouroboros.diloco.coordinator_cli import parse_args
from ouroboros.diloco.coordinator_dispatch import reconcile_after_dispatch, trigger_kaggle_workers
from ouroboros.diloco.coordinator_io import (
    ANCHOR_PREFIX,
    hub_download_file,
    hub_download_json,
    hub_upload_json,
    load_adapter_weights_cpu,
    save_and_upload_anchor,
    weighted_average_deltas,
)
from ouroboros.diloco.hub_state import worker_status_path
from ouroboros.diloco.protocol import (
    WORKER_IDS as PROTOCOL_WORKER_IDS,
    compute_projected_shards,
    mode_from_active_workers as protocol_mode_from_active_workers,
    ordered_unique_worker_ids,
)

WORKER_IDS = ["A", "B", "C"]
ROUND_STATE_PATH = "diloco_state/round_state.json"


def _state_stage(round_state: Mapping[str, Any]) -> int:
    return int(round_state.get("stage_k", round_state.get("stage", 0)))


def _state_round(round_state: Mapping[str, Any]) -> int:
    return int(round_state.get("round_n", round_state.get("round", 0)))


def _repo_id_from_args(args: Any) -> str:
    repo_id = getattr(args, "hub_repo_id", None) or getattr(args, "repo_id", None)
    if not repo_id:
        raise ValueError("hub_repo_id/repo_id is required")
    return str(repo_id)


def _compute_projected_shards(*, total_samples: int, total_samples_seen: int, worker_ids: Sequence[str]) -> dict[str, int]:
    return compute_projected_shards(
        total_samples=total_samples,
        total_samples_seen=total_samples_seen,
        worker_ids=worker_ids,
    )


def _determine_round_mode(active_workers: Sequence[str]) -> str:
    return protocol_mode_from_active_workers(active_workers)


def _ordered_unique_worker_ids(worker_ids: Sequence[str]) -> list[str]:
    return ordered_unique_worker_ids(worker_ids)


def _status_matches_round(status: Mapping[str, Any], *, stage: int | None, round_n: int | None) -> bool:
    if stage is not None and int(status.get("stage_k", status.get("stage", -1))) != int(stage):
        return False
    if round_n is not None and int(status.get("round_n", status.get("round", -1))) != int(round_n):
        return False
    return True


def _status_is_ready(status: Mapping[str, Any], *, stage: int | None = None, round_n: int | None = None) -> bool:
    if not _status_matches_round(status, stage=stage, round_n=round_n):
        return False
    return str(status.get("status", "")).strip().lower() in {"done", "ready", "complete", "completed"}


def _partition_ready_workers(
    statuses: Mapping[str, Mapping[str, Any]],
    *,
    min_ready_workers: int,
    expected_workers: Sequence[str] | None = None,
    stage: int | None = None,
    round_n: int | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    expected = ordered_unique_worker_ids(expected_workers or statuses.keys() or PROTOCOL_WORKER_IDS)
    ready = [
        worker
        for worker in expected
        if worker in statuses and _status_is_ready(statuses[worker], stage=stage, round_n=round_n)
    ]
    missing = [worker for worker in expected if worker not in ready]
    if len(ready) < int(min_ready_workers):
        return (), tuple(missing or expected)
    return tuple(ready), tuple(missing)


def _download_json_compat(repo_id: str, path: str, *, token: str | None, default: Any) -> Any:
    """Call hub_download_json while tolerating older monkeypatched signatures."""

    try:
        value = hub_download_json(repo_id, path, token=token, default=default)
    except TypeError:
        try:
            value = hub_download_json(repo_id, path, token)  # type: ignore[misc]
        except TypeError:
            value = hub_download_json(repo_id, path)  # type: ignore[misc]
    return default if value is None else value


def _initial_round_state(args: Any) -> dict[str, Any]:
    total_samples = int(getattr(args, "total_samples", getattr(args, "total_train_samples", 0)))
    total_seen = int(getattr(args, "total_samples_seen", 0))
    projected = _compute_projected_shards(
        total_samples=total_samples,
        total_samples_seen=total_seen,
        worker_ids=WORKER_IDS,
    )
    active_workers = [
        worker
        for worker, shard_size in projected.items()
        if int(shard_size) >= int(getattr(args, "min_shard_samples", 1))
    ] or list(WORKER_IDS)
    now = time.time()
    return {
        "stage": int(getattr(args, "stage", getattr(args, "stage_k", 0))),
        "stage_k": int(getattr(args, "stage", getattr(args, "stage_k", 0))),
        "round": int(getattr(args, "round", getattr(args, "round_n", 0))),
        "round_n": int(getattr(args, "round", getattr(args, "round_n", 0))),
        "mode": _determine_round_mode(active_workers),
        "active_workers": active_workers,
        "triggered_workers": [],
        "attendance_workers": [],
        "triggered_at": 0.0,
        "projected_shards": projected,
        "total_samples": total_samples,
        "total_train_samples": total_samples,
        "total_samples_seen": total_seen,
        "last_updated": now,
        "updated_at": now,
    }


def collect_ready_workers(args_or_repo_id: Any, round_state_or_token: Any = None, **kwargs: Any) -> Any:
    """Collect worker status JSON.

    Supports both the production call ``collect_ready_workers(args, round_state)``
    and the legacy/test seam ``collect_ready_workers(repo, token, stage_k=..., round_n=..., expected_workers=...)``.
    """

    if isinstance(args_or_repo_id, str):
        repo_id = args_or_repo_id
        token = round_state_or_token
        stage = int(kwargs["stage_k"] if "stage_k" in kwargs else kwargs.get("stage", 0))
        round_n = int(kwargs["round_n"] if "round_n" in kwargs else kwargs.get("round", 0))
        workers = ordered_unique_worker_ids(kwargs.get("expected_workers") or PROTOCOL_WORKER_IDS)
        ready_statuses: list[dict[str, Any]] = []
        for worker in workers:
            status = _download_json_compat(repo_id, worker_status_path(worker), token=token, default={})
            if isinstance(status, Mapping) and _status_is_ready(status, stage=stage, round_n=round_n):
                ready_statuses.append(dict(status))
        return ready_statuses

    args = args_or_repo_id
    round_state = dict(round_state_or_token or {})
    repo_id = _repo_id_from_args(args)
    token = getattr(args, "hf_token", None)
    workers = ordered_unique_worker_ids(
        round_state.get("triggered_workers"),
        round_state.get("active_workers"),
        PROTOCOL_WORKER_IDS,
    )
    statuses: dict[str, dict[str, Any]] = {}
    for worker in workers:
        status = _download_json_compat(repo_id, worker_status_path(worker), token=token, default={})
        if isinstance(status, Mapping) and status:
            statuses[worker] = dict(status)
    return statuses


def _download_weights(repo_id: str, weights_path: str, *, token: str | None) -> dict[str, Any]:
    if Path(weights_path).exists():
        return load_adapter_weights_cpu(weights_path)
    local_path = hub_download_file(repo_id, weights_path, token=token)
    return load_adapter_weights_cpu(local_path)


def _aggregate_ready_workers(
    args: Any,
    round_state: Mapping[str, Any],
    statuses: Mapping[str, Mapping[str, Any]],
    ready_workers: Sequence[str],
) -> dict[str, Any]:
    repo_id = _repo_id_from_args(args)
    token = getattr(args, "hf_token", None)
    worker_weights: list[Mapping[str, Any]] = []
    worker_samples: list[int] = []
    weighted_for_average: list[tuple[Mapping[str, Any], float]] = []
    samples_by_worker: dict[str, int] = {}

    for worker in ordered_unique_worker_ids(ready_workers):
        status = statuses[worker]
        weights_path = status.get("weights_path") or status.get("adapter_delta_path") or status.get("adapter_weights_path")
        if not weights_path:
            continue
        weights = _download_weights(repo_id, str(weights_path), token=token)
        samples = int(status.get("samples_seen", status.get("samples", status.get("shard_size", 1))) or 1)
        samples = max(samples, 1)
        worker_weights.append(weights)
        worker_samples.append(samples)
        weighted_for_average.append((weights, float(samples)))
        samples_by_worker[worker] = samples
    if not worker_weights:
        raise RuntimeError("no ready worker adapter deltas found")

    anchor_source = round_state.get("anchor_path") or round_state.get("previous_anchor_path")
    if anchor_source:
        anchor_weights = _download_weights(repo_id, str(anchor_source), token=token)
        anchor = apply_diloco_outer_update(
            anchor_weights,
            worker_weights,
            worker_samples,
            float(getattr(args, "outer_lr", 1.0)),
        )
    else:
        # Bootstrap case: no prior anchor is recorded, so average the completed
        # worker weights instead of crashing or applying an undefined outer update.
        anchor = weighted_average_deltas(weighted_for_average)

    stage = _state_stage(round_state)
    round_n = _state_round(round_state)
    anchor_path = f"{ANCHOR_PREFIX}/stage_{stage}/round_{round_n}/anchor.safetensors"
    save_and_upload_anchor(repo_id, anchor_path, anchor, token=token)
    return {
        "anchor_path": anchor_path,
        "contributing_workers": ordered_unique_worker_ids(ready_workers),
        "samples_by_worker": samples_by_worker,
    }


def _next_round_state(args: Any, current: Mapping[str, Any], aggregate_result: Mapping[str, Any]) -> dict[str, Any]:
    total_samples = int(getattr(args, "total_samples", getattr(args, "total_train_samples", current.get("total_samples", 0))))
    samples_by_worker = dict(aggregate_result.get("samples_by_worker", {}))
    completed = sum(int(v or 0) for v in samples_by_worker.values())
    if completed <= 0:
        completed = sum(int(v or 0) for v in current.get("projected_shards", {}).values())
    total_seen = min(total_samples, int(current.get("total_samples_seen", 0)) + completed)
    next_round = _state_round(current) + 1
    stage = _state_stage(current)
    projected = _compute_projected_shards(
        total_samples=total_samples,
        total_samples_seen=total_seen,
        worker_ids=WORKER_IDS,
    )
    active_workers = [
        worker
        for worker, shard_size in projected.items()
        if int(shard_size) >= int(getattr(args, "min_shard_samples", 1))
    ]
    now = time.time()
    return {
        "stage": stage,
        "stage_k": stage,
        "round": next_round,
        "round_n": next_round,
        "mode": _determine_round_mode(active_workers),
        "active_workers": active_workers,
        "triggered_workers": [],
        "attendance_workers": ordered_unique_worker_ids(current.get("attendance_workers")),
        "triggered_at": 0.0,
        "projected_shards": projected,
        "total_samples": total_samples,
        "total_train_samples": total_samples,
        "total_samples_seen": total_seen,
        "previous_anchor_path": aggregate_result.get("anchor_path"),
        "anchor_path": aggregate_result.get("anchor_path"),
        "last_updated": now,
        "updated_at": now,
    }


def _dispatch_and_save(args: Any, state: Mapping[str, Any], workers: Sequence[str]) -> dict[str, Any]:
    if getattr(args, "skip_trigger", False):
        return dict(state)
    dispatched = trigger_kaggle_workers(args, workers, state)
    next_state = reconcile_after_dispatch(state, dispatched)
    hub_upload_json(_repo_id_from_args(args), ROUND_STATE_PATH, next_state, token=getattr(args, "hf_token", None))
    return next_state


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repo_id = _repo_id_from_args(args)
    round_state = hub_download_json(
        repo_id,
        ROUND_STATE_PATH,
        token=getattr(args, "hf_token", None),
        default=_initial_round_state(args),
    )

    if args.force_worker_ids:
        forced = _ordered_unique_worker_ids(str(args.force_worker_ids).split(","))
        round_state = dict(round_state)
        round_state["active_workers"] = list(forced)
        round_state["triggered_workers"] = list(forced)
        round_state["mode"] = _determine_round_mode(forced)

    if args.dry_run:
        print({"round_state": round_state})
        return

    start = time.time()
    while True:
        stage = _state_stage(round_state)
        round_n = _state_round(round_state)
        expected = ordered_unique_worker_ids(round_state.get("triggered_workers"), round_state.get("active_workers"))
        statuses = collect_ready_workers(args, round_state)
        ready, missing = _partition_ready_workers(
            statuses,
            min_ready_workers=int(args.min_ready_workers),
            expected_workers=expected,
            stage=stage,
            round_n=round_n,
        )
        if ready:
            break

        if expected and float(round_state.get("triggered_at", 0.0) or 0.0) <= 0.0:
            _dispatch_and_save(args, round_state, missing or expected)
            return

        if args.max_wait_seconds and time.time() - start >= int(args.max_wait_seconds):
            _dispatch_and_save(args, round_state, missing or expected)
            return
        time.sleep(max(int(args.poll_seconds), 1))

    aggregate_result = _aggregate_ready_workers(args, round_state, statuses, ready)
    next_state = _next_round_state(args, round_state, aggregate_result)
    hub_upload_json(repo_id, ROUND_STATE_PATH, next_state, token=getattr(args, "hf_token", None))
    if next_state.get("active_workers") and not getattr(args, "skip_trigger", False):
        _dispatch_and_save(args, next_state, next_state.get("active_workers", WORKER_IDS))


__all__ = [
    "ROUND_STATE_PATH",
    "WORKER_IDS",
    "collect_ready_workers",
    "hub_download_file",
    "hub_download_json",
    "main",
    "weighted_average_deltas",
    "_compute_projected_shards",
    "_determine_round_mode",
    "_ordered_unique_worker_ids",
    "_partition_ready_workers",
]
