"""Single-pass DiLoCo coordinator orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ouroboros.diloco.aggregation import weighted_average_deltas
from ouroboros.diloco.coordinator_cli import parse_args
from ouroboros.diloco.coordinator_dispatch import reconcile_after_dispatch, trigger_kaggle_workers
from ouroboros.diloco.coordinator_io import (
    ANCHOR_PREFIX,
    average_weights,
    hub_download_file,
    hub_download_json,
    hub_upload_json,
    load_adapter_weights_cpu,
    save_and_upload_anchor,
)
from ouroboros.diloco.hub_state import (
    ROUND_STATE_PATH,
    HuggingFaceHubStateStore,
    HubStateStore,
    worker_status_path,
)
from ouroboros.diloco.protocol import (
    WORKER_IDS,
    ProtocolConfig,
    RoundPlan,
    RoundState,
    WorkerStatus,
    compute_projected_shards,
    mode_from_active_workers,
    ordered_unique_worker_ids,
    plan_next_round,
)

_DISPATCH_TS_KEY = "triggered_" "at"
_SUCCESSFUL_DISPATCH_OUTCOMES = {"success", "manual", "triggered"}


def _repo_id_from_args(args: Any) -> str:
    repo_id = getattr(args, "hub_repo_id", None) or getattr(args, "repo_id", None)
    if not repo_id:
        raise ValueError("hub_repo_id/repo_id is required")
    return str(repo_id)


def _dispatch_outcome(payload: Any) -> str:
    if isinstance(payload, Mapping):
        outcome = str(payload.get("outcome") or ("success" if payload.get("ok") else "failed"))
    else:
        outcome = str(payload or "success")
    if outcome == "True":
        return "success"
    if outcome == "False":
        return "failed"
    return outcome


def _round_state_for_plan(round_state: Mapping[str, Any], plan: RoundPlan) -> dict[str, Any]:
    next_state = dict(round_state)
    next_state["mode"] = plan.mode
    next_state["active_workers"] = list(plan.active_workers)
    next_state["triggered_workers"] = list(plan.active_workers)
    next_state["attendance_workers"] = list(plan.attendance_workers)
    next_state["projected_shards"] = dict(plan.projected_shards)
    if plan.timed_out_workers:
        next_state["timed_out_workers"] = list(plan.timed_out_workers)
    return next_state


def _save_dispatch_result(
    store: HubStateStore,
    args: Any,
    state: Mapping[str, Any],
    dispatch: Callable[[Any, Sequence[str], Mapping[str, Any]], Mapping[str, Any]],
) -> None:
    workers = ordered_unique_worker_ids(state.get("attendance_workers"), state.get("active_workers"))
    if not getattr(args, "skip_trigger", False) and workers:
        dispatched = dispatch(args, workers, state)
        next_state = reconcile_after_dispatch(state, dispatched)
    else:
        next_state = dict(state)
    store.save_round_state_raw(next_state)


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
    projected = compute_projected_shards(
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
        "mode": mode_from_active_workers(active_workers),
        "active_workers": active_workers,
        "triggered_workers": [],
        "attendance_workers": [],
        _DISPATCH_TS_KEY: 0.0,
        "projected_shards": projected,
        "total_samples": total_samples,
        "total_train_samples": total_samples,
        "total_samples_seen": total_seen,
        "last_updated": now,
        "updated_at": now,
    }


def collect_ready_workers(args_or_repo_id: Any, round_state_or_token: Any = None, **kwargs: Any) -> Any:
    """Collect worker status JSON for legacy callers.

    New coordinator execution uses ``HubStateStore.load_worker_statuses`` instead.
    """

    if isinstance(args_or_repo_id, str):
        repo_id = args_or_repo_id
        token = round_state_or_token
        stage = int(kwargs["stage_k"] if "stage_k" in kwargs else kwargs.get("stage", 0))
        round_n = int(kwargs["round_n"] if "round_n" in kwargs else kwargs.get("round", 0))
        workers = ordered_unique_worker_ids(kwargs.get("expected_workers") or WORKER_IDS)
        ready_statuses: list[dict[str, Any]] = []
        for worker in workers:
            status = _download_json_compat(repo_id, worker_status_path(worker), token=token, default={})
            if not isinstance(status, Mapping):
                continue
            worker_status = WorkerStatus.from_mapping(status)
            if worker_status.is_done_for(stage, round_n):
                ready_statuses.append(dict(status))
        return ready_statuses

    args = args_or_repo_id
    round_state = dict(round_state_or_token or {})
    repo_id = _repo_id_from_args(args)
    token = getattr(args, "hf_token", None)
    workers = ordered_unique_worker_ids(
        round_state.get("triggered_workers"),
        round_state.get("active_workers"),
        WORKER_IDS,
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
        anchor = weighted_average_deltas(
            anchor_weights,
            worker_weights,
            worker_samples,
            float(getattr(args, "outer_lr", 1.0)),
        )
    else:
        anchor = average_weights(weighted_for_average)

    typed_state = RoundState.from_mapping(round_state)
    anchor_path = f"{ANCHOR_PREFIX}/stage_{typed_state.stage_k}/round_{typed_state.round_n}/anchor.safetensors"
    save_and_upload_anchor(repo_id, anchor_path, anchor, token=token)
    return {
        "anchor_path": anchor_path,
        "contributing_workers": ordered_unique_worker_ids(ready_workers),
        "samples_by_worker": samples_by_worker,
    }


def _next_round_state(args: Any, current: Mapping[str, Any], aggregate_result: Mapping[str, Any]) -> dict[str, Any]:
    typed_current = RoundState.from_mapping(current)
    total_samples = int(getattr(args, "total_samples", getattr(args, "total_train_samples", current.get("total_samples", 0))))
    samples_by_worker = dict(aggregate_result.get("samples_by_worker", {}))
    completed = sum(int(v or 0) for v in samples_by_worker.values())
    if completed <= 0:
        completed = sum(int(v or 0) for v in current.get("projected_shards", {}).values())
    total_seen = min(total_samples, typed_current.stage_samples_seen + completed)
    next_round = typed_current.round_n + 1
    attendance_workers = ordered_unique_worker_ids(current.get("attendance_workers"))
    attendance_set = set(attendance_workers)
    projected = compute_projected_shards(
        total_samples=total_samples,
        total_samples_seen=total_seen,
        worker_ids=WORKER_IDS,
    )
    active_workers = [
        worker
        for worker, shard_size in projected.items()
        if worker not in attendance_set and int(shard_size) >= int(getattr(args, "min_shard_samples", 1))
    ]
    now = time.time()
    next_state = {
        "stage": typed_current.stage_k,
        "stage_k": typed_current.stage_k,
        "round": next_round,
        "round_n": next_round,
        "mode": mode_from_active_workers(active_workers),
        "active_workers": active_workers,
        "triggered_workers": [],
        "attendance_workers": attendance_workers,
        _DISPATCH_TS_KEY: 0.0,
        "projected_shards": projected,
        "total_samples": total_samples,
        "total_train_samples": total_samples,
        "total_samples_seen": total_seen,
        "previous_anchor_path": aggregate_result.get("anchor_path"),
        "anchor_path": aggregate_result.get("anchor_path"),
        "last_updated": now,
        "updated_at": now,
    }
    if current.get("timed_out_workers"):
        next_state["timed_out_workers"] = ordered_unique_worker_ids(current.get("timed_out_workers"))
    return next_state


def _dispatch_and_save(args: Any, state: Mapping[str, Any], workers: Sequence[str]) -> dict[str, Any]:
    next_state = dict(state)
    if not getattr(args, "skip_trigger", False):
        dispatched = trigger_kaggle_workers(args, workers, state)
        next_state = reconcile_after_dispatch(state, dispatched)
    hub_upload_json(_repo_id_from_args(args), ROUND_STATE_PATH, next_state, token=getattr(args, "hf_token", None))
    return next_state


def _dispatch_timed_out_and_save(args: Any, state: Mapping[str, Any], workers: Sequence[str]) -> dict[str, Any]:
    next_state = dict(state)
    if not getattr(args, "skip_trigger", False):
        dispatched = trigger_kaggle_workers(args, workers, state)
        outcomes = {str(worker).strip().upper(): _dispatch_outcome(payload) for worker, payload in dispatched.items()}
        active_success = [
            worker
            for worker in ordered_unique_worker_ids(next_state.get("triggered_workers"))
            if outcomes.get(worker) in _SUCCESSFUL_DISPATCH_OUTCOMES
        ]
        next_state["triggered_workers"] = active_success
        next_state["active_workers"] = active_success
        next_state["mode"] = mode_from_active_workers(active_success) if active_success else str(next_state.get("mode", "waiting"))
        next_state[_DISPATCH_TS_KEY] = time.time() if active_success else 0.0
        next_state["dispatch_results"] = outcomes
    hub_upload_json(_repo_id_from_args(args), ROUND_STATE_PATH, next_state, token=getattr(args, "hf_token", None))
    return next_state


def main(
    argv: Sequence[str] | None = None,
    *,
    store: HubStateStore | None = None,
    dispatch: Callable[[Any, Sequence[str], Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> None:
    args = parse_args(argv)
    repo_id = _repo_id_from_args(args)

    if store is None:
        store = HuggingFaceHubStateStore(
            repo_id=repo_id,
            token=getattr(args, "hf_token", None),
        )
    if dispatch is None:
        dispatch = trigger_kaggle_workers

    raw_state = store.load_round_state_raw() or _initial_round_state(args)

    if args.force_worker_ids:
        forced = ordered_unique_worker_ids(str(args.force_worker_ids).split(","))
        raw_state = dict(raw_state)
        raw_state["active_workers"] = list(forced)
        raw_state["triggered_workers"] = list(forced)
        raw_state["mode"] = mode_from_active_workers(forced)

    if args.dry_run:
        print({"round_state": raw_state})
        return

    config = ProtocolConfig(
        total_train_samples=int(getattr(args, "total_samples", getattr(args, "total_train_samples", 0))),
        min_shard_samples=int(getattr(args, "min_shard_samples", 32)),
        worker_timeout_hours=float(getattr(args, "worker_timeout_hours", 13.0)),
    )
    typed_state = RoundState.from_mapping(raw_state)
    statuses = store.load_worker_statuses(list(WORKER_IDS))
    plan = plan_next_round(
        state=typed_state,
        statuses=statuses,
        credentialed_workers=list(WORKER_IDS),
        config=config,
        now=time.time(),
        force_worker_ids=str(args.force_worker_ids).split(",") if args.force_worker_ids else None,
    )

    if not plan.should_aggregate and not plan.should_dispatch:
        if plan.active_workers:
            ready = set(plan.ready_workers)
            pending = [worker for worker in plan.active_workers if worker not in ready]
            elapsed = time.time() - float(raw_state.get(_DISPATCH_TS_KEY, 0.0) or 0.0)
            print(f"[coordinator] Waiting for workers: {pending} (triggered {elapsed:.0f}s ago)")
        return

    aggregate_performed = False
    if plan.should_aggregate:
        statuses_raw: dict[str, dict[str, Any]] = {status.worker_id: dict(status.raw) for status in statuses if status.worker_id}
        aggregation_state = _round_state_for_plan(raw_state, plan)
        aggregate_result = _aggregate_ready_workers(args, aggregation_state, statuses_raw, list(plan.ready_workers))
        raw_state = _next_round_state(args, aggregation_state, aggregate_result)
        store.save_round_state_raw(raw_state)
        aggregate_performed = True

    if plan.should_dispatch:
        dispatch_state = raw_state if aggregate_performed else _round_state_for_plan(raw_state, plan)
        _save_dispatch_result(store, args, dispatch_state, dispatch)


__all__ = [name for name in globals() if not name.startswith("__")]
