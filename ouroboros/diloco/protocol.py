"""Deep DiLoCo round-protocol module.

The coordinator script should not need to know the details of worker ordering,
projected shard math, attendance demotion, dispatch reconciliation, or the
"triggered_at == 0" sentinel. Those rules live here so the interface is the test
surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

WORKER_IDS: Tuple[str, ...] = ("A", "B", "C")
RoundMode = Literal["complete", "solo", "diloco", "waiting"]
DispatchOutcome = Literal["success", "manual", "failed", "triggered", "missing_credentials", "quota_exhausted"]


@dataclass(frozen=True)
class WorkerStatus:
    """Normalized worker status uploaded by a DiLoCo worker."""

    worker_id: str
    stage_k: int
    round_n: int
    status: str
    samples_seen: int = 0
    weights_path: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "WorkerStatus":
        return cls(
            worker_id=str(payload.get("worker_id", "")).strip().upper(),
            stage_k=int(payload.get("stage_k", -1)),
            round_n=int(payload.get("round_n", -1)),
            status=str(payload.get("status", "")),
            samples_seen=int(payload.get("samples_seen", 0) or 0),
            weights_path=(
                str(payload.get("weights_path"))
                if payload.get("weights_path") is not None
                else None
            ),
            raw=dict(payload),
        )

    def is_done_for(self, stage_k: int, round_n: int) -> bool:
        return self.stage_k == int(stage_k) and self.round_n == int(round_n) and self.status == "done"


@dataclass(frozen=True)
class RoundState:
    """Normalized subset of round_state.json used for protocol decisions."""

    stage_k: int
    round_n: int
    mode: RoundMode
    total_samples_seen: Mapping[str, int]
    triggered_workers: Tuple[str, ...] = ()
    attendance_workers: Tuple[str, ...] = ()
    triggered_at: float = 0.0
    seed: int = 42
    completed_stages: Tuple[int, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RoundState":
        return cls(
            stage_k=int(payload.get("stage_k", 0)),
            round_n=int(payload.get("round_n", 0)),
            mode=str(payload.get("mode", "diloco")),  # type: ignore[arg-type]
            total_samples_seen={str(k): int(v) for k, v in dict(payload.get("total_samples_seen", {})).items()},
            triggered_workers=tuple(ordered_unique_worker_ids(payload.get("triggered_workers"))),
            attendance_workers=tuple(ordered_unique_worker_ids(payload.get("attendance_workers"))),
            triggered_at=float(payload.get("triggered_at", 0.0) or 0.0),
            seed=int(payload.get("seed", 42)),
            completed_stages=tuple(int(x) for x in payload.get("completed_stages", []) or []),
            raw=dict(payload),
        )

    @property
    def stage_samples_seen(self) -> int:
        return int(dict(self.total_samples_seen).get(str(self.stage_k), 0))


@dataclass(frozen=True)
class ProtocolConfig:
    total_train_samples: int
    min_shard_samples: int = 32
    worker_timeout_hours: float = 13.0
    worker_ids: Tuple[str, ...] = WORKER_IDS

    @property
    def worker_timeout_seconds(self) -> float:
        return float(self.worker_timeout_hours) * 3600.0


@dataclass(frozen=True)
class RoundPlan:
    mode: RoundMode
    active_workers: Tuple[str, ...]
    attendance_workers: Tuple[str, ...]
    projected_shards: Mapping[str, int]
    should_aggregate: bool
    should_dispatch: bool
    should_advance_stage: bool
    timed_out_workers: Tuple[str, ...] = ()
    ready_workers: Tuple[str, ...] = ()
    attendance_ready_workers: Tuple[str, ...] = ()
    reason: str = ""


def _normalize_worker_id(worker_id: Any) -> Optional[str]:
    wid = str(worker_id).strip().upper()
    return wid if wid in WORKER_IDS else None


def ordered_unique_worker_ids(*groups: Optional[Iterable[Any]]) -> List[str]:
    """Return valid worker IDs in first-seen order."""

    ordered: List[str] = []
    seen = set()
    for group in groups:
        if group is None:
            continue
        for worker_id in group:
            wid = _normalize_worker_id(worker_id)
            if wid is None or wid in seen:
                continue
            ordered.append(wid)
            seen.add(wid)
    return ordered


def compute_projected_shards(
    *,
    total_samples: int,
    total_samples_seen: int,
    worker_ids: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """Compute deterministic three-way projected shard sizes.

    The worker partition count intentionally remains 3 even when some workers
    are inactive. This preserves shard determinism across A/B/C and matches the
    existing worker-side partitioning contract.
    """

    ids = ordered_unique_worker_ids(worker_ids or WORKER_IDS)
    index = {wid: idx for idx, wid in enumerate(WORKER_IDS)}
    remaining = max(int(total_samples) - int(total_samples_seen), 0)
    base = remaining // len(WORKER_IDS)
    remainder = remaining % len(WORKER_IDS)
    return {
        wid: max(base + (1 if index.get(wid, 0) < remainder else 0), 0)
        for wid in ids
    }


def determine_round_mode(
    *,
    projected_shards: Mapping[str, int],
    credentialed_workers: Sequence[str],
    min_shard_samples: int,
    force_worker_ids: Optional[Sequence[str]] = None,
) -> Tuple[RoundMode, List[str]]:
    """Determine the next mode and active workers for a training round."""

    credentialed = ordered_unique_worker_ids(credentialed_workers)
    if force_worker_ids:
        active = [w for w in ordered_unique_worker_ids(force_worker_ids) if w in credentialed]
        if not active:
            return "complete", []
        return ("solo" if len(active) == 1 else "diloco"), active

    active = [
        worker_id
        for worker_id in credentialed
        if int(projected_shards.get(worker_id, 0) or 0) >= int(min_shard_samples)
    ]
    if not active:
        return "complete", []
    return ("solo" if len(active) == 1 else "diloco"), active


def partition_ready_workers(
    ready_workers: Sequence[Mapping[str, Any]],
    *,
    expected_workers: Optional[Sequence[str]],
    attendance_workers: Optional[Sequence[str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split ready statuses into active completions and attendance check-ins."""

    expected_set = set(ordered_unique_worker_ids(expected_workers))
    attendance_set = set(ordered_unique_worker_ids(attendance_workers))
    active_ready: List[Dict[str, Any]] = []
    attendance_ready: List[Dict[str, Any]] = []
    for status in ready_workers:
        payload = dict(status)
        worker_id = _normalize_worker_id(payload.get("worker_id"))
        if worker_id in attendance_set and worker_id not in expected_set:
            attendance_ready.append(payload)
        else:
            active_ready.append(payload)
    return active_ready, attendance_ready


def mode_from_active_workers(active_workers: Sequence[str], fallback: RoundMode = "complete") -> RoundMode:
    active = ordered_unique_worker_ids(active_workers)
    if not active:
        return fallback
    return "solo" if len(active) == 1 else "diloco"


def reconcile_post_dispatch_state(
    *,
    state: Mapping[str, Any],
    planned_active_workers: Sequence[str],
    planned_attendance_workers: Sequence[str],
    dispatch_results: Mapping[str, str],
    now: float,
) -> Optional[Dict[str, Any]]:
    """Correct round_state after one or more Kaggle dispatch failures."""

    failed_workers = [w for w, status in dispatch_results.items() if status == "failed"]
    if not failed_workers:
        return None

    dispatched_active = [
        w for w in planned_active_workers if dispatch_results.get(w) in {"success", "manual", "triggered"}
    ]
    dispatched_attendance = [
        w for w in planned_attendance_workers if dispatch_results.get(w) in {"success", "manual", "triggered"}
    ]
    failed_active = [w for w in planned_active_workers if dispatch_results.get(w) == "failed"]
    failed_attendance = [w for w in planned_attendance_workers if dispatch_results.get(w) == "failed"]

    corrected_triggered_workers = ordered_unique_worker_ids(dispatched_active)
    corrected_attendance_workers = ordered_unique_worker_ids(
        dispatched_attendance,
        failed_active,
        failed_attendance,
    )
    corrected_mode: RoundMode = str(state.get("mode", "complete"))  # type: ignore[assignment]
    if corrected_triggered_workers:
        corrected_mode = mode_from_active_workers(corrected_triggered_workers, fallback=corrected_mode)
    elif corrected_attendance_workers:
        corrected_mode = "waiting"

    outstanding_dispatches = ordered_unique_worker_ids(corrected_triggered_workers, corrected_attendance_workers)
    successful_or_manual = [
        worker_id
        for worker_id in outstanding_dispatches
        if dispatch_results.get(worker_id) in {"success", "manual", "triggered"}
    ]

    return {
        **dict(state),
        "mode": corrected_mode,
        "triggered_workers": corrected_triggered_workers,
        "attendance_workers": corrected_attendance_workers,
        "triggered_at": float(now) if successful_or_manual else 0.0,
        "last_updated": float(now),
        "dispatch_failures": failed_workers,
    }


def plan_next_round(
    *,
    state: RoundState,
    statuses: Sequence[WorkerStatus],
    credentialed_workers: Sequence[str],
    config: ProtocolConfig,
    now: float,
    force_worker_ids: Optional[Sequence[str]] = None,
) -> RoundPlan:
    """Plan the next coordinator action without touching Hub, W&B, or Kaggle."""

    expected_workers = ordered_unique_worker_ids(state.triggered_workers)
    attendance_prev = [
        worker_id
        for worker_id in ordered_unique_worker_ids(state.attendance_workers)
        if worker_id not in set(expected_workers)
    ]
    projected_shards = compute_projected_shards(
        total_samples=config.total_train_samples,
        total_samples_seen=state.stage_samples_seen,
        worker_ids=config.worker_ids,
    )
    remaining = max(config.total_train_samples - state.stage_samples_seen, 0)

    ready_payloads = [dict(status.raw or {
        "worker_id": status.worker_id,
        "stage_k": status.stage_k,
        "round_n": status.round_n,
        "status": status.status,
        "samples_seen": status.samples_seen,
    }) for status in statuses if status.is_done_for(state.stage_k, state.round_n)]
    active_ready, attendance_ready = partition_ready_workers(
        ready_payloads,
        expected_workers=expected_workers,
        attendance_workers=attendance_prev,
    )
    ready_ids = set(ordered_unique_worker_ids([s.get("worker_id") for s in active_ready]))
    attendance_ready_ids = set(ordered_unique_worker_ids([s.get("worker_id") for s in attendance_ready]))

    if remaining < int(config.min_shard_samples):
        return RoundPlan(
            mode="complete",
            active_workers=(),
            attendance_workers=tuple(attendance_prev),
            projected_shards=projected_shards,
            should_aggregate=False,
            should_dispatch=False,
            should_advance_stage=True,
            ready_workers=tuple(sorted(ready_ids)),
            attendance_ready_workers=tuple(sorted(attendance_ready_ids)),
            reason="remaining samples below min_shard_samples",
        )

    if expected_workers:
        missing = [w for w in expected_workers if w not in ready_ids]
        if missing:
            if state.triggered_at <= 0:
                return RoundPlan(
                    mode=state.mode,
                    active_workers=tuple(expected_workers),
                    attendance_workers=tuple(attendance_prev),
                    projected_shards=projected_shards,
                    should_aggregate=False,
                    should_dispatch=True,
                    should_advance_stage=False,
                    ready_workers=tuple(sorted(ready_ids)),
                    attendance_ready_workers=tuple(sorted(attendance_ready_ids)),
                    reason="dispatch timestamp sentinel is zero; retry dispatch",
                )
            timed_out = (float(now) - float(state.triggered_at)) > config.worker_timeout_seconds
            if not timed_out:
                return RoundPlan(
                    mode=state.mode,
                    active_workers=tuple(expected_workers),
                    attendance_workers=tuple(attendance_prev),
                    projected_shards=projected_shards,
                    should_aggregate=False,
                    should_dispatch=False,
                    should_advance_stage=False,
                    ready_workers=tuple(sorted(ready_ids)),
                    attendance_ready_workers=tuple(sorted(attendance_ready_ids)),
                    reason=f"waiting for active workers: {missing}",
                )
            survivors = [w for w in expected_workers if w in ready_ids]
            demoted = ordered_unique_worker_ids(attendance_prev, missing)
            if not survivors:
                return RoundPlan(
                    mode="waiting",
                    active_workers=(),
                    attendance_workers=tuple(demoted),
                    projected_shards=projected_shards,
                    should_aggregate=False,
                    should_dispatch=True,
                    should_advance_stage=False,
                    timed_out_workers=tuple(missing),
                    ready_workers=tuple(sorted(ready_ids)),
                    attendance_ready_workers=tuple(sorted(attendance_ready_ids)),
                    reason="all active workers timed out; enter attendance waiting mode",
                )
            return RoundPlan(
                mode=mode_from_active_workers(survivors),
                active_workers=tuple(survivors),
                attendance_workers=tuple(demoted),
                projected_shards=projected_shards,
                should_aggregate=True,
                should_dispatch=True,
                should_advance_stage=False,
                timed_out_workers=tuple(missing),
                ready_workers=tuple(sorted(ready_ids)),
                attendance_ready_workers=tuple(sorted(attendance_ready_ids)),
                reason="timed-out workers demoted to attendance",
            )

    eligible = [w for w in credentialed_workers if w not in set(attendance_prev)]
    next_mode, active = determine_round_mode(
        projected_shards=projected_shards,
        credentialed_workers=eligible,
        min_shard_samples=config.min_shard_samples,
        force_worker_ids=force_worker_ids,
    )
    return RoundPlan(
        mode=next_mode,
        active_workers=tuple(active),
        attendance_workers=tuple(attendance_prev),
        projected_shards=projected_shards,
        should_aggregate=False,
        should_dispatch=bool(active or attendance_prev),
        should_advance_stage=next_mode == "complete",
        ready_workers=tuple(sorted(ready_ids)),
        attendance_ready_workers=tuple(sorted(attendance_ready_ids)),
        reason="fresh round planning",
    )
