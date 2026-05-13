"""Pure DiLoCo worker lifecycle classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

from ouroboros.runtime_env import parse_worker_id_list, require_known_worker_id


class WorkerLifecycleKind(str, Enum):
    NORMAL_DILOCO_WORKER = "normal-diloco-worker"
    DGAC_DILOCO_WORKER = "dgac-diloco-worker"
    ATTENDANCE_ONLY = "attendance-only"
    EMPTY_SHARD_PASSTHROUGH = "empty-shard-passthrough"
    UNSUPPORTED = "unsupported"
    NOOP = "noop"


@dataclass(frozen=True)
class WorkerLifecyclePlan:
    kind: WorkerLifecycleKind
    worker_id: str
    stage_k: int
    round_n: int
    should_train: bool
    should_upload_status: bool
    should_push_signal: bool
    contributes_to_aggregation: bool
    attendance_only: bool = False
    skip_pre_validation: bool = False
    shard_samples: int = 0
    reason: str = ""


def classify_worker_lifecycle(
    *,
    worker_id: str,
    round_state: Mapping[str, Any],
    shard_samples: int,
    use_halt_gate: bool = False,
    resume_from_diloco_anchor: bool = False,
) -> WorkerLifecyclePlan:
    """Classify how a worker should behave for the current round."""
    wid = require_known_worker_id(worker_id)
    if use_halt_gate and not resume_from_diloco_anchor:
        raise ValueError("DGAC DiLoCo requires resume_from_diloco_anchor")

    stage_k = int(round_state.get("stage_k", 0))
    round_n = int(round_state.get("round_n", 0))
    triggered_raw = round_state.get("triggered_workers")
    triggered_workers: Optional[list[str]] = (
        parse_worker_id_list(triggered_raw) if triggered_raw is not None else None
    )
    attendance_workers = [
        worker for worker in parse_worker_id_list(round_state.get("attendance_workers"))
        if triggered_workers is None or worker not in set(triggered_workers)
    ]

    selected_for_training = triggered_workers is None or wid in triggered_workers
    attendance_only = wid in attendance_workers and not selected_for_training

    if not selected_for_training and not attendance_only:
        return WorkerLifecyclePlan(
            kind=WorkerLifecycleKind.NOOP,
            worker_id=wid,
            stage_k=stage_k,
            round_n=round_n,
            should_train=False,
            should_upload_status=False,
            should_push_signal=False,
            contributes_to_aggregation=False,
            shard_samples=int(shard_samples),
            reason="worker not scheduled for this round",
        )

    if attendance_only:
        return WorkerLifecyclePlan(
            kind=WorkerLifecycleKind.ATTENDANCE_ONLY,
            worker_id=wid,
            stage_k=stage_k,
            round_n=round_n,
            should_train=False,
            should_upload_status=True,
            should_push_signal=True,
            contributes_to_aggregation=False,
            attendance_only=True,
            shard_samples=0,
            reason="attendance responder only",
        )

    if int(shard_samples) <= 0:
        return WorkerLifecyclePlan(
            kind=WorkerLifecycleKind.EMPTY_SHARD_PASSTHROUGH,
            worker_id=wid,
            stage_k=stage_k,
            round_n=round_n,
            should_train=False,
            should_upload_status=True,
            should_push_signal=True,
            contributes_to_aggregation=False,
            shard_samples=0,
            reason="empty shard passthrough",
        )

    is_dgac = bool(use_halt_gate and resume_from_diloco_anchor)
    return WorkerLifecyclePlan(
        kind=WorkerLifecycleKind.DGAC_DILOCO_WORKER if is_dgac else WorkerLifecycleKind.NORMAL_DILOCO_WORKER,
        worker_id=wid,
        stage_k=stage_k,
        round_n=round_n,
        should_train=True,
        should_upload_status=True,
        should_push_signal=True,
        contributes_to_aggregation=True,
        skip_pre_validation=False,
        shard_samples=int(shard_samples),
        reason="DGAC DiLoCo active shard" if is_dgac else "normal DiLoCo active shard",
    )
