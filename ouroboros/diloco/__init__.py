"""DiLoCo coordination modules for Project Ouroboros."""

from .protocol import (
    WORKER_IDS,
    ProtocolConfig,
    RoundPlan,
    RoundState,
    WorkerStatus,
    compute_projected_shards,
    determine_round_mode,
    ordered_unique_worker_ids,
    partition_ready_workers,
    plan_next_round,
    reconcile_post_dispatch_state,
)

__all__ = [
    "WORKER_IDS",
    "ProtocolConfig",
    "RoundPlan",
    "RoundState",
    "WorkerStatus",
    "compute_projected_shards",
    "determine_round_mode",
    "ordered_unique_worker_ids",
    "partition_ready_workers",
    "plan_next_round",
    "reconcile_post_dispatch_state",
]
