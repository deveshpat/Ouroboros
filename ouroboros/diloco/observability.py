"""Observability module for DiLoCo run identity and metric axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class WandbIdentity:
    project: str
    entity: Optional[str]
    run_id: str
    group: Optional[str]
    name: str
    step_offset: int = 0


def worker_wandb_identity(
    *,
    worker_id: str,
    stage_k: int,
    round_n: int,
    shard_step_estimate: int,
    project: str,
    entity: Optional[str] = None,
) -> WandbIdentity:
    """Return the per-round worker W&B identity.

    The run ID is unique per round to avoid accidental resume collisions; the
    group remains stable per worker/stage so the UI still clusters related runs.
    """

    wid = str(worker_id).strip().upper()
    return WandbIdentity(
        project=project,
        entity=entity,
        run_id=f"diloco-worker-{wid}-s{int(stage_k)}-r{int(round_n)}",
        group=f"diloco-worker-{wid}-s{int(stage_k)}",
        name=f"Worker {wid} | Stage {int(stage_k)} | Round {int(round_n)}",
        step_offset=max(int(round_n), 0) * max(int(shard_step_estimate), 0),
    )


def coordinator_wandb_identity(
    *,
    stage_k: int,
    project: str,
    entity: Optional[str] = None,
) -> WandbIdentity:
    """Return the coordinator W&B identity for a stage."""

    return WandbIdentity(
        project=project,
        entity=entity,
        run_id=f"diloco-coordinator-s{int(stage_k)}",
        group="diloco-coordinator",
        name=f"Coordinator | Stage {int(stage_k)}",
        step_offset=0,
    )


def redact_mapping(values: Mapping[str, object]) -> dict[str, object]:
    """Redact common token/key fields before logging configs or env maps."""

    redacted: dict[str, object] = {}
    for key, value in values.items():
        lower = str(key).lower()
        if any(marker in lower for marker in ("token", "key", "secret", "password")):
            redacted[str(key)] = "***" if value else value
        else:
            redacted[str(key)] = value
    return redacted
