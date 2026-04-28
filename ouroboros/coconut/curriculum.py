"""Coconut curriculum module.

This module keeps stage/sample/shard behavior local to the Coconut domain instead
of scattering it across coordinator and worker scripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ouroboros.diloco.protocol import WORKER_IDS, compute_projected_shards


@dataclass(frozen=True)
class Shard:
    worker_id: str
    start: int
    end: int
    size: int
    remaining: int


@dataclass(frozen=True)
class StageSample:
    question: str
    visible_steps: List[str]
    latent_steps: List[str]
    answer_full: str
    answer_norm: str
    stage_k: int


def normalize_steps(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            return [value]
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def build_stage_sample(example: Mapping[str, Any], stage_k: int) -> StageSample:
    steps = normalize_steps(example.get("steps", []))
    k = max(int(stage_k), 0)
    latent_steps = steps[:k]
    visible_steps = steps[k:]
    return StageSample(
        question=str(example.get("question", "")),
        visible_steps=visible_steps,
        latent_steps=latent_steps,
        answer_full=str(example.get("answer_full", example.get("answer", ""))),
        answer_norm=str(example.get("answer_norm", "")),
        stage_k=k,
    )


def partition_stage_shard(
    *,
    total_samples: int,
    total_seen: int,
    worker_id: str,
    worker_ids: Sequence[str] = WORKER_IDS,
) -> Shard:
    wid = str(worker_id).strip().upper()
    if wid not in WORKER_IDS:
        raise ValueError(f"invalid worker_id: {worker_id!r}")
    remaining = max(int(total_samples) - int(total_seen), 0)
    projected = compute_projected_shards(total_samples=total_samples, total_samples_seen=total_seen, worker_ids=worker_ids)
    index = {w: i for i, w in enumerate(WORKER_IDS)}[wid]
    start_offset = sum(projected.get(w, 0) for w in WORKER_IDS[:index])
    start = int(total_seen) + start_offset
    size = int(projected.get(wid, 0))
    return Shard(worker_id=wid, start=start, end=start + size, size=size, remaining=remaining)


def partition_stage_shards(*, total_samples: int, total_seen: int, worker_ids: Sequence[str] = WORKER_IDS) -> Dict[str, Shard]:
    return {
        wid: partition_stage_shard(total_samples=total_samples, total_seen=total_seen, worker_id=wid, worker_ids=worker_ids)
        for wid in worker_ids
    }
