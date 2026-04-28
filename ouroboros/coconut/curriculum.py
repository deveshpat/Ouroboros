"""Coconut curriculum, dataset, and shard utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:
    from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
except Exception:  # pragma: no cover - optional in unit tests
    Dataset = Any  # type: ignore
    DatasetDict = Any  # type: ignore
    load_dataset = None  # type: ignore

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
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return [value]
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [str(parsed)]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
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


def get_max_stage(records: Iterable[Mapping[str, Any]]) -> int:
    max_steps = 0
    for record in records:
        if "n_steps" in record:
            try:
                max_steps = max(max_steps, int(record["n_steps"]))
                continue
            except Exception:
                pass
        max_steps = max(max_steps, len(normalize_steps(record.get("steps", []))))
    return max_steps


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_canonical_dataset(data_dir: str | Path) -> Any:
    """Load the canonical Coconut dataset from a directory or JSONL file.

    Supports the legacy local layout without importing ``datasets`` at module
    import time. When Hugging Face datasets is installed, returns a Dataset or
    DatasetDict; otherwise returns a plain list for tests/lightweight tooling.
    """

    path = Path(data_dir)
    if path.is_file() and path.suffix == ".jsonl":
        rows = _load_jsonl(path)
        return Dataset.from_list(rows) if hasattr(Dataset, "from_list") else rows

    for name in ("train.jsonl", "dataset.jsonl", "data.jsonl"):
        candidate = path / name
        if candidate.exists():
            rows = _load_jsonl(candidate)
            return Dataset.from_list(rows) if hasattr(Dataset, "from_list") else rows

    if load_dataset is None:
        raise FileNotFoundError(f"no JSONL dataset found under {path}")
    return load_dataset(str(path))


def partition_stage_shard(
    *,
    total_samples: int,
    total_seen: int,
    worker_id: str,
    worker_ids: Sequence[str] = WORKER_IDS,
) -> Shard:
    wid = str(worker_id).strip().upper()
    ordered = tuple(str(w).strip().upper() for w in worker_ids)
    if wid not in ordered:
        raise ValueError(f"invalid worker_id: {worker_id!r}")
    remaining = max(int(total_samples) - int(total_seen), 0)
    projected = compute_projected_shards(
        total_samples=total_samples,
        total_samples_seen=total_seen,
        worker_ids=ordered,
    )
    index = {w: i for i, w in enumerate(ordered)}[wid]
    start_offset = sum(int(projected.get(w, 0)) for w in ordered[:index])
    start = int(total_seen) + start_offset
    size = int(projected.get(wid, 0))
    return Shard(worker_id=wid, start=start, end=start + size, size=size, remaining=remaining)


def partition_stage_shards(
    *, total_samples: int, total_seen: int, worker_ids: Sequence[str] = WORKER_IDS
) -> Dict[str, Shard]:
    return {
        str(wid).strip().upper(): partition_stage_shard(
            total_samples=total_samples,
            total_seen=total_seen,
            worker_id=wid,
            worker_ids=worker_ids,
        )
        for wid in worker_ids
    }


__all__ = [
    "Shard",
    "StageSample",
    "build_stage_sample",
    "get_max_stage",
    "load_canonical_dataset",
    "normalize_steps",
    "partition_stage_shard",
    "partition_stage_shards",
]
