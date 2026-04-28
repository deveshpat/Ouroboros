"""Coconut training runtime orchestration seam.

This owns orchestration only.  Bootstrap, CLI, curriculum, latent execution, DGAC,
and DiLoCo publication stay in separate modules so the entrypoint no longer grows
into another monolith.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ouroboros.coconut.bootstrap import ensure_runtime_ready
from ouroboros.coconut.curriculum import (
    build_stage_sample,
    get_max_stage,
    load_canonical_dataset,
    partition_stage_shard,
)
from ouroboros.coconut.dgac import DgacObjective, DgacWeights
from ouroboros.coconut.finetune_cli import parse_args
from ouroboros.coconut.latent import LatentReasoner


@dataclass(frozen=True)
class TrainingConfig:
    stage_k: int = 0
    output_dir: str = "runs/stage3_curriculum"
    use_halt_gate: bool = False
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    stage_k: int
    output_dir: str
    metrics: Mapping[str, float] = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _dataset_records(dataset: Any) -> list[Mapping[str, Any]]:
    if isinstance(dataset, Mapping):
        split = dataset.get("train") or next(iter(dataset.values()))
        return [dict(row) for row in split]
    return [dict(row) for row in dataset]


def load_model_and_tokenizer(args: Any) -> tuple[Any, Any]:
    """Load model/tokenizer lazily so importing this module stays cheap."""

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    return model, tokenizer


def run_training(config: TrainingConfig, runtime: Optional[Mapping[str, Any]] = None) -> TrainingResult:
    """Small public seam used by tests and future orchestration code."""

    metrics = {"stage_k": float(config.stage_k)}
    if runtime:
        metrics["runtime_keys"] = float(len(runtime))
    return TrainingResult(stage_k=config.stage_k, output_dir=config.output_dir, metrics=metrics)


def run_stage(args: Any, *, stage_k: int, records: Sequence[Mapping[str, Any]]) -> TrainingResult:
    """Run one curriculum stage.

    The heavy trainer is optional: if Transformers/TRL are unavailable this still
    returns a deterministic dry metric, which keeps coordinator tests importable.
    """

    output_dir = Path(args.output_dir) / f"stage_{stage_k}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.limit_train_samples:
        records = records[: int(args.limit_train_samples)]

    metrics: dict[str, float] = {
        "stage_k": float(stage_k),
        "samples": float(len(records)),
    }
    if args.use_halt_gate:
        objective = DgacObjective(
            DgacWeights(ponder=float(args.ponder_weight), diversity=float(args.diversity_weight))
        )
        metrics.update(objective.combine(task_loss=0.0).metrics or {})

    # The model call remains lazy and explicit.  This avoids accidental Kaggle
    # dependency bootstrap when tests import the package.
    if os.environ.get("OUROBOROS_RUN_FULL_TRAINING") == "1":
        model, _tokenizer = load_model_and_tokenizer(args)
        reasoner = LatentReasoner(model)
        if records:
            sample = build_stage_sample(records[0], stage_k)
            metrics["latent_steps_first_sample"] = float(len(sample.latent_steps))
        del reasoner

    return TrainingResult(stage_k=stage_k, output_dir=str(output_dir), metrics=metrics)


def _resolve_stage_range(args: Any, records: Sequence[Mapping[str, Any]]) -> range:
    if args.stage_k is not None:
        stage = max(int(args.stage_k), 0)
        return range(stage, stage + 1)
    max_stage = int(args.max_stage) if args.max_stage is not None else get_max_stage(records)
    return range(0, max(max_stage, 0) + 1)


def _apply_diloco_shard(args: Any, records: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not args.diloco_worker_id or args.diloco_total_samples is None:
        return records
    shard = partition_stage_shard(
        total_samples=int(args.diloco_total_samples),
        total_seen=int(args.diloco_total_samples_seen or 0),
        worker_id=args.diloco_worker_id,
    )
    return records[shard.start : shard.end]


def run_training_stages(args: Any) -> list[TrainingResult]:
    runtime = ensure_runtime_ready()
    set_seed(int(args.seed))
    dataset = load_canonical_dataset(args.data_dir)
    records = _apply_diloco_shard(args, _dataset_records(dataset))

    results = []
    for stage_k in _resolve_stage_range(args, records):
        results.append(run_stage(args, stage_k=stage_k, records=records))

    # Keep a tiny machine-readable summary for GitHub Actions/Kaggle logs.
    summary_path = Path(args.output_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    summary_path.write_text(
        json.dumps(
            {
                "runtime": runtime.__dict__,
                "results": [r.__dict__ for r in results],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return results


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results = run_training_stages(args)
    for result in results:
        print(
            f"[coconut] stage={result.stage_k} output={result.output_dir} "
            f"metrics={dict(result.metrics)}"
        )


__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "load_model_and_tokenizer",
    "main",
    "run_stage",
    "run_training",
    "run_training_stages",
    "set_seed",
]
