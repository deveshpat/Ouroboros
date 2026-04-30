"""Coconut training runtime orchestration seam."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


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


def run_training(config: TrainingConfig, runtime: Optional[Mapping[str, Any]] = None) -> TrainingResult:
    """Minimal orchestration hook for future extraction from the legacy script."""

    metrics = {"stage_k": float(config.stage_k)}
    if runtime:
        metrics["runtime_keys"] = float(len(runtime))
    return TrainingResult(stage_k=config.stage_k, output_dir=config.output_dir, metrics=metrics)
