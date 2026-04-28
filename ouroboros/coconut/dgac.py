"""DGAC halt-gate objective seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class DgacLoss:
    total: object
    task: object
    ponder: object | float = 0.0
    diversity: object | float = 0.0
    metrics: Mapping[str, float] | None = None


@dataclass(frozen=True)
class DgacWeights:
    ponder: float = 0.01
    diversity: float = 0.01


class DgacObjective:
    def __init__(self, weights: DgacWeights | None = None) -> None:
        self.weights = weights or DgacWeights()

    def combine(self, *, task_loss, ponder_loss=0.0, diversity_loss=0.0, metrics: Optional[Mapping[str, float]] = None) -> DgacLoss:
        total = task_loss + self.weights.ponder * ponder_loss + self.weights.diversity * diversity_loss
        return DgacLoss(total=total, task=task_loss, ponder=ponder_loss, diversity=diversity_loss, metrics=dict(metrics or {}))
