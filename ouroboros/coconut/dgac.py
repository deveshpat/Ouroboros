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

    def combine(
        self,
        *,
        task_loss,
        ponder_loss=0.0,
        diversity_loss=0.0,
        metrics: Optional[Mapping[str, float]] = None,
    ) -> DgacLoss:
        total = task_loss + self.weights.ponder * ponder_loss + self.weights.diversity * diversity_loss
        merged = {
            "dgac_ponder_weight": float(self.weights.ponder),
            "dgac_diversity_weight": float(self.weights.diversity),
        }
        merged.update(dict(metrics or {}))
        return DgacLoss(
            total=total,
            task=task_loss,
            ponder=ponder_loss,
            diversity=diversity_loss,
            metrics=merged,
        )


class HaltGate:
    """Small halt-gate head wrapper isolated from the training loop."""

    def __init__(self, hidden_size: int) -> None:
        try:
            import torch.nn as nn  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("HaltGate requires torch") from exc
        self.module = nn.Linear(hidden_size, 1)

    def __call__(self, hidden_states):
        return self.module(hidden_states).squeeze(-1)


def compute_dgac_lambda1(stage_k: int, *, base: float = 0.01, max_value: float = 0.1) -> float:
    """Progressively increase ponder regularization with curriculum depth."""

    return min(float(max_value), float(base) * max(int(stage_k), 1))


__all__ = [
    "DgacLoss",
    "DgacObjective",
    "DgacWeights",
    "HaltGate",
    "compute_dgac_lambda1",
]
