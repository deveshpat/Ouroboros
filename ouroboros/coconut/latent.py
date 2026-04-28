"""Latent reasoning seam for Coconut stage execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ForwardResult:
    loss: Any
    logits: Any = None
    hidden_states: Any = None
    metrics: Mapping[str, float] | None = None


class LatentReasoner:
    """Adapter around the model-specific latent forward path."""

    def __init__(self, model: Any) -> None:
        if not hasattr(model, "__call__"):
            raise TypeError("LatentReasoner requires a callable model")
        self.model = model

    def forward_stage(self, batch: Mapping[str, Any], stage_k: int) -> ForwardResult:
        result = self.model(**dict(batch))
        return ForwardResult(
            loss=getattr(result, "loss", None),
            logits=getattr(result, "logits", None),
            hidden_states=getattr(result, "hidden_states", None),
            metrics={"stage_k": float(stage_k)},
        )


def build_question_context(question: str, visible_steps: list[str]) -> str:
    parts = [f"Question: {question.strip()}", "Reasoning:"]
    parts.extend(step.strip() for step in visible_steps if str(step).strip())
    return "\n".join(parts).strip()


def compute_ce_sum_and_count(loss, labels) -> tuple[object, int]:
    """Return a comparable CE sum/count without forcing torch at import time."""

    try:
        count = int((labels != -100).sum().item())
    except Exception:
        count = 0
    return loss * max(count, 1), count


__all__ = [
    "ForwardResult",
    "LatentReasoner",
    "build_question_context",
    "compute_ce_sum_and_count",
]
