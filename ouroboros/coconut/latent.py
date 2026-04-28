"""Latent reasoning seam for Coconut stage execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ForwardResult:
    loss: Any
    logits: Any = None
    hidden_states: Any = None
    metrics: Mapping[str, float] | None = None


class LatentReasoner:
    """Adapter around the model-specific latent forward path.

    The full Jamba/Coconut implementation remains in the legacy script until the
    behavior is characterized. This seam gives the training runtime a stable name
    for the concept and a testable interface for future extraction.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def forward_stage(self, batch: Mapping[str, Any], stage_k: int) -> ForwardResult:
        if not hasattr(self.model, "__call__"):
            raise TypeError("LatentReasoner requires a callable model")
        result = self.model(**dict(batch))
        return ForwardResult(
            loss=getattr(result, "loss", None),
            logits=getattr(result, "logits", None),
            hidden_states=getattr(result, "hidden_states", None),
            metrics={"stage_k": float(stage_k)},
        )
