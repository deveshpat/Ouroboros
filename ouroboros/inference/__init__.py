"""Inference public interface: prompt-to-generation runtime."""
from __future__ import annotations

from .generation import (
    InferenceResult,
    load_components,
    main,
    resolve_device,
    resolve_dtype,
    resolve_prompt,
    run_single_prompt,
)

__all__ = [
    "InferenceResult",
    "load_components",
    "main",
    "resolve_device",
    "resolve_dtype",
    "resolve_prompt",
    "run_single_prompt",
]
