"""Inference public interface: prompt-to-generation runtime."""
from __future__ import annotations
from . import generation as _generation

def _export_module(module):
    for name in dir(module):
        if name.startswith("__"):
            continue
        globals()[name] = getattr(module, name)

_export_module(_generation)

__all__ = (
    "InferenceResult",
    "resolve_prompt",
    "resolve_device",
    "resolve_dtype",
    "load_components",
    "run_single_prompt",
    "main",
)
