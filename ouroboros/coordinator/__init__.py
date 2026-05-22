"""Coordinator public interface: orchestration, workers, dispatch, and aggregation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "main": ("coordinator", "main"),
    "parse_args": ("coordinator", "parse_args"),
    "plan_round_start": ("decision", "plan_round_start"),
    "trigger_kaggle_workers": ("dispatch", "trigger_kaggle_workers"),
    "run_diloco_worker": ("worker", "run_diloco_worker"),
    "RoundState": ("shared", "RoundState"),
    "CoordinatorTransitionDecision": ("decision", "CoordinatorTransitionDecision"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attr_name)
    globals()[name] = value
    return value
