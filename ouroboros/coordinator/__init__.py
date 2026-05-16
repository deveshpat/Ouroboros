"""Coordinator public interface: orchestration, workers, dispatch, aggregation, and launch planning."""

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
    "build_launch_command": ("kaggle_launch_matrix", "build_launch_command"),
    "build_diloco_training_command": ("kaggle_commands", "build_diloco_training_command"),
    "format_shell_command": ("kaggle_commands", "format_shell_command"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attr_name)
    globals()[name] = value
    return value
