"""Eval public interface: benchmark, lm-eval, diagnostics, and smoke quality gates."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "build_lm_eval_argv": ("benchmark_harness", "build_lm_eval_argv"),
    "run_lm_eval": ("benchmark_harness", "run_lm_eval"),
    "main": ("benchmark_harness", "main"),
    "build_sharded_lm_eval_benchmark_commands": ("benchmark_multi_gpu", "build_sharded_lm_eval_benchmark_commands"),
    "run_cpu_smoke_validation": ("smoke", "run_cpu_smoke_validation"),
    "build_cpu_smoke_validation_command": ("smoke", "build_cpu_smoke_validation_command"),
    "workflow_validation_mode": ("smoke", "workflow_validation_mode"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attr_name)
    globals()[name] = value
    return value
