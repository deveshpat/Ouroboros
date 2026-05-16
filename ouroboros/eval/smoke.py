"""Eval smoke checks.

Small import-level gate for local validation. No CPU workflow-validation branch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module


@dataclass(frozen=True)
class EvalSmokeReport:
    """Result from lightweight eval package smoke."""

    imported: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_eval_smoke() -> EvalSmokeReport:
    """Import eval backends used by current quality gates."""

    modules = (
        "ouroboros.eval.benchmark_harness",
        "ouroboros.eval.benchmark_multi_gpu",
    )
    for module in modules:
        import_module(module)
    return EvalSmokeReport(imported=modules)


__all__ = ["EvalSmokeReport", "run_eval_smoke"]
