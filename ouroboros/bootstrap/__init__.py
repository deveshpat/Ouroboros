"""Bootstrap public interface: runtime readiness, device setup, dependency checks, and executable guardrails."""
from __future__ import annotations
from . import runtime as _runtime
from . import guardrails as _guardrails

def _export_module(module):
    for name in dir(module):
        if name.startswith("__"):
            continue
        globals()[name] = getattr(module, name)

_export_module(_runtime)
_export_module(_guardrails)

__all__ = (
    "ensure_environment",
    "install_or_verify_dependencies",
    "detect_runtime_context",
    "classify_failure_log",
    "triage_failure_log",
    "format_triage",
    "documented_hard_lesson_symptoms",
)
