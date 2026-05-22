"""Bootstrap public interface: runtime readiness and executable guardrails."""
from __future__ import annotations

from .guardrails import (
    HARD_LESSON_GUARDRAILS,
    HardLessonGuardrail,
    classify_failure_log,
    documented_hard_lesson_symptoms,
    duplicate_guardrail_symptoms,
    format_triage,
    guardrail_by_symptom,
    triage_failure_log,
    triage_failure_log_path,
    unguarded_documented_lessons,
)
from .runtime import ensure_environment, run_shared_install_preflight

__all__ = [
    "HARD_LESSON_GUARDRAILS",
    "HardLessonGuardrail",
    "classify_failure_log",
    "documented_hard_lesson_symptoms",
    "duplicate_guardrail_symptoms",
    "ensure_environment",
    "format_triage",
    "guardrail_by_symptom",
    "run_shared_install_preflight",
    "triage_failure_log",
    "triage_failure_log_path",
    "unguarded_documented_lessons",
]
