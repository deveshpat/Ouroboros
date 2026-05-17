"""Named lm-eval task suites for Ouroboros benchmark launches."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

ANCHOR_TASKS = "arc_easy,hellaswag,winogrande"
REASONING_TASKS = "arc_challenge,openbookqa,piqa,gsm8k,truthfulqa_mc2"
DEFAULT_BENCHMARK_SUITE = "anchor"

BENCHMARK_TASK_SUITES: Mapping[str, str] = MappingProxyType(
    {
        "anchor": ANCHOR_TASKS,
        "reasoning": REASONING_TASKS,
    }
)


def normalize_suite_name(value: object | None) -> str:
    """Return a supported benchmark suite name, defaulting to the anchor suite."""
    text = str(value).strip().lower().replace("-", "_") if value is not None else ""
    if not text:
        return DEFAULT_BENCHMARK_SUITE
    if text not in BENCHMARK_TASK_SUITES:
        expected = ", ".join(sorted(BENCHMARK_TASK_SUITES))
        raise ValueError(f"Unknown benchmark suite {text!r}. Expected one of: {expected}.")
    return text


def resolve_benchmark_tasks(*, suite: object | None = None, tasks: object | None = None) -> str:
    """Resolve explicit task CSV or the task CSV for a named benchmark suite.

    Explicit ``tasks`` always wins so existing custom workflow/manual launches keep
    working. ``suite`` is the lower-friction path for common benchmark bundles.
    """
    task_text = str(tasks).strip() if tasks is not None else ""
    if task_text:
        return task_text
    suite_name = normalize_suite_name(suite)
    return BENCHMARK_TASK_SUITES[suite_name]


__all__ = [
    "ANCHOR_TASKS",
    "BENCHMARK_TASK_SUITES",
    "DEFAULT_BENCHMARK_SUITE",
    "REASONING_TASKS",
    "normalize_suite_name",
    "resolve_benchmark_tasks",
]
