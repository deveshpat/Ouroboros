"""Artifact-only gate before widening architecture experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ouroboros.eval.artifacts import write_json


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _count_jsonl_rows(path: Path) -> int | None:
    try:
        with path.open(encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return None


def _check(name: str, passed: bool, detail: str, *, severity: str = "blocker", **extra: Any) -> dict[str, Any]:
    status = "passed" if passed else ("warning" if severity == "warning" else "failed")
    payload: dict[str, Any] = {
        "name": name,
        "status": status,
        "severity": severity,
        "detail": detail,
    }
    payload.update(extra)
    return payload


def _diagnostics(summary: dict[str, Any]) -> dict[str, Any]:
    diagnostics = summary.get("diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else {}


def _token_budget_clean(summary: dict[str, Any]) -> bool:
    token_budget = _diagnostics(summary).get("token_budget")
    if not isinstance(token_budget, dict):
        return False
    return not bool(token_budget.get("any_prompt_truncated"))


def _release_gate_passed(summary: dict[str, Any]) -> bool:
    gate = summary.get("release_gate")
    return isinstance(gate, dict) and bool(gate.get("passed"))


def _halt_gate_suspect(summary: dict[str, Any]) -> bool:
    return bool(_diagnostics(summary).get("halt_gate_suspect"))


def _comparison_checks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    root = Path(args.comparison_dir).expanduser()
    summary_path = root / "summary.json"
    run_config_path = root / "run_config.json"
    results_path = root / "results.jsonl"
    preflight_path = root / "token_budget.preflight.json"

    checks = [
        _check("comparison_dir_present", root.exists(), f"comparison artifact folder: {root}", path=str(root)),
        _check("comparison_summary_present", summary_path.exists(), "generated-answer summary.json is present", path=str(summary_path)),
        _check("comparison_run_config_present", run_config_path.exists(), "run_config.json is present", path=str(run_config_path)),
        _check("comparison_results_present", results_path.exists(), "results.jsonl is present", path=str(results_path)),
        _check("token_budget_preflight_present", preflight_path.exists(), "token-budget preflight artifact is present", path=str(preflight_path)),
    ]
    summary = _read_json(summary_path) if summary_path.exists() else None
    run_config = _read_json(run_config_path) if run_config_path.exists() else None
    preflight = _read_json(preflight_path) if preflight_path.exists() else None

    if summary is None:
        checks.append(
            _check("comparison_summary_readable", False, "summary.json must be valid JSON written by compare-coconut-val")
        )
        return checks, None

    checks.append(
        _check(
            "comparison_summary_readable",
            True,
            "summary.json is readable",
            status_value=summary.get("status"),
            score_type=summary.get("score_type"),
        )
    )
    checks.append(
        _check(
            "comparison_mode_is_generated_answer",
            isinstance(run_config, dict) and run_config.get("mode") == "compare_coconut_val",
            "artifact folder comes from compare-coconut-val, not a dry run",
            mode=run_config.get("mode") if isinstance(run_config, dict) else None,
        )
    )

    release_score_valid = bool(summary.get("release_score_valid"))
    checks.append(
        _check(
            "release_score_valid",
            release_score_valid,
            "score is a full clean generated-answer run",
            severity="warning" if args.allow_diagnostic_score and not release_score_valid else "blocker",
            release_score_valid=release_score_valid,
        )
    )

    gate_passed = _release_gate_passed(summary)
    checks.append(
        _check(
            "candidate_not_regressed",
            gate_passed,
            "candidate meets the configured baseline margin",
            severity="warning" if args.allow_candidate_regression and not gate_passed else "blocker",
            release_gate=summary.get("release_gate"),
        )
    )

    budget_clean = _token_budget_clean(summary)
    checks.append(
        _check(
            "prompt_budget_clean",
            budget_clean,
            "baseline and candidate prompts were not truncated",
            token_budget=_diagnostics(summary).get("token_budget"),
        )
    )
    if isinstance(preflight, dict):
        checks.append(
            _check(
                "preflight_budget_clean",
                not bool(preflight.get("any_prompt_truncated")),
                "tokenizer-only preflight did not detect truncation before model load",
                preflight_status=preflight.get("status"),
            )
        )

    suspect_halt_gate = _halt_gate_suspect(summary)
    checks.append(
        _check(
            "halt_gate_not_suspect",
            not suspect_halt_gate,
            "HaltGate did not collapse to one-latent behavior",
            severity="warning" if args.allow_halt_gate_suspect and suspect_halt_gate else "blocker",
            halt_gate_suspect=suspect_halt_gate,
            reason=_diagnostics(summary).get("halt_gate_suspect_reason"),
        )
    )

    expected_rows = summary.get("n_samples")
    observed_rows = _count_jsonl_rows(results_path) if results_path.exists() else None
    row_count_ok = isinstance(expected_rows, int) and observed_rows == expected_rows
    checks.append(
        _check(
            "result_rows_match_summary",
            row_count_ok,
            "results.jsonl row count matches summary.n_samples",
            expected_rows=expected_rows,
            observed_rows=observed_rows,
        )
    )
    return checks, summary


def _lm_eval_checks(args: argparse.Namespace) -> list[dict[str, Any]]:
    lm_eval_dir = (getattr(args, "lm_eval_dir", "") or "").strip()
    if not lm_eval_dir:
        return [
            _check(
                "lm_eval_smoke_checked",
                False,
                "lm-eval smoke artifacts were not requested for this gate",
                severity="blocker" if args.require_lm_eval else "warning",
            )
        ]

    root = Path(lm_eval_dir).expanduser()
    config_path = root / "ouroboros_lm_eval_run_config.json"
    config = _read_json(config_path) if config_path.exists() else None
    result_jsons = sorted(
        str(path)
        for path in root.rglob("*.json")
        if path.name != "ouroboros_lm_eval_run_config.json"
    ) if root.exists() else []
    return [
        _check("lm_eval_dir_present", root.exists(), f"lm-eval artifact folder: {root}", path=str(root)),
        _check(
            "lm_eval_run_config_present",
            config_path.exists() and isinstance(config, dict),
            "Ouroboros lm-eval launch config is present and readable",
            path=str(config_path),
            tasks=config.get("tasks") if isinstance(config, dict) else None,
            limit=config.get("limit") if isinstance(config, dict) else None,
        ),
        _check(
            "lm_eval_result_artifacts_present",
            bool(result_jsons),
            "lm-evaluation-harness wrote JSON result artifacts",
            severity="blocker" if args.require_lm_eval else "warning",
            result_jsons_preview=result_jsons[:10],
        ),
    ]


def run_experiment_readiness_gate(args: argparse.Namespace) -> None:
    """Read existing eval artifacts and decide whether architecture work is unblocked."""
    comparison_checks, summary = _comparison_checks(args)
    checks = comparison_checks + _lm_eval_checks(args)
    blocker_failures = [check for check in checks if check["severity"] == "blocker" and check["status"] == "failed"]
    ready = not blocker_failures
    report: dict[str, Any] = {
        "status": "ready" if ready else "blocked",
        "ready_for_architecture_experiment": ready,
        "claim_boundary": "artifact sanity only; no SOTA or model-quality claim",
        "artifact_sources": {
            "comparison_dir": str(Path(args.comparison_dir).expanduser()),
            "lm_eval_dir": (getattr(args, "lm_eval_dir", "") or None),
        },
        "comparison": {
            "status": summary.get("status") if isinstance(summary, dict) else None,
            "score_type": summary.get("score_type") if isinstance(summary, dict) else None,
            "n_samples": summary.get("n_samples") if isinstance(summary, dict) else None,
            "release_gate": summary.get("release_gate") if isinstance(summary, dict) else None,
        },
        "checks": checks,
        "blocking_checks": [check["name"] for check in blocker_failures],
        "next_allowed_step": (
            "Write the architecture PRD/tracer slice on an isolated branch."
            if ready
            else "Rerun the missing or failed eval smoke steps, then run this gate again."
        ),
    }
    output_path = (getattr(args, "output_path", "") or "").strip()
    if output_path:
        write_json(Path(output_path).expanduser(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if blocker_failures and not bool(getattr(args, "allow_not_ready", False)):
        raise SystemExit(2)


__all__ = ("run_experiment_readiness_gate",)
