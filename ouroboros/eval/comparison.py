"""Generated-answer comparison runtime for Coconut validation.

This is the deep Eval seam for the release gate.  The CLI delegates here so row
selection, paired generation, scoring, and artifact writing stay local to one
module instead of leaking across the CLI, inference, and model-loading layers.
"""

from __future__ import annotations

import gc
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from ouroboros.eval import generation_runtime
from ouroboros.eval.coconut_val import (
    ANSWER_FIELD,
    CLAIM_BOUNDARY,
    ID_FIELD,
    PRIMARY_METRIC,
    QUESTION_FIELD,
    SOURCE_FIELD,
    _actual_latents_mean,
    _ensure_required_halt_gate,
    _iter_validation_rows,
    _truncation_audit_failure_message,
    audit_token_budget_for_rows,
    dataset_metadata,
    ensure_output_dir,
    inspect_local_validation,
    normalize_generated_answer,
    write_json,
    write_jsonl,
)


def _comparison_run_config(args: Any, local_inspection: dict[str, Any]) -> dict[str, Any]:
    disable_candidate_halt_gate = bool(getattr(args, "disable_candidate_halt_gate", False))
    candidate_flow = (
        "question -> base + <|lat|> + adapter + fixed-depth latent runtime "
        "-> greedy decode -> normalize_pred -> exact match"
        if disable_candidate_halt_gate
        else "question -> base + <|lat|> + adapter + HaltGate + latent runtime -> greedy decode -> normalize_pred -> exact match"
    )
    return {
        "mode": "compare_coconut_val",
        "dataset": dataset_metadata(args),
        "prompt_policy": {
            "prompt_field": QUESTION_FIELD,
            "forbidden_prompt_fields": ["steps", "answer_full", "stage labels", "latent supervision"],
            "baseline_flow": "question -> true base Jamba -> greedy decode -> normalize_pred -> exact match",
            "candidate_flow": candidate_flow,
        },
        "runtime": {
            "device": str(getattr(args, "device", "auto")),
            "dtype": str(getattr(args, "dtype", "auto")),
            "stage_k": int(getattr(args, "stage_k", 10)),
            "max_seq_len": int(getattr(args, "max_seq_len", 8192)),
            "halt_threshold": float(getattr(args, "halt_threshold", 0.5)),
            "use_chat_template": bool(getattr(args, "use_chat_template", True)),
            "disable_mamba_kernels": bool(getattr(args, "disable_mamba_kernels", False)),
            "disable_candidate_halt_gate": disable_candidate_halt_gate,
            "limit_samples": getattr(args, "limit_samples", None),
            "cleanup_every_n_samples": int(getattr(args, "cleanup_every_n_samples", 25)),
            "model_device_map": str(getattr(args, "model_device_map", "single")),
            "sample_strategy": str(getattr(args, "sample_strategy", "first")),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "decode": {"gen_max_tokens": int(args.gen_max_tokens), "do_sample": False},
        "input_truncation_policy": {
            "bounded_context_required": True,
            "fail_on_truncation": not bool(getattr(args, "allow_truncated_eval", False)),
            "allow_truncated_eval": bool(getattr(args, "allow_truncated_eval", False)),
        },
        "local_validation": local_inspection,
        "baseline": {"model_id": args.baseline_model_id, "mode": "true_base"},
        "candidate": {
            "model_id": args.candidate_repo_id,
            "subdir": args.candidate_subdir,
            "halt_gate_required": bool(args.candidate_requires_halt_gate),
            "halt_gate_disabled_for_fixed_depth": disable_candidate_halt_gate,
        },
        "scoring": {
            "primary_metric": PRIMARY_METRIC,
            "answer_field": ANSWER_FIELD,
            "min_candidate_margin": float(getattr(args, "min_candidate_margin", 0.0)),
            "allow_candidate_regression": bool(getattr(args, "allow_candidate_regression", False)),
        },
    }


def _base_result_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row[ID_FIELD],
        "source": row.get(SOURCE_FIELD, ""),
        "answer_norm": normalize_generated_answer(str(row[ANSWER_FIELD])),
    }


def _score_baseline_rows(rows: list[dict[str, Any]], baseline: Any, args: Any) -> tuple[list[dict[str, Any]], int]:
    result_rows: list[dict[str, Any]] = []
    baseline_correct = 0

    for idx, row in enumerate(rows, start=1):
        question = str(row[QUESTION_FIELD])
        result = _base_result_row(row)
        answer_norm = str(result["answer_norm"])
        baseline_generation = generation_runtime.generate_baseline_result(baseline, question, args)
        baseline_text = baseline_generation.text
        baseline_pred_norm = normalize_generated_answer(baseline_text)
        baseline_ok = baseline_pred_norm == answer_norm
        baseline_correct += int(baseline_ok)
        result.update(
            {
                "baseline_text": baseline_text,
                "baseline_pred_norm": baseline_pred_norm,
                "baseline_correct": baseline_ok,
                "baseline_prompt_budget": baseline_generation.prompt_budget,
            }
        )
        result_rows.append(result)
        _maybe_release_accelerator_memory(idx, args)
    return result_rows, baseline_correct


def _score_candidate_rows(
    rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    candidate: Any,
    args: Any,
) -> tuple[int, list[Any]]:
    candidate_latents: list[Any] = []
    candidate_correct = 0

    for idx, (row, result) in enumerate(zip(rows, result_rows, strict=True), start=1):
        question = str(row[QUESTION_FIELD])
        answer_norm = str(result["answer_norm"])
        candidate_result = generation_runtime.generate_candidate(candidate, question, args)
        candidate_pred_norm = normalize_generated_answer(candidate_result.text)
        candidate_ok = candidate_pred_norm == answer_norm
        candidate_correct += int(candidate_ok)
        candidate_latents.append(candidate_result.actual_latents)
        result.update(
            {
                "candidate_text": candidate_result.text,
                "candidate_pred_norm": candidate_pred_norm,
                "candidate_correct": candidate_ok,
                "candidate_actual_latents": candidate_result.actual_latents,
                "candidate_stage_k": candidate_result.stage_k,
                "candidate_used_halt_gate": candidate_result.used_halt_gate,
                "candidate_prompt_budget": candidate_result.prompt_budget,
            }
        )
        _maybe_release_accelerator_memory(idx, args)
    return candidate_correct, candidate_latents


def _release_accelerator_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _maybe_release_accelerator_memory(row_index: int, args: Any) -> None:
    every_n = int(getattr(args, "cleanup_every_n_samples", 25) or 0)
    if every_n > 0 and row_index % every_n == 0:
        _release_accelerator_memory()


def _token_budget_diagnostics(result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_phase: dict[str, dict[str, Any]] = {}
    truncated_ids: set[str] = set()
    for phase in ("baseline", "candidate"):
        key = f"{phase}_prompt_budget"
        reports = [row.get(key) for row in result_rows if isinstance(row.get(key), dict)]
        truncated_reports = [report for report in reports if bool(report.get("truncated"))]
        phase_truncated_ids = [
            str(row.get("id", ""))
            for row in result_rows
            if isinstance(row.get(key), dict) and bool(row[key].get("truncated"))
        ]
        truncated_ids.update(sample_id for sample_id in phase_truncated_ids if sample_id)
        by_phase[phase] = {
            "rows_seen": len(reports),
            "truncated_rows": len(truncated_reports),
            "truncated_fraction": (len(truncated_reports) / len(reports)) if reports else 0.0,
            "max_original_tokens": max((int(report.get("original_tokens", 0)) for report in reports), default=0),
            "max_dropped_tokens": max((int(report.get("dropped_tokens", 0)) for report in reports), default=0),
            "total_dropped_tokens": sum(int(report.get("dropped_tokens", 0)) for report in reports),
            "truncated_ids_preview": phase_truncated_ids[:25],
        }
    any_truncated = any(phase["truncated_rows"] for phase in by_phase.values())
    return {
        "status": "truncated_inputs" if any_truncated else "clean",
        "any_prompt_truncated": any_truncated,
        "truncated_unique_rows": len(truncated_ids),
        "truncated_ids_preview": sorted(truncated_ids)[:25],
        "by_phase": by_phase,
    }


def _truncation_failure_message(token_budget: dict[str, Any]) -> str:
    return (
        "Eval prompt truncation detected; generated-answer score is not clean. "
        f"truncated_unique_rows={token_budget['truncated_unique_rows']}. "
        "Artifacts were still written. Increase --max_seq_len, reduce prompt length, or rerun with "
        "--allow_truncated_eval only when a truncated-input score is acceptable."
    )


def _flatten_latents(values: list[Any]) -> list[int]:
    flat: list[int] = []
    for value in values:
        if isinstance(value, list):
            flat.extend(int(v) for v in value)
        elif value is not None:
            flat.append(int(value))
    return flat


def _latent_diagnostics(candidate_latents: list[Any], *, stage_k: int, halt_gate_used: bool) -> dict[str, Any]:
    flat = _flatten_latents(candidate_latents)
    total = len(flat)
    histogram = {str(key): count for key, count in sorted(Counter(flat).items())}
    one_latent_fraction = (histogram.get("1", 0) / total) if total else 0.0
    fixed_depth_fraction = (histogram.get(str(stage_k), 0) / total) if total else 0.0
    halt_gate_suspect = bool(halt_gate_used and stage_k > 1 and total and one_latent_fraction >= 0.5)
    return {
        "actual_latents_histogram": histogram,
        "one_latent_fraction": one_latent_fraction,
        "fixed_depth_fraction": fixed_depth_fraction,
        "halt_gate_suspect": halt_gate_suspect,
        "halt_gate_suspect_reason": (
            "HaltGate stopped at one latent for >=50% of rows; run fixed-depth ablation before promotion."
            if halt_gate_suspect
            else None
        ),
    }


def _gate_status(
    *,
    baseline_score: float,
    candidate_score: float,
    min_candidate_margin: float,
) -> dict[str, Any]:
    margin = candidate_score - baseline_score
    required_margin = float(min_candidate_margin)
    passed = margin >= required_margin
    return {
        "status": "passed" if passed else "failed_candidate_regression",
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "candidate_minus_baseline": margin,
        "required_candidate_margin": required_margin,
        "passed": passed,
    }


def run_generated_answer_comparison(args: Any) -> Path:
    """Run the faithful generated-answer comparison and write artifacts."""
    local_inspection = inspect_local_validation(args.data_dir)
    if local_inspection["status"] == "invalid":
        raise SystemExit(f"Invalid validation file; refusing comparison: {local_inspection}")
    rows = _iter_validation_rows(
        args.data_dir,
        args.limit_samples,
        sample_strategy=getattr(args, "sample_strategy", "first"),
        tokenizer_model_id=str(args.baseline_model_id),
        max_seq_len=int(getattr(args, "max_seq_len", 8192)),
        stage_k=int(getattr(args, "stage_k", 10)),
        use_chat_template=bool(getattr(args, "use_chat_template", True)),
    )
    if not rows:
        raise SystemExit("No validation rows selected for compare-coconut-val.")
    if bool(args.candidate_requires_halt_gate) and args.candidate_adapter_dir:
        _ensure_required_halt_gate(Path(args.candidate_adapter_dir))

    output_dir = ensure_output_dir(args.output_dir)
    run_config = _comparison_run_config(args, local_inspection)
    preflight_token_budget, preflight_rows = audit_token_budget_for_rows(
        rows,
        tokenizer_model_id=str(args.baseline_model_id),
        max_seq_len=int(getattr(args, "max_seq_len", 8192)),
        stage_k=int(getattr(args, "stage_k", 10)),
        use_chat_template=bool(getattr(args, "use_chat_template", True)),
    )
    run_config["input_truncation_policy"]["preflight"] = {
        "enabled": True,
        "loads_model_weights": False,
        "artifact_summary": "token_budget.preflight.json",
        "artifact_rows": "token_budget.preflight.jsonl",
    }
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "token_budget.preflight.json", preflight_token_budget)
    write_jsonl(output_dir / "token_budget.preflight.jsonl", preflight_rows)
    if preflight_token_budget["any_prompt_truncated"] and not bool(getattr(args, "allow_truncated_eval", False)):
        summary = {
            "status": "failed_input_truncation_preflight",
            "primary_metric": PRIMARY_METRIC,
            "claim_boundary": CLAIM_BOUNDARY,
            "n_samples": len(rows),
            "dataset": dataset_metadata(args),
            "baseline": {"model_id": args.baseline_model_id, PRIMARY_METRIC: None, "correct_count": None},
            "candidate": {
                "model_id": args.candidate_repo_id,
                "subdir": args.candidate_subdir,
                "halt_gate_required": bool(args.candidate_requires_halt_gate),
                "halt_gate_used": None,
                "halt_gate_disabled_for_fixed_depth": bool(getattr(args, "disable_candidate_halt_gate", False)),
                PRIMARY_METRIC: None,
                "correct_count": None,
                "actual_latents_mean": None,
            },
            "release_score_valid": False,
            "score_type": "diagnostic",
            "release_gate": {
                "status": "blocked_before_model_load",
                "passed": False,
                "reason": "input_truncation_preflight",
            },
            "diagnostics": {"token_budget": preflight_token_budget},
            "artifacts": {
                "run_config": "run_config.json",
                "token_budget_preflight": "token_budget.preflight.json",
                "token_budget_rows": "token_budget.preflight.jsonl",
            },
        }
        write_json(output_dir / "summary.json", summary)
        write_jsonl(output_dir / "results.jsonl", [])
        raise SystemExit(_truncation_audit_failure_message(preflight_token_budget))

    baseline = generation_runtime.load_baseline_runtime(args)
    baseline_device_map = baseline.device_map
    try:
        result_rows, baseline_correct = _score_baseline_rows(rows, baseline, args)
        write_jsonl(output_dir / "results.baseline.jsonl", result_rows)
    finally:
        del baseline
        _release_accelerator_memory()

    candidate = generation_runtime.load_candidate_runtime(args)
    candidate_device_map = candidate.device_map
    halt_gate_used = bool(candidate.halt_gate is not None)
    try:
        candidate_correct, candidate_latents = _score_candidate_rows(rows, result_rows, candidate, args)
    finally:
        del candidate
        _release_accelerator_memory()

    n = len(result_rows)
    baseline_score = baseline_correct / max(n, 1)
    candidate_score = candidate_correct / max(n, 1)
    gate = _gate_status(
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        min_candidate_margin=float(getattr(args, "min_candidate_margin", 0.0)),
    )
    diagnostics = _latent_diagnostics(
        candidate_latents,
        stage_k=int(getattr(args, "stage_k", 10)),
        halt_gate_used=halt_gate_used,
    )
    token_budget = _token_budget_diagnostics(result_rows)
    model_device_maps = {
        "baseline": baseline_device_map,
        "candidate": candidate_device_map,
    }
    is_full_split = getattr(args, "limit_samples", None) is None and str(getattr(args, "sample_strategy", "first")) == "first"
    release_score_valid = bool(
        is_full_split
        and not token_budget["any_prompt_truncated"]
        and not bool(getattr(args, "allow_truncated_eval", False))
    )
    status = "failed_input_truncation" if token_budget["any_prompt_truncated"] and not bool(getattr(args, "allow_truncated_eval", False)) else gate["status"]
    summary = {
        "status": status,
        "primary_metric": PRIMARY_METRIC,
        "claim_boundary": CLAIM_BOUNDARY,
        "n_samples": n,
        "baseline": {
            "model_id": args.baseline_model_id,
            PRIMARY_METRIC: baseline_score,
            "correct_count": baseline_correct,
        },
        "candidate": {
            "model_id": args.candidate_repo_id,
            "subdir": args.candidate_subdir,
            "halt_gate_required": bool(args.candidate_requires_halt_gate),
            "halt_gate_used": halt_gate_used,
            "halt_gate_disabled_for_fixed_depth": bool(getattr(args, "disable_candidate_halt_gate", False)),
            PRIMARY_METRIC: candidate_score,
            "correct_count": candidate_correct,
            "actual_latents_mean": _actual_latents_mean(candidate_latents),
        },
        "release_score_valid": release_score_valid,
        "score_type": "full_release" if release_score_valid else "diagnostic",
        "release_gate": gate,
        "diagnostics": {
            **diagnostics,
            "token_budget": token_budget,
            "model_device_maps": model_device_maps,
        },
        "health_metrics": {
            "teacher_forced": "optional side metric only; not used for claims"
        },
    }
    write_jsonl(output_dir / "results.jsonl", result_rows)
    write_json(output_dir / "summary.json", summary)
    print(f"wrote comparison artifacts -> {output_dir}")
    if diagnostics["halt_gate_suspect"]:
        print(f"[eval] WARNING: {diagnostics['halt_gate_suspect_reason']}")
    if token_budget["any_prompt_truncated"]:
        message = _truncation_failure_message(token_budget)
        if bool(getattr(args, "allow_truncated_eval", False)):
            print(f"[eval] WARNING: {message}")
        else:
            raise SystemExit(message)
    if not gate["passed"]:
        message = (
            f"Candidate failed generated-answer gate: candidate={candidate_score:.6f}, "
            f"baseline={baseline_score:.6f}, required_margin={gate['required_candidate_margin']:.6f}. "
            "Artifacts were still written for inspection."
        )
        if bool(getattr(args, "allow_candidate_regression", False)):
            print(f"[eval] WARNING: {message}")
        else:
            raise SystemExit(message)
    return output_dir


__all__ = ["run_generated_answer_comparison"]
