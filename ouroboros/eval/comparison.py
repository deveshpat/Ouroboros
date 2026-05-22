"""Generated-answer comparison runtime for Coconut validation.

This is the deep Eval seam for the release gate.  The CLI delegates here so row
selection, paired generation, scoring, and artifact writing stay local to one
module instead of leaking across the CLI, inference, and model-loading layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    dataset_metadata,
    ensure_output_dir,
    inspect_local_validation,
    normalize_generated_answer,
    write_json,
    write_jsonl,
)


def _comparison_run_config(args: Any, local_inspection: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "compare_coconut_val",
        "dataset": dataset_metadata(args),
        "prompt_policy": {
            "prompt_field": QUESTION_FIELD,
            "forbidden_prompt_fields": ["steps", "answer_full", "stage labels", "latent supervision"],
            "baseline_flow": "question -> true base Jamba -> greedy decode -> normalize_pred -> exact match",
            "candidate_flow": "question -> base + <|lat|> + adapter + HaltGate + latent runtime -> greedy decode -> normalize_pred -> exact match",
        },
        "runtime": {
            "device": str(getattr(args, "device", "auto")),
            "dtype": str(getattr(args, "dtype", "auto")),
            "stage_k": int(getattr(args, "stage_k", 10)),
            "max_seq_len": int(getattr(args, "max_seq_len", 512)),
            "halt_threshold": float(getattr(args, "halt_threshold", 0.5)),
            "use_chat_template": bool(getattr(args, "use_chat_template", True)),
            "disable_mamba_kernels": bool(getattr(args, "disable_mamba_kernels", False)),
            "limit_samples": getattr(args, "limit_samples", None),
        },
        "decode": {"gen_max_tokens": int(args.gen_max_tokens), "do_sample": False},
        "local_validation": local_inspection,
        "baseline": {"model_id": args.baseline_model_id, "mode": "true_base"},
        "candidate": {
            "model_id": args.candidate_repo_id,
            "subdir": args.candidate_subdir,
            "halt_gate_required": bool(args.candidate_requires_halt_gate),
        },
        "scoring": {"primary_metric": PRIMARY_METRIC, "answer_field": ANSWER_FIELD},
    }


def _score_rows(rows: list[dict[str, Any]], baseline: Any, candidate: Any, args: Any) -> tuple[list[dict[str, Any]], int, int, list[Any]]:
    result_rows: list[dict[str, Any]] = []
    candidate_latents: list[Any] = []
    baseline_correct = 0
    candidate_correct = 0

    for row in rows:
        question = str(row[QUESTION_FIELD])
        answer_norm = normalize_generated_answer(str(row[ANSWER_FIELD]))
        baseline_text = generation_runtime.generate_baseline(baseline, question, args)
        candidate_result = generation_runtime.generate_candidate(candidate, question, args)
        baseline_pred_norm = normalize_generated_answer(baseline_text)
        candidate_pred_norm = normalize_generated_answer(candidate_result.text)
        baseline_ok = baseline_pred_norm == answer_norm
        candidate_ok = candidate_pred_norm == answer_norm
        baseline_correct += int(baseline_ok)
        candidate_correct += int(candidate_ok)
        candidate_latents.append(candidate_result.actual_latents)
        result_rows.append(
            {
                "id": row[ID_FIELD],
                "source": row.get(SOURCE_FIELD, ""),
                "answer_norm": answer_norm,
                "baseline_text": baseline_text,
                "baseline_pred_norm": baseline_pred_norm,
                "baseline_correct": baseline_ok,
                "candidate_text": candidate_result.text,
                "candidate_pred_norm": candidate_pred_norm,
                "candidate_correct": candidate_ok,
                "candidate_actual_latents": candidate_result.actual_latents,
            }
        )
    return result_rows, baseline_correct, candidate_correct, candidate_latents


def run_generated_answer_comparison(args: Any) -> Path:
    """Run the faithful generated-answer comparison and write artifacts."""
    local_inspection = inspect_local_validation(args.data_dir)
    if local_inspection["status"] == "invalid":
        raise SystemExit(f"Invalid validation file; refusing comparison: {local_inspection}")
    rows = _iter_validation_rows(args.data_dir, args.limit_samples)
    if not rows:
        raise SystemExit("No validation rows selected for compare-coconut-val.")
    if bool(args.candidate_requires_halt_gate) and args.candidate_adapter_dir:
        _ensure_required_halt_gate(Path(args.candidate_adapter_dir))

    output_dir = ensure_output_dir(args.output_dir)
    write_json(output_dir / "run_config.json", _comparison_run_config(args, local_inspection))

    baseline = generation_runtime.load_baseline_runtime(args)
    candidate = generation_runtime.load_candidate_runtime(args)
    result_rows, baseline_correct, candidate_correct, candidate_latents = _score_rows(rows, baseline, candidate, args)

    n = len(result_rows)
    summary = {
        "primary_metric": PRIMARY_METRIC,
        "claim_boundary": CLAIM_BOUNDARY,
        "n_samples": n,
        "baseline": {
            "model_id": args.baseline_model_id,
            PRIMARY_METRIC: baseline_correct / max(n, 1),
        },
        "candidate": {
            "model_id": args.candidate_repo_id,
            "subdir": args.candidate_subdir,
            "halt_gate_required": bool(args.candidate_requires_halt_gate),
            "halt_gate_used": bool(candidate.halt_gate is not None),
            PRIMARY_METRIC: candidate_correct / max(n, 1),
            "actual_latents_mean": _actual_latents_mean(candidate_latents),
        },
        "health_metrics": {
            "teacher_forced": "optional side metric only; not used for claims"
        },
    }
    write_jsonl(output_dir / "results.jsonl", result_rows)
    write_json(output_dir / "summary.json", summary)
    print(f"wrote comparison artifacts -> {output_dir}")
    return output_dir


__all__ = ["run_generated_answer_comparison"]
