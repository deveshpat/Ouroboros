"""Coconut validation inspection, dry-run artifacts, and faithful comparison."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from ouroboros.eval.artifacts import ensure_output_dir, write_json, write_jsonl

DEFAULT_DATASET_REPO = "WeirdRunner/Ouroboros"
DEFAULT_DATASET_CONFIG = "coconut-v1"
DEFAULT_DATASET_SPLIT = "validation"
DEFAULT_DATASET_REVISION = "6a52cd0c47be1e7b85d9018225387950aefc4631"
CLAIM_BOUNDARY = "ID-backed in-domain holdout; not external benchmark"
PRIMARY_METRIC = "generated_answer_exact_match"
ID_FIELD = "id"
SOURCE_FIELD = "source"
QUESTION_FIELD = "question"
ANSWER_FIELD = "answer_norm"


def dataset_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset_repo": args.dataset_repo,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "dataset_revision": args.dataset_revision,
        "id_field": ID_FIELD,
        "source_field": SOURCE_FIELD,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _val_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "val.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object row, got {type(row).__name__}")
            rows.append(row)
    return rows


def inspect_local_validation(data_dir: str | Path) -> dict[str, Any]:
    path = _val_path(data_dir)
    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
            "row_count": 0,
            "source_counts": {},
            "missing_id_count": 0,
            "missing_ids": [],
            "duplicate_id_count": 0,
            "duplicate_ids": [],
        }

    rows = _load_jsonl(path)
    ids = [str(row.get(ID_FIELD, "")).strip() for row in rows]
    sources = [str(row.get(SOURCE_FIELD, "")).strip() or "<missing>" for row in rows]
    answers = [str(row.get(ANSWER_FIELD, "")).strip() for row in rows]
    missing_positions = [idx for idx, value in enumerate(ids) if not value]
    duplicate_ids = sorted([value for value, count in Counter(ids).items() if value and count > 1])
    missing_answer_ids = [
        ids[idx] or f"<row:{idx}>"
        for idx, value in enumerate(answers)
        if value == ""
    ]
    missing_answer_sources = Counter(
        sources[idx]
        for idx, value in enumerate(answers)
        if value == ""
    )
    status = "ok" if not missing_positions and not duplicate_ids else "invalid"
    return {
        "status": status,
        "path": str(path),
        "row_count": len(rows),
        "source_counts": dict(sorted(Counter(sources).items())),
        "scorable_answer_count": len(rows) - len(missing_answer_ids),
        "missing_answer_norm_count": len(missing_answer_ids),
        "missing_answer_norm_ids": missing_answer_ids[:50],
        "missing_answer_norm_by_source": dict(sorted(missing_answer_sources.items())),
        "missing_id_count": len(missing_positions),
        "missing_ids": missing_positions[:50],
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids[:50],
    }


def inspect_coconut_val(args: argparse.Namespace) -> None:
    report = inspect_local_validation(args.data_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] == "invalid":
        raise SystemExit(2)


def dry_run_coconut_val(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    local_inspection = inspect_local_validation(args.data_dir)
    run_config = {
        "mode": "dry_run_coconut_val",
        "dataset": dataset_metadata(args),
        "runtime": {
            "loads_model_weights": False,
            "auto_downloads_dataset": False,
            "source_of_truth": "existing Ouroboros Coconut/inference runtime",
        },
        "scoring": {
            "primary_metric": PRIMARY_METRIC,
            "prompt_field": QUESTION_FIELD,
            "answer_field": ANSWER_FIELD,
            "forbidden_prompt_fields": ["steps", "answer_full", "stage labels", "latent supervision"],
        },
        "local_validation": local_inspection,
    }
    summary = {
        "status": "dry_run_complete",
        "primary_metric": PRIMARY_METRIC,
        "claim_boundary": CLAIM_BOUNDARY,
        "dataset": dataset_metadata(args),
        "local_validation": local_inspection,
        "artifacts": {
            "run_config": "run_config.json",
            "summary": "summary.json",
            "results": "results.jsonl",
        },
    }
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "results.jsonl", [])
    print(f"wrote dry-run artifacts -> {output_dir}")




def _load_tokenizer_only(tokenizer_model_id: str):
    """Load only the tokenizer needed for prompt-budget accounting.

    This intentionally avoids base/adaptor model loading so release eval can fail
    fast on CPU when a proposed max_seq_len would truncate validation prompts.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_id,
        use_fast=True,
        trust_remote_code=True,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _measure_prompt_budget(
    tokenizer: Any,
    prompt: str,
    *,
    max_seq_len: int,
    reserve_tokens: int = 0,
    context: str = "prompt",
) -> dict[str, Any]:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids:
        raise ValueError(f"{context} encoded to an empty token sequence.")
    max_seq_len = max(1, int(max_seq_len))
    reserve_tokens = max(0, int(reserve_tokens))
    budget = max(1, max_seq_len - reserve_tokens)
    original_tokens = len(input_ids)
    dropped = max(0, original_tokens - budget)
    return {
        "context": context,
        "original_tokens": original_tokens,
        "budget_tokens": budget,
        "max_seq_len": max_seq_len,
        "reserve_tokens": reserve_tokens,
        "final_tokens": original_tokens - dropped,
        "truncated": bool(dropped),
        "dropped_tokens": dropped,
        "required_max_seq_len_for_no_truncation": original_tokens + reserve_tokens,
    }


def _token_budget_summary(result_rows: list[dict[str, Any]], *, max_seq_len: int, stage_k: int) -> dict[str, Any]:
    by_phase: dict[str, dict[str, Any]] = {}
    truncated_ids: set[str] = set()
    required_max_seq_len = 0
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
        phase_required = max(
            (int(report.get("required_max_seq_len_for_no_truncation", 0)) for report in reports),
            default=0,
        )
        required_max_seq_len = max(required_max_seq_len, phase_required)
        by_phase[phase] = {
            "rows_seen": len(reports),
            "truncated_rows": len(truncated_reports),
            "truncated_fraction": (len(truncated_reports) / len(reports)) if reports else 0.0,
            "max_original_tokens": max((int(report.get("original_tokens", 0)) for report in reports), default=0),
            "max_dropped_tokens": max((int(report.get("dropped_tokens", 0)) for report in reports), default=0),
            "total_dropped_tokens": sum(int(report.get("dropped_tokens", 0)) for report in reports),
            "required_max_seq_len_for_no_truncation": phase_required,
            "truncated_ids_preview": phase_truncated_ids[:25],
        }
    any_truncated = any(phase["truncated_rows"] for phase in by_phase.values())
    return {
        "status": "truncated_inputs" if any_truncated else "clean",
        "loads_model_weights": False,
        "current_max_seq_len": int(max_seq_len),
        "stage_k": int(stage_k),
        "candidate_reserve_tokens": max(0, int(stage_k)),
        "required_max_seq_len_for_no_truncation": required_max_seq_len,
        "recommended_action": (
            f"increase max_seq_len to at least {required_max_seq_len} before generated-answer scoring, "
            "or rerun with --allow_truncated_eval only for a knowingly truncated score"
            if any_truncated
            else "run generated-answer eval; prompt budget is clean"
        ),
        "any_prompt_truncated": any_truncated,
        "truncated_unique_rows": len(truncated_ids),
        "truncated_ids_preview": sorted(truncated_ids)[:25],
        "by_phase": by_phase,
    }


def audit_token_budget_for_rows(
    rows: list[dict[str, Any]],
    *,
    tokenizer_model_id: str,
    max_seq_len: int,
    stage_k: int,
    use_chat_template: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure baseline/candidate prompt budgets without loading model weights."""
    from ouroboros.inference.generation import format_prompt

    tokenizer = _load_tokenizer_only(tokenizer_model_id)
    result_rows: list[dict[str, Any]] = []
    for row in rows:
        question = str(row[QUESTION_FIELD])
        formatted_prompt = format_prompt(tokenizer, question, use_chat_template=use_chat_template)
        result_rows.append(
            {
                "id": row[ID_FIELD],
                "source": row.get(SOURCE_FIELD, ""),
                "baseline_prompt_budget": _measure_prompt_budget(
                    tokenizer,
                    formatted_prompt,
                    max_seq_len=max_seq_len,
                    reserve_tokens=0,
                    context="baseline prompt",
                ),
                "candidate_prompt_budget": _measure_prompt_budget(
                    tokenizer,
                    formatted_prompt,
                    max_seq_len=max_seq_len,
                    reserve_tokens=max(0, int(stage_k)),
                    context="candidate prompt",
                ),
            }
        )
    return _token_budget_summary(result_rows, max_seq_len=max_seq_len, stage_k=stage_k), result_rows




def _select_rows_by_budget(
    rows: list[dict[str, Any]],
    *,
    limit_samples: int | None,
    sample_strategy: str,
    tokenizer_model_id: str,
    max_seq_len: int,
    stage_k: int,
    use_chat_template: bool,
) -> list[dict[str, Any]]:
    strategy = str(sample_strategy or "first").strip().lower()
    if limit_samples is None or strategy == "first":
        return rows[: int(limit_samples)] if limit_samples is not None else rows
    if strategy != "longest":
        raise ValueError(f"Unsupported sample_strategy={sample_strategy!r}. Use 'first' or 'longest'.")

    _, budget_rows = audit_token_budget_for_rows(
        rows,
        tokenizer_model_id=tokenizer_model_id,
        max_seq_len=max_seq_len,
        stage_k=stage_k,
        use_chat_template=use_chat_template,
    )
    required_by_id: dict[str, int] = {}
    for row in budget_rows:
        reports = [row.get("baseline_prompt_budget"), row.get("candidate_prompt_budget")]
        required_by_id[str(row.get(ID_FIELD, ""))] = max(
            (int(report.get("required_max_seq_len_for_no_truncation", 0)) for report in reports if isinstance(report, dict)),
            default=0,
        )
    indexed_rows = list(enumerate(rows))
    indexed_rows.sort(key=lambda item: (-required_by_id.get(str(item[1].get(ID_FIELD, "")), 0), item[0]))
    selected = [row for _, row in indexed_rows[: int(limit_samples)]]
    if selected:
        longest_required = max(required_by_id.get(str(row.get(ID_FIELD, "")), 0) for row in selected)
        shortest_required = min(required_by_id.get(str(row.get(ID_FIELD, "")), 0) for row in selected)
        print(
            f"[eval] sample_strategy=longest selected {len(selected)} rows "
            f"with required_max_seq_len range {shortest_required}..{longest_required}."
        )
    return selected


def _truncation_audit_failure_message(summary: dict[str, Any]) -> str:
    return (
        "Eval prompt truncation would occur before any model load; refusing generated-answer scoring. "
        f"truncated_unique_rows={summary['truncated_unique_rows']}; "
        f"current_max_seq_len={summary['current_max_seq_len']}; "
        f"required_max_seq_len_for_no_truncation={summary['required_max_seq_len_for_no_truncation']}. "
        "Increase --max_seq_len or rerun with --allow_truncated_eval only when a truncated-input score is acceptable."
    )


def audit_coconut_val_budget(args: argparse.Namespace) -> None:
    """Run the CPU/tokenizer-only truncation audit and optionally write artifacts."""
    local_inspection = inspect_local_validation(args.data_dir)
    if local_inspection["status"] == "invalid":
        raise SystemExit(f"Invalid validation file; refusing token-budget audit: {local_inspection}")
    rows = _iter_validation_rows(
        args.data_dir,
        args.limit_samples,
        sample_strategy=getattr(args, "sample_strategy", "first"),
        tokenizer_model_id=str(args.tokenizer_model_id),
        max_seq_len=int(args.max_seq_len),
        stage_k=int(args.stage_k),
        use_chat_template=bool(args.use_chat_template),
    )
    if not rows:
        raise SystemExit("No validation rows selected for audit-coconut-val-budget.")
    summary, result_rows = audit_token_budget_for_rows(
        rows,
        tokenizer_model_id=str(args.tokenizer_model_id),
        max_seq_len=int(args.max_seq_len),
        stage_k=int(args.stage_k),
        use_chat_template=bool(args.use_chat_template),
    )
    output_dir_arg = getattr(args, "output_dir", None)
    if output_dir_arg:
        output_dir = ensure_output_dir(output_dir_arg)
        write_json(output_dir / "token_budget.summary.json", {"local_validation": local_inspection, "token_budget": summary})
        write_jsonl(output_dir / "token_budget.results.jsonl", result_rows)
        print(f"wrote token-budget artifacts -> {output_dir}")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary["any_prompt_truncated"] and not bool(getattr(args, "allow_truncated_eval", False)):
        raise SystemExit(_truncation_audit_failure_message(summary))


def normalize_generated_answer(text: str) -> str:
    """Normalize generated answers with the existing Coconut answer extractor."""
    from ouroboros.coconut.dgac import normalize_pred

    value = normalize_pred(str(text))
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^[\s:;,.!?]+|[\s:;,.!?]+$", "", value)
    return value


def _iter_validation_rows(
    data_dir: str | Path,
    limit_samples: int | None,
    *,
    sample_strategy: str = "first",
    tokenizer_model_id: str | None = None,
    max_seq_len: int = 1024,
    stage_k: int = 10,
    use_chat_template: bool = True,
) -> list[dict[str, Any]]:
    path = _val_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"validation file not found: {path}. Compare does not auto-download datasets; "
            "prepare data/coconut_v1/val.jsonl first."
        )
    rows = _load_jsonl(path)
    valid_rows: list[dict[str, Any]] = []
    skipped_missing_answers: list[str] = []
    for row in rows:
        sample_id = str(row.get(ID_FIELD, "")).strip()
        question = str(row.get(QUESTION_FIELD, "")).strip()
        answer_norm = str(row.get(ANSWER_FIELD, "")).strip()
        if not sample_id:
            raise ValueError(f"validation row missing required {ID_FIELD!r}")
        if not question:
            raise ValueError(f"validation row {sample_id!r} missing required {QUESTION_FIELD!r}")
        if answer_norm == "":
            skipped_missing_answers.append(sample_id)
            continue
        valid_rows.append(row)

    selected_rows = _select_rows_by_budget(
        valid_rows,
        limit_samples=limit_samples,
        sample_strategy=sample_strategy,
        tokenizer_model_id=tokenizer_model_id or "",
        max_seq_len=max_seq_len,
        stage_k=stage_k,
        use_chat_template=use_chat_template,
    )
    if skipped_missing_answers:
        preview = ", ".join(skipped_missing_answers[:10])
        suffix = "" if len(skipped_missing_answers) <= 10 else f", +{len(skipped_missing_answers) - 10} more"
        print(
            f"[eval] Skipped {len(skipped_missing_answers)} validation rows without {ANSWER_FIELD!r} "
            f"before selecting {len(selected_rows)} scorable rows: {preview}{suffix}"
        )
    return selected_rows


def _actual_latents_mean(values: list[Any]) -> float:
    flat: list[float] = []
    for value in values:
        if isinstance(value, list):
            flat.extend(float(v) for v in value)
        elif value is not None:
            flat.append(float(value))
    return sum(flat) / len(flat) if flat else 0.0


def _ensure_required_halt_gate(adapter_dir: Path) -> None:
    gate = adapter_dir / "halt_gate.pt"
    if not gate.exists():
        raise FileNotFoundError(
            f"candidate_requires_halt_gate was set, but required halt_gate.pt is missing: {gate}"
        )


def compare_coconut_val(args: argparse.Namespace) -> None:
    """Run the faithful generated-answer comparison. Heavy imports happen inside."""
    from ouroboros.eval.comparison import run_generated_answer_comparison

    run_generated_answer_comparison(args)

