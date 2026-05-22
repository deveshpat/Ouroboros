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


def normalize_generated_answer(text: str) -> str:
    """Normalize generated answers with the existing Coconut answer extractor."""
    from ouroboros.coconut.dgac import normalize_pred

    value = normalize_pred(str(text))
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^[\s:;,.!?]+|[\s:;,.!?]+$", "", value)
    return value


def _iter_validation_rows(data_dir: str | Path, limit_samples: int | None) -> list[dict[str, Any]]:
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
        if limit_samples is not None and len(valid_rows) >= int(limit_samples):
            break
    if skipped_missing_answers:
        preview = ", ".join(skipped_missing_answers[:10])
        suffix = "" if len(skipped_missing_answers) <= 10 else f", +{len(skipped_missing_answers) - 10} more"
        print(
            f"[eval] Skipped {len(skipped_missing_answers)} validation rows without {ANSWER_FIELD!r} "
            f"before selecting {len(valid_rows)} scorable rows: {preview}{suffix}"
        )
    return valid_rows


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

