"""Dataset runtime helpers for Coconut finetuning.

This module owns dataset download/loading and stage-limit inference so the
finetune entrypoint keeps orchestration locality while delegating data details.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.coconut.curriculum import normalize_steps


def download_dataset_from_hub(
    data_dir: Path,
    *,
    is_main_process: Callable[[], bool],
    hf_repo_id: str = "WeirdRunner/Ouroboros",
    hf_config: str = "coconut-v1",
) -> None:
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    is_main = is_main_process()
    if is_main:
        print(f"  [data] Local files missing. Downloading {hf_repo_id}[{hf_config}] from Hub...")

    data_dir.mkdir(parents=True, exist_ok=True)

    def _write_split(split_name: str, hf_split: str, out_path: Path) -> List[Dict[str, Any]]:
        try:
            ds = hf_load_dataset(hf_repo_id, hf_config, split=hf_split, token=True)
        except Exception as exc:
            if is_main:
                print(f"  [data] Could not load split '{hf_split}': {exc}")
            return []

        rows: List[Dict[str, Any]] = []
        with out_path.open("w", encoding="utf-8") as fh:
            for row in ds:
                steps = normalize_steps(row.get("steps", []))
                sample = {
                    "id": row.get("id", ""),
                    "source": row.get("source", ""),
                    "question": row.get("question", ""),
                    "steps": steps,
                    "answer_full": row.get("answer_full", ""),
                    "answer_norm": row.get("answer_norm", ""),
                    "n_steps": int(row.get("n_steps", len(steps))),
                }
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                rows.append(sample)

        if is_main:
            print(f"  [data] {split_name}: {len(rows)} samples -> {out_path}")
        return rows

    train_rows = _write_split("train", "train", data_dir / "train.jsonl")
    val_rows = _write_split("val", "validation", data_dir / "val.jsonl")

    def _quick_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {}
        n_steps = [r["n_steps"] for r in rows]
        by_source: Dict[str, int] = {}
        for r in rows:
            by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        sorted_steps = sorted(n_steps)
        return {
            "n_samples": len(rows),
            "n_steps_mean": round(sum(n_steps) / len(n_steps), 2),
            "n_steps_min": min(n_steps),
            "n_steps_max": max(n_steps),
            "n_steps_median": sorted_steps[len(sorted_steps) // 2],
            "by_source": by_source,
        }

    stats = {"train": _quick_stats(train_rows), "val": _quick_stats(val_rows)}
    with (data_dir / "stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    if is_main:
        t = stats.get("train", {})
        print(
            f"  [data] stats.json written. "
            f"median_steps={t.get('n_steps_median')}  "
            f"recommended --max_stage={t.get('n_steps_median')}"
        )


def load_canonical_dataset(
    data_dir: Path,
    max_samples: Optional[int],
    *,
    is_main_process: Callable[[], bool],
    download_from_hub: Callable[[Path], None],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    stats_path = data_dir / "stats.json"

    if not train_path.exists():
        download_from_hub(data_dir)

    if not train_path.exists():
        raise FileNotFoundError(
            f"train.jsonl not found at {train_path} and Hub download failed.\n"
            "Run: python prepare_coconut_dataset.py --output_dir data/coconut_v1 --push_to_hub"
        )

    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["steps"] = normalize_steps(row.get("steps", []))
                rows.append(row)
        return rows

    train = _load_jsonl(train_path)
    val = _load_jsonl(val_path) if val_path.exists() else []
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

    if max_samples is not None:
        n_val = max(1, max_samples // 20) if val else 0
        n_train = max(max_samples - n_val, 0)
        train = train[:n_train]
        val = val[:n_val] if n_val else []

    if is_main_process():
        print(f"  Loaded {len(train)} train / {len(val)} val from {data_dir}")
        if stats:
            t = stats.get("train", {})
            print(
                f"  Step stats: median={t.get('n_steps_median')} "
                f"mean={t.get('n_steps_mean')} max={t.get('n_steps_max')}"
            )
    return train, val, stats


def get_max_stage(args_max_stage: Optional[int], stats: Dict[str, Any], *, is_main_process: Callable[[], bool]) -> int:
    if args_max_stage is not None:
        return int(args_max_stage)
    median = stats.get("train", {}).get("n_steps_median")
    if median is not None:
        if is_main_process():
            print(f"  --max_stage not set; using n_steps_median={median} from stats.json")
        return int(median)
    if is_main_process():
        print("  [warn] --max_stage not set and stats.json absent; defaulting to 10")
    return 10
