"""Dataset loading and Coconut stage sample construction."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ouroboros.model import _is_main_process, _maybe_apply_chat_template

def _download_dataset_from_hub(
    data_dir: Path,
    hf_repo_id: str = "WeirdRunner/Ouroboros",
    hf_config: str = "coconut-v1",
) -> None:
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    is_main = _is_main_process()
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
                steps = row.get("steps", [])
                if isinstance(steps, str):
                    try:
                        steps = json.loads(steps)
                    except json.JSONDecodeError:
                        steps = [steps]
                sample = {
                    "id":          row.get("id", ""),
                    "source":      row.get("source", ""),
                    "question":    row.get("question", ""),
                    "steps":       steps,
                    "answer_full": row.get("answer_full", ""),
                    "answer_norm": row.get("answer_norm", ""),
                    "n_steps":     int(row.get("n_steps", len(steps))),
                }
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                rows.append(sample)

        if is_main:
            print(f"  [data] {split_name}: {len(rows)} samples -> {out_path}")
        return rows

    train_rows = _write_split("train", "train",      data_dir / "train.jsonl")
    val_rows   = _write_split("val",   "validation", data_dir / "val.jsonl")

    def _quick_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {}
        n_steps = [r["n_steps"] for r in rows]
        by_source: Dict[str, int] = {}
        for r in rows:
            by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        sorted_steps = sorted(n_steps)
        return {
            "n_samples":      len(rows),
            "n_steps_mean":   round(sum(n_steps) / len(n_steps), 2),
            "n_steps_min":    min(n_steps),
            "n_steps_max":    max(n_steps),
            "n_steps_median": sorted_steps[len(sorted_steps) // 2],
            "by_source":      by_source,
        }

    stats = {"train": _quick_stats(train_rows), "val": _quick_stats(val_rows)}
    with (data_dir / "stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    if is_main:
        t = stats.get("train", {})
        print(f"  [data] stats.json written. "
              f"median_steps={t.get('n_steps_median')}  "
              f"recommended --max_stage={t.get('n_steps_median')}")


def load_canonical_dataset(
    data_dir: Path,
    max_samples: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    train_path = data_dir / "train.jsonl"
    val_path   = data_dir / "val.jsonl"
    stats_path = data_dir / "stats.json"

    if not train_path.exists():
        _download_dataset_from_hub(data_dir)

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
                steps = row.get("steps")
                if isinstance(steps, str):
                    try:
                        row["steps"] = json.loads(steps)
                    except json.JSONDecodeError:
                        row["steps"] = [steps]
                rows.append(row)
        return rows

    train = _load_jsonl(train_path)
    val   = _load_jsonl(val_path) if val_path.exists() else []
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

    if max_samples is not None:
        n_val   = max(1, max_samples // 20) if val else 0
        n_train = max(max_samples - n_val, 0)
        train   = train[:n_train]
        val     = val[:n_val] if n_val else []

    if _is_main_process():
        print(f"  Loaded {len(train)} train / {len(val)} val from {data_dir}")
        if stats:
            t = stats.get("train", {})
            print(
                f"  Step stats: median={t.get('n_steps_median')} "
                f"mean={t.get('n_steps_mean')} max={t.get('n_steps_max')}"
            )
    return train, val, stats


def get_max_stage(args: argparse.Namespace, stats: Dict[str, Any]) -> int:
    if args.max_stage is not None:
        return int(args.max_stage)
    median = stats.get("train", {}).get("n_steps_median")
    if median is not None:
        if _is_main_process():
            print(f"  --max_stage not set; using n_steps_median={median} from stats.json")
        return int(median)
    if _is_main_process():
        print("  [warn] --max_stage not set and stats.json absent; defaulting to 10")
    return 10


def build_sample_at_stage(
    tokenizer,
    sample: Dict[str, Any],
    stage_k: int,
    lat_token_id: int,
    max_seq_len: int,
) -> Optional[Dict[str, Any]]:
    question = str(sample.get("question", "")).strip()
    if not question:
        return None

    prefix_text = _maybe_apply_chat_template(tokenizer, question)
    q_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    steps_raw = sample.get("steps") or []
    if isinstance(steps_raw, str):
        try:
            steps_raw = json.loads(steps_raw)
        except json.JSONDecodeError:
            steps_raw = [steps_raw]
    steps = [str(s) for s in steps_raw if str(s).strip()]

    n_latent = min(int(stage_k), len(steps))
    remaining_steps = steps[n_latent:]

    supervised_ids: List[int] = []
    for step_text in remaining_steps:
        supervised_ids.extend(tokenizer.encode(step_text + "\n", add_special_tokens=False))

    answer_text = str(sample.get("answer_full", ""))
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer_ids.append(int(tokenizer.eos_token_id))
    supervised_ids.extend(answer_ids)

    if not supervised_ids:
        return None

    total = len(q_ids) + n_latent + len(supervised_ids)
    if total > max_seq_len:
        allowed = max_seq_len - len(q_ids) - n_latent
        if allowed < 4:
            return None
        supervised_ids = supervised_ids[:allowed]

    full_ids = q_ids + [lat_token_id] * n_latent + supervised_ids
    labels = [-100] * len(q_ids) + [-100] * n_latent + supervised_ids
    assert len(full_ids) == len(labels)

    return {
        "full_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "q_len": len(q_ids),
        "n_latent": n_latent,
        "answer_norm": str(sample.get("answer_norm", "")),
    }


def collate_stage_k(samples: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(s["full_ids"].size(0) for s in samples)
    batch_size = len(samples)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    q_lens = torch.zeros(batch_size, dtype=torch.long)
    n_latents = torch.zeros(batch_size, dtype=torch.long)
    for i, sample in enumerate(samples):
        seq_len = sample["full_ids"].size(0)
        input_ids[i, :seq_len] = sample["full_ids"]
        labels[i, :seq_len] = sample["labels"]
        attn_mask[i, :seq_len] = True
        q_lens[i] = int(sample["q_len"])
        n_latents[i] = int(sample["n_latent"])
    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "labels": labels,
        "q_lens": q_lens,
        "n_latents": n_latents,
        "pad_id": torch.tensor(int(pad_id), dtype=torch.long),
    }