#!/usr/bin/env python3
"""
prepare_coconut_dataset.py — Offline Coconut dataset preparation for Project Ouroboros.

Produces a canonical JSONL dataset from the same five sources used in sft-mix-v1,
re-processed from scratch with reasoning traces segmented into discrete steps.
Each sample is suitable for Coconut's progressive replacement curriculum.

Run once before training:
    python prepare_coconut_dataset.py --output_dir data/coconut_v1

To also push to the HuggingFace Hub (recommended for Kaggle resume):
    python prepare_coconut_dataset.py --output_dir data/coconut_v1 \\
        --push_to_hub --hf_token YOUR_TOKEN

Output files:
    data/coconut_v1/train.jsonl
    data/coconut_v1/val.jsonl
    data/coconut_v1/stats.json

Hub config pushed as:  WeirdRunner/Ouroboros  (config: coconut-v1, type: dataset)

Canonical sample format (one JSON object per line):
    {
        "id":          "bespoke_0042",
        "source":      "bespoke_stratos",
        "question":    "Solve: ...",
        "steps":       ["Step 1 text", "Step 2 text", ...],
        "answer_full": "The answer is 42.",
        "answer_norm": "42",
        "n_steps":     3
    }

Training script (jamba_coconut_finetune.py) reads ONLY this output —
it never loads raw HuggingFace datasets. This strict separation means the
preprocessing can be re-run or audited without touching the training code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Colab-friendly token resolution ──────────────────────────────────────────
# ── Environment-friendly token resolution ────────────────────────────────────
# Priority: 1) Kaggle Secrets 2) Colab Secrets 3) env var 4) CLI 5) placeholder

_KAGGLE_HF_TOKEN: Optional[str] = None
try:
    from kaggle_secrets import UserSecretsClient
    _user_secrets = UserSecretsClient()
    _KAGGLE_HF_TOKEN = _user_secrets.get_secret("HF_TOKEN")
except ImportError:
    pass

_COLAB_HF_TOKEN: Optional[str] = None
try:
    from google.colab import userdata as _colab_userdata
    _COLAB_HF_TOKEN = _colab_userdata.get("HF_TOKEN")
except Exception:
    pass

# Dynamic Backup path depending on environment (Kaggle vs Colab/Local)
if os.path.exists("/kaggle/working"):
    _DEFAULT_BACKUP_PATH = "/kaggle/working/Ouroboros/coconut_dataset_backup"
else:
    _DEFAULT_BACKUP_PATH = "/content/drive/MyDrive/Ouroboros/coconut_dataset_backup"

try:
    from datasets import Dataset, load_dataset
    from tqdm.auto import tqdm
except ImportError:
    sys.exit("pip install datasets tqdm")


# ── Regex helpers ────────────────────────────────────────────────────────────

_THINK_RE       = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL)
_THINK2_RE      = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_SOLUTION_RE    = re.compile(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL)
_BOXED_RE       = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_ANS_RE   = re.compile(r"(?:The answer is|answer:|=\s*)\**\s*([\d,\.\-]+)", re.IGNORECASE)
_NUMBER_RE      = re.compile(r"[\d,]+(?:\.\d+)?")
_NUMBERED_RE    = re.compile(r"^(?:\d+[\.\)]\s+|Step\s+\d+[:\.]?\s+)", re.MULTILINE | re.IGNORECASE)


# ── Step segmentation ─────────────────────────────────────────────────────────

def _clean_step(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(?:\d+[\.\)]\s+|Step\s+\d+[:\.]?\s+)", "", text, flags=re.IGNORECASE)
    return text.strip()


def segment_into_steps(text: str, min_step_chars: int = 15) -> List[str]:
    """
    Segment a reasoning trace into discrete steps.

    Priority hierarchy:
      1. Numbered/labelled steps ("Step 1:", "1.", "1)") — highest quality signal
      2. Double-newline paragraphs (most common in <think> blocks)
      3. Single-newline lines when text is compact
      4. Sentence boundaries as last resort

    Steps shorter than min_step_chars are folded into the preceding step.
    """
    if not text or not text.strip():
        return []

    if _NUMBERED_RE.search(text):
        raw = _NUMBERED_RE.split(text)
        steps = [_clean_step(s) for s in raw if s.strip()]
    elif "\n\n" in text:
        steps = [_clean_step(s) for s in text.split("\n\n") if s.strip()]
    elif "\n" in text and text.count("\n") >= 2:
        steps = [_clean_step(s) for s in text.split("\n") if s.strip()]
    else:
        raw = re.split(r"(?<=[.!?])\s+", text)
        steps = [_clean_step(s) for s in raw if s.strip()]

    merged: List[str] = []
    for step in steps:
        if not step:
            continue
        if len(step) < min_step_chars and merged:
            merged[-1] = merged[-1] + " " + step
        else:
            merged.append(step)

    return [s for s in merged if len(s) >= min_step_chars]


def normalize_answer(text: str) -> str:
    if not text:
        return ""
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _FINAL_ANS_RE.search(text)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences[-1] if sentences else text.strip()[:120]


def _doc_hash(q: str) -> str:
    return hashlib.sha1(q.encode("utf-8", errors="replace")).hexdigest()


# ── Per-source extractors ─────────────────────────────────────────────────────

def _extract_think_and_answer(assistant_blob: str) -> Tuple[str, str]:
    m = _THINK_RE.search(assistant_blob)
    if m:
        reasoning = m.group(1).strip()
        m2 = _SOLUTION_RE.search(assistant_blob)
        answer = m2.group(1).strip() if m2 else assistant_blob[m.end():].strip()
        return reasoning, answer
    m = _THINK2_RE.search(assistant_blob)
    if m:
        reasoning = m.group(1).strip()
        answer = assistant_blob[m.end():].strip()
        return reasoning, answer
    return "", assistant_blob.strip()


def _chat_pair(turns: Any) -> Tuple[str, str]:
    q, a = "", ""
    for turn in (turns or []):
        role = str(turn.get("role") or turn.get("from") or "").lower().strip()
        value = str(turn.get("content") or turn.get("value") or "").strip()
        if role in {"user", "human"} and not q:
            q = value
        elif role in {"assistant", "gpt"} and not a:
            a = value
    return q.strip(), a.strip()


def extract_bespoke(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    q, blob = _chat_pair(ex.get("conversations"))
    reasoning, answer = _extract_think_and_answer(blob)
    return q.strip(), reasoning.strip(), answer.strip()


def extract_metamath(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    q = str(ex.get("original_question") or ex.get("query") or "").strip()
    blob = str(ex.get("response") or ex.get("output") or "").strip()
    reasoning = blob
    answer = normalize_answer(blob)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", blob) if s.strip()]
    if sentences:
        answer_full = sentences[-1]
        reasoning = blob[: blob.rfind(answer_full)].strip() or blob
    else:
        answer_full = answer
    return q, reasoning.strip(), answer_full.strip()


def extract_openhermes(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    q, blob = _chat_pair(ex.get("conversations"))
    return q.strip(), blob.strip(), ""


def extract_openr1_math(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    q, blob = _chat_pair(ex.get("messages"))
    if not q:
        q = str(ex.get("problem") or ex.get("question") or "").strip()
    generations = ex.get("generations") or []
    if generations and not blob:
        math_v = list(ex.get("correctness_math_verify") or [])
        llama_v = list(ex.get("correctness_llama") or [])
        for i, gen in enumerate(generations):
            if (i < len(math_v) and math_v[i]) or (i < len(llama_v) and llama_v[i]):
                blob = str(gen).strip()
                break
        if not blob:
            blob = str(generations[0]).strip()
    if not blob:
        blob = str(ex.get("solution") or "").strip()
    reasoning, answer = _extract_think_and_answer(blob)
    if not reasoning:
        reasoning = blob
    if not answer:
        answer = str(ex.get("answer") or "").strip()
    return q.strip(), reasoning.strip(), answer.strip()


def extract_openr1_code(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    q, blob = _chat_pair(ex.get("messages"))
    if not q:
        title = str(ex.get("title") or "").strip()
        desc  = str(ex.get("description") or "").strip()
        q = f"{title}\n\n{desc}".strip()
    if not blob:
        blob = str(ex.get("generation") or ex.get("solution") or ex.get("editorial") or "").strip()
    reasoning, answer = _extract_think_and_answer(blob)
    if not reasoning:
        reasoning = blob
    return q.strip(), reasoning.strip(), answer.strip()


# ── Source registry ────────────────────────────────────────────────────────────

SOURCES = [
    {
        "name": "bespoke_stratos",
        "candidates": [("bespokelabs/Bespoke-Stratos-17k", None)],
        "split": "train",
        "extractor": extract_bespoke,
        "weight": 0.30,
    },
    {
        "name": "metamath",
        "candidates": [("meta-math/MetaMathQA", None)],
        "split": "train",
        "extractor": extract_metamath,
        "weight": 0.20,
    },
    {
        "name": "openhermes",
        "candidates": [("teknium/OpenHermes-2.5", None)],
        "split": "train",
        "extractor": extract_openhermes,
        "weight": 0.10,
    },
    {
        "name": "openr1_math",
        "candidates": [
            ("open-r1/OpenR1-Math-220k", "default"),
            ("open-r1/OpenR1-Math-220k", None),
        ],
        "split": "train",
        "extractor": extract_openr1_math,
        "weight": 0.25,
    },
    {
        "name": "openr1_code",
        "candidates": [
            ("open-r1/OpenR1-Code", None),
            ("open-r1/codeforces-cots", "solutions_w_editorials_py"),
            ("open-r1/codeforces-cots", "solutions_py"),
        ],
        "split": "train",
        "extractor": extract_openr1_code,
        "weight": 0.15,
    },
]


def _load_first_available(candidates, split):
    for ds_name, config in candidates:
        try:
            ds = load_dataset(ds_name, config, split=split) if config else load_dataset(ds_name, split=split)
            label = f"{ds_name}[{config}]" if config else ds_name
            return ds, label
        except Exception:
            continue
    return None, None


# ── Main processing logic ─────────────────────────────────────────────────────

def process_source(
    source: Dict,
    max_samples: Optional[int],
    min_steps: int,
    max_steps: int,
    min_answer_chars: int,
    id_prefix_counter: List[int],
    seen_hashes: set,
) -> List[Dict[str, Any]]:
    ds, label = _load_first_available(source["candidates"], source["split"])
    if ds is None:
        print(f"  [warn] {source['name']}: could not load from any candidate — skipping.")
        return []

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"  Processing {label} ({len(ds)} rows) ...")
    extractor = source["extractor"]
    results: List[Dict[str, Any]] = []
    skipped = 0

    for ex in tqdm(ds, desc=f"    {source['name']}", leave=False):
        q, reasoning, answer_text = extractor(ex)
        if not q:
            skipped += 1
            continue

        think_text = reasoning if reasoning else answer_text
        answer_full = answer_text if answer_text else reasoning

        steps = segment_into_steps(think_text)

        if len(steps) < min_steps:
            skipped += 1
            continue
        if len(steps) > max_steps:
            steps = steps[:max_steps]

        if not answer_full:
            answer_full = steps[-1]
            steps = steps[:-1]
            if len(steps) < min_steps:
                skipped += 1
                continue

        if len(answer_full) < min_answer_chars:
            skipped += 1
            continue

        h = _doc_hash(q)
        if h in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(h)

        sample_id = f"{source['name']}_{id_prefix_counter[0]:07d}"
        id_prefix_counter[0] += 1

        results.append({
            "id":          sample_id,
            "source":      source["name"],
            "question":    q,
            "steps":       steps,
            "answer_full": answer_full,
            "answer_norm": normalize_answer(answer_full),
            "n_steps":     len(steps),
        })

    print(
        f"    kept {len(results)}, skipped {skipped} "
        f"(too few steps / empty / duplicate / short answer)"
    )
    return results


def build_balanced_mix(
    all_by_source: Dict[str, List[Dict]],
    source_weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    available = {name: len(samples) for name, samples in all_by_source.items() if samples}
    if not available:
        return []

    weight_sum = sum(source_weights[n] for n in available)
    weights = {n: source_weights[n] / weight_sum for n in available}

    target_total = max(1, int(min(available[n] / weights[n] for n in available)))
    targets = {n: min(available[n], int(math.floor(target_total * weights[n]))) for n in available}

    remaining = target_total - sum(targets.values())
    if remaining > 0:
        by_frac = sorted(available.keys(), key=lambda n: target_total * weights[n] - targets[n], reverse=True)
        for n in by_frac:
            if remaining <= 0:
                break
            if targets[n] < available[n]:
                targets[n] += 1
                remaining -= 1

    mixed: List[Dict] = []
    for name, samples in all_by_source.items():
        t = targets.get(name, 0)
        mixed.extend(samples[:t])
        print(f"    {name}: {t} samples selected (available: {available.get(name, 0)})")

    random.seed(42)
    random.shuffle(mixed)
    return mixed


def split_train_val(
    samples: List[Dict],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    by_source: Dict[str, List[Dict]] = {}
    for s in samples:
        by_source.setdefault(s["source"], []).append(s)

    train, val = [], []
    for src_samples in by_source.values():
        n_val = max(1, int(len(src_samples) * val_fraction))
        idx = list(range(len(src_samples)))
        rng.shuffle(idx)
        val.extend(src_samples[i] for i in idx[:n_val])
        train.extend(src_samples[i] for i in idx[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_jsonl(samples: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Written: {path}  ({len(samples)} samples)")


def compute_stats(train: List[Dict], val: List[Dict]) -> Dict[str, Any]:
    def _stats(samples):
        if not samples:
            return {}
        n_steps = [s["n_steps"] for s in samples]
        sources: Dict[str, int] = {}
        for s in samples:
            sources[s["source"]] = sources.get(s["source"], 0) + 1
        return {
            "n_samples":      len(samples),
            "n_steps_mean":   round(sum(n_steps) / len(n_steps), 2),
            "n_steps_min":    min(n_steps),
            "n_steps_max":    max(n_steps),
            "n_steps_median": sorted(n_steps)[len(n_steps) // 2],
            "by_source":      sources,
        }
    return {"train": _stats(train), "val": _stats(val)}


# ── Hub push ──────────────────────────────────────────────────────────────────

def upload_to_hub(
    train: List[Dict],
    val: List[Dict],
    hf_token: str,
    hf_repo_id: str,
    hf_dataset_config: str,
    backup_path: str,
) -> None:
    """
    Push the canonical Coconut JSONL dataset to the HuggingFace Hub as a dataset repo.

    Schema pushed: {id, source, question, steps, answer_full, answer_norm, n_steps}
    steps is stored as a JSON string per row (HF datasets doesn't natively support
    list-of-strings columns with variable length across rows without Arrow overhead).
    jamba_coconut_finetune.py loads from local JSONL and never touches this Hub copy
    directly — the Hub copy exists solely for backup/resume across sessions.

    Local Google Drive backup is saved first to guard against 504 Hub errors.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=hf_repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=hf_token,
    )
    print(f"  HF dataset repo ready: {hf_repo_id}")

    def _to_hf_dict(samples: List[Dict]) -> Dict[str, List]:
        return {
            "id":          [s["id"] for s in samples],
            "source":      [s["source"] for s in samples],
            "question":    [s["question"] for s in samples],
            # Serialize steps list to JSON string to avoid Arrow nested-type issues
            "steps":       [json.dumps(s["steps"], ensure_ascii=False) for s in samples],
            "answer_full": [s["answer_full"] for s in samples],
            "answer_norm": [s["answer_norm"] for s in samples],
            "n_steps":     [s["n_steps"] for s in samples],
        }

    train_ds = Dataset.from_dict(_to_hf_dict(train))
    val_ds   = Dataset.from_dict(_to_hf_dict(val))

    # ── Local Drive backup first (guards against 504 Hub errors) ─────────────
    backup = Path(backup_path)
    backup.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Saving local backup to {backup} ...")
    train_ds.save_to_disk(str(backup / "train"))
    val_ds.save_to_disk(str(backup / "val"))
    print("  Local backup saved.")

    # ── Push train split ─────────────────────────────────────────────────────
    print(f"  Pushing train ({len(train_ds):,} rows) to {hf_repo_id}[{hf_dataset_config}] ...")
    train_ds.push_to_hub(
        repo_id=hf_repo_id,
        config_name=hf_dataset_config,
        split="train",
        token=hf_token,
        private=True,
    )

    # ── Push val split ───────────────────────────────────────────────────────
    print(f"  Pushing val ({len(val_ds):,} rows) to {hf_repo_id}[{hf_dataset_config}] ...")
    val_ds.push_to_hub(
        repo_id=hf_repo_id,
        config_name=hf_dataset_config,
        split="validation",
        token=hf_token,
        private=True,
    )

    print(f"\n  Done. Reload with:")
    print(f'    load_dataset("{hf_repo_id}", "{hf_dataset_config}", split="train")')
    print(f"  Note: 'steps' column is JSON-encoded strings. Deserialize with json.loads().")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Coconut dataset preparation for Project Ouroboros",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", default="data/coconut_v1")
    parser.add_argument(
        "--max_samples_per_source",
        type=int,
        default=None,
        help="Cap per-source samples before filtering. None = use all. "
             "Set to 5000 for a quick dry-run.",
    )
    parser.add_argument("--min_steps",  type=int, default=3,
        help="Minimum reasoning steps. Samples with fewer are dropped.")
    parser.add_argument("--max_steps",  type=int, default=16,
        help="Maximum reasoning steps. Longer samples are truncated, not dropped.")
    parser.add_argument("--min_answer_chars", type=int, default=5,
        help="Minimum answer_full length in characters.")
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    # ── Hub push args (mirrors prepare_sft_dataset.py pattern) ───────────────
    parser.add_argument("--push_to_hub", action="store_true",
        help="Push canonical JSONL dataset to HuggingFace Hub after building.")
    parser.add_argument("--hf_token", default=None,
        help="HF write token. Falls back to Colab Secrets → HF_TOKEN env var.")
    parser.add_argument("--hf_repo_id", default="WeirdRunner/Ouroboros",
        help="HuggingFace dataset repo to push to.")
    parser.add_argument("--hf_dataset_config", default="coconut-v1",
        help="Dataset config name (appears as a tab in the HF dataset viewer).")
    parser.add_argument("--backup_path", default=_DEFAULT_BACKUP_PATH,
        help="Local path for workspace/drive backup before Hub push.")

    return parser.parse_args()


def _resolve_hf_token(cli_value: Optional[str]) -> Optional[str]:
    """Resolve HF token: CLI > Kaggle Secrets > Colab Secrets > env var."""
    if cli_value:
        return cli_value
    if _KAGGLE_HF_TOKEN:
        return _KAGGLE_HF_TOKEN
    if _COLAB_HF_TOKEN:
        return _COLAB_HF_TOKEN
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    random.seed(args.seed)

    hf_token: Optional[str] = None
    if args.push_to_hub:
        hf_token = _resolve_hf_token(args.hf_token)
        if not hf_token:
            sys.exit(
                "ERROR: --push_to_hub requires an HF write token.\n"
                "  Provide via --hf_token, Kaggle/Colab Secrets (HF_TOKEN), or env var HF_TOKEN."
            )


    print("=" * 64)
    print("  Coconut Dataset Preparation — Project Ouroboros")
    print("=" * 64)
    print(f"  output_dir          : {output_dir}")
    print(f"  max_samples/source  : {args.max_samples_per_source or 'all'}")
    print(f"  min_steps           : {args.min_steps}")
    print(f"  max_steps           : {args.max_steps}")
    print(f"  min_answer_chars    : {args.min_answer_chars}")
    print(f"  val_fraction        : {args.val_fraction}")
    print(f"  push_to_hub         : {args.push_to_hub}")
    if args.push_to_hub:
        print(f"  hf_repo_id          : {args.hf_repo_id}")
        print(f"  hf_dataset_config   : {args.hf_dataset_config}")
        print(f"  backup_path         : {args.backup_path}")
    print()

    seen_hashes: set = set()
    id_counter: List[int] = [0]
    source_weights = {s["name"]: s["weight"] for s in SOURCES}
    all_by_source: Dict[str, List] = {}

    for source in SOURCES:
        samples = process_source(
            source=source,
            max_samples=args.max_samples_per_source,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
            min_answer_chars=args.min_answer_chars,
            id_prefix_counter=id_counter,
            seen_hashes=seen_hashes,
        )
        all_by_source[source["name"]] = samples

    print()
    print("  Balancing mix ...")
    mixed = build_balanced_mix(all_by_source, source_weights)

    if not mixed:
        sys.exit("No samples produced. Check dataset connectivity and --max_samples_per_source.")

    train, val = split_train_val(mixed, args.val_fraction, args.seed)

    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(val,   output_dir / "val.jsonl")

    stats = compute_stats(train, val)
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  Written: {output_dir / 'stats.json'}")

    print()
    print("=" * 64)
    print("  Summary")
    print("=" * 64)
    t = stats["train"]
    v = stats["val"]
    print(f"  Train  : {t['n_samples']:,} samples  steps: "
          f"min={t['n_steps_min']} median={t['n_steps_median']} "
          f"mean={t['n_steps_mean']} max={t['n_steps_max']}")
    print(f"  Val    : {v['n_samples']:,} samples  steps: "
          f"min={v['n_steps_min']} median={v['n_steps_median']} "
          f"mean={v['n_steps_mean']} max={v['n_steps_max']}")
    print()
    print(f"  Recommended --max_stage for jamba_coconut_finetune.py : {t['n_steps_median']}")
    print(f"  (Run with --max_stage {t['n_steps_median']} to cover median sample fully)")

    if args.push_to_hub and hf_token:
        print()
        print("  Pushing to HuggingFace Hub ...")
        upload_to_hub(
            train=train,
            val=val,
            hf_token=hf_token,
            hf_repo_id=args.hf_repo_id,
            hf_dataset_config=args.hf_dataset_config,
            backup_path=args.backup_path,
        )

    print()
    print("  Dataset ready. Feed to training script:")
    print(f"    python jamba_coconut_finetune.py --data_dir {output_dir} ...")
    print("=" * 64)


if __name__ == "__main__":
    main()
