#!/usr/bin/env python3
"""
prepare_coconut_dataset.py — Offline Coconut dataset preparation for Project Ouroboros.

Produces a canonical JSONL dataset from the same five sources used in sft-mix-v1,
re-processed from scratch with reasoning traces segmented into discrete steps.
Each sample is suitable for Coconut's progressive replacement curriculum.

Run once before training:
    python prepare_coconut_dataset.py --output_dir data/coconut_v1

Output files:
    data/coconut_v1/train.jsonl
    data/coconut_v1/val.jsonl
    data/coconut_v1/stats.json

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

Sources (same as sft-mix-v1, raw not tokenized):
    1. bespokelabs/Bespoke-Stratos-17k   — rich <think> blocks
    2. meta-math/MetaMathQA              — augmented math CoT
    3. teknium/OpenHermes-2.5            — general instruction; mostly filtered at min_steps
    4. open-r1/OpenR1-Math-220k          — RLVR math reasoning
    5. open-r1/OpenR1-Code               — code with reasoning traces

Samples that segment into fewer than --min_steps steps are dropped automatically.
No source requires special-casing; the filter handles quality uniformly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from datasets import load_dataset
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
    """Strip leading/trailing whitespace and remove step prefixes like '1. ' or 'Step 3: '."""
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

    # 1. Numbered / labelled steps
    if _NUMBERED_RE.search(text):
        raw = _NUMBERED_RE.split(text)
        steps = [_clean_step(s) for s in raw if s.strip()]
    # 2. Double-newline paragraphs
    elif "\n\n" in text:
        steps = [_clean_step(s) for s in text.split("\n\n") if s.strip()]
    # 3. Single-newline lines
    elif "\n" in text and text.count("\n") >= 2:
        steps = [_clean_step(s) for s in text.split("\n") if s.strip()]
    # 4. Sentence boundaries
    else:
        raw = re.split(r"(?<=[.!?])\s+", text)
        steps = [_clean_step(s) for s in raw if s.strip()]

    # Merge very short steps into the previous one
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
    """
    Extract a normalized final answer string for accuracy evaluation.

    Priority:
      1. \\boxed{...} — LaTeX math answer (GSM8K, competition math)
      2. "The answer is X" / "= X" patterns
      3. Last standalone number in the text
      4. Last non-empty sentence (general fallback)
    """
    if not text:
        return ""

    # \\boxed{...}
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()

    # "The answer is X" / "answer: X"
    m = _FINAL_ANS_RE.search(text)
    if m:
        return m.group(1).strip().replace(",", "")

    # Last standalone number
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "")

    # Last non-empty sentence
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences[-1] if sentences else text.strip()[:120]


def _doc_hash(q: str) -> str:
    """Stable hash for deduplication."""
    return hashlib.sha1(q.encode("utf-8", errors="replace")).hexdigest()


# ── Per-source extractors ─────────────────────────────────────────────────────

def _extract_think_and_answer(assistant_blob: str) -> Tuple[str, str]:
    """
    Extract (reasoning_text, answer_text) from an assistant response
    that may contain <think>...</think> or <|begin_of_thought|>...<|end_of_thought|>.
    """
    # Stratos / OpenR1 style
    m = _THINK_RE.search(assistant_blob)
    if m:
        reasoning = m.group(1).strip()
        m2 = _SOLUTION_RE.search(assistant_blob)
        answer = m2.group(1).strip() if m2 else assistant_blob[m.end():].strip()
        return reasoning, answer

    # Standard <think> style
    m = _THINK2_RE.search(assistant_blob)
    if m:
        reasoning = m.group(1).strip()
        answer = assistant_blob[m.end():].strip()
        return reasoning, answer

    # No thinking block — full blob is the answer
    return "", assistant_blob.strip()


def _chat_pair(turns: Any) -> Tuple[str, str]:
    """Extract (user_question, assistant_blob) from a turns list."""
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
    """bespokelabs/Bespoke-Stratos-17k — (question, reasoning, answer)."""
    q, blob = _chat_pair(ex.get("conversations"))
    reasoning, answer = _extract_think_and_answer(blob)
    return q.strip(), reasoning.strip(), answer.strip()


def extract_metamath(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    """meta-math/MetaMathQA — no explicit think block; full response is reasoning+answer."""
    q = str(ex.get("original_question") or ex.get("query") or "").strip()
    blob = str(ex.get("response") or ex.get("output") or "").strip()
    # MetaMathQA interleaves reasoning and answer; treat full blob as reasoning steps
    # and extract last sentence / number as answer.
    reasoning = blob
    answer = normalize_answer(blob)
    # If answer_full is very short or same as full blob, split differently
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", blob) if s.strip()]
    if sentences:
        answer_full = sentences[-1]
        reasoning = blob[: blob.rfind(answer_full)].strip() or blob
    else:
        answer_full = answer
    return q, reasoning.strip(), answer_full.strip()


def extract_openhermes(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    """teknium/OpenHermes-2.5 — no reasoning; most samples filtered at min_steps."""
    q, blob = _chat_pair(ex.get("conversations"))
    # No think block — treat as step-less; will be filtered unless blob has paragraphs
    return q.strip(), blob.strip(), ""


def extract_openr1_math(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    """open-r1/OpenR1-Math-220k."""
    q, blob = _chat_pair(ex.get("messages"))
    if not q:
        q = str(ex.get("problem") or ex.get("question") or "").strip()

    # Some rows have a 'generations' list; pick a verified one
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
    """open-r1/OpenR1-Code (or codeforces-cots fallbacks)."""
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
        "weight": 0.10,  # low weight — most samples will be filtered
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
    """Try each (dataset_name, config) candidate until one loads."""
    from datasets import load_dataset as _load
    for ds_name, config in candidates:
        try:
            ds = _load(ds_name, config, split=split) if config else _load(ds_name, split=split)
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
    """Load, extract, segment, and filter one source. Returns canonical samples."""
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

        # For sources with no reasoning block (OpenHermes), use answer_text as reasoning
        think_text = reasoning if reasoning else answer_text
        answer_full = answer_text if answer_text else reasoning

        # Segment reasoning into steps
        steps = segment_into_steps(think_text)

        if len(steps) < min_steps:
            skipped += 1
            continue
        if len(steps) > max_steps:
            # Truncate to max_steps — don't drop long samples entirely
            steps = steps[:max_steps]

        # answer_full: if we used reasoning as think_text, answer_full is the last step
        # otherwise it's the actual answer text
        if not answer_full:
            answer_full = steps[-1]
            steps = steps[:-1]
            if len(steps) < min_steps:
                skipped += 1
                continue

        if len(answer_full) < min_answer_chars:
            skipped += 1
            continue

        # Deduplication
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
    """
    Balance sources by weight, anchoring to the scarcest source relative to its weight.
    Same logic as sft-mix-v1 to match diversity.
    """
    import math as _math
    available = {name: len(samples) for name, samples in all_by_source.items() if samples}
    if not available:
        return []

    weight_sum = sum(source_weights[n] for n in available)
    weights = {n: source_weights[n] / weight_sum for n in available}

    # Anchor to scarcest source
    target_total = max(1, int(min(
        available[n] / weights[n] for n in available
    )))
    targets = {n: min(available[n], int(_math.floor(target_total * weights[n]))) for n in available}

    # Distribute remainder
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

    import random
    random.seed(42)
    random.shuffle(mixed)
    return mixed


def split_train_val(
    samples: List[Dict],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Deterministic stratified train/val split preserving source distribution."""
    import random
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
    """Compute and return dataset statistics for the stats.json manifest."""
    def _stats(samples):
        if not samples:
            return {}
        n_steps = [s["n_steps"] for s in samples]
        sources = {}
        for s in samples:
            sources[s["source"]] = sources.get(s["source"], 0) + 1
        return {
            "n_samples":     len(samples),
            "n_steps_mean":  round(sum(n_steps) / len(n_steps), 2),
            "n_steps_min":   min(n_steps),
            "n_steps_max":   max(n_steps),
            "n_steps_median": sorted(n_steps)[len(n_steps) // 2],
            "by_source":     sources,
        }

    return {"train": _stats(train), "val": _stats(val)}


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
             "Set to 5000 for a quick dry-run."
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=3,
        help="Minimum number of reasoning steps. Samples with fewer are dropped.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=16,
        help="Maximum number of reasoning steps. Longer samples are truncated, not dropped.",
    )
    parser.add_argument(
        "--min_answer_chars",
        type=int,
        default=5,
        help="Minimum answer_full length in characters.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 64)
    print("  Coconut Dataset Preparation — Project Ouroboros")
    print("=" * 64)
    print(f"  output_dir          : {output_dir}")
    print(f"  max_samples/source  : {args.max_samples_per_source or 'all'}")
    print(f"  min_steps           : {args.min_steps}")
    print(f"  max_steps           : {args.max_steps}")
    print(f"  min_answer_chars    : {args.min_answer_chars}")
    print(f"  val_fraction        : {args.val_fraction}")
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
          f"min={t['n_steps_min']} median={t['n_steps_median']} mean={t['n_steps_mean']} max={t['n_steps_max']}")
    print(f"  Val    : {v['n_samples']:,} samples  steps: "
          f"min={v['n_steps_min']} median={v['n_steps_median']} mean={v['n_steps_mean']} max={v['n_steps_max']}")
    print()
    print(f"  Recommended --max_stage for jamba_coconut_finetune.py : {t['n_steps_median']}")
    print(f"  (Run with --max_stage {t['n_steps_median']} to cover median sample fully)")
    print()
    print("  Dataset ready. Feed to training script:")
    print(f"    python jamba_coconut_finetune.py --data_dir {output_dir} ...")
    print("=" * 64)


if __name__ == "__main__":
    main()
