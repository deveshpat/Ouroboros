#!/usr/bin/env python3
"""
prepare_sft_dataset.py — One-time dataset builder for Project Ouroboros Stage 2
================================================================================
Processes all 5 source datasets (same logic as train_sft.py load_mixed_dataset),
saves the result as a HuggingFace dataset, and uploads it to the Hub.

Run ONCE on any machine with internet access (Kaggle, Colab, local).
After this, train_sft.py --dataset_mix=cached skips the 25-45 min processing step
and loads the pre-built dataset directly.

Usage:
    python prepare_sft_dataset.py --hf_token $HF_TOKEN

Output:
    WeirdRunner/Ouroboros (dataset repo, config="sft-mix-v1", split="train")
    Columns: input_ids (list[int]), prompt_len (int), source (str)

Install:
    pip install transformers datasets huggingface_hub tqdm
    (mamba-ssm NOT required — this script has no model dependencies)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    sys.exit("PyTorch required: pip install torch")

try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("transformers required: pip install transformers")

try:
    from datasets import Dataset, load_dataset
except ImportError:
    sys.exit("datasets required: pip install datasets")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("tqdm required: pip install tqdm")

# ── Constants (must match train_sft.py exactly) ───────────────────────────────

TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"
HF_REPO_ID     = "WeirdRunner/Ouroboros"
DATASET_CONFIG = "sft-mix-v1"
MAX_SEQ_LEN    = 2048
SEED           = 42

SOURCE_SPECS = [
    {
        "name": "bespokelabs/Bespoke-Stratos-17k",
        "candidates": [("bespokelabs/Bespoke-Stratos-17k", None)],
        "split": "train",
        "ratio": 0.30,
    },
    {
        "name": "meta-math/MetaMathQA",
        "candidates": [("meta-math/MetaMathQA", None)],
        "split": "train",
        "ratio": 0.20,
    },
    {
        "name": "teknium/OpenHermes-2.5",
        "candidates": [("teknium/OpenHermes-2.5", None)],
        "split": "train",
        "ratio": 0.15,
    },
    {
        "name": "open-r1/OpenR1-Math-220k",
        "candidates": [
            ("open-r1/OpenR1-Math-220k", "default"),
            ("open-r1/OpenR1-Math-220k", None),
        ],
        "split": "train",
        "ratio": 0.20,
    },
    {
        "name": "open-r1/OpenR1-Code",
        "candidates": [
            ("open-r1/OpenR1-Code", None),
            ("open-r1/codeforces-cots", "solutions_w_editorials_py"),
            ("open-r1/codeforces-cots", "solutions_py"),
        ],
        "split": "train",
        "ratio": 0.15,
    },
]

# ── Parsing helpers (copied verbatim from train_sft.py) ──────────────────────

_THINK_RE   = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL)
_SOLUTION_RE= re.compile(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL)
_CLEAN_RE   = re.compile(
    r"<\|begin_of_thought\|>|<\|end_of_thought\|>|"
    r"<\|begin_of_solution\|>|<\|end_of_solution\|>|"
    r"<think>|</think>"
)


def _parse_assistant_blob(blob: str) -> Tuple[str, str]:
    blob = str(blob or "").strip()
    if not blob:
        return "", ""
    m = _THINK_RE.search(blob)
    sm = _SOLUTION_RE.search(blob)
    if m:
        r = m.group(1).strip()
        a = sm.group(1).strip() if sm else blob[m.end():].strip()
        return r, a
    if "<think>" in blob and "</think>" in blob:
        parts = blob.split("</think>", 1)
        return parts[0].replace("<think>", "").strip(), parts[1].strip()
    return "", _CLEAN_RE.sub("", blob).strip()


def _extract_chat_pair(turns: Any) -> Tuple[str, str]:
    q = a = ""
    for t in (turns or []):
        role  = str(t.get("role") or t.get("from") or "").lower().strip()
        value = str(t.get("content") or t.get("value") or "").strip()
        if role in {"user", "human"} and not q:
            q = value
        elif role in {"assistant", "gpt"} and not a:
            a = value
    return q, a


def _extract_bespoke(ex: Dict) -> Tuple[str, str, str]:
    q, blob = _extract_chat_pair(ex.get("conversations"))
    if not q or not blob:
        return "", "", ""
    r, a = _parse_assistant_blob(blob)
    return q.strip(), r.strip(), a.strip()


def _extract_metamath(ex: Dict) -> Tuple[str, str, str]:
    q = str(ex.get("original_question") or ex.get("query") or "").strip()
    a = str(ex.get("response") or ex.get("output") or "").strip()
    return q, "", a


def _extract_openhermes(ex: Dict) -> Tuple[str, str, str]:
    q = a = ""
    for t in (ex.get("conversations") or []):
        role  = str(t.get("from", "")).lower().strip()
        value = str(t.get("value", "")).strip()
        if role == "human" and not q:
            q = value
        elif role == "gpt" and not a:
            a = value
    return q, "", a


def _pick_openr1_math_gen(ex: Dict) -> str:
    gens = ex.get("generations") or []
    if not gens:
        return ""
    mv = list(ex.get("correctness_math_verify") or [])
    lv = list(ex.get("correctness_llama") or [])
    for i, g in enumerate(gens):
        if (i < len(mv) and mv[i]) or (i < len(lv) and lv[i]):
            return str(g).strip()
    return str(gens[0]).strip()


def _extract_openr1_math(ex: Dict) -> Tuple[str, str, str]:
    q, blob = _extract_chat_pair(ex.get("messages"))
    if not q:
        q = str(ex.get("problem") or ex.get("question") or "").strip()
    gen = _pick_openr1_math_gen(ex)
    if gen:
        blob = gen
    r = a = ""
    if blob:
        r, a = _parse_assistant_blob(blob)
    if not r and ex.get("solution"):
        r = str(ex.get("solution") or "").strip()
    if not a and ex.get("answer"):
        a = str(ex.get("answer") or "").strip()
    return q.strip(), r.strip(), a.strip()


def _extract_openr1_code(ex: Dict) -> Tuple[str, str, str]:
    q, blob = _extract_chat_pair(ex.get("messages"))
    if not q:
        q = str(ex.get("prompt") or "").strip()
    if not q:
        t = str(ex.get("title") or "").strip()
        d = str(ex.get("description") or "").strip()
        q = f"{t}\n\n{d}".strip()
    if not blob:
        blob = str(ex.get("generation") or ex.get("solution") or ex.get("editorial") or "").strip()
    if not q or not blob:
        return "", "", ""
    r, a = _parse_assistant_blob(blob)
    return q.strip(), r.strip(), a.strip()


EXTRACTORS = {
    "bespokelabs/Bespoke-Stratos-17k": _extract_bespoke,
    "meta-math/MetaMathQA":            _extract_metamath,
    "teknium/OpenHermes-2.5":          _extract_openhermes,
    "open-r1/OpenR1-Math-220k":        _extract_openr1_math,
    "open-r1/OpenR1-Code":             _extract_openr1_code,
}

# ── Tokenization helpers ──────────────────────────────────────────────────────

def _format_text(q: str, r: str, a: str, eos: str) -> str:
    if r:
        return f"User: {q}\n\nAssistant: <think>\n{r}\n</think>\n{a}{eos}"
    return f"User: {q}\n\nAssistant: {a}{eos}"


def _build_sample(tokenizer, q: str, r: str, a: str, eos: str) -> Optional[Dict]:
    """Build and tokenize one sample, with truncation of reasoning if needed."""
    if not r:
        text = _format_text(q, "", a, eos)
        prefix = f"User: {q}\n\nAssistant: "
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 4 or len(ids) > MAX_SEQ_LEN:
            return None
        pl = len(tokenizer.encode(prefix, add_special_tokens=False))
        if pl >= len(ids):
            return None
        return {"input_ids": ids, "prompt_len": pl}

    # Attempt with full reasoning
    sample = _try_build(tokenizer, q, r, a, eos)
    if sample:
        return sample

    # Truncate reasoning
    q_ids   = tokenizer.encode(f"User: {q}\n\nAssistant: ", add_special_tokens=False)
    a_ids   = tokenizer.encode(f"{a}{eos}", add_special_tokens=False)
    open_ids= tokenizer.encode("<think>\n", add_special_tokens=False)
    close_ids=tokenizer.encode("\n</think>\n", add_special_tokens=False)
    r_ids   = tokenizer.encode(r, add_special_tokens=False)
    base_len= len(q_ids) + len(a_ids)
    if base_len > MAX_SEQ_LEN:
        return None
    budget  = MAX_SEQ_LEN - base_len - len(open_ids) - len(close_ids)
    if budget <= 0:
        return _try_build(tokenizer, q, "", a, eos)
    truncated = r_ids[:budget] if random.random() < 0.5 else r_ids[-budget:]
    return _try_build(tokenizer, q, tokenizer.decode(truncated, skip_special_tokens=False), a, eos)


def _try_build(tokenizer, q: str, r: str, a: str, eos: str) -> Optional[Dict]:
    text   = _format_text(q, r, a, eos)
    prefix = f"User: {q}\n\nAssistant: "
    ids    = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 4 or len(ids) > MAX_SEQ_LEN:
        return None
    pl = len(tokenizer.encode(prefix, add_special_tokens=False))
    if pl >= len(ids):
        return None
    return {"input_ids": ids, "prompt_len": pl}


# ── Dataset loading ───────────────────────────────────────────────────────────

import math

def _load_first(candidates, split):
    for ds_name, cfg_name in candidates:
        try:
            ds = load_dataset(ds_name, cfg_name, split=split) if cfg_name else load_dataset(ds_name, split=split)
            return ds, ds_name, f"{ds_name}[{cfg_name}]" if cfg_name else ds_name
        except Exception:
            continue
    return None, None, None


def build_mixed_samples(tokenizer) -> List[Dict]:
    eos = tokenizer.eos_token or "<|endoftext|>"
    source_specs = []
    for spec in SOURCE_SPECS:
        print(f"  Loading {spec['name']} ...")
        ds, resolved, label = _load_first(spec["candidates"], spec["split"])
        if ds is None:
            print(f"  [warn] Could not load {spec['name']} — skipping.")
            continue
        print(f"    resolved -> {label}  ({len(ds):,} rows)")
        source_specs.append({**spec, "dataset": ds, "resolved": resolved, "label": label})

    if not source_specs:
        sys.exit("No source datasets could be loaded.")

    weight_sum = sum(s["ratio"] for s in source_specs)
    for s in source_specs:
        s["weight"] = s["ratio"] / weight_sum

    target_total = max(1, int(min(len(s["dataset"]) / s["weight"] for s in source_specs)))
    target_counts = {
        s["name"]: min(len(s["dataset"]), int(math.floor(target_total * s["weight"])))
        for s in source_specs
    }

    all_samples = []
    for spec in source_specs:
        extractor = EXTRACTORS.get(spec["name"], _extract_bespoke)
        target    = target_counts[spec["name"]]
        kept = skipped = 0
        for ex in tqdm(spec["dataset"], desc=f"  {spec['label'].split('/')[-1]}", leave=False):
            q, r, a = extractor(ex)
            if not q or not a:
                skipped += 1
                continue
            sample = _build_sample(tokenizer, q, r, a, eos)
            if sample is None:
                skipped += 1
                continue
            sample["source"] = spec["name"]
            all_samples.append(sample)
            kept += 1
            if kept >= target:
                break
        print(f"    kept {kept} / target {target} (skipped {skipped})")

    random.shuffle(all_samples)
    print(f"  Total: {len(all_samples):,} samples")
    return all_samples


# ── Upload ────────────────────────────────────────────────────────────────────

def upload_to_hub(samples: List[Dict], hf_token: str) -> None:
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    # Create as dataset repo (not model repo)
    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=hf_token,
    )
    print(f"  HF dataset repo ready: {HF_REPO_ID}")

    hf_dict = {
        "input_ids": [s["input_ids"] for s in samples],
        "prompt_len": [s["prompt_len"] for s in samples],
        "source":    [s.get("source", "") for s in samples],
    }
    ds = Dataset.from_dict(hf_dict)

    print(f"  Pushing {len(ds):,} samples to {HF_REPO_ID} (config={DATASET_CONFIG}) ...")
    ds.push_to_hub(
        repo_id=HF_REPO_ID,
        config_name=DATASET_CONFIG,
        split="train",
        token=hf_token,
        private=True,
    )
    print(f"  Done. Load with:")
    print(f'    load_dataset("{HF_REPO_ID}", "{DATASET_CONFIG}", split="train")')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build and upload Stage 2 SFT dataset.")
    parser.add_argument("--hf_token", default=None, help="HF write token (or set HF_TOKEN env var).")
    parser.add_argument("--dry_run", action="store_true", help="Process but do not upload.")
    parser.add_argument("--save_local", default=None, help="Save processed samples locally as JSON (path).")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not args.dry_run and not hf_token:
        sys.exit("--hf_token or HF_TOKEN env var required (unless --dry_run).")

    random.seed(SEED)

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  vocab: {len(tokenizer):,}")

    print("\nBuilding dataset mix ...")
    samples = build_mixed_samples(tokenizer)

    if args.save_local:
        p = Path(args.save_local)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Save as JSON for inspection; strip tensors
        serializable = [{"input_ids": s["input_ids"], "prompt_len": s["prompt_len"], "source": s.get("source","")} for s in samples]
        p.write_text(json.dumps(serializable[:100], indent=2))  # save first 100 for inspection
        print(f"  Saved first 100 samples to {p} for inspection.")

    if args.dry_run:
        print(f"\n[dry_run] Would upload {len(samples):,} samples. Exiting.")
        return

    print("\nUploading to HuggingFace Hub ...")
    upload_to_hub(samples, hf_token)
    print("\nAll done. Future train_sft.py runs can use --dataset_mix=cached.")


if __name__ == "__main__":
    main()
