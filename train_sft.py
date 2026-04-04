#!/usr/bin/env python3
"""
Stage 2 SFT Training Script - Project Ouroboros / SERF Framework
================================================================
Trains BaselineTRMMamba on Bespoke-Stratos-17k, or on a mixed Stage 2
instruction corpus, using standard next-token cross-entropy loss.

Key Stage 2 features:
  - Proper <think>...</think> target formatting.
  - Cosine LR schedule with linear warmup.
  - Exponential Moving Average (EMA) of weights for stable inference.
  - Periodic validation CE and generation callbacks.
  - Checkpoint saves resolved_backbone_config.json for inference.
  - Optional local-first Hugging Face Hub checkpoint sync.
  - Resume from direct checkpoint paths, local output dirs, or Hub fallback.

Stage 2 success criterion:
  Validation CE < 1.5 on answer tokens only AND generated text is semantically coherent
  (not just grammatical) on the test prompts at --gen_every.

Hardware presets (choose via --preset):
  nano   d_model=512   n_groups=1  heads=8/4   ~92M  -> T4 / quick sanity runs
  small  d_model=1024  n_groups=2  heads=16/8  ~270M -> T4 / Kaggle single GPU
  medium d_model=2048  n_groups=2  heads=16/8  ~760M -> dual T4 / larger runs

Install:
  pip install "causal-conv1d>=1.4.0" mamba-ssm --no-build-isolation
  pip install transformers datasets wandb tqdm huggingface_hub

Run:
  python train_sft.py --preset small --output_dir ./runs/stage2_small

Dry-run style sanity check:
  python train_sft.py --preset nano --max_samples 300 --max_steps 100 \
    --val_every 50 --gen_every 50 --wandb_mode disabled
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
except ImportError:
    sys.exit("PyTorch not found. Install: pip install torch")


from training_utils import (
    ModelEMA,
    autocast_context,
    build_adamw_optimizer,
    checkpoint_step_from_name,
    cleanup_temporary_checkpoints,
    cosine_with_warmup,
    download_checkpoint_from_hub,
    ema_scope,
    list_local_checkpoints,
    list_remote_checkpoint_names,
    pad_vocab_size,
    resolve_hf_token,
    set_seed,
    sync_checkpoint_to_hub,
    try_load_state,
    vram_gb,
)

if not torch.cuda.is_available():
    sys.exit(
        "No CUDA GPU found. mamba-ssm requires CUDA kernels.\n"
        "Run on Kaggle (T4/Dual-T4) or Google Colab."
    )

try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("transformers required: pip install transformers")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("datasets required: pip install datasets")

try:
    from tqdm.auto import tqdm
except ImportError:
    sys.exit("tqdm required: pip install tqdm")

try:
    from baseline_trm_mamba import BaselineConfig, BaselineTRMMamba, count_parameters
except ImportError as exc:
    sys.exit(
        f"Cannot import baseline_trm_mamba: {exc}\n"
        "baseline_trm_mamba.py must be in the current working directory."
    )

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 SFT trainer for BaselineTRMMamba",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--preset",
        choices=["nano", "small", "medium"],
        default="small",
        help="Model size preset. nano=92M, small=270M, medium=760M.",
    )
    parser.add_argument("--max_seq_len", type=int, default=512)

    # Dataset
    parser.add_argument("--dataset_name", default="bespokelabs/Bespoke-Stratos-17k")
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap samples for dry-runs. None = use full dataset or full per-source cap.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.05,
        help="Fraction of data held out for validation CE.",
    )
    parser.add_argument(
        "--dataset_mix",
        default="stratos",
        choices=["stratos", "full"],
        help="'stratos' = Bespoke-Stratos-17k only. 'full' = 40/30/30 mix with MetaMathQA and OpenHermes-2.5.",
    )

    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override total steps. -1 = derived from epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Micro-batch size per gradient-accumulation step.",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
        help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.1,
        help="LR at end of cosine decay = lr * min_lr_ratio.",
    )
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)

    # I/O
    parser.add_argument("--output_dir", default="runs/stage2")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--keep_last", type=int, default=3)
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Path to a checkpoint directory or parent directory to resume from.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push finalized checkpoints to the Hugging Face Hub.",
    )
    parser.add_argument("--hf_repo_id", default="WeirdRunner/Ouroboros")
    parser.add_argument(
        "--hf_token",
        default=None,
        help="HF token for Hub sync. Falls back to HF_TOKEN/HUGGINGFACE_HUB_TOKEN.",
    )

    # Monitoring
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=250)
    parser.add_argument("--gen_every", type=int, default=250)
    parser.add_argument("--gen_max_tokens", type=int, default=120)

    # wandb
    parser.add_argument("--wandb_project", default="ouroboros-serf-phase2")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )

    return parser.parse_args()


PRESETS: Dict[str, Dict[str, Any]] = {
    "nano": dict(d_model=512, n_groups=1, n_heads=8, n_kv_heads=4),
    "small": dict(d_model=1024, n_groups=2, n_heads=16, n_kv_heads=8),
    "medium": dict(d_model=2048, n_groups=2, n_heads=16, n_kv_heads=8),
}

GEN_PROMPTS = [
    "What is 15 + 27?",
    "Write a Python function that returns the factorial of n.",
    "What is the capital of Japan?",
    "Explain what a neural network is in simple terms.",
    "Solve for x: 3x + 6 = 21.",
]


def pad_vocab(actual: int, multiple: int = 128) -> int:
    return math.ceil(actual / multiple) * multiple


def unique_word_ratio(text: str) -> float:
    words = text.split()
    return len(set(words)) / max(len(words), 1)


def sanitize_args_for_serialization(args: argparse.Namespace) -> Dict[str, Any]:
    """Return a JSON-safe args dict with secrets removed."""
    cfg = dict(vars(args))
    cfg.pop("hf_token", None)
    return cfg


_THINK_RE = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL)
_SOLUTION_RE = re.compile(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL)
_CLEAN_RE = re.compile(
    r"<\|begin_of_thought\|>|<\|end_of_thought\|>|"
    r"<\|begin_of_solution\|>|<\|end_of_solution\|>|"
    r"<think>|</think>"
)


def _extract_bespoke(example: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract (question, reasoning_chain, final_answer) from a Bespoke-Stratos row."""
    question = ""
    assistant_blob = ""
    for turn in (example.get("conversations") or []):
        role = str(turn.get("from", "")).lower().strip()
        value = str(turn.get("value", "")).strip()
        if role == "user" and not question:
            question = value
        elif role == "assistant" and not assistant_blob:
            assistant_blob = value

    if not question or not assistant_blob:
        return "", "", ""

    think_match = _THINK_RE.search(assistant_blob)
    solution_match = _SOLUTION_RE.search(assistant_blob)

    if think_match:
        reasoning = think_match.group(1).strip()
        answer = solution_match.group(1).strip() if solution_match else assistant_blob[think_match.end():].strip()
    elif "<think>" in assistant_blob and "</think>" in assistant_blob:
        parts = assistant_blob.split("</think>", 1)
        reasoning = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
    else:
        reasoning = ""
        answer = _CLEAN_RE.sub("", assistant_blob).strip()

    return question.strip(), reasoning.strip(), answer.strip()


def _format_training_text(question: str, reasoning: str, answer: str, eos: str) -> str:
    """Build the exact Stage 2 target format taught to the model."""
    if reasoning:
        return (
            f"User: {question}\n\n"
            f"Assistant: <think>\n{reasoning}\n</think>\n"
            f"{answer}{eos}"
        )
    return f"User: {question}\n\nAssistant: {answer}{eos}"

def _build_prompt_prefix(question: str, reasoning: str) -> str:
    """Build the exact non-supervised prompt prefix for one SFT sample."""
    prefix = f"User: {question}\n\nAssistant: "
    if reasoning:
        prefix += "<think>\n"
    return prefix


def _build_sft_sample(
    tokenizer,
    question: str,
    reasoning: str,
    answer: str,
    eos: str,
    max_seq_len: int,
) -> Dict[str, Any]:
    """Tokenize one SFT sample and record the prompt length for loss masking."""
    text = _format_training_text(question, reasoning, answer, eos)
    prefix_text = _build_prompt_prefix(question, reasoning)
    ids = tokenizer.encode(text, add_special_tokens=False)
    prompt_len = len(tokenizer.encode(prefix_text, add_special_tokens=False))
    ids = ids[:max_seq_len]
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "prompt_len": min(prompt_len, len(ids)),
    }


def load_and_tokenize(
    dataset_name: str,
    tokenizer,
    max_samples: Optional[int],
    max_seq_len: int,
) -> List[Dict[str, Any]]:
    """Load one dataset, format rows, tokenize, and filter invalid samples."""
    print(f"Loading {dataset_name} ...")
    raw = load_dataset(dataset_name, split="train")
    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    eos = tokenizer.eos_token or "<|endoftext|>"
    samples: List[Dict[str, Any]] = []
    skipped = 0

    for ex in tqdm(raw, desc="Formatting + tokenizing", leave=False):
        q, r, a = _extract_bespoke(ex)
        if not q or not a:
            skipped += 1
            continue
        text = _format_training_text(q, r, a, eos)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 4:
            skipped += 1
            continue
        samples.append(_build_sft_sample(tokenizer, q, r, a, eos, max_seq_len))

    print(f"  {len(samples)} samples kept, {skipped} skipped (missing fields / too short).")
    return samples


def _extract_metamath(example: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract (question, reasoning, answer) from MetaMathQA."""
    question = str(example.get("original_question") or example.get("query") or "").strip()
    answer = str(example.get("response") or example.get("output") or "").strip()
    return question, "", answer


def _extract_openhermes(example: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract (question, reasoning, answer) from OpenHermes-2.5 conversations."""
    question = ""
    answer = ""
    for turn in (example.get("conversations") or []):
        role = str(turn.get("from", "")).lower().strip()
        value = str(turn.get("value", "")).strip()
        if role == "human" and not question:
            question = value
        elif role == "gpt" and not answer:
            answer = value
    return question, "", answer


def load_mixed_dataset(
    tokenizer,
    max_samples_per_source: Optional[int],
    max_seq_len: int,
) -> List[Dict[str, Any]]:
    """Load a practical 40/30/30 Stratos/MetaMath/OpenHermes training mix."""
    sources = [
        ("bespokelabs/Bespoke-Stratos-17k", "train", _extract_bespoke, 0.40),
        ("meta-math/MetaMathQA", "train", _extract_metamath, 0.30),
        ("teknium/OpenHermes-2.5", "train", _extract_openhermes, 0.30),
    ]
    eos = tokenizer.eos_token or "<|endoftext|>"
    source_specs: List[Dict[str, Any]] = []

    for ds_name, split, extractor, ratio in sources:
        print(f"  Loading {ds_name} ...")
        try:
            raw = load_dataset(ds_name, split=split)
        except Exception as exc:
            print(f"  [warn] Could not load {ds_name}: {exc} - skipping.")
            continue

        available = len(raw)
        if max_samples_per_source is not None:
            available = min(available, max_samples_per_source)
            raw = raw.select(range(available))

        source_specs.append(
            {
                "name": ds_name,
                "dataset": raw,
                "extractor": extractor,
                "ratio": ratio,
                "available": available,
            }
        )

    if not source_specs:
        return []

    weight_sum = sum(spec["ratio"] for spec in source_specs)
    for spec in source_specs:
        spec["weight"] = spec["ratio"] / weight_sum

    # reason: anchor the actual mix to the scarcest available source after any per-source cap.
    target_total = max(1, int(min(spec["available"] / spec["weight"] for spec in source_specs)))
    raw_counts = {spec["name"]: target_total * spec["weight"] for spec in source_specs}
    target_counts = {
        spec["name"]: min(spec["available"], int(math.floor(raw_counts[spec["name"]])))
        for spec in source_specs
    }

    remaining = target_total - sum(target_counts.values())
    if remaining > 0:
        by_fraction = sorted(
            source_specs,
            key=lambda spec: raw_counts[spec["name"]] - target_counts[spec["name"]],
            reverse=True,
        )
        while remaining > 0:
            progressed = False
            for spec in by_fraction:
                name = spec["name"]
                if target_counts[name] < spec["available"]:
                    target_counts[name] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                break

    all_samples: List[Dict[str, Any]] = []
    actual_counts: Dict[str, int] = {}

    for spec in source_specs:
        ds_name = spec["name"]
        extractor = spec["extractor"]
        target = target_counts[ds_name]
        kept = 0
        for ex in tqdm(spec["dataset"], desc=f"  {ds_name.split('/')[-1]}", leave=False):
            q, r, a = extractor(ex)
            if not q or not a:
                continue
            text = _format_training_text(q, r, a, eos)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 4:
                continue
            sample = _build_sft_sample(tokenizer, q, r, a, eos, max_seq_len)
            sample["source"] = ds_name
            all_samples.append(sample)
            kept += 1
            if kept >= target:
                break
        actual_counts[ds_name] = kept
        print(f"    kept {kept} / target {target} samples.")

    random.shuffle(all_samples)
    print(f"  Total mixed samples: {len(all_samples)}")
    if actual_counts:
        pretty_counts = ", ".join(
            f"{name.split('/')[-1]}={count}" for name, count in actual_counts.items()
        )
        print(f"  Mix counts: {pretty_counts}")
    return all_samples


def split_train_val_samples(
    all_samples: List[Dict[str, Any]],
    seed: int,
    val_fraction: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically split tokenized SFT samples into train and validation sets."""
    if len(all_samples) < 2:
        raise ValueError("Need at least 2 samples to create non-empty train and validation splits.")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(all_samples), generator=generator).tolist()
    n_val = min(len(all_samples) - 1, max(1, int(len(all_samples) * val_fraction)))
    val_idx = set(perm[:n_val])
    train_samples = [sample for idx, sample in enumerate(all_samples) if idx not in val_idx]
    val_samples = [sample for idx, sample in enumerate(all_samples) if idx in val_idx]
    return train_samples, val_samples


def collate(samples: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad a micro-batch to the longest example and mask padding in labels."""
    max_len = max(sample["input_ids"].size(0) for sample in samples)
    batch_size = len(samples)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for idx, sample in enumerate(samples):
        ids = sample["input_ids"]
        length = ids.size(0)
        input_ids[idx, :length] = ids
        pl = min(sample.get("prompt_len", 0), length)
        labels[idx, pl:length] = ids[pl:]
        mask[idx, :length] = True

    return {"input_ids": input_ids, "attention_mask": mask, "labels": labels}


@torch.no_grad()
def run_generation_callback(
    model: BaselineTRMMamba,
    ema: ModelEMA,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    step: int,
    max_new_tokens: int,
    max_seq_len: int,
    wandb_run=None,
) -> None:
    """Run greedy decoding on the fixed prompts using EMA weights."""
    model.eval()
    print(f"\n  -- Generation @ step {step} (EMA weights) --")
    wandb_log: Dict[str, str] = {}
    mean_uwr = 0.0

    with ema_scope(model, ema):
        for prompt in GEN_PROMPTS:
            prefix = f"User: {prompt}\n\nAssistant: <think>\n"
            ids = torch.tensor(
                tokenizer.encode(prefix, add_special_tokens=False),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            eos_id = tokenizer.eos_token_id
            generated: List[int] = []

            for _ in range(max_new_tokens):
                if ids.size(1) > max_seq_len:
                    ids = ids[:, -max_seq_len:]
                with autocast_context(device, dtype):
                    logits = model(ids)
                next_id = int(logits[:, -1, :].argmax(dim=-1).item())
                if eos_id is not None and next_id == eos_id:
                    break
                generated.append(next_id)
                ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

            output_text = tokenizer.decode(generated, skip_special_tokens=True)
            uwr = unique_word_ratio(output_text)
            mean_uwr += uwr
            display = output_text[:180].replace("\n", " ")
            print(f"  Q: {prompt}")
            print(f"  A: {display}")
            print(f"     uwr={uwr:.3f}")
            wandb_log[f"gen/{prompt[:40]}"] = output_text

    mean_uwr /= max(len(GEN_PROMPTS), 1)
    print(f"  Mean UWR: {mean_uwr:.3f}\n")

    if wandb_run is not None:
        import wandb

        wandb.log({"gen/mean_uwr": mean_uwr, **wandb_log}, step=step)

    model.train()


@torch.no_grad()
def compute_val_ce(
    model: BaselineTRMMamba,
    ema: ModelEMA,
    val_samples: List[Dict[str, Any]],
    pad_id: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    vocab_size: int,
) -> float:
    """Compute per-token CE on the validation split using EMA weights."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with ema_scope(model, ema):
        for start in range(0, len(val_samples), batch_size):
            batch_samples = val_samples[start : start + batch_size]
            batch = collate(batch_samples, pad_id)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast_context(device, dtype):
                logits = model(input_ids, attention_mask=attn_mask)

            shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size).float()
            shift_labels = labels[:, 1:].contiguous().view(-1)
            valid = shift_labels != -100
            if valid.any():
                total_loss += F.cross_entropy(
                    shift_logits[valid], shift_labels[valid], reduction="sum"
                ).item()
                total_tokens += int(valid.sum().item())

    model.train()
    return total_loss / max(total_tokens, 1)


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: BaselineTRMMamba,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    config: BaselineConfig,
    val_ce: Optional[float],
    keep_last: int,
    epoch: int,
    samples_seen: int,
    sft_config: Dict[str, Any],
    push_to_hub: bool = False,
    hf_repo_id: str = "WeirdRunner/Ouroboros",
    hf_token: Optional[str] = None,
) -> Optional[Path]:
    """Write a Stage 2 checkpoint locally first, then try a Hub push."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"checkpoint-{step:07d}"
    final_dir = output_dir / ckpt_name
    tmp_dir = output_dir / f"{ckpt_name}.tmp"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    shadow = dict(ema.shadow)
    if config.tie_embeddings and "lm_head.weight" not in shadow and "token_embedding.weight" in shadow:
        shadow["lm_head.weight"] = shadow["token_embedding.weight"]

    cfg_dict = {k: v for k, v in (sft_config or {}).items() if k != "hf_token"}
    state = {
        "stage": "sft",
        "step": step,
        "epoch": epoch,
        "samples_seen": samples_seen,
        "val_ce": val_ce,
        "model_state_dict": model.state_dict(),
        "ema_backbone_state_dict": shadow,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "ema": ema.state_dict(),
        "backbone_config": asdict(config),
        "sft_config": cfg_dict,
    }

    try:
        torch.save(state, tmp_dir / "training_state.pt")
        with (tmp_dir / "resolved_backbone_config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2)
    except Exception as exc:
        print(f"  [ckpt] ERROR: could not write checkpoint to {tmp_dir}: {exc}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    if final_dir.exists():
        shutil.rmtree(final_dir, ignore_errors=True)
    tmp_dir.replace(final_dir)
    print(f"  [ckpt] saved  -> {final_dir}")

    retain = max(int(keep_last), 1)
    existing = sorted(
        [
            entry
            for entry in output_dir.iterdir()
            if entry.is_dir()
            and entry.name.startswith("checkpoint-")
            and not entry.name.endswith(".tmp")
            and checkpoint_step_from_name(entry.name) >= 0
        ],
        key=lambda entry: checkpoint_step_from_name(entry.name),
    )
    for old in existing[:-retain]:
        shutil.rmtree(old, ignore_errors=True)
        print(f"  [ckpt] pruned -> {old.name}")

    if push_to_hub and hf_token:
        uploaded = sync_checkpoint_to_hub(final_dir, hf_repo_id, hf_token)
        if not uploaded:
            print(
                f"  [hub]  warn: step {step} Hub sync failed; "
                f"local checkpoint retained at {final_dir}"
            )

    return final_dir


def _looks_like_pretrain_checkpoint(state: Dict[str, Any]) -> bool:
    """Return True when a checkpoint looks like a Stage 1 pre-training save."""
    return ("pretrain_config" in state) or (
        "tokens_processed" in state and "chunks_in_epoch" in state
    )


def _load_ema_shadow_from_alias(ema: ModelEMA, shadow_state: Dict[str, torch.Tensor]) -> None:
    """Best-effort load of EMA shadow tensors keyed by parameter name."""
    for name, tensor in shadow_state.items():
        if name in ema.shadow and ema.shadow[name].shape == tensor.shape:
            ema.shadow[name].copy_(tensor)


def load_latest_checkpoint(
    output_dir: Path,
    model: BaselineTRMMamba,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    device: torch.device,
    push_to_hub: bool = False,
    hf_repo_id: str = "WeirdRunner/Ouroboros",
    hf_token: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[int, int, int]:
    """Load the newest valid Stage 2 checkpoint or initialize from the newest Stage 1 checkpoint."""

    def _restore(state: Dict[str, Any], label: str) -> Optional[Tuple[int, int, int]]:
        if "model_state_dict" not in state:
            if verbose:
                print(f"  [resume] incomplete checkpoint {label} - skipping")
            return None

        pretrain_init = _looks_like_pretrain_checkpoint(state) and "sft_config" not in state
        epoch = int(state.get("epoch", 0))
        samples_seen = int(state.get("samples_seen", 0))
        model.load_state_dict(state["model_state_dict"])
        if state.get("ema"):
            ema.load_state_dict(state["ema"])
        elif state.get("ema_backbone_state_dict"):
            _load_ema_shadow_from_alias(ema, state["ema_backbone_state_dict"])

        if pretrain_init:
            if verbose:
                print(
                    f"  [init]   {label}  loaded Stage 1 weights; "
                    "resetting optimizer/scheduler for Stage 2."
                )
            return 0, 0, 0

        if any(key not in state for key in ("optimizer", "scheduler")):
            if verbose:
                print(f"  [resume] incomplete Stage 2 checkpoint {label} - skipping")
            return None

        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        step = int(state.get("step", 0))
        if verbose:
            print(
                f"  [resume] {label}  step={step}  epoch={epoch}  "
                f"samples_seen={samples_seen}  val_ce={state.get('val_ce')}"
            )
        return step, epoch, samples_seen

    search_root = Path(output_dir)
    direct_candidates: List[Path] = []

    if search_root.is_file() and search_root.name == "training_state.pt":
        direct_candidates.append(search_root.parent)
        if search_root.parent.name.startswith("checkpoint-"):
            search_root = search_root.parent.parent
        else:
            search_root = search_root.parent
    elif (search_root / "training_state.pt").exists():
        direct_candidates.append(search_root)
        if search_root.name.startswith("checkpoint-"):
            search_root = search_root.parent
    elif search_root.name.startswith("checkpoint-"):
        search_root = search_root.parent

    local_root = search_root if search_root.exists() else search_root.parent
    local_records: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()

    for candidate in direct_candidates + list_local_checkpoints(local_root):
        candidate_str = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if candidate_str in seen_paths:
            continue
        seen_paths.add(candidate_str)
        state = try_load_state(candidate, device)
        if state is None:
            continue
        local_records.append(
            {
                "step": int(state.get("step", checkpoint_step_from_name(candidate.name))),
                "source": "local",
                "priority": 1,
                "label": candidate.name,
                "state": state,
            }
        )

    remote_records: List[Dict[str, Any]] = []
    if hf_token:
        for ckpt_name in list_remote_checkpoint_names(hf_repo_id, hf_token):
            remote_records.append(
                {
                    "step": checkpoint_step_from_name(ckpt_name),
                    "source": "hub",
                    "priority": 0,
                    "label": ckpt_name,
                }
            )

    if verbose:
        local_latest = local_records[0]["step"] if local_records else None
        hub_latest = remote_records[0]["step"] if remote_records else None
        print(
            "  [resume] candidate summary: "
            f"local_latest={local_latest if local_latest is not None else 'none'}  "
            f"hub_latest={hub_latest if hub_latest is not None else 'none'}"
        )

    candidates = sorted(
        local_records + remote_records,
        key=lambda record: (int(record.get("step", -1)), int(record.get("priority", 0))),
        reverse=True,
    )

    for record in candidates:
        if record["source"] == "local":
            restored = _restore(record["state"], record["label"])
            if restored is not None:
                return restored
            continue

        if verbose:
            print(f"  [hub]  downloading {record['label']} ...")
        downloaded = download_checkpoint_from_hub(record["label"], local_root, hf_repo_id, hf_token)
        if downloaded is None:
            continue
        state = try_load_state(downloaded, device)
        if state is None:
            continue
        restored = _restore(state, record["label"])
        if restored is not None:
            return restored

    if verbose:
        print("  [resume] No checkpoint found - starting from scratch.")
    return 0, 0, 0



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    hf_token = resolve_hf_token(args.hf_token)
    if args.push_to_hub and not hf_token:
        sys.exit("--push_to_hub requires --hf_token or HF_TOKEN/HUGGINGFACE_HUB_TOKEN")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_temporary_checkpoints(output_dir)

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats(device)

    wandb_run = None
    if args.wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config=sanitize_args_for_serialization(args),
            )
        except ImportError:
            print("[warn] wandb not installed - logging to stdout only.")

    print()
    print("=" * 64)
    print("  Stage 2 SFT - Project Ouroboros")
    print("=" * 64)
    print(f"  preset        : {args.preset}")
    print(f"  seq_len       : {args.max_seq_len}")
    print(f"  batch x accum : {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  lr            : {args.lr}  warmup={args.warmup_steps}")
    print(f"  dtype         : {dtype}")
    print(f"  output_dir    : {output_dir}")
    print(f"  dataset_mix   : {args.dataset_mix}")
    print(f"  answer_only_ce: True")
    print(f"  push_to_hub   : {args.push_to_hub}")
    print("=" * 64)
    print()

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"  vocab: {len(tokenizer):,}  pad_token: '{tokenizer.pad_token}'")
    print()

    if args.dataset_mix == "full":
        all_samples = load_mixed_dataset(tokenizer, args.max_samples, args.max_seq_len)
    else:
        all_samples = load_and_tokenize(args.dataset_name, tokenizer, args.max_samples, args.max_seq_len)
    if not all_samples:
        sys.exit("No samples loaded. Check dataset connectivity and --max_samples.")

    train_samples, val_samples = split_train_val_samples(
        all_samples=all_samples,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    pad_id = tokenizer.pad_token_id or 0

    print(f"  train: {len(train_samples)}  val: {len(val_samples)}")
    print()

    preset_kwargs = PRESETS[args.preset]
    padded_vocab = pad_vocab_size(len(tokenizer), 128)
    config = BaselineConfig(vocab_size=padded_vocab, max_seq_len=args.max_seq_len, dropout=0.0, **preset_kwargs)
    model = BaselineTRMMamba(config).to(device=device, dtype=dtype)
    model.train()

    n_params = count_parameters(model)
    print(f"Model  : {n_params / 1e6:.1f}M parameters  (preset={args.preset})")
    print(
        f"Config : d_model={config.d_model}  n_groups={config.n_groups} "
        f" heads={config.n_heads}/{config.n_kv_heads}  mlp_hidden={config.mlp_hidden}"
    )
    print(f"Vocab  : {padded_vocab:,} (padded from {len(tokenizer):,})")
    print()

    optimizer, fused_enabled = build_adamw_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
        prefer_fused=True,
    )
    print(f"Optimizer : AdamW ({'fused CUDA kernel' if fused_enabled else 'standard'})")

    steps_per_epoch = math.ceil(len(train_samples) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.num_epochs
    scheduler = cosine_with_warmup(optimizer, args.warmup_steps, total_steps, args.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    ema = ModelEMA(model, decay=args.ema_decay)

    print(
        f"Schedule  : cosine warmup={args.warmup_steps} total={total_steps} "
        f" ({args.num_epochs} epochs x {steps_per_epoch} steps/epoch)"
    )
    print()

    resume_search = Path(args.resume_from) if args.resume_from else output_dir
    if args.resume_from and list_local_checkpoints(output_dir):
        resume_path = Path(args.resume_from).resolve()
        output_root = output_dir.resolve()
        resume_inside_output = (resume_path == output_root) or (output_root in resume_path.parents)
        if not resume_inside_output:
            print("  [resume] found local Stage 2 checkpoints in output_dir; preferring them over external --resume_from")
            resume_search = output_dir
    if resume_search.is_dir():
        cleanup_temporary_checkpoints(resume_search)

    start_step, start_epoch, samples_seen = load_latest_checkpoint(
        resume_search,
        model,
        ema,
        optimizer,
        scheduler,
        scaler,
        device,
        push_to_hub=args.push_to_hub,
        hf_repo_id=args.hf_repo_id,
        hf_token=hf_token,
        verbose=True,
    )

    torch.manual_seed(args.seed)
    if train_samples:
        samples_per_step = args.batch_size * args.grad_accum
        derived_samples_seen = start_step * samples_per_step
        if samples_seen <= 0:
            samples_seen = derived_samples_seen
        elif samples_seen != derived_samples_seen:
            print(
                f"  [resume] warning: checkpoint samples_seen={samples_seen} differs from "
                f"derived value {derived_samples_seen}; trusting checkpoint."
            )
        completed_epochs = samples_seen // len(train_samples)
        sample_idx = samples_seen % len(train_samples)
        perm_train = []
        for _ in range(completed_epochs + 1):
            perm_train = torch.randperm(len(train_samples)).tolist()
        if start_epoch <= 0:
            start_epoch = completed_epochs
    else:
        sample_idx = 0
        perm_train = []

    col_hdr = f"{'Step':>7}  {'Train CE':>9}  {'Val CE':>9}  {'GNorm':>7}  {'LR':>9}  {'VRAM':>7}  {'Tok/s':>7}"
    print(col_hdr)
    print("-" * 72)

    pbar = tqdm(total=total_steps, initial=start_step, desc="Training", dynamic_ncols=True)
    step = start_step
    micro_step = 0
    accum_loss = 0.0
    last_val_ce: Optional[float] = None
    t0 = time.perf_counter()
    tokens_seen = 0

    optimizer.zero_grad(set_to_none=True)

    logical_epoch = start_epoch

    while step < total_steps:
        micro_batch: List[Dict[str, Any]] = []
        for _ in range(args.batch_size):
            if sample_idx >= len(train_samples):
                sample_idx = 0
                logical_epoch += 1
                perm_train = torch.randperm(len(train_samples)).tolist()
            micro_batch.append(train_samples[perm_train[sample_idx]])
            sample_idx += 1

        batch = collate(micro_batch, pad_id)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        tokens_seen += int(attn_mask.sum().item())
        samples_seen += len(micro_batch)

        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(input_ids, attention_mask=attn_mask)

        shift_logits = logits[:, :-1, :].contiguous().view(-1, config.vocab_size).float()
        shift_labels = labels[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        if dtype == torch.float16:
            scaler.scale(loss / args.grad_accum).backward()
        else:
            (loss / args.grad_accum).backward()

        accum_loss += loss.detach().item()
        micro_step += 1
        if micro_step % args.grad_accum != 0:
            continue

        if dtype == torch.float16:
            scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))

        if dtype == torch.float16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)

        step += 1
        mean_ce = accum_loss / args.grad_accum
        accum_loss = 0.0

        pbar.update(1)
        pbar.set_postfix(ce=f"{mean_ce:.3f}", gn=f"{grad_norm:.3f}")

        if step % args.log_every == 0 or step == 1:
            elapsed = max(time.perf_counter() - t0, 1e-6)
            tok_s = tokens_seen / elapsed
            cur_lr = scheduler.get_last_lr()[0]
            vram = vram_gb(device)
            val_str = f"{last_val_ce:.4f}" if last_val_ce is not None else "     -"

            tqdm.write(
                f"{step:>7}  {mean_ce:>9.4f}  {val_str:>9}  "
                f"{grad_norm:>7.4f}  {cur_lr:>9.2e}  {vram:>7.3f}  {tok_s:>7.0f}"
            )

            if wandb_run is not None:
                import wandb

                wandb.log(
                    {
                        "train/ce": mean_ce,
                        "train/grad_norm": grad_norm,
                        "train/lr": cur_lr,
                        "train/tok_s": tok_s,
                        "train/vram_gb": vram,
                    },
                    step=step,
                )

        if step % args.val_every == 0 or step == total_steps:
            last_val_ce = compute_val_ce(
                model,
                ema,
                val_samples,
                pad_id,
                device,
                dtype,
                args.batch_size,
                config.vocab_size,
            )
            tqdm.write(f"  [val] step={step}  val_ce={last_val_ce:.4f}")

            if wandb_run is not None:
                import wandb

                wandb.log({"val/ce": last_val_ce}, step=step)

            if last_val_ce < 1.5:
                tqdm.write(
                    "  * val_ce < 1.5 - Stage 2 success criterion met. "
                    "Consider proceeding to Phase 3."
                )

        if step % args.gen_every == 0 or step == total_steps:
            run_generation_callback(
                model,
                ema,
                tokenizer,
                device,
                dtype,
                step=step,
                max_new_tokens=args.gen_max_tokens,
                max_seq_len=args.max_seq_len,
                wandb_run=wandb_run,
            )

        if step % args.save_every == 0 or step == total_steps:
            save_checkpoint(
                output_dir=output_dir,
                step=step,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if dtype == torch.float16 else None,
                config=config,
                val_ce=last_val_ce,
                keep_last=args.keep_last,
                epoch=logical_epoch,
                samples_seen=samples_seen,
                sft_config=sanitize_args_for_serialization(args),
                push_to_hub=args.push_to_hub,
                hf_repo_id=args.hf_repo_id,
                hf_token=hf_token,
            )

    pbar.close()

    total_time = time.perf_counter() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    print()
    print("=" * 64)
    print("  Stage 2 Training Complete")
    print("=" * 64)
    print(f"  Total steps  : {step}")
    print(f"  Total time   : {total_time / 60:.1f} min")
    print(f"  Peak VRAM    : {peak_vram_gb:.2f} GB")
    if last_val_ce is not None:
        print(f"  Final val CE : {last_val_ce:.4f}  (answer tokens only)")
        if last_val_ce < 1.5:
            print("  Status       : SUCCESS - proceed to Phase 3 (incremental recursion)")
        else:
            print("  Status       : val_ce >= 1.5 - extend training or check data quality")
    print("=" * 64)

    if wandb_run is not None:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
