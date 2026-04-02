#!/usr/bin/env python3
"""
Phase 2 SFT Training Script — Project Ouroboros / SERF Framework
=================================================================
Trains BaselineTRMMamba on the full Bespoke-Stratos-17k dataset
using standard next-token cross-entropy loss.

Key differences from Phase 1 (the viability gate):
  - Full dataset (16 710 samples) with a 90/10 train/val split.
  - Proper <think>...</think> target format.
  - Cosine LR schedule with linear warmup.
  - Exponential Moving Average (EMA) of weights for stable inference.
  - generate_sample callback every --gen_every steps — the single most
    important monitoring signal for a language model.
  - Periodic validation CE (not just training CE).
  - Checkpoint saves resolved_backbone_config.json for inference.

Phase 2 success criterion (proceed to Phase 3 when met):
  Validation CE < 1.5  AND  generated text is semantically coherent
  (not just grammatical) on the test prompts at --gen_every.

Hardware presets (choose via --preset):
  nano   d_model=512   n_groups=1  heads=8/4   ~92M  → T4 (Colab free)
  small  d_model=1024  n_groups=2  heads=16/8  ~270M → T4 (Kaggle single)
  medium d_model=2048  n_groups=2  heads=16/8  ~760M → Dual T4 (Kaggle)

Default preset is 'small'. Nano is useful for a quick end-to-end sanity
check before committing to a multi-hour small/medium run.

Install:
  pip install "causal-conv1d>=1.4.0" mamba-ssm --no-build-isolation
  pip install transformers datasets wandb tqdm

Run (Kaggle / Colab / any single GPU):
  python train_sft_phase2.py --preset small --output_dir ./runs/phase2_small

Dry-run to verify pipeline without full data download:
  python train_sft_phase2.py --preset nano --max_samples 200 --max_steps 50
    --wandb_mode disabled
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Dependency checks ────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
except ImportError:
    sys.exit("PyTorch not found. Install: pip install torch")

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 SFT trainer for BaselineTRMMamba",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--preset", choices=["nano", "small", "medium"], default="small",
                   help="Model size preset. nano=92M, small=270M, medium=760M.")
    p.add_argument("--max_seq_len", type=int, default=512)

    # ── Dataset ───────────────────────────────────────────────────────────────
    p.add_argument("--dataset_name", default="bespokelabs/Bespoke-Stratos-17k")
    p.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap samples for dry-runs. None = use full dataset.")
    p.add_argument("--val_fraction", type=float, default=0.05,
                   help="Fraction of data held out for validation CE.")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Override total steps. -1 = derived from epochs.")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Micro-batch size per gradient-accumulation step.")
    p.add_argument("--grad_accum", type=int, default=8,
                   help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1,
                   help="LR at end of cosine decay = lr * min_lr_ratio.")
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)

    # ── I/O ───────────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", default="runs/phase2")
    p.add_argument("--save_every", type=int, default=500,
                   help="Save a checkpoint every N optimizer steps.")
    p.add_argument("--keep_last", type=int, default=3,
                   help="Number of recent checkpoints to keep on disk.")
    p.add_argument("--resume_from", default=None,
                   help="Path to a checkpoint directory to resume from.")

    # ── Monitoring ────────────────────────────────────────────────────────────
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every", type=int, default=250,
                   help="Compute validation CE every N optimizer steps.")
    p.add_argument("--gen_every", type=int, default=250,
                   help="Run the generation callback every N optimizer steps.")
    p.add_argument("--gen_max_tokens", type=int, default=120)

    # ── wandb ─────────────────────────────────────────────────────────────────
    p.add_argument("--wandb_project", default="ouroboros-serf-phase2")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"],
                   default="online")

    return p.parse_args()


# ── Model presets ─────────────────────────────────────────────────────────────

PRESETS: Dict[str, Dict[str, Any]] = {
    "nano":   dict(d_model=512,  n_groups=1, n_heads=8,  n_kv_heads=4),
    "small":  dict(d_model=1024, n_groups=2, n_heads=16, n_kv_heads=8),
    "medium": dict(d_model=2048, n_groups=2, n_heads=16, n_kv_heads=8),
}


# ── Fixed generation prompts ──────────────────────────────────────────────────
# These never change across runs so quality is directly comparable.

GEN_PROMPTS = [
    "What is 15 + 27?",
    "Write a Python function that returns the factorial of n.",
    "What is the capital of Japan?",
    "Explain what a neural network is in simple terms.",
    "Solve for x: 3x + 6 = 21.",
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vram_gb(device: torch.device) -> float:
    return torch.cuda.memory_allocated(device) / (1024 ** 3)


def pad_vocab(actual: int, multiple: int = 128) -> int:
    return math.ceil(actual / multiple) * multiple


def unique_word_ratio(text: str) -> float:
    words = text.split()
    return len(set(words)) / max(len(words), 1)


# ── Dataset ───────────────────────────────────────────────────────────────────

_THINK_RE        = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL)
_SOLUTION_RE     = re.compile(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL)
_CLEAN_RE        = re.compile(r"<\|begin_of_thought\|>|<\|end_of_thought\|>|"
                               r"<\|begin_of_solution\|>|<\|end_of_solution\|>|"
                               r"<think>|</think>")


def _extract_bespoke(example: Dict) -> Tuple[str, str, str]:
    """
    Extract (question, reasoning_chain, final_answer) from a Bespoke-Stratos row.
    Returns three empty strings if the row is unusable.
    """
    question = assistant_blob = ""
    for turn in (example.get("conversations") or []):
        role = str(turn.get("from", "")).lower().strip()
        val  = str(turn.get("value", "")).strip()
        if role == "user" and not question:
            question = val
        elif role == "assistant" and not assistant_blob:
            assistant_blob = val

    if not question or not assistant_blob:
        return "", "", ""

    # Try DeepSeek R1 tag format first, then plain <think> tags
    think_m    = _THINK_RE.search(assistant_blob)
    solution_m = _SOLUTION_RE.search(assistant_blob)

    if think_m:
        reasoning = think_m.group(1).strip()
        answer    = solution_m.group(1).strip() if solution_m else (
            assistant_blob[think_m.end():].strip()
        )
    elif "<think>" in assistant_blob and "</think>" in assistant_blob:
        parts     = assistant_blob.split("</think>", 1)
        reasoning = parts[0].replace("<think>", "").strip()
        answer    = parts[1].strip()
    else:
        # No tags — treat entire assistant turn as answer, no chain-of-thought
        reasoning = ""
        answer    = _CLEAN_RE.sub("", assistant_blob).strip()

    return question.strip(), reasoning.strip(), answer.strip()


def _format_training_text(question: str, reasoning: str, answer: str, eos: str) -> str:
    """
    Build the exact target format taught to the model.
    The <think> scaffold is included so the model learns deliberation structure
    even though Phase 2 does not supervise the reasoning quality explicitly.
    """
    if reasoning:
        return (
            f"User: {question}\n\n"
            f"Assistant: <think>\n{reasoning}\n</think>\n"
            f"{answer}{eos}"
        )
    return f"User: {question}\n\nAssistant: {answer}{eos}"


def load_and_tokenize(
    dataset_name: str,
    tokenizer,
    max_samples: Optional[int],
    max_seq_len: int,
) -> List[Dict]:
    """
    Load the full dataset, format each row, tokenize, and filter.
    Returns a list of {'input_ids': Tensor[T]} dicts (variable length ≤ max_seq_len).
    """
    print(f"Loading {dataset_name} …")
    raw = load_dataset(dataset_name, split="train")
    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    eos = tokenizer.eos_token or "<|endoftext|>"
    samples = []
    skipped = 0

    for ex in tqdm(raw, desc="Formatting + tokenizing", leave=False):
        q, r, a = _extract_bespoke(ex)
        if not q or not a:
            skipped += 1
            continue
        text = _format_training_text(q, r, a, eos)
        ids  = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 4:
            skipped += 1
            continue
        samples.append({
            "input_ids": torch.tensor(ids[:max_seq_len], dtype=torch.long)
        })

    print(f"  {len(samples)} samples kept, {skipped} skipped (missing fields / too short).")
    return samples


# ── Collation ─────────────────────────────────────────────────────────────────

def collate(samples: List[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad a micro-batch to the longest example. Mask padding in labels."""
    max_len = max(s["input_ids"].size(0) for s in samples)
    B       = len(samples)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels    = torch.full((B, max_len), -100,   dtype=torch.long)
    mask      = torch.zeros(B, max_len, dtype=torch.bool)

    for i, s in enumerate(samples):
        ids = s["input_ids"]
        T   = ids.size(0)
        input_ids[i, :T] = ids
        labels[i, :T]    = ids     # labels = input shifted inside loss
        mask[i, :T]      = True

    return {"input_ids": input_ids, "attention_mask": mask, "labels": labels}


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    """
    Linear warmup then cosine decay to min_lr_ratio × base_lr.
    This is the standard schedule for all LLaMA-family training runs.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        if total_steps <= warmup_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ── EMA ───────────────────────────────────────────────────────────────────────

class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Shadow weights are used for inference and saved in every checkpoint.
    EMA provides ~0.5–1.0% accuracy gain on downstream evals vs live weights.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay  = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        one_minus = 1.0 - self.decay
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=one_minus)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.decay = float(sd.get("decay", self.decay))
        for name, tensor in sd.get("shadow", {}).items():
            if name in self.shadow and self.shadow[name].shape == tensor.shape:
                self.shadow[name].copy_(tensor)


# ── Generation callback ───────────────────────────────────────────────────────

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
    """
    Run greedy decoding on the fixed GEN_PROMPTS using EMA weights.
    Logs results to stdout and (optionally) wandb.

    This is the single most important monitoring tool for a language model.
    A falling CE score that never improves generation quality is a warning
    sign of overfitting or distribution shift.
    """
    # Temporarily copy EMA weights into the model for generation
    live_backup: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name in ema.shadow:
            live_backup[name] = param.data.clone()
            param.data.copy_(ema.shadow[name].to(dtype=param.data.dtype))

    model.eval()

    print(f"\n  ── Generation @ step {step} (EMA weights) ──")
    wandb_log: Dict[str, str] = {}
    mean_uwr = 0.0

    for prompt in GEN_PROMPTS:
        # Format the prompt the same way training data is formatted
        prefix = f"User: {prompt}\n\nAssistant: <think>\n"
        ids    = torch.tensor(
            tokenizer.encode(prefix, add_special_tokens=False),
            dtype=torch.long, device=device,
        ).unsqueeze(0)

        eos_id    = tokenizer.eos_token_id
        generated: List[int] = []

        for _ in range(max_new_tokens):
            if ids.size(1) > max_seq_len:
                ids = ids[:, -max_seq_len:]
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(ids)
            next_id = int(logits[:, -1, :].argmax(dim=-1).item())
            if eos_id is not None and next_id == eos_id:
                break
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

        output_text = tokenizer.decode(generated, skip_special_tokens=True)
        uwr = unique_word_ratio(output_text)
        mean_uwr += uwr

        # Truncate for display but log the full text to wandb
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

    # Restore live weights
    model.train()
    for name, param in model.named_parameters():
        if name in live_backup:
            param.data.copy_(live_backup[name])


# ── Validation CE ─────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_val_ce(
    model: BaselineTRMMamba,
    ema: ModelEMA,
    val_samples: List[Dict],
    pad_id: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    vocab_size: int,
) -> float:
    """
    Compute per-token cross-entropy on the validation split using EMA weights.
    Uses a fresh forward pass on each batch (no KV cache).
    """
    # Swap to EMA weights for eval
    for name, param in model.named_parameters():
        if name in ema.shadow:
            param.data.copy_(ema.shadow[name].to(dtype=param.data.dtype))

    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for start in range(0, len(val_samples), batch_size):
        batch_samples = val_samples[start : start + batch_size]
        batch         = collate(batch_samples, pad_id)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(input_ids, attention_mask=attn_mask)

        shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size).float()
        shift_labels = labels[:, 1:].contiguous().view(-1)
        valid        = (shift_labels != -100)

        if valid.any():
            loss = F.cross_entropy(
                shift_logits[valid], shift_labels[valid], reduction="sum"
            )
            total_loss   += loss.item()
            total_tokens += int(valid.sum().item())

    val_ce = total_loss / max(total_tokens, 1)
    model.train()
    return val_ce


# ── Checkpointing ─────────────────────────────────────────────────────────────

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
) -> Path:
    """
    Save checkpoint to output_dir/checkpoint-{step:07d}/.
    Always writes resolved_backbone_config.json alongside so the inference
    script can reconstruct the model without knowing the original CLI args.
    Automatically removes old checkpoints beyond keep_last.
    """
    ckpt_dir = output_dir / f"checkpoint-{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # The ema.shadow dict uses named_parameters() keys, which omit lm_head.weight
    # when tie_embeddings=True. Materialize the alias so inference scripts find it.
    shadow = dict(ema.shadow)
    if config.tie_embeddings:
        if "lm_head.weight" not in shadow and "token_embedding.weight" in shadow:
            shadow["lm_head.weight"] = shadow["token_embedding.weight"]

    state = {
        "step":                    step,
        "val_ce":                  val_ce,
        "model_state_dict":        model.state_dict(),
        "ema_backbone_state_dict": shadow,
        "optimizer":               optimizer.state_dict(),
        "scheduler":               scheduler.state_dict(),
        "scaler":                  scaler.state_dict() if scaler else None,
        "ema":                     ema.state_dict(),
        "backbone_config":         asdict(config),
    }
    torch.save(state, ckpt_dir / "training_state.pt")

    config_dict = asdict(config)
    with (ckpt_dir / "resolved_backbone_config.json").open("w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"  [ckpt] saved  → {ckpt_dir}")

    # Prune old checkpoints
    existing = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for old in existing[:-keep_last]:
        import shutil
        shutil.rmtree(old, ignore_errors=True)

    return ckpt_dir


def load_latest_checkpoint(
    output_dir: Path,
    model: BaselineTRMMamba,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    device: torch.device,
) -> int:
    """
    Load the most recent checkpoint. Returns the step to resume from (0 if none found).
    """
    candidates = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not candidates:
        return 0

    ckpt_dir = candidates[-1]
    state_path = ckpt_dir / "training_state.pt"
    if not state_path.exists():
        print(f"  [resume] {ckpt_dir} is incomplete, skipping.")
        return 0

    print(f"  [resume] loading {ckpt_dir}")
    state = torch.load(state_path, map_location=device)

    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if state.get("ema"):
        ema.load_state_dict(state["ema"])

    step = int(state.get("step", 0))
    val_ce = state.get("val_ce")
    print(f"  [resume] step={step}  val_ce={val_ce}")
    return step


# ── Main training loop ────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.cuda.reset_peak_memory_stats(device)

    # ── wandb ─────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_mode != "disabled":
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config=vars(args),
            )
        except ImportError:
            print("[warn] wandb not installed — logging to stdout only.")

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print("═" * 64)
    print("  Phase 2 SFT — Project Ouroboros")
    print("═" * 64)
    print(f"  preset        : {args.preset}")
    print(f"  seq_len       : {args.max_seq_len}")
    print(f"  batch×accum   : {args.batch_size}×{args.grad_accum}"
          f" = {args.batch_size * args.grad_accum}")
    print(f"  lr            : {args.lr}  warmup={args.warmup_steps}")
    print(f"  dtype         : {dtype}")
    print(f"  output_dir    : {output_dir}")
    print("═" * 64)
    print()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"  vocab: {len(tokenizer):,}  pad_token: '{tokenizer.pad_token}'")
    print()

    # ── Dataset ───────────────────────────────────────────────────────────────
    all_samples = load_and_tokenize(
        args.dataset_name, tokenizer, args.max_samples, args.max_seq_len
    )
    if not all_samples:
        sys.exit("No samples loaded. Check dataset connectivity and --max_samples.")

    # Deterministic train/val split — same split every run for comparability
    torch.manual_seed(args.seed)
    perm    = torch.randperm(len(all_samples)).tolist()
    n_val   = max(1, int(len(all_samples) * args.val_fraction))
    val_idx = set(perm[:n_val])

    train_samples = [s for i, s in enumerate(all_samples) if i not in val_idx]
    val_samples   = [s for i, s in enumerate(all_samples) if i in val_idx]
    pad_id        = tokenizer.pad_token_id or 0

    print(f"  train: {len(train_samples)}  val: {len(val_samples)}")
    print()

    # ── Model ─────────────────────────────────────────────────────────────────
    preset_kwargs = PRESETS[args.preset]
    padded_vocab  = pad_vocab(len(tokenizer), 128)

    config = BaselineConfig(
        vocab_size=padded_vocab,
        max_seq_len=args.max_seq_len,
        dropout=0.0,
        **preset_kwargs,
    )
    model = BaselineTRMMamba(config).to(device=device, dtype=dtype)
    model.train()

    n_params = count_parameters(model)
    print(f"Model  : {n_params / 1e6:.1f}M parameters  (preset={args.preset})")
    print(f"Config : d_model={config.d_model}  n_groups={config.n_groups}"
          f"  heads={config.n_heads}/{config.n_kv_heads}"
          f"  mlp_hidden={config.mlp_hidden}")
    print(f"Vocab  : {padded_vocab:,} (padded from {len(tokenizer):,})")
    print()

    # ── Optimizer & Schedule ──────────────────────────────────────────────────
    # Separate weight-decay from non-decay parameters (nanoGPT convention)
    decay_params    = [p for n, p in model.named_parameters()
                       if p.ndim >= 2 and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.ndim <  2 and p.requires_grad]

    adamw_base = dict(lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    try:
        optimizer = AdamW(
            [{"params": decay_params,    "weight_decay": args.weight_decay},
             {"params": no_decay_params, "weight_decay": 0.0}],
            fused=True, **adamw_base,
        )
        print("Optimizer : AdamW (fused CUDA kernel)")
    except TypeError:
        optimizer = AdamW(
            [{"params": decay_params,    "weight_decay": args.weight_decay},
             {"params": no_decay_params, "weight_decay": 0.0}],
            **adamw_base,
        )
        print("Optimizer : AdamW (standard)")

    steps_per_epoch = math.ceil(len(train_samples) / (args.batch_size * args.grad_accum))
    total_steps     = (
        args.max_steps if args.max_steps > 0
        else steps_per_epoch * args.num_epochs
    )

    scheduler = cosine_with_warmup(
        optimizer, args.warmup_steps, total_steps, args.min_lr_ratio
    )

    # fp16 only uses scaler; bf16 does not need it
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    ema = ModelEMA(model, decay=args.ema_decay)

    print(f"Schedule  : cosine warmup={args.warmup_steps} total={total_steps}"
          f"  ({args.num_epochs} epochs × {steps_per_epoch} steps/epoch)")
    print()

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume_from:
        output_dir_resume = Path(args.resume_from)
        start_step = load_latest_checkpoint(
            output_dir_resume, model, ema, optimizer, scheduler, scaler, device
        )
    else:
        start_step = load_latest_checkpoint(
            output_dir, model, ema, optimizer, scheduler, scaler, device
        )

    # ── Prepare data index ────────────────────────────────────────────────────
    # We cycle through training data indefinitely; epoch tracking is for logging.
    sample_idx = (start_step * args.batch_size * args.grad_accum) % max(len(train_samples), 1)
    torch.manual_seed(args.seed)
    perm_train = torch.randperm(len(train_samples)).tolist()

    # ── Training loop ──────────────────────────────────────────────────────────
    col_hdr = (
        f"{'Step':>7}  {'Train CE':>9}  {'Val CE':>9}  "
        f"{'GNorm':>7}  {'LR':>9}  {'VRAM':>7}  {'Tok/s':>7}"
    )
    print(col_hdr)
    print("─" * 72)

    pbar        = tqdm(total=total_steps, initial=start_step, desc="Training",
                       dynamic_ncols=True)
    step        = start_step
    micro_step  = 0
    accum_loss  = 0.0
    last_val_ce: Optional[float] = None
    t0          = time.perf_counter()
    tokens_seen = 0

    optimizer.zero_grad(set_to_none=True)

    while step < total_steps:
        # ── Micro-batch ───────────────────────────────────────────────────────
        micro_batch = []
        for _ in range(args.batch_size):
            if sample_idx >= len(train_samples):
                sample_idx  = 0
                perm_train  = torch.randperm(len(train_samples)).tolist()
            micro_batch.append(train_samples[perm_train[sample_idx]])
            sample_idx += 1

        batch     = collate(micro_batch, pad_id)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)

        tokens_seen += int(attn_mask.sum().item())

        # ── Forward ───────────────────────────────────────────────────────────
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(input_ids, attention_mask=attn_mask)

        # Standard single-step next-token CE — no deep supervision
        shift_logits = (
            logits[:, :-1, :].contiguous().view(-1, config.vocab_size).float()
        )
        shift_labels = labels[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        # ── Backward ──────────────────────────────────────────────────────────
        if dtype == torch.float16:
            scaler.scale(loss / args.grad_accum).backward()
        else:
            (loss / args.grad_accum).backward()

        accum_loss += loss.detach().item()
        micro_step += 1

        if micro_step % args.grad_accum != 0:
            continue

        # ── Optimizer step ────────────────────────────────────────────────────
        if dtype == torch.float16:
            scaler.unscale_(optimizer)

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        )

        if dtype == torch.float16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)

        step      += 1
        mean_ce    = accum_loss / args.grad_accum
        accum_loss = 0.0

        pbar.update(1)
        pbar.set_postfix(ce=f"{mean_ce:.3f}", gn=f"{grad_norm:.3f}")

        # ── Console log ───────────────────────────────────────────────────────
        if step % args.log_every == 0 or step == 1:
            elapsed  = max(time.perf_counter() - t0, 1e-6)
            tok_s    = tokens_seen / elapsed
            cur_lr   = scheduler.get_last_lr()[0]
            vram     = vram_gb(device)
            val_str  = f"{last_val_ce:.4f}" if last_val_ce is not None else "     -"

            tqdm.write(
                f"{step:>7}  {mean_ce:>9.4f}  {val_str:>9}  "
                f"{grad_norm:>7.4f}  {cur_lr:>9.2e}  {vram:>7.3f}  {tok_s:>7.0f}"
            )

            if wandb_run is not None:
                import wandb
                wandb.log({
                    "train/ce":        mean_ce,
                    "train/grad_norm": grad_norm,
                    "train/lr":        cur_lr,
                    "train/tok_s":     tok_s,
                    "train/vram_gb":   vram,
                }, step=step)

        # ── Validation CE ─────────────────────────────────────────────────────
        if step % args.val_every == 0 or step == total_steps:
            last_val_ce = compute_val_ce(
                model, ema, val_samples, pad_id, device, dtype,
                args.batch_size, config.vocab_size,
            )
            tqdm.write(f"  [val] step={step}  val_ce={last_val_ce:.4f}")

            if wandb_run is not None:
                import wandb
                wandb.log({"val/ce": last_val_ce}, step=step)

            # Phase 2 success criterion check
            if last_val_ce < 1.5:
                tqdm.write(
                    "  ★ val_ce < 1.5 — Phase 2 success criterion met. "
                    "Consider proceeding to Phase 3."
                )

        # ── Generation callback ───────────────────────────────────────────────
        if step % args.gen_every == 0 or step == total_steps:
            run_generation_callback(
                model, ema, tokenizer, device, dtype,
                step=step,
                max_new_tokens=args.gen_max_tokens,
                max_seq_len=args.max_seq_len,
                wandb_run=wandb_run,
            )

        # ── Checkpoint ───────────────────────────────────────────────────────
        if step % args.save_every == 0 or step == total_steps:
            save_checkpoint(
                output_dir, step, model, ema, optimizer,
                scheduler, scaler if dtype == torch.float16 else None,
                config, last_val_ce, args.keep_last,
            )

    pbar.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time   = time.perf_counter() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    print()
    print("═" * 64)
    print("  Phase 2 Training Complete")
    print("═" * 64)
    print(f"  Total steps  : {step}")
    print(f"  Total time   : {total_time / 60:.1f} min")
    print(f"  Peak VRAM    : {peak_vram_gb:.2f} GB")
    if last_val_ce is not None:
        print(f"  Final val CE : {last_val_ce:.4f}")
        if last_val_ce < 1.5:
            print("  Status       : SUCCESS — proceed to Phase 3 (incremental recursion)")
        else:
            print("  Status       : val_ce ≥ 1.5 — extend training or check data quality")
    print("═" * 64)

    if wandb_run is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
