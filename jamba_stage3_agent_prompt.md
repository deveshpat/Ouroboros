# Jamba Reasoning 3B — Coconut-Ouroboros Fine-tuning Agent Prompt
## Project Ouroboros / Stage 3 — `jamba_coconut_finetune.py`

> **Self-contained. Feed this entire file to a coding agent.**
> Generates one script: `jamba_coconut_finetune.py`
> Do NOT reference `baseline_trm_mamba.py`, `training_utils.py`, or any nano-specific file.
> Dataset is pre-processed by `prepare_coconut_dataset.py`. This script reads ONLY its output.

---

## Background & Design Rationale

### Model: Jamba Reasoning 3B, not Jamba2-3B

`ai21labs/AI21-Jamba-Reasoning-3B` (Oct 2025) is the correct model. It was trained with RLVR on math, code, and structured-output tasks, and already generates explicit `<think>` reasoning traces. Coconut's curriculum works by *replacing* those traces progressively with latent passes — the model already knows how to reason; we are teaching it to reason without tokens.

Jamba2-3B is enterprise-grounding-focused (RAG, IFBench) with no reasoning traces. Wrong model.

Architecture: 28 layers (26 Mamba, 2 Attention) — a **13:1 Mamba:Attention ratio**. More Mamba-heavy than Jamba 1.5, which is better for Coconut since more SSM state accumulates latent reasoning.

### Why the Coconut curriculum must match dataset step structure

```
Stage 0:  [Q][S1][S2][S3][A]   ← standard CoT fine-tune; labels on S1..S3 + A
Stage 1:  [Q][●][S2][S3][A]    ← S1 replaced by latent; labels on S2..S3 + A
Stage 2:  [Q][●][●][S3][A]     ← S1,S2 replaced; labels on S3 + A
Stage 3:  [Q][●][●][●][A]      ← all steps replaced; labels on A only
```

`●` = injected latent (last hidden state from sequential prefix pass).

The paper tested "pause tokens at question end" and found it **inferior** to step-replacement. K = median n_steps from stats.json.

### Stage advancement: epoch-based with best-checkpoint selection

Train for `--epochs_per_stage` epochs per curriculum stage. At stage end, select the checkpoint with the highest validation accuracy and load it before advancing to stage k+1.

### DGAC: Diversity-Gated Adaptive Coconut

Phase 3.4 only (after K-stage curriculum). The gate learns to halt early on easy problems.

---

## Part 0 — Critical Implementation Notes (READ BEFORE CODING)

These are hard lessons from prior training sessions that **must** be reflected in the generated script.

### 0.1 — `attn_implementation` must use try/except fallback

```python
# CORRECT — never hardcode flash_attention_2 without fallback
_attn_impl = "eager"
try:
    import flash_attn  # noqa
    _attn_impl = "flash_attention_2"
except ImportError:
    pass
load_kwargs["attn_implementation"] = _attn_impl
```

Hardcoding `"flash_attention_2"` crashes at model load on Colab T4 free tier and any environment where flash-attn is not pre-installed.

### 0.2 — `use_mamba_kernels=False` must be version-guarded

```python
import transformers as _tf
_tf_version = tuple(int(x) for x in _tf.__version__.split(".")[:2])
if _tf_version >= (4, 54):
    load_kwargs["use_mamba_kernels"] = False
# On older versions, the kwarg doesn't exist and raises TypeError
```

Required for disabling CUDA Mamba kernels on TPU/CPU-only paths. Breaks silently on transformers < 4.54.

### 0.3 — `inputs_embeds` must assert `last_hidden_state` is not None

```python
out = model.model(inputs_embeds=prefix_embeds, attention_mask=prefix_mask, use_cache=False)
assert out.last_hidden_state is not None, (
    "model.model() returned None for last_hidden_state. "
    "Add output_hidden_states=True and use out.hidden_states[-1] instead."
)
h_j = out.last_hidden_state[:, -1:, :]
```

If `last_hidden_state` is None (which can happen if the model's forward doesn't populate it by default), the Coconut mechanism injects zeros silently with no error. The assert must fire on the first forward pass.

### 0.4 — Graceful session timeout is MANDATORY for Kaggle Dual T4

Sessions 5–7 of Stage 2 were all killed by the NCCL watchdog because validation/generation took longer than the NCCL timeout. The script MUST implement:
- Wall-clock session budget tracking (`--session_timeout_hours`, `--graceful_exit_buffer_minutes`)
- Emergency checkpoint save when `elapsed + buffer >= budget`
- `os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"` set before DDP init
- `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` set at start
- `torch.cuda.empty_cache()` before every validation pass (Stage 2 OOM'd at first val)

### 0.5 — LoRA target modules: `conv1d` is intentionally excluded

```python
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",       # Attention (2 layers)
    "in_proj", "x_proj", "dt_proj", "out_proj",    # Mamba SSM (26 layers)
    # conv1d intentionally excluded: shape incompatible with standard LoRA rank decomposition
]
```

Do NOT add `conv1d` to this list. PEFT will silently create malformed adapters for conv layers.

### 0.6 — NCCL timeout must be raised before DDP init

```python
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    timeout=timedelta(minutes=60),  # default 30min killed S5–S7
)
```

---

## Task

Create `jamba_coconut_finetune.py` from scratch. Read the canonical dataset from `--data_dir` (output of `prepare_coconut_dataset.py`). Do not import or reference `baseline_trm_mamba.py`, `training_utils.py`, or `prepare_coconut_dataset.py`.

---

## Part 1 — File Header

```python
#!/usr/bin/env python3
"""
Stage 3 Coconut-Ouroboros Fine-tuning for Jamba Reasoning 3B
=============================================================
Implements Meta's Coconut progressive replacement curriculum + DGAC halt gate.

Curriculum (run in sequence, each resuming from previous best checkpoint):
  Stage 0:   standard CoT fine-tune; labels on all reasoning steps + answer
  Stage 1..K: replace first k reasoning steps with k latent passes;
              labels shift to supervise only the remaining (k+1..n) steps + answer

Phase 3.4 (--use_halt_gate): DGAC adaptive halt gate added on top of Stage K.

Dataset: read from --data_dir (output of prepare_coconut_dataset.py).
         Canonical format: {id, source, question, steps, answer_full, answer_norm, n_steps}
         Note: 'steps' column from Hub is JSON-encoded; local JSONL has native lists.

References:
  Coconut (Meta, arXiv:2412.06769)
  Jamba Reasoning 3B (AI21, ai21labs/AI21-Jamba-Reasoning-3B, Oct 2025)

Install:
  pip install "transformers>=4.54.0" peft datasets tqdm wandb bitsandbytes accelerate
  pip install flash-attn --no-build-isolation   # optional but recommended
  pip install causal-conv1d>=1.2.0 mamba-ssm    # optional; use_mamba_kernels=False if absent

Run (smoke test, Colab T4):
  python jamba_coconut_finetune.py \\
    --data_dir data/coconut_v1 --use_4bit \\
    --epochs_per_stage 1 --max_stage 2 --max_samples 200 \\
    --session_timeout_hours 1.5 --wandb_mode disabled --output_dir runs/smoke

Run (Phase 3.1 through 3.K, Kaggle Dual T4):
  torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \\
    --data_dir data/coconut_v1 --use_4bit \\
    --epochs_per_stage 3 --batch_size 2 --grad_accum 8 \\
    --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \\
    --output_dir runs/stage3_curriculum

Run (Phase 3.4, DGAC gate, from Stage K best checkpoint):
  python jamba_coconut_finetune.py \\
    --data_dir data/coconut_v1 --use_4bit \\
    --use_halt_gate --resume_from runs/stage3_curriculum/stage_K/best \\
    --epochs_per_stage 3 --output_dir runs/stage3_dgac
"""
```

---

## Part 2 — Imports and Constants

```python
import argparse
import json
import math
import os
import random
import re as _re
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set critical env vars BEFORE any torch/nccl import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import transformers as _tf
    _TF_VERSION = tuple(int(x) for x in _tf.__version__.split(".")[:2])
    from peft import LoraConfig, get_peft_model
    from tqdm.auto import tqdm
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\npip install 'transformers>=4.54.0' peft tqdm wandb bitsandbytes accelerate")

MODEL_ID = "ai21labs/AI21-Jamba-Reasoning-3B"

# conv1d intentionally excluded — shape incompatible with standard LoRA
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj", "x_proj", "dt_proj", "out_proj",
]

GEN_PROMPTS = [
    "What is 15 + 27?",
    "Write a Python function that returns the factorial of n.",
    "What is the capital of Japan?",
    "Explain what a neural network is in simple terms.",
    "Solve for x: 3x + 6 = 21.",
]

_LAST_NUM = _re.compile(r"[\d,]+(?:\.\d+)?")
```

---

## Part 3 — CLI Arguments

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jamba Reasoning 3B Coconut-Ouroboros fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--max_seq_len", type=int, default=512)

    # LoRA / QLoRA
    parser.add_argument("--use_4bit", action="store_true",
                        help="QLoRA (4-bit NF4). Requires CUDA + bitsandbytes.")
    parser.add_argument("--lora_r",       type=int,   default=32)
    parser.add_argument("--lora_alpha",   type=int,   default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Dataset
    parser.add_argument("--data_dir", default="data/coconut_v1")
    parser.add_argument("--max_samples", type=int, default=None)

    # Curriculum
    parser.add_argument("--max_stage", type=int, default=None,
                        help="Override K. None = read n_steps_median from stats.json.")
    parser.add_argument("--epochs_per_stage", type=int, default=3)
    parser.add_argument("--stage_0_epochs",   type=int, default=None)

    # DGAC halt gate (Phase 3.4)
    parser.add_argument("--use_halt_gate", action="store_true")
    parser.add_argument("--halt_threshold",           type=float, default=0.5)
    parser.add_argument("--dgac_lambda_ponder_max",   type=float, default=0.01)
    parser.add_argument("--dgac_lambda_diversity",    type=float, default=0.1)
    parser.add_argument("--dgac_tau",                 type=float, default=0.9)
    parser.add_argument("--dgac_warmup_steps",        type=int,   default=200)
    parser.add_argument("--dgac_ramp_steps",          type=int,   default=300)

    # Training
    parser.add_argument("--batch_size",   type=int,   default=2)
    parser.add_argument("--grad_accum",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int,   default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm",type=float, default=1.0)
    parser.add_argument("--grad_checkpoint", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    # Session timeout (MANDATORY for Kaggle)
    parser.add_argument("--session_timeout_hours",       type=float, default=11.0)
    parser.add_argument("--graceful_exit_buffer_minutes",type=float, default=20.0)

    # I/O
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--output_dir",  default="runs/stage3")
    parser.add_argument("--keep_checkpoints_per_stage", type=int, default=2)

    # Monitoring
    parser.add_argument("--log_every",        type=int, default=20)
    parser.add_argument("--val_batch_size",   type=int, default=1,
                        help="Batch size for val forward passes. Keep at 1 to avoid OOM.")
    parser.add_argument("--gen_every_stage",  action="store_true", default=True)
    parser.add_argument("--gen_max_tokens",   type=int, default=200)

    # wandb
    parser.add_argument("--wandb_project",  default="ouroboros-stage3-jamba")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_mode",
                        choices=["online", "offline", "disabled"], default="online")

    return parser.parse_args()
```

---

## Part 4 — Model Loading Helpers

```python
def _get_embed_tokens(model):
    """
    Robustly retrieve embed_tokens regardless of PEFT wrapping depth.
    Jamba Reasoning 3B: model.model.embed_tokens
    After PEFT wrap:    model.base_model.model.model.embed_tokens
    """
    for path_fn in [
        lambda m: m.model.embed_tokens,
        lambda m: m.base_model.model.model.embed_tokens,
        lambda m: m.base_model.model.embed_tokens,
    ]:
        try:
            e = path_fn(model)
            if e is not None:
                return e
        except AttributeError:
            continue
    raise AttributeError(
        "Cannot locate embed_tokens. Inspect:\n"
        "  print([n for n, _ in model.named_modules()][:40])"
    )


def _get_lm_head(model):
    for path_fn in [
        lambda m: m.lm_head,
        lambda m: m.base_model.model.lm_head,
    ]:
        try:
            h = path_fn(model)
            if h is not None:
                return h
        except AttributeError:
            continue
    raise AttributeError("Cannot locate lm_head. Inspect model.named_modules().")


def load_model_and_tokenizer(args, device):
    """
    Load Jamba Reasoning 3B with QLoRA (--use_4bit) or standard bfloat16 LoRA.

    Returns (model, tokenizer, d_model, lat_token_id).

    Key implementation notes (see Part 0 for rationale):
    - attn_implementation: tries flash_attention_2, falls back to 'eager'
    - use_mamba_kernels=False: only passed on transformers>=4.54.0
    - <|lat|> token: added if absent; embed_tokens resized accordingly
    """
    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    LAT_TOKEN = "<|lat|>"
    existing_id = tokenizer.convert_tokens_to_ids(LAT_TOKEN)
    if existing_id == tokenizer.unk_token_id or existing_id is None:
        tokenizer.add_special_tokens({"additional_special_tokens": [LAT_TOKEN]})
    lat_token_id = tokenizer.convert_tokens_to_ids(LAT_TOKEN)
    print(f"  <|lat|> token id: {lat_token_id}  vocab: {len(tokenizer)}")

    # Determine attn_implementation with fallback (see Part 0.1)
    _attn_impl = "eager"
    try:
        import flash_attn  # noqa
        _attn_impl = "flash_attention_2"
        print("  flash-attn available: using flash_attention_2")
    except ImportError:
        print("  flash-attn not installed: falling back to eager attention")

    print(f"Loading model: {args.model_id}")
    load_kwargs: Dict[str, Any] = dict(
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=_attn_impl,
    )
    # use_mamba_kernels only on transformers>=4.54.0 (see Part 0.2)
    if _TF_VERSION >= (4, 54):
        load_kwargs["use_mamba_kernels"] = False
    else:
        print(f"  [warn] transformers {_tf.__version__} < 4.54.0; "
              "use_mamba_kernels kwarg skipped (upgrade for reliable Mamba CPU fallback)")

    if args.use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)

    # Resize if <|lat|> is newly added (see Part 0.1)
    embed_size = model.model.embed_tokens.num_embeddings
    if len(tokenizer) > embed_size:
        print(f"  Resizing embed_tokens: {embed_size} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    if args.use_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.grad_checkpoint
        )
    elif args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Verify embed_tokens path works after PEFT wrap (early failure is better than silent zeros)
    _ = _get_embed_tokens(model)
    _ = _get_lm_head(model)
    print("  embed_tokens and lm_head paths verified after PEFT wrap.")

    d_model = model.config.hidden_size
    print(f"  d_model={d_model}  layers={model.config.num_hidden_layers}")
    return model, tokenizer, d_model, lat_token_id
```

---

## Part 5 — Dataset Loading

```python
def load_canonical_dataset(
    data_dir: Path,
    max_samples: Optional[int],
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Load train.jsonl, val.jsonl, and stats.json from prepare_coconut_dataset.py output.
    Each sample: {id, source, question, steps (list), answer_full, answer_norm, n_steps}
    """
    def _load_jsonl(path: Path) -> List[Dict]:
        out = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    # steps may be a JSON string if loaded from Hub backup
                    if isinstance(row.get("steps"), str):
                        row["steps"] = json.loads(row["steps"])
                    out.append(row)
        return out

    train_path = data_dir / "train.jsonl"
    val_path   = data_dir / "val.jsonl"
    stats_path = data_dir / "stats.json"

    if not train_path.exists():
        raise FileNotFoundError(
            f"train.jsonl not found at {train_path}. "
            "Run prepare_coconut_dataset.py first."
        )

    train = _load_jsonl(train_path)
    val   = _load_jsonl(val_path) if val_path.exists() else []
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    if max_samples is not None:
        n_val   = max(1, max_samples // 20)
        n_train = max_samples - n_val
        train, val = train[:n_train], val[:n_val]

    print(f"  Loaded {len(train)} train / {len(val)} val from {data_dir}")
    if stats:
        t = stats.get("train", {})
        print(f"  Step stats: median={t.get('n_steps_median')} "
              f"mean={t.get('n_steps_mean')} max={t.get('n_steps_max')}")
    return train, val, stats


def get_max_stage(args: argparse.Namespace, stats: Dict) -> int:
    if args.max_stage is not None:
        return args.max_stage
    median = stats.get("train", {}).get("n_steps_median")
    if median is not None:
        print(f"  --max_stage not set; using n_steps_median={median} from stats.json")
        return int(median)
    print("  [warn] --max_stage not set and stats.json absent; defaulting to 4")
    return 4
```

---

## Part 6 — Sample Building (Stage-Aware)

```python
def build_sample_at_stage(
    tokenizer,
    sample: Dict[str, Any],
    stage_k: int,
    lat_token_id: int,
    max_seq_len: int,
) -> Optional[Dict[str, Any]]:
    """
    Build a tokenized Coconut sample for curriculum stage k.

    Sequence layout (stage k > 0):
        [Q_ids] [lat_id * k] [S_{k+1}_ids ... S_n_ids] [answer_ids + eos]

    Labels:
        -100  for Q positions and k latent positions
        token ids  for S_{k+1..n} and answer (supervised)

    Stage 0 (k=0): no latent slots; labels on ALL steps + answer.
    Truncation: supervised tail is truncated if total > max_seq_len.
    Returns None if < 4 supervised tokens remain after truncation.
    """
    eos = tokenizer.eos_token or "<|endoftext|>"
    messages = [{"role": "user", "content": sample["question"]}]
    prefix_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    q_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    steps = sample["steps"]
    remaining_steps = steps[stage_k:]

    supervised_ids: List[int] = []
    for s in remaining_steps:
        supervised_ids.extend(tokenizer.encode(s + "\n", add_special_tokens=False))
    supervised_ids.extend(
        tokenizer.encode(sample["answer_full"] + eos, add_special_tokens=False)
    )

    if not supervised_ids:
        return None

    total = len(q_ids) + stage_k + len(supervised_ids)
    if total > max_seq_len:
        allowed = max_seq_len - len(q_ids) - stage_k
        if allowed < 4:
            return None
        supervised_ids = supervised_ids[:allowed]

    full_ids = q_ids + [lat_token_id] * stage_k + supervised_ids
    labels   = [-100] * len(q_ids) + [-100] * stage_k + supervised_ids
    assert len(full_ids) == len(labels)

    return {
        "full_ids":    torch.tensor(full_ids, dtype=torch.long),
        "labels":      torch.tensor(labels,   dtype=torch.long),
        "q_len":       len(q_ids),
        "n_latent":    stage_k,
        "answer_norm": sample.get("answer_norm", ""),
    }


def collate_stage_k(samples: List[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad a micro-batch to the longest example."""
    max_len = max(s["full_ids"].size(0) for s in samples)
    B = len(samples)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels    = torch.full((B, max_len), -100,   dtype=torch.long)
    attn_mask = torch.zeros(B, max_len,           dtype=torch.bool)
    q_lens    = torch.zeros(B,                    dtype=torch.long)
    n_latents = torch.zeros(B,                    dtype=torch.long)
    for i, s in enumerate(samples):
        T = s["full_ids"].size(0)
        input_ids[i, :T] = s["full_ids"]
        labels[i, :T]    = s["labels"]
        attn_mask[i, :T] = True
        q_lens[i]        = s["q_len"]
        n_latents[i]     = s["n_latent"]
    return {
        "input_ids": input_ids, "attention_mask": attn_mask,
        "labels": labels, "q_lens": q_lens, "n_latents": n_latents,
    }
```

---

## Part 7 — DGAC Halt Gate

```python
class HaltGate(nn.Module):
    """
    Halt gate for DGAC. Zero-initialized: outputs ~0.5 at start of Phase 3.4.
    Input: h_curr [B, D] + h_prev [B, D] at question-end position.
    Output: halt_prob [B].
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h_curr: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            self.gate(torch.cat([h_curr, h_prev], dim=-1))
        ).squeeze(-1)


def compute_dgac_lambda1(step: int, warmup: int, ramp: int, lmax: float) -> float:
    """λ₁=0 for warmup steps, then linearly ramps to lmax over ramp steps."""
    if step < warmup:
        return 0.0
    return lmax * min((step - warmup) / max(ramp, 1), 1.0)
```

---

## Part 8 — Coconut Forward Pass

```python
def coconut_forward(
    model,
    batch: Dict[str, torch.Tensor],
    stage_k: int,
    device: torch.device,
    halt_gate: Optional[HaltGate],
    args: argparse.Namespace,
    step_in_phase: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Coconut forward pass for curriculum stage k.

    Stage 0 (k=0): standard forward pass. No prefix passes, no latent injection.

    Stage k > 0:
      Phase A: embed all tokens.
      Phase B: k sequential prefix passes.
               Pass j: prefix = Q + latents[0..j-1].
               Extract h_j = last hidden state at prefix end.
               CRITICAL: assert last_hidden_state is not None (see Part 0.3).
               Patch embedding at latent slot j with h_j.
      Phase C: full forward over patched sequence. CE on supervised positions.
      Phase D: DGAC gate loss if --use_halt_gate.

    All passes use use_cache=False. Caching causes shape mismatches with patched embeddings.
    """
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    labels    = batch["labels"].to(device)
    q_lens    = batch["q_lens"].to(device)
    B, T      = input_ids.shape
    metrics: Dict[str, float] = {}

    embed_fn   = _get_embed_tokens(model)
    lm_head_fn = _get_lm_head(model)

    # ── Stage 0: efficient standard forward ─────────────────────────────────
    if stage_k == 0:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            out    = model.model(input_ids=input_ids, attention_mask=attn_mask,
                                 use_cache=False)
            assert out.last_hidden_state is not None, (
                "model.model() returned None for last_hidden_state at stage 0. "
                "Add output_hidden_states=True and use out.hidden_states[-1]."
            )
            logits = lm_head_fn(out.last_hidden_state).float()
        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[:, 1:].contiguous().view(-1)
        valid = shift_labels != -100
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True), {"ce": 0.0}
        ce = F.cross_entropy(shift_logits[valid], shift_labels[valid])
        return ce, {"ce": ce.item()}

    # ── Stage k > 0: latent injection ───────────────────────────────────────
    q_len = int(q_lens[0].item())

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=(device.type == "cuda")):
        all_embeds = embed_fn(input_ids)   # [B, T, D]

    patched = all_embeds.clone()
    hidden_at_q_end: List[torch.Tensor] = []

    # Phase B: k sequential prefix passes
    _first_pass = True
    for j in range(stage_k):
        prefix_len    = q_len + j
        prefix_embeds = patched[:, :prefix_len, :]
        prefix_mask   = attn_mask[:, :prefix_len]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            out = model.model(inputs_embeds=prefix_embeds,
                              attention_mask=prefix_mask, use_cache=False)

        # Critical: verify last_hidden_state populated (see Part 0.3)
        if _first_pass:
            assert out.last_hidden_state is not None, (
                "model.model(inputs_embeds=...) returned None for last_hidden_state. "
                "Pass output_hidden_states=True and use out.hidden_states[-1] instead."
            )
            _first_pass = False

        h_j = out.last_hidden_state[:, -1:, :]  # [B, 1, D]
        if halt_gate is not None:
            hidden_at_q_end.append(h_j.squeeze(1))

        inject_pos = q_len + j
        if inject_pos < T:
            patched = torch.cat(
                [patched[:, :inject_pos, :], h_j, patched[:, inject_pos + 1:, :]],
                dim=1,
            )

    # Phase C: full forward
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=(device.type == "cuda")):
        out    = model.model(inputs_embeds=patched, attention_mask=attn_mask,
                             use_cache=False)
        logits = lm_head_fn(out.last_hidden_state).float()

    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1)
    valid = shift_labels != -100
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True), {"ce": 0.0}
    ce = F.cross_entropy(shift_logits[valid], shift_labels[valid])
    metrics["ce"] = ce.item()

    # Phase D: DGAC gate loss (Phase 3.4 only)
    if halt_gate is None or len(hidden_at_q_end) < 2:
        return ce, metrics

    lam1     = compute_dgac_lambda1(
        step_in_phase, args.dgac_warmup_steps, args.dgac_ramp_steps,
        args.dgac_lambda_ponder_max,
    )
    ponder    = torch.zeros(B, device=device)
    div_loss  = torch.zeros(B, device=device)
    remainder = torch.ones(B,  device=device)
    halt_steps = torch.zeros(B, device=device)

    for k in range(1, len(hidden_at_q_end)):
        h_c = hidden_at_q_end[k].to(dtype=torch.float32)
        h_p = hidden_at_q_end[k - 1].to(dtype=torch.float32)
        hp  = halt_gate(h_c, h_p)
        ponder = ponder + remainder
        if k < len(hidden_at_q_end) - 1:
            remainder = remainder * (1.0 - hp.detach())
        div_loss = div_loss + F.relu(
            F.cosine_similarity(h_c, h_p, dim=-1) - args.dgac_tau
        )
        with torch.no_grad():
            halted = (hp > args.halt_threshold) & (halt_steps == 0)
            halt_steps = torch.where(
                halted, torch.full_like(halt_steps, float(k)), halt_steps
            )

    halt_steps = torch.where(
        halt_steps == 0, torch.full_like(halt_steps, float(stage_k)), halt_steps
    )
    total_loss = (
        ce
        + lam1 * ponder.mean()
        + args.dgac_lambda_diversity * div_loss.mean()
    )
    metrics.update({
        "ponder":         ponder.mean().item(),
        "diversity":      div_loss.mean().item(),
        "halt_step_mean": halt_steps.mean().item(),
        "lambda1":        lam1,
    })
    return total_loss, metrics
```

---

## Part 9 — Accuracy Evaluation

```python
def normalize_pred(text: str) -> str:
    m = _re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = _re.search(r"(?:answer is|=)\s*\**\s*([\d,\.\-]+)", text, _re.IGNORECASE)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = _LAST_NUM.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    return text.strip()[-40:]


@torch.no_grad()
def evaluate_stage(
    model, val_samples, tokenizer, lat_token_id, stage_k, device, args,
    halt_gate=None,
) -> Tuple[float, float]:
    """
    Compute val CE and exact-match accuracy at stage k using EMA-free live weights.
    torch.cuda.empty_cache() called before to avoid OOM (lesson from Stage 2 S8).
    Returns (val_ce, val_acc). Accuracy is primary for best-ckpt selection.
    """
    torch.cuda.empty_cache()   # MANDATORY — see Part 0.4
    model.eval()
    pad_id     = tokenizer.pad_token_id or 0
    ce_numer, ce_denom = 0.0, 0
    n_correct, n_total = 0, 0
    embed_fn   = _get_embed_tokens(model)
    lm_head_fn = _get_lm_head(model)
    batch_size = args.val_batch_size  # default 1 to avoid OOM

    # CE pass
    for start in range(0, len(val_samples), batch_size):
        batch_raw = val_samples[start : start + batch_size]
        built = [
            build_sample_at_stage(tokenizer, s, stage_k, lat_token_id, args.max_seq_len)
            for s in batch_raw
        ]
        built = [b for b in built if b is not None]
        if not built:
            continue
        batch = collate_stage_k(built, pad_id)
        loss, _ = coconut_forward(model, batch, stage_k, device,
                                  halt_gate=None, args=args, step_in_phase=999999)
        n_valid = int((batch["labels"][:, 1:].contiguous().view(-1) != -100).sum().item())
        ce_numer += loss.item() * n_valid
        ce_denom += n_valid

    # Accuracy pass (cap at 200 samples for speed)
    for sample in val_samples[:200]:
        built = build_sample_at_stage(tokenizer, sample, stage_k, lat_token_id, args.max_seq_len)
        if built is None:
            continue
        q_ids    = built["full_ids"][:built["q_len"]]
        q_tensor = q_ids.unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            ctx    = embed_fn(q_tensor)
            h_prev = None
            for j in range(stage_k):
                out    = model.model(inputs_embeds=ctx, use_cache=False)
                h_curr = out.last_hidden_state[:, -1:, :]
                if halt_gate is not None and h_prev is not None:
                    hp = halt_gate(h_curr.squeeze(1), h_prev.squeeze(1)).item()
                    if hp > args.halt_threshold:
                        break
                ctx    = torch.cat([ctx, h_curr], dim=1)
                h_prev = h_curr

        generated = []
        eos_id = tokenizer.eos_token_id
        for _ in range(args.gen_max_tokens):
            if ctx.size(1) > args.max_seq_len:
                ctx = ctx[:, -args.max_seq_len:, :]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                out   = model.model(inputs_embeds=ctx, use_cache=False)
                logit = lm_head_fn(out.last_hidden_state)
            nid = int(logit[:, -1, :].argmax(-1).item())
            if eos_id is not None and nid == eos_id:
                break
            generated.append(nid)
            ctx = torch.cat(
                [ctx, embed_fn(torch.tensor([[nid]], device=device))], dim=1
            )

        pred = normalize_pred(tokenizer.decode(generated, skip_special_tokens=True))
        if pred and sample.get("answer_norm") and pred == sample["answer_norm"]:
            n_correct += 1
        n_total += 1

    model.train()
    return ce_numer / max(ce_denom, 1), n_correct / max(n_total, 1)
```

---

## Part 10 — Checkpointing

```python
def save_checkpoint(
    output_dir: Path, step: int, epoch: int, stage_k: int,
    model, halt_gate: Optional[HaltGate], optimizer: AdamW, scheduler: LambdaLR,
    args: argparse.Namespace, val_ce: Optional[float], val_acc: Optional[float],
    tag: str = "",
) -> Path:
    """Save to output_dir/stage_{k}/{tag or checkpoint-{step}}/"""
    stage_dir = output_dir / f"stage_{stage_k}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    name = "best" if tag == "best" else f"checkpoint-{step:07d}"
    ckpt = stage_dir / name
    tmp  = stage_dir / f"{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()

    raw = model.module if hasattr(model, "module") else model
    raw.save_pretrained(str(tmp / "adapter_model"))
    if halt_gate is not None:
        torch.save(halt_gate.state_dict(), tmp / "halt_gate.pt")
    torch.save({
        "stage_k":      stage_k,
        "step":         step,
        "epoch":        epoch,
        "val_ce":       val_ce,
        "val_acc":      val_acc,
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
        "use_halt_gate":args.use_halt_gate,
    }, tmp / "training_state.pt")

    if ckpt.exists():
        shutil.rmtree(ckpt, ignore_errors=True)
    tmp.replace(ckpt)
    label = "best " if tag == "best" else "saved"
    print(f"  [ckpt] {label}-> {ckpt}  acc={val_acc}  ce={val_ce}")
    return ckpt


def load_checkpoint(
    ckpt_dir: Path, model, halt_gate, optimizer, scheduler, device, verbose=True
) -> Tuple[int, int, int]:
    """Load checkpoint. Returns (step, epoch, stage_k)."""
    state = torch.load(ckpt_dir / "training_state.pt", map_location=device)
    adapter_dir = ckpt_dir / "adapter_model"
    if adapter_dir.exists():
        from peft import set_peft_model_state_dict
        for fname in ["adapter_model.safetensors", "adapter_model.bin"]:
            fpath = adapter_dir / fname
            if fpath.exists():
                if fname.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    weights = load_file(str(fpath))
                else:
                    weights = torch.load(fpath, map_location=device)
                set_peft_model_state_dict(model, weights)
                break
    hgp = ckpt_dir / "halt_gate.pt"
    if halt_gate is not None and hgp.exists():
        halt_gate.load_state_dict(torch.load(hgp, map_location=device))
    if "optimizer" in state:
        try:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
        except Exception as exc:
            if verbose:
                print(f"  [resume] optimizer/scheduler mismatch ({exc}); resetting.")
    s = int(state.get("step", 0))
    e = int(state.get("epoch", 0))
    k = int(state.get("stage_k", 0))
    if verbose:
        print(f"  [resume] step={s}  epoch={e}  stage_k={k}  val_acc={state.get('val_acc')}")
    return s, e, k


def prune_epoch_checkpoints(stage_dir: Path, keep: int) -> None:
    ckpts = sorted(
        [p for p in stage_dir.iterdir()
         if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    for old in ckpts[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
```

---

## Part 11 — Optimizer and Scheduler

```python
def build_optimizer_and_scheduler(
    model, halt_gate, args, total_steps
) -> Tuple[AdamW, LambdaLR]:
    trainable = [p for p in model.parameters() if p.requires_grad]
    if halt_gate is not None:
        trainable.extend(halt_gate.parameters())
    decay    = [p for p in trainable if p.ndim >= 2]
    no_decay = [p for p in trainable if p.ndim < 2]
    optimizer = AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )
    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
    return optimizer, LambdaLR(optimizer, lr_lambda)
```

---

## Part 12 — Session Timeout Helper

```python
def make_timeout_checker(args: argparse.Namespace, rank: int):
    """
    Returns a callable that returns True when the session is within the
    graceful exit buffer. Only rank 0 checks wall time; other ranks follow
    via broadcast in the main loop.

    Mandatory for Kaggle Dual T4 sessions — Stage 2 S5–S7 were all killed
    by the NCCL watchdog because training ran past session limits without saving.
    """
    session_start    = time.perf_counter()
    timeout_limit_s  = args.session_timeout_hours * 3600.0
    timeout_buffer_s = args.graceful_exit_buffer_minutes * 60.0
    triggered        = [False]

    def check() -> bool:
        if triggered[0]:
            return True
        if rank != 0:
            return False
        elapsed = time.perf_counter() - session_start
        if elapsed + timeout_buffer_s >= timeout_limit_s:
            remaining = (timeout_limit_s - elapsed) / 60.0
            print(
                f"\n  [timeout] {elapsed / 3600:.2f}h elapsed — "
                f"{remaining:.1f} min remaining "
                f"(< {args.graceful_exit_buffer_minutes:.0f} min buffer)."
            )
            triggered[0] = True
        return triggered[0]

    return check
```

---

## Part 13 — Generation Callback

```python
@torch.no_grad()
def run_generation_callback(
    model, tokenizer, halt_gate, stage_k, device, args, step, wandb_run=None
) -> float:
    model.eval()
    print(f"\n  -- Generation @ step {step} stage={stage_k} --")
    embed_fn, lm_head_fn = _get_embed_tokens(model), _get_lm_head(model)
    mean_uwr = 0.0
    for prompt in GEN_PROMPTS:
        msgs   = [{"role": "user", "content": prompt}]
        prefix = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        q_ids  = tokenizer.encode(prefix, add_special_tokens=False)
        ctx    = embed_fn(torch.tensor(q_ids, device=device).unsqueeze(0))
        h_prev = None
        actual_k = 0
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            for j in range(stage_k):
                out    = model.model(inputs_embeds=ctx, use_cache=False)
                h_curr = out.last_hidden_state[:, -1:, :]
                if halt_gate is not None and h_prev is not None:
                    if halt_gate(h_curr.squeeze(1), h_prev.squeeze(1)).item() > args.halt_threshold:
                        break
                ctx    = torch.cat([ctx, h_curr], dim=1)
                h_prev = h_curr
                actual_k += 1

        generated = []
        eos_id = tokenizer.eos_token_id
        for _ in range(args.gen_max_tokens):
            if ctx.size(1) > args.max_seq_len:
                ctx = ctx[:, -args.max_seq_len:, :]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                logit = lm_head_fn(model.model(inputs_embeds=ctx, use_cache=False).last_hidden_state)
            nid = int(logit[:, -1, :].argmax(-1).item())
            if eos_id is not None and nid == eos_id:
                break
            generated.append(nid)
            ctx = torch.cat([ctx, embed_fn(torch.tensor([[nid]], device=device))], dim=1)

        text = tokenizer.decode(generated, skip_special_tokens=True)
        uwr  = len(set(text.split())) / max(len(text.split()), 1)
        mean_uwr += uwr
        display = text[:200].replace("\n", " ")
        print(f"  Q: {prompt}")
        print(f"  A: {display}  [k_actual={actual_k} uwr={uwr:.3f}]")

    mean_uwr /= max(len(GEN_PROMPTS), 1)
    print(f"  Mean UWR: {mean_uwr:.3f}\n")
    if wandb_run:
        import wandb
        wandb.log({"gen/mean_uwr": mean_uwr, "gen/stage": stage_k}, step=step)
    model.train()
    return mean_uwr
```

---

## Part 14 — Main

```python
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rank        = int(os.environ.get("RANK", "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank  = int(os.environ.get("LOCAL_RANK", str(rank)))
    distributed = world_size > 1
    is_main     = rank == 0

    if distributed:
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=60),  # raised from default 30 min (S5–S7 lesson)
        )

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if is_main and args.wandb_mode != "disabled":
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project, name=args.wandb_run_name,
                mode=args.wandb_mode, config=vars(args),
            )
        except ImportError:
            print("[warn] wandb not installed")

    train_samples, val_samples, stats = load_canonical_dataset(
        Path(args.data_dir), args.max_samples
    )
    max_stage = get_max_stage(args, stats)

    model, tokenizer, d_model, lat_token_id = load_model_and_tokenizer(args, device)
    pad_id = tokenizer.pad_token_id or 0

    halt_gate = None
    if args.use_halt_gate:
        halt_gate = HaltGate(d_model).to(device=device, dtype=torch.float32)
        print(f"  DGAC HaltGate: d_model={d_model}  "
              f"params={sum(p.numel() for p in halt_gate.parameters())}")

    # Determine starting stage
    start_stage = 0
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if (ckpt_path / "training_state.pt").exists():
            tmp_o, tmp_s = build_optimizer_and_scheduler(model, halt_gate, args, 1)
            _, _, loaded_k = load_checkpoint(ckpt_path, model, halt_gate, tmp_o, tmp_s, device)
            start_stage = loaded_k if args.use_halt_gate else loaded_k + 1
        else:
            print(f"  [warn] resume_from {ckpt_path} not found; starting from scratch.")

    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, find_unused_parameters=True)

    stages = [max_stage] if args.use_halt_gate else list(range(start_stage, max_stage + 1))
    global_step   = 0
    step_in_phase = 0

    # Session timeout checker (MANDATORY — see Part 0.4)
    check_timeout = make_timeout_checker(args, rank)

    for stage_k in stages:
        n_epochs          = (args.stage_0_epochs or args.epochs_per_stage) if stage_k == 0 \
                            else args.epochs_per_stage
        steps_per_epoch   = max(1, math.ceil(
            len(train_samples) / max(args.batch_size * args.grad_accum, 1)
        ))
        total_stage_steps = n_epochs * steps_per_epoch
        raw_model_ref     = model.module if distributed else model
        optimizer, scheduler = build_optimizer_and_scheduler(
            raw_model_ref, halt_gate, args, total_stage_steps
        )

        if is_main:
            print()
            print("=" * 64)
            print(f"  Stage {stage_k}/{max_stage}  "
                  f"{'(CoT warmup)' if stage_k == 0 else f'{stage_k} latent pass(es)'}"
                  f"{'  + DGAC' if args.use_halt_gate else ''}")
            print(f"  Epochs: {n_epochs}  Steps/epoch: {steps_per_epoch}  "
                  f"Total: {total_stage_steps}")
            print("=" * 64)

        best_val_acc, best_ckpt = -1.0, None
        timeout_triggered = False

        for epoch in range(n_epochs):
            perm = list(range(len(train_samples)))
            random.shuffle(perm)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            local_bs = args.batch_size // world_size if distributed else args.batch_size
            pbar = tqdm(total=steps_per_epoch, desc=f"S{stage_k}E{epoch}",
                        dynamic_ncols=True) if is_main else None

            for step_idx in range(steps_per_epoch):
                # ── Timeout check ────────────────────────────────────────────
                if check_timeout():
                    timeout_triggered = True
                if distributed:
                    import torch.distributed as dist
                    t_tensor = torch.tensor(
                        [int(timeout_triggered)], device=device, dtype=torch.int32
                    )
                    dist.broadcast(t_tensor, src=0)
                    timeout_triggered = bool(t_tensor.item())
                if timeout_triggered:
                    if is_main:
                        print(f"  [timeout] saving emergency checkpoint at step {global_step} ...")
                        save_checkpoint(
                            output_dir, global_step, epoch, stage_k,
                            raw_model_ref, halt_gate, optimizer, scheduler, args,
                            None, best_val_acc,
                        )
                    break

                accum_loss, accum_ce = 0.0, 0.0

                for micro in range(args.grad_accum):
                    idx       = (step_idx * args.grad_accum + micro) % len(train_samples)
                    batch_raw = [
                        train_samples[perm[(idx + j) % len(train_samples)]]
                        for j in range(local_bs)
                    ]
                    built = [
                        build_sample_at_stage(
                            tokenizer, s, stage_k, lat_token_id, args.max_seq_len
                        )
                        for s in batch_raw
                    ]
                    built = [b for b in built if b is not None]
                    if not built:
                        continue
                    batch = collate_stage_k(built, pad_id)
                    raw_m = model.module if distributed else model
                    loss, mets = coconut_forward(
                        raw_m, batch, stage_k, device, halt_gate, args, step_in_phase
                    )
                    (loss / args.grad_accum).backward()
                    accum_loss += loss.item()
                    accum_ce   += mets.get("ce", loss.item())

                gnorm = float(torch.nn.utils.clip_grad_norm_(
                    list(model.parameters())
                    + (list(halt_gate.parameters()) if halt_gate else []),
                    args.max_grad_norm,
                ))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step   += 1
                step_in_phase += 1
                mean_ce = accum_ce / args.grad_accum

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(ce=f"{mean_ce:.3f}", gn=f"{gnorm:.3f}")

                if is_main and global_step % args.log_every == 0:
                    log = {
                        "train/ce":        mean_ce,
                        "train/grad_norm": gnorm,
                        "train/lr":        scheduler.get_last_lr()[0],
                        "train/stage":     stage_k,
                        **{f"train/{k}": v for k, v in mets.items()},
                    }
                    tqdm.write(
                        f"  step={global_step:6d} s={stage_k} ep={epoch} "
                        f"ce={mean_ce:.4f} gn={gnorm:.4f}"
                    )
                    if wandb_run:
                        import wandb; wandb.log(log, step=global_step)

            if pbar:
                pbar.close()

            if timeout_triggered:
                break

            # ── End-of-epoch validation ──────────────────────────────────────
            if is_main:
                raw_m = model.module if distributed else model
                val_ce, val_acc = evaluate_stage(
                    raw_m, val_samples, tokenizer, lat_token_id,
                    stage_k, device, args, halt_gate,
                )
                tqdm.write(
                    f"  [val] s={stage_k} ep={epoch} "
                    f"val_ce={val_ce:.4f} val_acc={val_acc:.4f}"
                )
                if wandb_run:
                    import wandb
                    wandb.log({"val/ce": val_ce, "val/acc": val_acc,
                               "val/stage": stage_k}, step=global_step)

                save_checkpoint(
                    output_dir, global_step, epoch, stage_k,
                    raw_m, halt_gate, optimizer, scheduler, args, val_ce, val_acc,
                )
                prune_epoch_checkpoints(
                    output_dir / f"stage_{stage_k}", args.keep_checkpoints_per_stage
                )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_ckpt = save_checkpoint(
                        output_dir, global_step, epoch, stage_k,
                        raw_m, halt_gate, optimizer, scheduler, args,
                        val_ce, val_acc, tag="best",
                    )
                    tqdm.write(f"  [best] stage={stage_k} new best acc={best_val_acc:.4f}")

            if distributed:
                import torch.distributed as dist; dist.barrier()

        if timeout_triggered:
            break

        # ── Generation callback at end of stage ─────────────────────────────
        if is_main and args.gen_every_stage:
            raw_m = model.module if distributed else model
            run_generation_callback(
                raw_m, tokenizer, halt_gate, stage_k, device, args,
                global_step, wandb_run,
            )

        # ── Load best checkpoint before advancing to next stage ──────────────
        if best_ckpt is not None and not args.use_halt_gate:
            if is_main:
                print(f"  [stage] Stage {stage_k} done. Best acc={best_val_acc:.4f}. "
                      "Loading best ckpt before advancing.")
            raw_m = model.module if distributed else model
            tmp_o, tmp_s = build_optimizer_and_scheduler(raw_m, halt_gate, args, 1)
            load_checkpoint(best_ckpt, raw_m, halt_gate, tmp_o, tmp_s, device,
                            verbose=is_main)

        if distributed:
            import torch.distributed as dist; dist.barrier()

    # ── Final summary ────────────────────────────────────────────────────────
    if is_main:
        print("\n" + "=" * 64)
        if timeout_triggered:
            print("  [timeout] Session budget exhausted — checkpoint saved.")
            print("  Re-run with same --output_dir to resume from last checkpoint.")
        else:
            print(f"  Curriculum complete. Stages: {stages}  Global steps: {global_step}")
            if not args.use_halt_gate:
                best_k_dir = output_dir / f"stage_{max_stage}" / "best"
                print(f"  Phase 3.4 (DGAC):\n"
                      f"    python jamba_coconut_finetune.py --use_halt_gate "
                      f"--resume_from {best_k_dir} "
                      f"--output_dir {args.output_dir}_dgac [...]")
        print("=" * 64)

    if distributed:
        import torch.distributed as dist; dist.destroy_process_group()
    if wandb_run is not None:
        import wandb; wandb.finish()


if __name__ == "__main__":
    main()
```

---

## Part 15 — Verification Checklist

### Smoke test (Colab T4, ~10 min):
```bash
python jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit \
  --epochs_per_stage 1 --max_stage 2 --max_samples 200 \
  --batch_size 1 --grad_accum 2 \
  --session_timeout_hours 1.5 --val_batch_size 1 \
  --wandb_mode disabled --output_dir runs/smoke
```

**Must pass before Kaggle run:**
- [ ] No import errors; trainable parameters printed by PEFT
- [ ] `attn_implementation` log shows either flash_attention_2 or eager (not crash)
- [ ] `use_mamba_kernels` log: present on TF>=4.54, skipped warning on older
- [ ] `<|lat|>` token added; embed_tokens resized if needed
- [ ] `embed_tokens and lm_head paths verified` printed after PEFT wrap
- [ ] Stage 0 forward: `last_hidden_state is not None` assert passes; loss finite
- [ ] Stage 1 forward: assert passes on first prefix pass; CE finite
- [ ] Stage 2 forward: two prefix passes visible; no shape errors
- [ ] `evaluate_stage`: `torch.cuda.empty_cache()` called; no OOM; val_acc logged
- [ ] `stage_0/best/` created with `adapter_model/` + `training_state.pt`
- [ ] Stage advancement loads best_ckpt before Stage 1 begins (verify log line)
- [ ] Timeout: set `--session_timeout_hours 0.005` and verify emergency checkpoint fires
- [ ] Resume: `--resume_from runs/smoke/stage_1/best` starts at stage 2 without error

### Full curriculum (Kaggle Dual T4):
- [ ] Stage 0 val_acc > 0.30 within 1 epoch
- [ ] Each stage: val_acc within 5% of previous stage best
- [ ] VRAM stable across stages (no retained graphs from prefix passes)
- [ ] `halt_step_mean` only logged during Phase 3.4 (not in stages 0..K)
- [ ] Timeout saves checkpoint cleanly; re-run resumes from correct stage

---

## Part 16 — Known Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `last_hidden_state` is None for Jamba base model | Low | `assert` in Phase B fires immediately on first pass; clear error message |
| `embed_tokens` path wrong after PEFT wrap | High | `_get_embed_tokens()` tries 3 paths; verification call after PEFT wrap |
| `adapter_model.bin` vs `.safetensors` (PEFT version) | Medium | `load_checkpoint` checks both; use PEFT ≥ 0.13.0 |
| DDP + PEFT gradient sync | Medium | `find_unused_parameters=True`; PEFT ≥ 0.9.0 |
| Loss increases between stages | Expected | Normal; best-epoch selection within each stage handles this |
| `use_mamba_kernels` kwarg missing on old TF | Low | Version guard in `load_model_and_tokenizer` |
| flash-attn absent on Colab | High | try/except fallback to `eager`; fully handled |
| NCCL watchdog killing DDP run | Was S5–S7 | `timeout=timedelta(minutes=60)` + graceful timeout saves checkpoint first |
| OOM at first val step | Was S8 | `torch.cuda.empty_cache()` + `val_batch_size=1` default |
| `conv1d` in LoRA targets | Prevented | Explicitly excluded with comment |
