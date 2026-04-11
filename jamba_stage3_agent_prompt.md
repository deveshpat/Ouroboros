# Jamba2-3B Coconut-Ouroboros Fine-tuning Agent Prompt
## Project Ouroboros / Stage 3 — `jamba_coconut_finetune.py`

> **Self-contained. Feed this entire file to a coding agent.**
> Generates one script: `jamba_coconut_finetune.py`
> Do NOT reference `baseline_trm_mamba.py`, `training_utils.py`, or `stage3_agent_prompt.md` — this is a clean, Jamba2-specific implementation.

---

## Background & Design Rationale

### Why Jamba2-3B-Instruct, not from scratch

AI21's Jamba2-3B uses the identical 1:7 Transformer:Mamba ratio that our nano architecture validated, with Mamba 1 (confirmed superior to Mamba 2 in hybrid — Jamba 1.5 paper Appendix C.2). Using the Instruct variant means:
- We skip re-teaching instruction following (already production-grade)
- The research comparison is clean: Jamba2-Instruct (greedy, K=0) vs. Jamba2-Instruct + Coconut-Ouroboros (K=16 + DGAC)

### Why LoRA / QLoRA, not full fine-tuning

3B params × bfloat16 = 6GB weights; full fine-tuning with optimizer states would require ~24GB per GPU. With QLoRA (4-bit quantization + LoRA adapters in bfloat16) the memory footprint drops to ~2GB weights + ~0.3GB LoRA, fitting comfortably on a T4 (15GB).

QLoRA requires bitsandbytes, which is CUDA-only. Use the `--use_4bit` flag to switch between QLoRA (GPU) and standard LoRA (TPU). The code path is otherwise identical.

### Why Coconut (Meta, arXiv:2412.06769)

Coconut replaces explicit chain-of-thought tokens with latent thought passes: the hidden state at the last question position is injected back as the embedding for the next input position, bypassing the token vocabulary. The Mamba SSM recurrent state accumulates a compressed O(d_state) summary across K passes — a genuine architectural advantage over pure Transformer Coconut.

### Why DGAC (our contribution)

Standard learned halt gates collapse to K=1 in training because the ponder cost dominates before the task loss has incentivized multi-pass refinement. DGAC (Diversity-Gated Adaptive Coconut) adds a **diversity regularizer** that penalizes latent passes where the hidden state barely changed (cosine similarity > τ). Together with ACT-style ponder cost (annealed from zero), this enforces genuine refinement as a precondition for halting credit.

---

## Task

Create `jamba_coconut_finetune.py` from scratch.
Do not use `baseline_trm_mamba.py` or `training_utils.py`.
Use standard libraries only: `transformers`, `peft`, `datasets`, `torch`, `tqdm`, `wandb`.

---

## Part 1 — File Header

```python
#!/usr/bin/env python3
"""
Stage 3 Coconut-Ouroboros Fine-tuning for Jamba2-3B-Instruct
=============================================================
Implements Coconut-style latent thought injection + DGAC halt gate
on top of Jamba2-3B-Instruct using LoRA/QLoRA via PEFT.

Phases:
  3.1  K=1  fixed latent passes, no gate
  3.2  K=4  fixed latent passes, no gate
  3.3  K=16 fixed latent passes, no gate
  3.4  K=16 with DGAC halt gate (--use_halt_gate)

Each phase resumes from the previous checkpoint via --resume_from.
Start Phase 3.1 from the base Jamba2-3B-Instruct model (no --resume_from).

References:
  Coconut (Meta, arXiv:2412.06769)
  Jamba (AI21, arXiv:2403.19887 / Jamba2 2024)

Hardware targets:
  Smoke test : Free Colab T4 (15 GB), --use_4bit, batch=1, seq=128, K=1, max_steps=20
  K=1→4     : Kaggle Dual T4 (2×16 GB), --use_4bit, DDP
  K=16+DGAC : TRC A100 (80 GB), full bfloat16 or --use_4bit

Install:
  pip install transformers peft datasets tqdm wandb bitsandbytes accelerate

Run (smoke test on Colab T4):
  python jamba_coconut_finetune.py \\
    --n_latent 1 --max_steps 20 --max_samples 50 \\
    --use_4bit --wandb_mode disabled --output_dir runs/smoke

Run (Phase 3.1, Kaggle Dual T4, torchrun):
  torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \\
    --n_latent 1 --num_epochs 2 --batch_size 2 --grad_accum 8 \\
    --use_4bit --output_dir runs/stage3_k1

Run (Phase 3.2, from Phase 3.1 checkpoint):
  torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \\
    --n_latent 4 --resume_from runs/stage3_k1/checkpoint-XXXXXXX \\
    --use_4bit --baseline_val_ce 1.42 --output_dir runs/stage3_k4

Run (Phase 3.4, DGAC gate):
  python jamba_coconut_finetune.py \\
    --n_latent 16 --use_halt_gate --resume_from runs/stage3_k16/checkpoint-XXXXXXX \\
    --baseline_val_ce 1.42 --output_dir runs/stage3_dgac
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
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel
    from datasets import load_dataset
    from tqdm.auto import tqdm
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\npip install transformers peft datasets tqdm wandb bitsandbytes accelerate")

MODEL_ID = "ai21labs/AI21-Jamba2-3B-Instruct"

LORA_TARGET_MODULES = [
    # Attention projections
    "q_proj", "k_proj", "v_proj", "o_proj",
    # Mamba SSM projections (primary carriers of latent state)
    "in_proj", "x_proj", "dt_proj", "out_proj",
]

# Generation prompts for qualitative callbacks
GEN_PROMPTS = [
    "What is 15 + 27?",
    "Write a Python function that returns the factorial of n.",
    "What is the capital of Japan?",
    "Explain what a neural network is in simple terms.",
    "Solve for x: 3x + 6 = 21.",
]
```

---

## Part 3 — CLI Arguments

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jamba2-3B Coconut-Ouroboros fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--max_seq_len", type=int, default=512)

    # LoRA / QLoRA
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use QLoRA (4-bit NF4 + bfloat16 compute). Requires CUDA + bitsandbytes. "
                             "Do NOT set on TPU.")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Latent thoughts
    parser.add_argument("--n_latent", type=int, default=4,
                        help="Number of latent thought passes (K). Curriculum: 1 → 4 → 16.")
    parser.add_argument("--use_halt_gate", action="store_true",
                        help="Enable DGAC halt gate (Phase 3.4 only). Requires a Phase 3.3 checkpoint.")
    parser.add_argument("--halt_threshold", type=float, default=0.5,
                        help="Hard-halt threshold at inference (0–1). Lower = more passes.")
    parser.add_argument("--dgac_lambda_ponder_max", type=float, default=0.01,
                        help="Maximum ponder cost weight (λ₁). Annealed from 0 over dgac_ramp_steps.")
    parser.add_argument("--dgac_lambda_diversity", type=float, default=0.1,
                        help="Diversity regularization weight (λ₂). Fixed throughout Phase 3.4.")
    parser.add_argument("--dgac_tau", type=float, default=0.9,
                        help="Cosine similarity threshold above which a latent pass is penalized.")
    parser.add_argument("--dgac_warmup_steps", type=int, default=200,
                        help="Steps at start of Phase 3.4 with λ₁=0 (diversity only).")
    parser.add_argument("--dgac_ramp_steps", type=int, default=300,
                        help="Steps over which λ₁ ramps from 0 to dgac_lambda_ponder_max.")

    # Dataset
    parser.add_argument("--dataset_name", default="bespokelabs/Bespoke-Stratos-17k")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.05)

    # Training
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Override total steps. -1 = epochs × steps_per_epoch.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Global batch size (split across GPUs in DDP).")
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_checkpoint", action="store_true", default=True,
                        help="Enable gradient checkpointing (strongly recommended for K≥4).")
    parser.add_argument("--seed", type=int, default=42)

    # Gate check
    parser.add_argument("--baseline_val_ce", type=float, default=None,
                        help="Jamba2-Instruct baseline val_ce at K=0. "
                             "Gate check warns when answer val_ce > baseline × 1.05.")

    # I/O
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a previous stage3 checkpoint directory to resume from.")
    parser.add_argument("--output_dir", default="runs/stage3")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--keep_last", type=int, default=3)

    # Monitoring
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=250)
    parser.add_argument("--gen_every", type=int, default=500)
    parser.add_argument("--gen_max_tokens", type=int, default=200)

    # wandb
    parser.add_argument("--wandb_project", default="ouroboros-stage3-jamba")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")

    return parser.parse_args()
```

---

## Part 4 — Model Loading

```python
def load_model_and_tokenizer(args, device):
    """Load Jamba2-3B-Instruct with LoRA or QLoRA.

    Returns: (model, tokenizer, d_model)

    Notes:
      - With --use_4bit: loads in 4-bit NF4 (QLoRA). Requires CUDA + bitsandbytes.
      - Without --use_4bit: loads in bfloat16 (standard LoRA). Works on GPU or TPU.
      - use_mamba_kernels=False: disables the CUDA mamba-ssm kernel, enabling pure PyTorch
        fallback compatible with TPU and devices without mamba-ssm installed.
      - The halt gate is a separate nn.Module, NOT a LoRA module. It is always trained
        from scratch in Phase 3.4 regardless of the --resume_from checkpoint.
    """
    print(f"Loading tokenizer from {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model from {args.model_id} ...")
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_mamba_kernels=False,  # pure PyTorch fallback; required when mamba-ssm unavailable
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_mamba_kernels=False,
        )

    # Prepare for k-bit training (only needed for QLoRA, harmless otherwise)
    if args.use_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.grad_checkpoint)
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

    # Verify inputs_embeds API works as expected
    d_model = model.config.hidden_size
    print(f"  d_model (hidden_size): {d_model}")

    return model, tokenizer, d_model
```

---

## Part 5 — Data Preparation

```python
def build_coconut_sample(
    tokenizer,
    question: str,
    answer: str,
    n_latent: int,
    lat_token_id: int,
    max_seq_len: int,
    eos: str,
) -> Optional[Dict[str, Any]]:
    """Build one Coconut-format sample.

    Layout of full_ids: [question_prefix | lat*K | answer_suffix]

    Labels: -100 everywhere except answer positions.
    At training time, the K lat_token positions are replaced with injected
    hidden states in coconut_forward — the token IDs there are irrelevant.

    Note: we use the Jamba2 chat template to format the question prefix.
    The answer portion is appended directly after the latent positions,
    matching the post-[/INST] region of the template.
    """
    # Build question prefix using tokenizer's chat template
    messages = [{"role": "user", "content": question}]
    # apply_chat_template with add_generation_prompt=True appends the
    # assistant header token sequence (e.g. "[/INST] "), giving us the
    # exact injection point for latent passes.
    prefix_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    answer_text = f"{answer}{eos}"

    q_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    a_ids = tokenizer.encode(answer_text, add_special_tokens=False)

    max_q = max_seq_len - n_latent - len(a_ids)
    max_a = max_seq_len - n_latent - len(q_ids)
    if max_q < 4 or max_a < 1:
        return None

    q_ids = q_ids[:max_q]
    a_ids = a_ids[:max_a]

    q_tensor = torch.tensor(q_ids, dtype=torch.long)
    a_tensor = torch.tensor(a_ids, dtype=torch.long)
    lat_tensor = torch.full((n_latent,), lat_token_id, dtype=torch.long)

    full_ids = torch.cat([q_tensor, lat_tensor, a_tensor])
    q_len = len(q_ids)
    a_start = q_len + n_latent

    return {
        "full_ids": full_ids,
        "q_len": q_len,
        "a_start": a_start,
    }


def get_lat_token_id(tokenizer) -> int:
    """Add <|lat|> as a special token if not present; return its token ID.

    This token's embedding is NEVER used during training (it is replaced by
    injected hidden states in coconut_forward). Its ID just serves as a
    placeholder in full_ids for correct padding and label masking.
    """
    LAT_TOKEN = "<|lat|>"
    existing_id = tokenizer.convert_tokens_to_ids(LAT_TOKEN)
    if existing_id != tokenizer.unk_token_id:
        return existing_id
    tokenizer.add_special_tokens({"additional_special_tokens": [LAT_TOKEN]})
    # NOTE: Jamba2's token_embedding is inside model.model.embed_tokens.
    # It must be resized to cover the new token ID.
    # Resize is handled in load_and_prepare_data() after this function returns.
    return tokenizer.convert_tokens_to_ids(LAT_TOKEN)


def _extract_qa(example: Dict[str, Any]) -> Tuple[str, str]:
    """Extract (question, answer) from a Bespoke-Stratos conversation row."""
    question = ""
    assistant_blob = ""
    for turn in (example.get("conversations") or []):
        role = str(turn.get("role") or turn.get("from") or "").lower().strip()
        value = str(turn.get("content") or turn.get("value") or "").strip()
        if role in {"user", "human"} and not question:
            question = value
        elif role in {"assistant", "gpt"} and not assistant_blob:
            assistant_blob = value
    # Extract final answer (drop <think> block if present)
    if "</think>" in assistant_blob:
        answer = assistant_blob.split("</think>", 1)[1].strip()
    else:
        answer = assistant_blob.strip()
    return question.strip(), answer.strip()


def load_and_prepare_data(
    args: argparse.Namespace,
    tokenizer,
    model,
    device: torch.device,
) -> Tuple[List[Dict], List[Dict], int]:
    """Load dataset, tokenize, split train/val. Return (train, val, lat_token_id).

    Also resizes model.model.embed_tokens if <|lat|> is a new token.
    """
    lat_token_id = get_lat_token_id(tokenizer)

    # Resize embedding if the new token extended the vocabulary
    current_vocab = len(tokenizer)
    embed_size = model.model.embed_tokens.num_embeddings
    if current_vocab > embed_size:
        print(f"  Resizing embed_tokens: {embed_size} → {current_vocab}")
        model.resize_token_embeddings(current_vocab)

    print(f"  <|lat|> token id: {lat_token_id}")
    eos = tokenizer.eos_token or "<|endoftext|>"

    print(f"Loading {args.dataset_name} ...")
    raw = load_dataset(args.dataset_name, split="train")
    if args.max_samples is not None:
        raw = raw.select(range(min(args.max_samples, len(raw))))

    samples = []
    skipped = 0
    for ex in tqdm(raw, desc="Tokenizing", leave=False):
        q, a = _extract_qa(ex)
        if not q or not a:
            skipped += 1
            continue
        sample = build_coconut_sample(
            tokenizer, q, a, args.n_latent, lat_token_id, args.max_seq_len, eos
        )
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    print(f"  {len(samples)} samples kept, {skipped} skipped.")

    # Deterministic train/val split
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(samples), generator=generator).tolist()
    n_val = min(len(samples) - 1, max(1, int(len(samples) * args.val_fraction)))
    val_idx = set(perm[:n_val])
    train_samples = [s for i, s in enumerate(samples) if i not in val_idx]
    val_samples = [s for i, s in enumerate(samples) if i in val_idx]
    print(f"  train: {len(train_samples)}  val: {len(val_samples)}")

    return train_samples, val_samples, lat_token_id
```

---

## Part 6 — Collate Function

```python
def collate_coconut(
    samples: List[Dict[str, Any]],
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    """Pad a micro-batch. Labels are -100 everywhere except answer positions.

    IMPORTANT: all samples in a micro-batch must have the same q_len.
    Enforced by the data loader (see fetch_micro_batch). If q_lens differ,
    the batch is rejected and a warning is printed. Use batch_size=1 during
    debugging to bypass this entirely.
    """
    q_lens = [s["q_len"] for s in samples]
    if len(set(q_lens)) > 1:
        # Mixed q_len batch: truncate all to minimum q_len so injection positions
        # are consistent. This loses some context but avoids silent errors.
        min_q = min(q_lens)
        import warnings
        warnings.warn(
            f"Mixed q_lens in batch: {q_lens}. Truncating all to min={min_q}. "
            "Consider sorting samples by q_len for efficiency."
        )
        adjusted = []
        for s in samples:
            if s["q_len"] > min_q:
                diff = s["q_len"] - min_q
                new_full = torch.cat([s["full_ids"][:min_q], s["full_ids"][s["q_len"]:]])
                adjusted.append({
                    "full_ids": new_full,
                    "q_len": min_q,
                    "a_start": min_q + (s["a_start"] - s["q_len"]),
                })
            else:
                adjusted.append(s)
        samples = adjusted

    max_len = max(s["full_ids"].size(0) for s in samples)
    B = len(samples)

    input_ids      = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels         = torch.full((B, max_len), -100,   dtype=torch.long)
    attention_mask = torch.zeros(B, max_len,           dtype=torch.bool)
    q_lens_t       = torch.zeros(B,                    dtype=torch.long)
    a_starts_t     = torch.zeros(B,                    dtype=torch.long)

    for i, s in enumerate(samples):
        ids = s["full_ids"]
        T = ids.size(0)
        a_start = s["a_start"]

        input_ids[i, :T]      = ids
        attention_mask[i, :T] = True
        if a_start < T:
            labels[i, a_start:T] = ids[a_start:T]
        q_lens_t[i]  = s["q_len"]
        a_starts_t[i] = a_start

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "q_lens":         q_lens_t,
        "a_starts":       a_starts_t,
    }
```

---

## Part 7 — DGAC Halt Gate

```python
class HaltGate(nn.Module):
    """Learned halt gate for DGAC (Diversity-Gated Adaptive Coconut).

    Takes the current and previous question-end hidden states as input.
    Outputs a per-sample halt probability ∈ (0, 1).

    Zero-initialized so the gate outputs 0.5 at the start of Phase 3.4,
    giving maximum uncertainty before any gradient signal.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        h_curr: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_curr: [B, D] — hidden state at question-end after current pass
            h_prev: [B, D] — hidden state at question-end after previous pass
        Returns:
            halt_prob: [B] — halt probability per sample
        """
        combined = torch.cat([h_curr, h_prev], dim=-1)  # [B, 2D]
        return torch.sigmoid(self.gate(combined)).squeeze(-1)  # [B]


def compute_dgac_lambda1(
    step_in_phase: int,
    warmup_steps: int,
    ramp_steps: int,
    lambda_max: float,
) -> float:
    """Return λ₁ (ponder cost weight) at a given step within Phase 3.4.

    Schedule:
      steps 0..warmup_steps:              λ₁ = 0
      steps warmup_steps..warmup+ramp:    λ₁ linearly 0 → lambda_max
      steps warmup+ramp..:                λ₁ = lambda_max
    """
    if step_in_phase < warmup_steps:
        return 0.0
    ramp_progress = min((step_in_phase - warmup_steps) / max(ramp_steps, 1), 1.0)
    return lambda_max * ramp_progress
```

---

## Part 8 — Coconut Forward Pass

```python
def coconut_forward(
    model,
    batch: Dict[str, torch.Tensor],
    n_latent: int,
    device: torch.device,
    halt_gate: Optional[HaltGate],
    args: argparse.Namespace,
    step_in_phase: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute Coconut-Ouroboros training loss for one micro-batch.

    Mechanism:
      Phase A: Embed all tokens (question + lat*K + answer).
      Phase B: Sequentially build latent embeddings.
               For pass k=0..K-1:
                 - Run model over prefix [question | lat*0..k] to get h_{q+k}.
                 - Patch the embedding at position (q_len + k + 1) with h_{q+k}.
      Phase C: Run full model over all patched embeddings.
               Compute CE loss on answer tokens only.
      [Optional Phase D]: If use_halt_gate, compute DGAC gate and extra loss terms.

    Returns:
        loss: scalar (backpropagatable)
        metrics: dict of float for logging (ponder_cost, diversity_loss, halt_step_mean)

    IMPORTANT: All forward passes in Phase B must use use_cache=False.
               Caching would cause silent shape mismatches with the patched embeddings.

    Variable q_len: This implementation assumes identical q_len within a batch
    (enforced by collate_coconut). If batch has B>1 with mixed q_lens after
    collate's truncation, the injection positions may be off by a few tokens
    for some samples. Set batch_size=1 during debugging to avoid this entirely.
    """
    input_ids  = batch["input_ids"].to(device)        # [B, T]
    attn_mask  = batch["attention_mask"].to(device)   # [B, T] bool
    labels     = batch["labels"].to(device)            # [B, T]
    q_lens     = batch["q_lens"].to(device)            # [B]
    B, T       = input_ids.shape
    metrics    = {}

    # q_len: assumed identical within batch after collate
    q_len = int(q_lens[0].item())

    # ── Phase A: get all token embeddings ───────────────────────────────────
    # model.model.embed_tokens is the token embedding table inside Jamba2.
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        all_embeds = model.model.embed_tokens(input_ids)  # [B, T, D]

    patched_embeds = all_embeds.clone()  # differentiable clone for in-place patching

    # ── Phase B: sequential latent injection ────────────────────────────────
    # Collect hidden states for DGAC gate (only needed when use_halt_gate=True)
    hidden_states_at_q_end: List[torch.Tensor] = []

    for k in range(n_latent):
        # Prefix: [question | lat*0..k-1]  (length = q_len + k)
        prefix_len = q_len + k
        if prefix_len == 0:
            break

        prefix_embeds = patched_embeds[:, :prefix_len, :]   # [B, prefix_len, D]
        prefix_mask   = attn_mask[:, :prefix_len]            # [B, prefix_len]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            prefix_out = model.model(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_mask,
                use_cache=False,
            )
        prefix_hidden = prefix_out.last_hidden_state  # [B, prefix_len, D]
        last_hidden   = prefix_hidden[:, -1:, :]      # [B, 1, D] — hidden at position q_len+k-1

        if halt_gate is not None:
            hidden_states_at_q_end.append(last_hidden.squeeze(1))  # [B, D]

        # Patch embedding at position (q_len + k)
        inject_pos = q_len + k
        if inject_pos < T:
            # torch.cat instead of in-place assignment to keep gradient flow intact
            patched_embeds = torch.cat([
                patched_embeds[:, :inject_pos, :],
                last_hidden,
                patched_embeds[:, inject_pos + 1:, :],
            ], dim=1)  # [B, T, D]

    # ── Phase C: full forward pass with patched embeddings ──────────────────
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        full_out = model.model(
            inputs_embeds=patched_embeds,
            attention_mask=attn_mask,
            use_cache=False,
        )
        full_hidden = full_out.last_hidden_state       # [B, T, D]
        logits      = model.lm_head(full_hidden).float()  # [B, T, V]

    # CE loss on answer tokens only (shifted by 1 for next-token prediction)
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1)
    valid = shift_labels != -100

    if not valid.any():
        dummy = torch.tensor(0.0, device=device, requires_grad=True)
        return dummy, {"ce": 0.0, "ponder": 0.0, "diversity": 0.0, "halt_step_mean": float(n_latent)}

    ce_loss = F.cross_entropy(shift_logits[valid], shift_labels[valid])
    metrics["ce"] = ce_loss.item()

    # ── Phase D: DGAC gate loss (Phase 3.4 only) ────────────────────────────
    if halt_gate is None or len(hidden_states_at_q_end) < 2:
        return ce_loss, metrics

    # Compute halt probabilities and ACT variables
    # hidden_states_at_q_end[k]: hidden state at question-end position after pass k+1
    # h_prev for k=0 is h_before_any_latent = prefix_hidden at q_len-1, pass 0
    # We use hidden_states_at_q_end[0] as h_prev for k=1, etc.

    lambda1 = compute_dgac_lambda1(
        step_in_phase=step_in_phase,
        warmup_steps=args.dgac_warmup_steps,
        ramp_steps=args.dgac_ramp_steps,
        lambda_max=args.dgac_lambda_ponder_max,
    )

    ponder_cost    = torch.zeros(B, device=device)
    diversity_loss = torch.zeros(B, device=device)
    remainder      = torch.ones(B, device=device)
    halt_steps     = torch.zeros(B, device=device)  # for logging only

    for k in range(1, len(hidden_states_at_q_end)):
        h_curr = hidden_states_at_q_end[k].to(device=device, dtype=torch.float32)
        h_prev = hidden_states_at_q_end[k - 1].to(device=device, dtype=torch.float32)

        halt_prob = halt_gate(h_curr, h_prev)  # [B]

        # ACT ponder cost contribution
        ponder_cost = ponder_cost + remainder

        # Update remainder (soft continuation probability)
        is_last = (k == len(hidden_states_at_q_end) - 1)
        if not is_last:
            remainder = remainder * (1.0 - halt_prob.detach())

        # Diversity penalty: penalize if this pass barely changed the hidden state
        cos_sim = F.cosine_similarity(h_curr, h_prev, dim=-1)  # [B]
        diversity_loss = diversity_loss + F.relu(cos_sim - args.dgac_tau)

        # Track mean halt step for logging (non-differentiable)
        with torch.no_grad():
            halted = (halt_prob > args.halt_threshold) & (halt_steps == 0)
            halt_steps = torch.where(halted, torch.full_like(halt_steps, float(k)), halt_steps)

    # Samples that never hit threshold: assign max halt step
    halt_steps = torch.where(halt_steps == 0, torch.full_like(halt_steps, float(n_latent)), halt_steps)

    ponder_mean    = ponder_cost.mean()
    diversity_mean = diversity_loss.mean()
    total_loss     = ce_loss + lambda1 * ponder_mean + args.dgac_lambda_diversity * diversity_mean

    metrics["ponder"]          = ponder_mean.item()
    metrics["diversity"]       = diversity_mean.item()
    metrics["halt_step_mean"]  = halt_steps.mean().item()
    metrics["lambda1"]         = lambda1

    return total_loss, metrics
```

---

## Part 9 — Validation

```python
@torch.no_grad()
def compute_val_ce(
    model,
    halt_gate: Optional[HaltGate],
    val_samples: List[Dict],
    pad_id: int,
    n_latent: int,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    """Compute answer-token validation CE using Coconut forward (no EMA — LoRA is always live).

    Note: for Phase 3.4 (halt gate), runs with gate disabled so the CE is
    comparable across phases (measures reasoning quality, not efficiency).
    """
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for start in range(0, len(val_samples), 1):  # batch_size=1 for val stability
        batch = collate_coconut([val_samples[start]], pad_id)
        loss, _ = coconut_forward(
            model=model,
            batch=batch,
            n_latent=n_latent,
            device=device,
            halt_gate=None,   # disable gate for CE comparison
            args=args,
            step_in_phase=999999,  # max step → lambda1 at max, but gate disabled so irrelevant
        )
        labels = batch["labels"].to(device)
        n_valid = int((labels[:, 1:].contiguous().view(-1) != -100).sum().item())
        total_loss   += loss.item() * n_valid
        total_tokens += n_valid

    model.train()
    return total_loss / max(total_tokens, 1)
```

---

## Part 10 — Generation Callback

```python
@torch.no_grad()
def run_generation_callback(
    model,
    tokenizer,
    halt_gate: Optional[HaltGate],
    n_latent: int,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    wandb_run=None,
) -> float:
    """Generate answers using Coconut latent injection at inference.

    Inference procedure:
      1. Tokenize question with chat template.
      2. Run K latent passes (inject hidden state; no token generated).
         If use_halt_gate: halt early when gate(h_k, h_{k-1}) > halt_threshold.
      3. Greedy decode answer from latent-enriched context.
    """
    model.eval()
    print(f"\n  -- Generation @ step {step} (K={n_latent}, halt_gate={halt_gate is not None}) --")
    mean_uwr = 0.0

    for prompt in GEN_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        prefix_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        q_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        q_tensor = torch.tensor(q_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T_q]

        # Embed question
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            current_embeds = model.model.embed_tokens(q_tensor)  # [1, T_q, D]
            out = model.model(inputs_embeds=current_embeds, use_cache=False)
            h_prev = out.last_hidden_state[:, -1, :]  # [1, D]

        # K latent passes
        actual_k = 0
        for k in range(n_latent):
            # Extend embeds with last hidden state
            current_embeds = torch.cat([current_embeds, h_prev.unsqueeze(1)], dim=1)  # [1, T+1, D]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model.model(inputs_embeds=current_embeds, use_cache=False)
                h_curr = out.last_hidden_state[:, -1, :]  # [1, D]

            actual_k = k + 1

            # DGAC halt check
            if halt_gate is not None and k > 0:
                hp = halt_gate(h_curr, h_prev).item()
                if hp > args.halt_threshold:
                    break

            h_prev = h_curr

        # Greedy decode
        generated = []
        eos_id = tokenizer.eos_token_id
        context = current_embeds

        for _ in range(args.gen_max_tokens):
            if context.size(1) > args.max_seq_len:
                context = context[:, -args.max_seq_len:, :]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model.model(inputs_embeds=context, use_cache=False)
                logits = model.lm_head(out.last_hidden_state)  # [1, T, V]
            next_id = int(logits[:, -1, :].argmax(dim=-1).item())
            if eos_id is not None and next_id == eos_id:
                break
            generated.append(next_id)
            next_embed = model.model.embed_tokens(
                torch.tensor([[next_id]], device=device)
            )  # [1, 1, D]
            context = torch.cat([context, next_embed], dim=1)

        output_text = tokenizer.decode(generated, skip_special_tokens=True)
        words = output_text.split()
        uwr = len(set(words)) / max(len(words), 1)
        mean_uwr += uwr
        print(f"  Q: {prompt}")
        print(f"  A: {output_text[:200].replace(chr(10),' ')}  [k_used={actual_k}  uwr={uwr:.3f}]")

    mean_uwr /= max(len(GEN_PROMPTS), 1)
    print(f"  Mean UWR: {mean_uwr:.3f}\n")
    model.train()
    return mean_uwr
```

---

## Part 11 — Checkpointing

```python
def save_checkpoint(
    output_dir: Path,
    step: int,
    phase: str,
    n_latent: int,
    model,
    halt_gate: Optional[HaltGate],
    optimizer: AdamW,
    scheduler: LambdaLR,
    args: argparse.Namespace,
    val_ce: Optional[float],
    baseline_val_ce: Optional[float],
) -> Optional[Path]:
    """Save LoRA adapters + halt gate + training metadata.

    Directory structure:
        checkpoint-XXXXXXX/
            adapter_model/      ← PEFT save_pretrained output
            halt_gate.pt        ← HaltGate state_dict (only if use_halt_gate)
            training_state.pt   ← metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"checkpoint-{step:07d}"
    final_dir = output_dir / ckpt_name
    tmp_dir   = output_dir / f"{ckpt_name}.tmp"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapters via PEFT
    model.save_pretrained(str(tmp_dir / "adapter_model"))

    # Save halt gate
    if halt_gate is not None:
        torch.save(halt_gate.state_dict(), tmp_dir / "halt_gate.pt")

    # Save training state
    state = {
        "stage":            "coconut_jamba",
        "phase":            phase,
        "step":             step,
        "n_latent":         n_latent,
        "use_halt_gate":    args.use_halt_gate,
        "halt_threshold":   args.halt_threshold,
        "val_ce":           val_ce,
        "baseline_val_ce":  baseline_val_ce,
        "optimizer":        optimizer.state_dict(),
        "scheduler":        scheduler.state_dict(),
        "jamba_model_id":   args.model_id,
        "lora_r":           args.lora_r,
        "lora_alpha":       args.lora_alpha,
    }
    torch.save(state, tmp_dir / "training_state.pt")

    if final_dir.exists():
        shutil.rmtree(final_dir, ignore_errors=True)
    tmp_dir.replace(final_dir)
    print(f"  [ckpt] saved -> {final_dir}")

    # Prune old checkpoints
    retain = max(int(args.keep_last), 1)
    existing = sorted(
        [p for p in output_dir.iterdir()
         if p.is_dir() and p.name.startswith("checkpoint-") and not p.name.endswith(".tmp")],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    for old in existing[:-retain]:
        shutil.rmtree(old, ignore_errors=True)
        print(f"  [ckpt] pruned -> {old.name}")

    return final_dir


def load_checkpoint(
    checkpoint_dir: Path,
    model,
    halt_gate: Optional[HaltGate],
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[int, str]:
    """Load checkpoint. Returns (step, phase).

    Handles two cases:
      (a) Resuming within the same phase: loads optimizer + scheduler.
      (b) Advancing to next phase (n_latent changed): loads model weights only,
          resets optimizer + scheduler (caller detects this via returned step=0
          when state["step"] is present but phase mismatches).

    LoRA adapters are loaded via PEFT's PeftModel.from_pretrained into the
    existing PEFT-wrapped model. This requires that the base model architecture
    and LoRA config are identical to the checkpoint.
    """
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"No training_state.pt in {checkpoint_dir}")

    state = torch.load(state_path, map_location=device)
    step  = int(state.get("step", 0))
    phase = str(state.get("phase", "3.x"))
    saved_n_latent = int(state.get("n_latent", -1))

    # Load LoRA adapter weights
    adapter_dir = checkpoint_dir / "adapter_model"
    if adapter_dir.exists():
        from peft import set_peft_model_state_dict
        adapter_weights = torch.load(adapter_dir / "adapter_model.bin", map_location=device)
        set_peft_model_state_dict(model, adapter_weights)
        if verbose:
            print(f"  [resume] loaded LoRA adapters from {adapter_dir.name}")

    # Load halt gate
    halt_gate_path = checkpoint_dir / "halt_gate.pt"
    if halt_gate is not None and halt_gate_path.exists():
        halt_gate.load_state_dict(torch.load(halt_gate_path, map_location=device))
        if verbose:
            print("  [resume] loaded halt_gate.pt")

    # Detect phase transition (different n_latent = new phase → reset optimizer)
    advancing_phase = (saved_n_latent != -1) and (saved_n_latent != state.get("n_latent", saved_n_latent))
    if advancing_phase:
        if verbose:
            print(f"  [resume] n_latent changed ({saved_n_latent} → current). "
                  "Resetting optimizer/scheduler for new phase.")
        return 0, phase

    # Resume within same phase: restore optimizer and scheduler
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    if verbose:
        print(f"  [resume] step={step}  phase={phase}  val_ce={state.get('val_ce')}")

    return step, phase
```

---

## Part 12 — Optimizer and Scheduler

```python
def build_optimizer_and_scheduler(
    model,
    halt_gate: Optional[HaltGate],
    args: argparse.Namespace,
    total_steps: int,
) -> Tuple[AdamW, LambdaLR]:
    """Build AdamW for LoRA parameters + halt gate, with cosine LR decay."""
    # Only optimize trainable parameters (LoRA adapters + halt gate)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if halt_gate is not None:
        trainable_params.extend(halt_gate.parameters())

    decay_params   = [p for p in trainable_params if p.ndim >= 2]
    no_decay_params = [p for p in trainable_params if p.ndim < 2]
    param_groups = [
        {"params": decay_params,    "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        if total_steps <= args.warmup_steps:
            return args.min_lr_ratio
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
```

---

## Part 13 — Main Training Loop

```python
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ── Distributed setup ────────────────────────────────────────────────────
    rank       = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    distributed = world_size > 1
    is_main = rank == 0

    if distributed:
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # ── wandb ────────────────────────────────────────────────────────────────
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

    # ── Load model ───────────────────────────────────────────────────────────
    model, tokenizer, d_model = load_model_and_tokenizer(args, device)

    # ── Build halt gate (Phase 3.4 only) ────────────────────────────────────
    halt_gate = None
    if args.use_halt_gate:
        halt_gate = HaltGate(d_model).to(device=device, dtype=torch.float32)
        print(f"  DGAC HaltGate enabled: d_model={d_model}, params={sum(p.numel() for p in halt_gate.parameters())}")

    # ── Load data ────────────────────────────────────────────────────────────
    train_samples, val_samples, lat_token_id = load_and_prepare_data(args, tokenizer, model, device)
    pad_id = tokenizer.pad_token_id or 0

    # ── Optimizer / scheduler ────────────────────────────────────────────────
    steps_per_epoch = max(1, math.ceil(len(train_samples) / max(args.batch_size * args.grad_accum, 1)))
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.num_epochs
    optimizer, scheduler = build_optimizer_and_scheduler(model, halt_gate, args, total_steps)

    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_step = 0
    phase = f"3.{['?','1','4','16'].index(str(args.n_latent)) if str(args.n_latent) in ['1','4','16'] else 'x'}"
    # (Correct phase label: use n_latent to determine)
    phase = {1: "3.1", 4: "3.2", 16: "3.3"}.get(args.n_latent, "3.x")
    if args.use_halt_gate:
        phase = "3.4"

    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if ckpt_path.is_dir():
            start_step, _ = load_checkpoint(
                ckpt_path, model, halt_gate, optimizer, scheduler, device, verbose=is_main
            )
        elif is_main:
            print(f"  [warn] resume_from {ckpt_path} not found — starting from scratch")

    # ── DDP wrap (after checkpoint load, after LoRA) ─────────────────────────
    # NOTE: DDP with QLoRA can have gradient-sync issues on some PEFT versions.
    # If you hit "Trying to backward through the graph a second time" errors,
    # add find_unused_parameters=True or update PEFT to >= 0.9.0.
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, find_unused_parameters=True)

    # ── Training loop ────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.train()

    step = start_step
    step_in_phase = 0  # tracks steps within Phase 3.4 for λ₁ annealing
    last_val_ce: Optional[float] = None
    last_saved_step = step
    perm = list(range(len(train_samples)))
    random.shuffle(perm)
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=total_steps, initial=step, desc=f"Phase {phase} K={args.n_latent}") if is_main else None

    local_batch = args.batch_size // world_size if distributed else args.batch_size

    while step < total_steps:
        # ── Micro-batch accumulation ─────────────────────────────────────────
        accum_loss  = 0.0
        accum_ce    = 0.0
        valid_steps = 0

        for _ in range(args.grad_accum):
            idx = (step * args.grad_accum + _) % len(train_samples)
            batch_samples = [train_samples[perm[(idx + j) % len(train_samples)]]
                             for j in range(local_batch)]
            batch = collate_coconut(batch_samples, pad_id)

            raw_model = model.module if distributed else model
            loss, metrics = coconut_forward(
                model=raw_model,
                batch=batch,
                n_latent=args.n_latent,
                device=device,
                halt_gate=halt_gate,
                args=args,
                step_in_phase=step_in_phase,
            )

            (loss / args.grad_accum).backward()
            accum_loss += loss.item()
            accum_ce   += metrics.get("ce", loss.item())
            valid_steps += 1

        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + (list(halt_gate.parameters()) if halt_gate else []),
            args.max_grad_norm,
        ))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        step_in_phase += 1
        mean_loss = accum_loss / max(valid_steps, 1)
        mean_ce   = accum_ce   / max(valid_steps, 1)

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(ce=f"{mean_ce:.3f}", gn=f"{grad_norm:.3f}")

        if is_main and step % args.log_every == 0:
            log = {
                "train/loss": mean_loss,
                "train/ce": mean_ce,
                "train/grad_norm": grad_norm,
                "train/lr": scheduler.get_last_lr()[0],
                **{f"train/{k}": v for k, v in metrics.items()},
            }
            tqdm.write(
                f"  step={step:6d}  loss={mean_loss:.4f}  ce={mean_ce:.4f}  "
                f"gnorm={grad_norm:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
            )
            if wandb_run:
                import wandb; wandb.log(log, step=step)

        # ── Validation ───────────────────────────────────────────────────────
        if is_main and step % args.val_every == 0:
            raw_model = model.module if distributed else model
            last_val_ce = compute_val_ce(
                raw_model, halt_gate, val_samples, pad_id, args.n_latent, device, args
            )
            gate_str = ""
            if args.baseline_val_ce is not None:
                threshold = args.baseline_val_ce * 1.05
                status = "✓ GATE PASS" if last_val_ce <= threshold else f"⚠ {last_val_ce:.4f} > {threshold:.4f}"
                gate_str = f"  [{status}]"
            tqdm.write(f"  [val] step={step}  val_ce={last_val_ce:.4f}{gate_str}")
            if wandb_run:
                import wandb; wandb.log({"val/ce": last_val_ce}, step=step)

        # ── Generation ───────────────────────────────────────────────────────
        if is_main and step % args.gen_every == 0:
            raw_model = model.module if distributed else model
            run_generation_callback(
                raw_model, tokenizer, halt_gate, args.n_latent, device, args, step, wandb_run
            )

        # ── Checkpoint ───────────────────────────────────────────────────────
        if is_main and (step % args.save_every == 0 or step == total_steps):
            raw_model = model.module if distributed else model
            save_checkpoint(
                output_dir, step, phase, args.n_latent, raw_model,
                halt_gate, optimizer, scheduler, args, last_val_ce, args.baseline_val_ce,
            )
            last_saved_step = step

    if pbar is not None:
        pbar.close()

    # Final checkpoint
    if is_main and step != last_saved_step:
        raw_model = model.module if distributed else model
        save_checkpoint(
            output_dir, step, phase, args.n_latent, raw_model,
            halt_gate, optimizer, scheduler, args, last_val_ce, args.baseline_val_ce,
        )

    # ── Success banner ───────────────────────────────────────────────────────
    if is_main:
        print("\n" + "=" * 64)
        print(f"  Phase {phase} complete  K={args.n_latent}  steps={step}")
        if last_val_ce is not None:
            print(f"  Final val_ce: {last_val_ce:.4f}")
            if args.baseline_val_ce is not None:
                threshold = args.baseline_val_ce * 1.05
                if last_val_ce <= threshold:
                    print(f"  GATE PASSED ✓  val_ce {last_val_ce:.4f} ≤ {threshold:.4f}")
                    if args.n_latent < 16 and not args.use_halt_gate:
                        next_k = {1: 4, 4: 16}[args.n_latent]
                        print(f"\n  Next sub-stage:\n"
                              f"    python jamba_coconut_finetune.py \\\n"
                              f"      --n_latent {next_k} \\\n"
                              f"      --resume_from {output_dir}/checkpoint-{step:07d} \\\n"
                              f"      --baseline_val_ce {args.baseline_val_ce:.4f} \\\n"
                              f"      --output_dir runs/stage3_k{next_k}")
                    elif args.n_latent == 16 and not args.use_halt_gate:
                        print(f"\n  Proceed to Phase 3.4 (DGAC):\n"
                              f"    python jamba_coconut_finetune.py \\\n"
                              f"      --n_latent 16 --use_halt_gate \\\n"
                              f"      --resume_from {output_dir}/checkpoint-{step:07d} \\\n"
                              f"      --baseline_val_ce {args.baseline_val_ce:.4f} \\\n"
                              f"      --output_dir runs/stage3_dgac")
                    elif args.use_halt_gate:
                        print("  DGAC gate phase complete. Proceed to Stage 4 (GRPO).")
        print("=" * 64)

    if distributed:
        import torch.distributed as dist
        dist.destroy_process_group()
    if wandb_run:
        import wandb; wandb.finish()


if __name__ == "__main__":
    main()
```

---

## Part 14 — Verification Checklist

### Smoke test (Free Colab T4, ~5 min):
```bash
python jamba_coconut_finetune.py \
  --n_latent 1 --max_steps 20 --max_samples 50 \
  --use_4bit --wandb_mode disabled \
  --output_dir runs/smoke
```

- [ ] No import errors
- [ ] Model loads (prints trainable parameters, ~0.3–0.5% of total)
- [ ] `<|lat|>` token added; `embed_tokens` resized if needed
- [ ] Chat template applied correctly (prefix ends with assistant header)
- [ ] `coconut_forward` runs without OOM (batch=1, seq=128, K=1)
- [ ] Loss is finite and in [0.5, 5.0] range
- [ ] Generation runs; output is non-degenerate (UWR > 0.1)
- [ ] Checkpoint saved: `checkpoint-0000020/adapter_model/` exists
- [ ] Resume works: `--resume_from runs/smoke/checkpoint-0000020` continues from step 20

### Functional check K=1 (Kaggle Dual T4, ~2–4h):
```bash
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --n_latent 1 --num_epochs 2 --batch_size 2 --grad_accum 8 \
  --use_4bit --baseline_val_ce <jamba_baseline_ce> \
  --output_dir runs/stage3_k1
```

- [ ] Loss decreases over first 100 steps
- [ ] val_ce < baseline_val_ce × 1.05 within 1 epoch
- [ ] Generated answers are coherent (not loops)
- [ ] VRAM stable (no graph retention; use `nvidia-smi` to verify)

### Phase 3.4 DGAC check (after K=16):
- [ ] Gate initialized at 0.5 (verify via `halt_gate.gate.weight.norm() < 1e-3` at step 0)
- [ ] `halt_step_mean` logs > 1.0 within 500 steps (not collapsing to K=1)
- [ ] `diversity` loss decreases over training (passes become more distinct)
- [ ] Inference with `--halt_threshold 0.3` uses more passes; `--halt_threshold 0.7` uses fewer

---

## Part 15 — Known Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `model.model.embed_tokens` attribute name wrong for this Jamba2 version | Medium | Check `model.base_model.model.model.embed_tokens` if the direct path fails; add a helper `_get_embed_tokens(model)` that tries both paths |
| DDP + PEFT gradient sync failure | Medium | Add `find_unused_parameters=True` to DDP constructor; update PEFT ≥ 0.9.0 |
| QLoRA + DDP OOM at K=4 on Dual T4 | Medium | Reduce seq_len to 128 or batch_size to 1; add gradient checkpointing |
| DGAC gate collapse to K=1 | Low (mitigated by λ₁ annealing) | Monitor `halt_step_mean`; if it drops below 1.5 after step 500, increase dgac_warmup_steps |
| `inputs_embeds` + `use_cache=False` not supported by this Jamba2 HF version | Low | Check transformers version ≥ 4.40; if `last_hidden_state` not in output, use `output_hidden_states=True` and take `outputs.hidden_states[-1]` |
| mamba-ssm CUDA kernel import error | Low | Set `use_mamba_kernels=False` in model loading (already done in this spec) |
