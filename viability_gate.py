#!/usr/bin/env python3
"""
Phase 1 Viability Gate — Project Ouroboros / SERF Framework
============================================================
Purpose : Confirm BaselineTRMMamba can learn language on a tiny dataset
          before committing GPU-weeks to Phase 2 SFT distillation.

Four mandatory gates — ALL must PASS to clear Phase 2:
  G1  Per-token CE < 3.5 at final step
        Proves the architecture converges. Random-init CE ≈ 11.93.
        Reaching 3.5 means the model has learned significant structure
        even though it won't yet be fluent.
  G2  Generation non-degenerate at final step
        Unique-word-ratio > 0.10 on majority of test prompts.
        Catches the comma-loop / "the the the" failure modes early.
  G3  Gradient norm stable in final 100 steps
        Max grad_norm < 10.0. Exploding norms indicate architecture bugs
        (vanishing residuals, broken init, or graph retention).
  G4  VRAM footprint flat
        Delta between step 1 and final step < 1.0 GB.
        A growing footprint means gradient graphs are being retained
        across optimizer steps — a silent training-loop bug.

Runtime  : ~8-15 min on a single T4.
Requires : CUDA GPU (mamba-ssm uses CUDA kernels).
           baseline_trm_mamba.py in the same directory.

Install:
  pip install "causal-conv1d>=1.4.0" mamba-ssm --no-build-isolation
  pip install transformers datasets tqdm
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Dependency checks (fast-fail with clear messages) ────────────────────────

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("PyTorch is not installed. Run: pip install torch")

if not torch.cuda.is_available():
    sys.exit(
        "No CUDA GPU found. mamba-ssm requires CUDA kernels.\n"
        "Run on Kaggle (Dual T4), Colab, or any CUDA-capable machine."
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
    from tqdm import tqdm
except ImportError:
    sys.exit("tqdm required: pip install tqdm")

try:
    from baseline_trm_mamba import BaselineConfig, BaselineTRMMamba, count_parameters
except ImportError as exc:
    sys.exit(
        f"Cannot import baseline_trm_mamba: {exc}\n"
        "baseline_trm_mamba.py must be in the current working directory."
    )

from training_utils import (
    autocast_context,
    build_adamw_optimizer,
    pad_vocab_size,
    set_seed,
    vram_gb,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── Hardcoded hyperparameters ─────────────────────────────────────────────────
# This is a smoke test, not a configurable trainer. All knobs are fixed
# so results are reproducible and comparable across architecture changes.

TOKENIZER_NAME  = "Qwen/Qwen2.5-0.5B"
DATASET_NAME    = "bespokelabs/Bespoke-Stratos-17k"
MAX_SAMPLES     = 500      # tiny slice — fast to load, enough to show convergence
MAX_SEQ_LEN     = 256      # short sequences keep VRAM and step time low
TRAIN_STEPS     = 300      # ~8-15 min on T4; enough to see a clear loss trend
BATCH_SIZE      = 2        # micro-batch per grad-accum step
GRAD_ACCUM      = 4        # effective batch = 8
LR              = 3e-4     # aggressive — we want signal, not final convergence
WEIGHT_DECAY    = 0.1
MAX_GRAD_NORM   = 1.0      # clip threshold during training
SEED            = 42

# Model preset: nano (d_model=512, n_groups=1) — same as smoke-test baseline
# The goal is NOT to train a good model. The goal is to confirm convergence.
MODEL_D_MODEL   = 512
MODEL_N_GROUPS  = 1
MODEL_N_HEADS   = 8
MODEL_N_KV_HEADS = 4

# Gate thresholds
G1_CE_MAX           = 3.5   # per-token CE must be below this at the final step
G2_UNIQUE_WORD_MIN  = 0.10  # unique-word-ratio in generated text
G3_GRAD_NORM_MAX    = 10.0  # max grad_norm in the final 100 steps
G4_VRAM_DELTA_MAX   = 1.0   # GB growth allowed from step 1 → final step

# Generation test prompts (fixed across all runs)
GEN_PROMPTS = [
    "What is 2 + 2?",
    "Write a Python function that returns the square of a number.",
    "What is the capital of France?",
    "Explain what a variable is in programming.",
]
GEN_EVERY_STEPS  = 50   # run generation at these intervals
GEN_MAX_TOKENS   = 80   # greedy decode this many new tokens per prompt


# ── Utility functions ─────────────────────────────────────────────────────────


def analyze_output(text: str) -> Tuple[bool, float, int]:
    """
    Return (is_degenerate, unique_word_ratio, unique_char_count).

    Degenerate = unique_word_ratio < G2 threshold OR < 4 unique characters.
    Both conditions are needed: 'the the the' has high char diversity but
    low word diversity; ',,,,,' has low diversity on both metrics.
    """
    text = text.strip()
    if len(text) < 8:
        return True, 0.0, len(set(text))
    words = text.split()
    uwr   = len(set(words)) / max(len(words), 1)
    ucc   = len(set(text))
    return (uwr < G2_UNIQUE_WORD_MIN or ucc < 4), uwr, ucc



# ── Dataset ───────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_qa(example: Dict) -> Tuple[str, str]:
    """
    Pull question + answer from a Bespoke-Stratos conversations entry.
    Think tags are stripped — Phase 1 only checks token-level convergence.
    """
    question, answer = "", ""
    for turn in (example.get("conversations") or []):
        role = str(turn.get("from", "")).lower().strip()
        val  = str(turn.get("value", "")).strip()
        if role == "user" and not question:
            question = val
        elif role == "assistant" and not answer:
            answer = _THINK_RE.sub("", val).strip()
    return question, answer


def load_samples(
    tokenizer,
    max_samples: int,
    max_seq_len: int,
) -> List[Dict]:
    """
    Load the dataset, format as plain QA text, tokenize, truncate.
    Returns a list of {'input_ids': Tensor[T]} dicts (variable length).
    """
    print(f"Loading {DATASET_NAME} ({max_samples} samples) …")
    raw = load_dataset(DATASET_NAME, split="train")
    raw = raw.select(range(min(max_samples, len(raw))))

    eos = tokenizer.eos_token or "<|endoftext|>"
    out = []
    for ex in tqdm(raw, desc="Formatting", leave=False):
        q, a = _extract_qa(ex)
        if not q or not a:
            continue
        # Simple QA format — no <think> scaffolding needed for Phase 1
        text = f"User: {q}\n\nAssistant: {a}{eos}"
        ids  = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 4:
            continue
        out.append({"input_ids": torch.tensor(ids[:max_seq_len], dtype=torch.long)})

    print(f"Kept {len(out)} valid samples after filtering.")
    return out


def collate_batch(
    samples: List[Dict],
    pad_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Pad a list of variable-length samples to the longest in the batch.
    Labels copy input_ids; padding positions are masked with -100 so
    cross-entropy ignores them.
    """
    max_len = max(s["input_ids"].size(0) for s in samples)
    B = len(samples)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels    = torch.full((B, max_len), -100,   dtype=torch.long)
    mask      = torch.zeros(B, max_len, dtype=torch.bool)

    for i, s in enumerate(samples):
        ids = s["input_ids"]
        T   = ids.size(0)
        input_ids[i, :T] = ids
        labels[i, :T]    = ids
        mask[i, :T]      = True

    return {"input_ids": input_ids, "attention_mask": mask, "labels": labels}


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_generate(
    model: BaselineTRMMamba,
    tokenizer,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int = GEN_MAX_TOKENS,
) -> str:
    """
    Run greedy decoding from a formatted prompt string.
    Returns only the newly generated token text (prompt excluded).
    """
    model.eval()

    prefix = f"User: {prompt}\n\nAssistant: "
    ids    = torch.tensor(
        tokenizer.encode(prefix, add_special_tokens=False),
        dtype=torch.long, device=device,
    ).unsqueeze(0)  # [1, T]

    eos_id    = tokenizer.eos_token_id
    generated = []

    for _ in range(max_new_tokens):
        # Left-truncate if context exceeds model capacity
        if ids.size(1) > model.config.max_seq_len:
            ids = ids[:, -model.config.max_seq_len:]

        with autocast_context(device, dtype):
            logits = model(ids)  # [1, T, V]

        next_id  = int(logits[:, -1, :].argmax(dim=-1).item())

        if eos_id is not None and next_id == eos_id:
            break

        generated.append(next_id)
        ids = torch.cat(
            [ids, torch.tensor([[next_id]], device=device)], dim=1
        )

    model.train()
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── Gate tracking ─────────────────────────────────────────────────────────────

@dataclass
class GenRecord:
    step: int
    prompt: str
    output: str
    degenerate: bool
    uwr: float      # unique-word-ratio
    ucc: int        # unique-char-count


@dataclass
class GateTracker:
    ce_history:       List[float]    = field(default_factory=list)
    grad_norm_history: List[float]   = field(default_factory=list)
    vram_step1:       Optional[float] = None
    vram_final:       Optional[float] = None
    gen_records:      List[GenRecord] = field(default_factory=list)

    def record_step(
        self,
        step: int,
        ce: float,
        grad_norm: float,
        device: torch.device,
    ) -> None:
        self.ce_history.append(ce)
        self.grad_norm_history.append(grad_norm)
        current_vram = vram_gb(device)
        if step == 1:
            self.vram_step1 = current_vram
        self.vram_final = current_vram

    def record_generation(
        self, step: int, prompt: str, output: str
    ) -> GenRecord:
        degen, uwr, ucc = analyze_output(output)
        rec = GenRecord(
            step=step, prompt=prompt, output=output,
            degenerate=degen, uwr=uwr, ucc=ucc,
        )
        self.gen_records.append(rec)
        return rec

    def evaluate(self) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """
        Return (gate_pass_dict, diagnostic_values_dict).
        All four gates must be True to clear Phase 2.
        """
        # G1 — final CE
        final_ce = self.ce_history[-1] if self.ce_history else float("inf")

        # G2 — non-degenerate generation at the last recorded step
        if self.gen_records:
            last_step = max(r.step for r in self.gen_records)
            last_recs = [r for r in self.gen_records if r.step == last_step]
            non_degen = sum(1 for r in last_recs if not r.degenerate)
            # Pass if strictly more than half are non-degenerate
            g2_pass = len(last_recs) > 0 and non_degen > len(last_recs) // 2
            last_mean_uwr = (
                sum(r.uwr for r in last_recs) / len(last_recs)
                if last_recs else 0.0
            )
        else:
            g2_pass, last_mean_uwr = False, 0.0

        # G3 — grad norm in final 100 steps
        tail_norms  = self.grad_norm_history[-100:] if self.grad_norm_history else []
        max_norm    = max(tail_norms) if tail_norms else float("inf")

        # G4 — VRAM growth
        vram_delta = (
            (self.vram_final - self.vram_step1)
            if (self.vram_step1 is not None and self.vram_final is not None)
            else float("inf")
        )

        gates = {
            "G1_ce_converged":         final_ce < G1_CE_MAX,
            "G2_generation_coherent":  g2_pass,
            "G3_grad_norm_stable":     max_norm < G3_GRAD_NORM_MAX,
            "G4_vram_stable":          vram_delta < G4_VRAM_DELTA_MAX,
        }
        metrics = {
            "final_ce":      final_ce,
            "max_grad_norm": max_norm,
            "vram_delta_gb": vram_delta,
            "last_mean_uwr": last_mean_uwr,
        }
        return gates, metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda")
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.cuda.reset_peak_memory_stats(device)

    # ── Header ───────────────────────────────────────────────────────────────
    print()
    print("═" * 62)
    print("  Phase 1 Viability Gate — Project Ouroboros")
    print("═" * 62)
    print(f"  steps         : {TRAIN_STEPS}")
    print(f"  samples       : {MAX_SAMPLES}")
    print(f"  seq_len       : {MAX_SEQ_LEN}")
    print(f"  batch×accum   : {BATCH_SIZE}×{GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  lr            : {LR}")
    print(f"  dtype         : {dtype}")
    print()
    print("  Gate thresholds:")
    print(f"    G1 CE       < {G1_CE_MAX}   (random-init ≈ 11.93)")
    print(f"    G2 UWR      > {G2_UNIQUE_WORD_MIN}   (unique-word-ratio at final gen step)")
    print(f"    G3 grad_norm < {G3_GRAD_NORM_MAX}  (max over final 100 steps)")
    print(f"    G4 VRAM Δ   < {G4_VRAM_DELTA_MAX} GB (growth from step 1 to final)")
    print("═" * 62)
    print()

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Tokenizer vocab  : {len(tokenizer):,} tokens")

    # ── Dataset ──────────────────────────────────────────────────────────────
    samples = load_samples(tokenizer, MAX_SAMPLES, MAX_SEQ_LEN)
    if len(samples) < BATCH_SIZE * GRAD_ACCUM:
        sys.exit(
            f"Only {len(samples)} samples loaded — need at least "
            f"{BATCH_SIZE * GRAD_ACCUM}. Check dataset connectivity."
        )
    pad_id = tokenizer.pad_token_id or 0
    print()

    # ── Model ─────────────────────────────────────────────────────────────────
    padded_vocab = pad_vocab_size(len(tokenizer), 128)
    config = BaselineConfig(
        vocab_size=padded_vocab,
        d_model=MODEL_D_MODEL,
        n_groups=MODEL_N_GROUPS,
        n_heads=MODEL_N_HEADS,
        n_kv_heads=MODEL_N_KV_HEADS,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
    )
    model = BaselineTRMMamba(config).to(device=device, dtype=dtype)
    model.train()

    n_params = count_parameters(model)
    print(f"Model            : {n_params / 1e6:.1f}M parameters")
    print(f"Config           : d_model={config.d_model}  n_groups={config.n_groups}"
          f"  heads={config.n_heads}/{config.n_kv_heads}  mlp_hidden={config.mlp_hidden}")
    print(f"Residual layers  : {config.total_residual_layers}")
    print(f"Padded vocab     : {padded_vocab:,}")
    print()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # No LR schedule — constant LR gives the cleanest convergence signal for
    # a smoke test. We want to know if the *architecture* converges, not if
    # the schedule is tuned correctly.
    optimizer, fused_enabled = build_adamw_optimizer(
        model=model,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-8,
        prefer_fused=True,
    )
    print(f"Optimizer        : AdamW ({'fused CUDA kernel' if fused_enabled else 'standard'})")

    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    print()

    # ── Training loop ──────────────────────────────────────────────────────────
    tracker    = GateTracker()
    step       = 0           # optimizer steps (each = GRAD_ACCUM micro-steps)
    micro_step = 0           # individual forward-backward calls
    sample_idx = 0           # cycles through the dataset
    accum_loss = 0.0
    t0         = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    col_header = (
        f"{'Step':>6}  {'CE Loss':>9}  {'Grad Norm':>10}  "
        f"{'VRAM GB':>8}  {'Tok/s':>8}  {'Note'}"
    )
    print(col_header)
    print("─" * 70)

    pbar = tqdm(total=TRAIN_STEPS, desc="Training", dynamic_ncols=True)

    while step < TRAIN_STEPS:
        # ── Micro-batch ───────────────────────────────────────────────────────
        micro_samples = []
        for _ in range(BATCH_SIZE):
            if sample_idx >= len(samples):
                sample_idx = 0  # cycle
            micro_samples.append(samples[sample_idx])
            sample_idx += 1

        batch     = collate_batch(micro_samples, pad_id)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)

        # ── Forward ───────────────────────────────────────────────────────────
        with autocast_context(device, dtype):
            logits = model(input_ids, attention_mask=attn_mask)  # [B, T, V]

        # Next-token cross-entropy (shift by 1)
        shift_logits = (
            logits[:, :-1, :].contiguous().view(-1, config.vocab_size).float()
        )
        shift_labels = labels[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        # ── Backward ──────────────────────────────────────────────────────────
        scaled_loss = loss / GRAD_ACCUM
        if dtype == torch.float16:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        accum_loss += loss.detach().item()
        micro_step += 1

        if micro_step % GRAD_ACCUM != 0:
            continue  # accumulate more gradients before stepping

        # ── Optimizer step ────────────────────────────────────────────────────
        if dtype == torch.float16:
            scaler.unscale_(optimizer)

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        )

        if dtype == torch.float16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        step += 1

        mean_ce    = accum_loss / GRAD_ACCUM
        accum_loss = 0.0

        tracker.record_step(step, mean_ce, grad_norm, device)
        pbar.update(1)
        pbar.set_postfix(ce=f"{mean_ce:.3f}", gn=f"{grad_norm:.3f}")

        # ── Per-step console log ──────────────────────────────────────────────
        if step % 10 == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            tok_s   = (step * BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN) / max(elapsed, 1e-6)
            vram    = vram_gb(device)
            note    = ""
            if mean_ce < G1_CE_MAX:
                note = "← G1 threshold crossed"
            tqdm.write(
                f"{step:>6}  {mean_ce:>9.4f}  {grad_norm:>10.4f}  "
                f"{vram:>8.3f}  {tok_s:>8.0f}  {note}"
            )

        # ── Generation check ──────────────────────────────────────────────────
        if step % GEN_EVERY_STEPS == 0 or step == TRAIN_STEPS:
            tqdm.write(f"\n  ── Generation @ step {step} ──")
            for prompt in GEN_PROMPTS:
                out = greedy_generate(model, tokenizer, prompt, device, dtype)
                rec = tracker.record_generation(step, prompt, out)
                flag = "  ⚠ DEGENERATE" if rec.degenerate else ""
                tqdm.write(f"  Q: {prompt}")
                tqdm.write(f"  A: {out[:140]}")
                tqdm.write(
                    f"     uwr={rec.uwr:.3f}  ucc={rec.ucc}{flag}"
                )
            tqdm.write("")

    pbar.close()

    # ── Gate evaluation ───────────────────────────────────────────────────────
    total_time = time.perf_counter() - t0
    gates, metrics = tracker.evaluate()

    vram_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    print()
    print("═" * 62)
    print("  Phase 1 Gate Results")
    print("═" * 62)
    print()

    def gate_line(key: str, label: str, measured: str) -> None:
        result = "PASS ✓" if gates[key] else "FAIL ✗"
        print(f"  {key}  {label:<35}  {measured:<20}  {result}")

    gate_line(
        "G1_ce_converged",
        f"CE < {G1_CE_MAX}",
        f"final CE = {metrics['final_ce']:.4f}",
    )
    gate_line(
        "G2_generation_coherent",
        f"UWR > {G2_UNIQUE_WORD_MIN} (last gen step)",
        f"mean UWR = {metrics['last_mean_uwr']:.3f}",
    )
    gate_line(
        "G3_grad_norm_stable",
        f"grad_norm < {G3_GRAD_NORM_MAX} (last 100 steps)",
        f"max = {metrics['max_grad_norm']:.4f}",
    )
    gate_line(
        "G4_vram_stable",
        f"VRAM Δ < {G4_VRAM_DELTA_MAX} GB",
        f"Δ = {metrics['vram_delta_gb']:.3f} GB",
    )

    print()
    print(f"  Total time  : {total_time / 60:.1f} min")
    print(f"  Peak VRAM   : {vram_peak:.2f} GB")
    print()

    all_pass = all(gates.values())

    if all_pass:
        print("  ╔══════════════════════════════════════════════════════╗")
        print("  ║  ALL GATES PASSED                                    ║")
        print("  ║  Architecture is viable. Proceed to Phase 2.         ║")
        print("  ╚══════════════════════════════════════════════════════╝")
    else:
        failed = [k for k, v in gates.items() if not v]
        print("  ╔══════════════════════════════════════════════════════╗")
        print(f"  ║  GATES FAILED: {', '.join(failed)}")
        print("  ║  DO NOT proceed to Phase 2. Diagnose below.          ║")
        print("  ╚══════════════════════════════════════════════════════╝")
        print()
        _print_diagnostics(gates, metrics, tracker)

    print("═" * 62)
    print()
    sys.exit(0 if all_pass else 1)


def _print_diagnostics(
    gates: Dict[str, bool],
    metrics: Dict[str, float],
    tracker: GateTracker,
) -> None:
    """Print targeted diagnostic hints for each failed gate."""

    if not gates["G1_ce_converged"]:
        ce = metrics["final_ce"]
        # Check if loss is at least decreasing
        if len(tracker.ce_history) > 50:
            early = sum(tracker.ce_history[:20]) / 20
            late  = sum(tracker.ce_history[-20:]) / 20
            drop  = early - late
        else:
            early, late, drop = float("nan"), float("nan"), 0.0

        print("  Diagnostic G1: CE did not converge below 3.5")
        print(f"    Early mean CE : {early:.4f}")
        print(f"    Late mean CE  : {late:.4f}")
        print(f"    Total drop    : {drop:.4f}")
        if drop < 0.5:
            print("    → Loss is barely moving. Check:")
            print("      - Data pipeline (are labels correctly shifted?)")
            print("      - Learning rate (try 5e-4 if loss is flat from step 1)")
            print("      - That mamba_ssm kernels compiled correctly (no silent fallback)")
        elif drop > 0:
            print("    → Loss IS decreasing but not fast enough in 300 steps.")
            print("      - Raise LR to 5e-4 in the GATE constants and re-run.")
            print("      - Extend TRAIN_STEPS to 500 for a slower dataset.")
        print()

    if not gates["G2_generation_coherent"]:
        print("  Diagnostic G2: Generation is degenerate (loops/repetition)")
        print(f"    Mean UWR at final step : {metrics['last_mean_uwr']:.3f}")
        print("    → Comma/word loops are a clear sign of CE non-convergence.")
        print("      G1 should also be failing. If not, the model learned something")
        print("      but generation code has a bug (check EOS handling, autocast).")
        print()

    if not gates["G3_grad_norm_stable"]:
        print("  Diagnostic G3: Gradient norm is exploding")
        print(f"    Max grad_norm in final 100 steps : {metrics['max_grad_norm']:.4f}")
        print("    → Check for NaN in activations (add torch.isnan checks).")
        print("    → The clip threshold MAX_GRAD_NORM=1.0 should prevent this —")
        print("      if grad_norm is still > 10.0 post-clip, something is wrong")
        print("      with the backward graph (check .detach() placement).")
        print()

    if not gates["G4_vram_stable"]:
        delta = metrics["vram_delta_gb"]
        print("  Diagnostic G4: VRAM is growing across steps")
        print(f"    VRAM growth : {delta:.3f} GB over {TRAIN_STEPS} steps")
        print("    → The computation graph is being retained between steps.")
        print("      Common causes:")
        print("      1. loss.item() called BEFORE .backward() — use .detach() instead")
        print("      2. Storing tensor objects (not scalars) in lists")
        print("      3. A .retain_graph=True call somewhere in the codebase")
        print()


if __name__ == "__main__":
    main()
