# Project Ouroboros — Master Blueprint

> **Thread-resume header.** In any new session: read Part 0 first, then jump to
> the current stage. This file is the single source of truth for architecture,
> status, bugs, and next actions. Append new findings — never rewrite history.

---

## Part 0 — Quick-Resume Context

### What this project is
Novel hybrid Transformer-Mamba language model ("TRM-Mamba", 1:7 ratio) pre-trained
from scratch on FineWeb-Edu, then fine-tuned for chain-of-thought reasoning. Target:
a ~1B parameter model with test-time compute scaling via recursive inference (SERF).
Zero budget — Kaggle Dual T4 + Google TRC application.

### Root failure on record
`checkpoint-950` (archived, do not use): randomly-initialized model trained directly
on SFT data. Output: comma-loops and "the the the". Three root causes:
1. No pre-training — SFT cannot teach language from random noise
2. EMA initialized to zeros — bias-correction diverged at step 1
3. Novel components stacked without per-component validation

### Current status

| Stage | Name | Status | Gate |
|---|---|---|---|
| 0 | Architecture & Viability | ✅ COMPLETE | All 4 gates passed |
| 1 | Pre-training | 🟡 IN PROGRESS on Kaggle Dual T4 — step ~9000 / 61,036 (~14.7%) | val_ce < 3.0 + UWR > 0.05 |
| 2 | SFT | 🔴 BLOCKED — Bug 3 (collate prompt masking) not fixed | Fix Bug 3, then run |
| 3 | Recursive Inference (Coconut-Ouroboros) | ⬜ NOT STARTED | Stage 2 answer val_ce < 1.5 |
| 4 | GRPO | ⬜ NOT STARTED | Stage 3 gate |
| 5 | Quantization | ⬜ NOT STARTED | Stage 4 or 3 gate |

### Immediate next action

Stage 1 is running. While it runs:
 
**Action 1 — Fix Bug 3 in `train_sft.py`** (prompt masking in collate).
See Part 7 → Bug 3 for the exact two-change fix.
Without this, Stage 2 val_ce includes prompt supervision and cannot be
used as a Stage 3 baseline.
 
**Action 2 — Increase `--shuffle_buffer` on next Kaggle session restart.**
Multiple 2–3 step spike clusters have now appeared regularly (see Live Observations).
Add `--shuffle_buffer 20000` to the resume command.
 
**Action 3 — Stage 3 architecture decisions are settled** (see Appendix B update).
No code action needed yet. `recursive_finetune.py` to be written after Stage 2 completes.
Use `stage3_agent_prompt.md` as the agent prompt.
 
```bash
# Stage 1 resume (next session) — add shuffle_buffer:
python pretrain.py \
  --preset nano \
  --resume_from runs/stage1 \
  --shuffle_buffer 20000 \
  --session_timeout_hours 12.0

# Stage 2 launch (after Stage 1 banner: val_ce < 3.0 AND mean_uwr > 0.05):
python train_sft.py \
  --preset nano \
  --resume_from runs/stage1/checkpoint-XXXXXXX \
  --ema_decay 0.995 \
  --dataset_mix stratos   # upgrade to full after first successful run
 
# Stage 3 launch (after Stage 2 answer val_ce < 1.5):
python recursive_finetune.py \
  --preset nano \
  --resume_from runs/stage2/checkpoint-XXXXXXX \
  --n_latent 1 \
  --stage2_val_ce <record_here> \
  --output_dir runs/stage3_k1
```

---

## Part 1 — Root Failure Record

| Cause | Effect | Fix |
|---|---|---|
| No pre-training | Model must learn language + reasoning simultaneously from 17k examples | Pre-train on 2B tokens first |
| EMA init = zeros | Bias-correction `ema/(1-decay^t)` blows up at t=1 | `pretrain.py`: EMA init from live weights |
| Too many new components | Cannot isolate failure | One change at a time, gate each |
| No non-recursive baseline | No reference to compare against | `baseline_trm_mamba.py` now the baseline |

**Loss=54.67 explained (not a bug):** `DeepSupervisedSERF` summed CE across 8 recursive
steps with gamma=0.8 → weight sum ≈4.57. Random-init CE = ln(151665) ≈ 11.93.
So 4.57 × 11.93 ≈ 54.5. Exactly right for random init.

---

## Part 2 — Architecture Specification

### Block structure

```
TokenEmbedding(vocab_size=151_680, d_model)
└─ n_groups × TRMMambaGroup:
   ├─ 1 × TRMBlock:
   │   ├─ RMSNorm + GQA Attention (RoPE) + residual
   │   └─ RMSNorm + SwiGLU MLP + residual
   └─ 7 × MambaLayer:          ← 1:7 ratio, hard-coded, DO NOT change
       └─ RMSNorm + mamba_ssm.Mamba + residual
FinalRMSNorm
LM Head (weight tied to TokenEmbedding)
```

### Configuration presets (verified against terminal_log.md)

| Preset | d_model | n_groups | n_heads | n_kv_heads | mlp_hidden | Params |
|---|---|---|---|---|---|---|
| nano | 512 | 1 | 8 | 4 | 1408 | 92.5M |
| small | 1024 | 2 | 16 | 8 | 2816 | ~270M |
| medium | 2048 | 2 | 16 | 8 | 5632 | ~760M |

**Nano checksum (verified Stage 0):**
```
parameters          = 92,477,440      (tied-embedding dedup working)
mlp_hidden          = 1408            = ceil((8/3×512)/64)×64
total_residual_layers = 9             = 1×(2+7)
initial_loss        = 11.9904         (theoretical 11.9295, within 0.5%)
grad_norm_total     = 6.4242
```

**Fixed config values (all presets):**
```python
vocab_size            = 151_680       # ceil(151_665/128)×128 — Qwen2.5-0.5B padded
rope_theta            = 1_000_000.0   # Qwen2.5 convention
mamba_d_state         = 16
mamba_d_conv          = 4
mamba_expand          = 2
max_seq_len           = 2048
dropout               = 0.0           # pre-training; 0.1 for SFT if overfitting
rms_norm_eps          = 1e-5
tie_embeddings        = True
mlp_hidden            = ceil((8/3 × d_model) / 64) × 64
total_residual_layers = n_groups × (2 + 7)
```

### Key implementation notes (from verified baseline_trm_mamba.py)

**RoPE** — Non-interleaved. Cache rebuilt lazily on device/dtype change. Theta=1e6.

**GQA** — `repeat_interleave` applied after RoPE on compact KV tensors, then expanded.
`n_repeats = n_heads / n_kv_heads`. Verified correct.

**Padding + causal attention** — SDPA raises on `is_causal=True` + custom mask. When
`attention_mask` is provided: build combined causal+padding additive bias, call with
`is_causal=False`. Verified in Stage 0 smoke test.

**Mamba padding** — Padded positions zeroed: `h = h * mask` before AND after Mamba
forward. Mask convention: `True` = valid token.

**Residual scaling (GPT-2 style)** — Residual writers (`o_proj`, `mlp.down_proj`,
`mamba.out_proj`) scaled by `0.02 / sqrt(total_residual_layers)` after all base inits.
Two-pass to avoid double-initializing tied weights.

---

## Part 3 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Tokenizer | Qwen2.5-0.5B; vocab_size=151_680 everywhere, no exceptions |
| Mamba version | `mamba_ssm.Mamba` (original). Mamba2 deferred to Stage 3+ |
| Stage 1 dataset | HuggingFaceFW/fineweb-edu sample-10BT, streamed |
| Stage 2 dataset | Bespoke-Stratos-17k (40%) + MetaMathQA (30%) + OpenHermes-2.5 (30%) |
| Hub repo | WeirdRunner/Ouroboros (private) |
| Pre-training strategy | Train from scratch. Apply to Google TRC immediately (async) |
| Recursion timing | Post-SFT only. Never train recursion from random init |
| GRPO implementation | TRL GRPOTrainer. Never reimplement from scratch |
| Stage 3 recursion mechanism | Coconut-Ouroboros (latent hidden-state injection, K=1→4→16 curriculum). NOT the TRM EMA-loop approach. See stage3_agent_prompt.md. |
| TRM paper applicability | TRM is an encoder for grid tasks; not directly applicable to autoregressive LLMs. Inspiration only. |
| Stage 3 Mamba SSM property | SSM state accumulates across latent passes, giving richer scratch-memory than Transformer-Coconut. This is a key architectural advantage. |

---

## Part 4 — File Registry

### Active files

| File | Stage | Status | Notes |
|---|---|---|---|
| `baseline_trm_mamba.py` | 0 | ✅ COMPLETE | All smoke checks passed |
| `viability_gate.py` | 0 | ✅ COMPLETE | All 4 gates passed |
| `pretrain.py` | 1 | 🟡 SCRIPT VERIFIED, not run on Kaggle | See Stage 1 |
| `train_sft.py` | 2 | 🔴 Bug 3 open — do not run | See Part 7 |
| `BLUEPRINT.md` | — | Living document | This file |
| `terminal_log.md` | — | Verified terminal outputs | Append only |
| `recursive_finetune.py` | 3 | ⬜ NOT CREATED | Use stage3_agent_prompt.md to generate |
| `stage3_agent_prompt.md` | 3 | ✅ CREATED | Complete Coconut-Ouroboros agent prompt |

### Renamed files (reflect manually)

| Was | Now |
|---|---|
| `Ouroboros_Blueprint_v3.md` | `BLUEPRINT.md` |
| `ouroboros_blueprint_v2.md` | archived / delete |
| `phase1_viability_gate.py` | `viability_gate.py` |
| `train_sft_phase2.py` | `train_sft.py` |

### Deprecated (do not use)

| File | Why |
|---|---|
| `recursive_trm_mamba.py` | Produced checkpoint-950 failure |
| `train_sft_deep_supervision.py` | Deep supervision on random init; obsolete |
| `inference_trm_mamba.py` | Tied to old recursive model |
| `architecture.py` | Superseded by `baseline_trm_mamba.py` |

---

## Part 5 — Stage Definitions

---

### Stage 0 — Architecture & Viability Gate ✅ COMPLETE

**Files:** `baseline_trm_mamba.py`, `viability_gate.py`

**Completed checks:**
- [x] Model instantiates on CPU and CUDA without error
- [x] Initial loss = 11.9904 (within ±0.5 nats of theoretical 11.9295)
- [x] `loss.backward()` — no NaN gradients, no missing gradients
- [x] `param_count = 92,477,440` — tied-embedding dedup verified
- [x] Padding mask path (`is_causal=False`) exercised and correct
- [x] G1: CE converged 12.04 → 2.00 in 300 steps (threshold < 3.5) ✅
- [x] G2: Mean UWR = 0.573 at step 300 (threshold > 0.10) ✅
- [x] G3: Max grad_norm = 4.03 in final 100 steps (threshold < 10.0) ✅
- [x] G4: VRAM delta = 0.000 GB (threshold < 1.0 GB) ✅

---

### Stage 1 — Pre-training 🟡 IN PROGRESS

**Gate:** Stage 0 all gates passed ✅

**Script:** `pretrain.py`
**Hardware:** Kaggle Dual T4, DDP world_size=2
**Status:** Running — step ~9000 / 61,036 (~14.7% complete). See terminal_log.md for full output.

**Bug 5 fix confirmed working:**
- Session 5 resumed from `checkpoint-0002000` (step=2000, tokens=65.5M) — clean local load ✅
- `checkpoint-0003000` saved locally first, then uploaded to Hub (commit=5e2ba2b8) ✅
- Local-first + Hub fire-and-forget is now the live behaviour.

**Dry-run checklist (completed):**
- [x] No import errors; FakeMamba smoke test passes (epoch_offset=7, loss decreasing)
- [x] Initial loss ≈ 11.93 (actual 11.98); decreases over 20 steps
- [x] Checkpoint round-trip verified (step=20 restored correctly)
- [x] `ema_backbone_state_dict` contains `lm_head.weight` alias
- [x] Val buffer builds (2M tokens, 1,887 docs, document-disjoint from training)
- [x] Epoch offset=228 (non-zero for epoch 0) ✓

**Key hyperparameters (nano, 2B tokens):**
```
batch_size=8, grad_accum=4, chunk_size=1024 → 32,768 tokens/step
total_steps = 61,036
lr=6e-4, warmup=200 steps, cosine decay to 6e-5
weight_decay=0.1, max_grad_norm=1.0, ema_decay=0.995
save_every=1000, val_every=500, gen_every=500
DDP throughput: ~5,800 tok/s. Full run ETA: ~9.4h
```

**Live observations (step 9000):**
- Val CE declining but plateaued: 6.38 → 5.85 → 5.68 → 5.56 → 5.48 → 5.42 → 5.36 → 5.34 → 5.32 → 5.30 → 5.30 → 5.31 → 5.30 → 5.31 → 5.30 → 5.29 → 5.29 ✅ (still decreasing but very slowly)
- **⚠ Val CE plateau:** Essentially flat 5.29–5.31 from step 4500–9000 (~147M–295M tokens, 4500 steps). Train CE continues declining (4.59→4.20), gap widening. Not alarming at 14.7% of token budget; expected to break through with more data. Monitor at step 10000.
- VRAM flat at 2.035 GB throughout — no graph retention ✅
- **⚠ UWR degradation:** Mean UWR chronically low since step 5000 (0.12–0.19 at most callbacks). "The capital of the city of the city" repetition loop recurring. Val CE is the primary signal and is healthy; UWR is a lagging indicator expected to recover as CE drops.
- Spike rate: 44 spikes in 9000 steps = 0.49% — within 10% threshold ✅
- **⚠ Spike clusters now regular:** (3570–3573), (6397–6400), (7971–7987–8000), (8060/8131), (8797/8847/8900). Multiple recurring clusters. Increase `--shuffle_buffer 20000` on next session resume.
- Code prompt degenerate (digit/letter loops) — expected, FineWeb-Edu has no code.

**Checkpoint status (as of step 9000):**
- Local (keep_last=3): checkpoint-6000, checkpoint-7000, checkpoint-8000
- Hub: checkpoint-3000 through checkpoint-8000

**Success criteria:**
- [ ] `val_ce < 3.0` AND `mean_uwr > 0.05` (script prints banner when met)
- [ ] No loss spikes > smoothed+0.5 for > 5% of steps
- [ ] Coherent text completions in gen callback

**Next action after Stage 1:**
```bash
python train_sft.py \
  --preset nano \
  --resume_from runs/stage1/checkpoint-XXXXXXX \
  --ema_decay 0.995
```
Fix Bug 3 in `train_sft.py` first (see Part 7).

**Loss curve summary (full run to step 9000):**

| Step | Train CE | Smoothed | Val CE | Tokens Seen | Notes |
|---|---|---|---|---|---|
| 1 | 11.98 | 11.98 | — | 32k | Random init |
| 500 | 5.46 | 5.78 | 6.38 | 16.4M | Phrases forming |
| 1000 | 4.97 | 5.14 | 5.85 | 32.8M | Real sentences |
| 1500 | 4.89 | 4.91 | 5.68 | 49.2M | Coherent prose |
| 2000 | — | — | 5.56 | 65.5M | Resumed (ckpt-2000) |
| 2500 | 4.57 | 4.68 | 5.48 | 82.0M | Consistent drop |
| 3000 | 4.48 | 4.60 | 5.42 | 98.3M | Hub sync working |
| 3500 | 4.42 | 4.58 | 5.36 | 114.7M | Spike cluster ⚠ |
| 4000 | 4.46 | 4.50 | 5.34 | 131.1M | Val drop slowing |
| 4500 | 4.59 | 4.50 | 5.32 | 147.5M | |
| 5000 | 4.47 | 4.45 | 5.30 | 163.8M | Val plateau begins ⚠ |
| 5500 | 4.37 | 4.39 | 5.30 | 180.2M | Flat |
| 6000 | 4.31 | 4.36 | 5.31 | 196.6M | Val ticked up slightly |
| 6500 | 4.32 | 4.35 | 5.30 | 213.0M | |
| 7000 | 4.45 | 4.34 | 5.31 | 229.4M | |
| 7500 | 4.32 | 4.32 | 5.30 | 245.8M | |
| 8000 | 4.92 | 4.31 | 5.29 | 262.1M | Spike cluster (7971/7987/8000) |
| 8500 | 4.33 | 4.30 | 5.29 | 278.5M | Plateau continues |

---

### Stage 2 — SFT 🔴 BLOCKED

**Gate:** Stage 1 `val_ce < 3.0` AND `mean_uwr > 0.05`

**Script:** `train_sft.py` — Bug 3 open. See Part 7.

**Agent prompt for fixing train_sft.py:** See Appendix A.

**Target format:**
```
User: {question}
Assistant: <think>
{reasoning_chain}
</think>
{final_answer}
```

**Hyperparameters:**
```
lr=3e-5 to 1e-4 (10× lower than pre-training)
warmup=100 steps, cosine decay, num_epochs=3 max
batch_size=2, grad_accum=8, max_seq_len=512
ema_decay=0.995
```

**Success criteria:**
- [ ] `val_ce < 1.5`
- [ ] Model generates `<think>` blocks before answering
- [ ] Semantically coherent answers on GEN_PROMPTS

---

### Stage 3 — Incremental Recursion ⬜ NOT STARTED

**Gate:** Stage 2 `val_ce < 1.5`

**Strategy:** Fine-tune the Stage 2 checkpoint into recursion. NEVER train recursion
from a randomly initialized model — this is what killed checkpoint-950.

```
Step 3.1: Fine-tune Stage 2 ckpt with n_loops=2
          Gate: CE returns within 5% of Stage 2 baseline CE
Step 3.2: n_loops=4. Same gate.
Step 3.3: n_loops=8 if budget permits.
```

**Critical EMA warm-start (fixes the original bug):**
```python
# CORRECT — warm-start from current hidden state
ema_state = base_hidden_state.detach().clone()

# WRONG — caused checkpoint-950 divergence
ema_state = torch.zeros_like(base_hidden_state)
```

**Recursive forward (inference-time wrapper, no architecture change):**
```python
@torch.no_grad()
def recursive_generate(model, input_ids, n_loops=4, ema_decay=0.9):
    logits, hidden = model(input_ids, return_hidden=True)
    ema = hidden.detach().clone()                     # warm start
    for step in range(1, n_loops):
        correction = 1.0 - ema_decay ** step
        loop_input = ema / correction                 # bias-corrected
        logits, hidden = model(loop_input_as_embeddings, ...)
        ema = ema_decay * ema + (1 - ema_decay) * hidden
    return logits
```

**Agent prompt:** See Appendix B.

---

### Stage 4 — GRPO ⬜ NOT STARTED

**Gate:** Stage 3 quality does not degrade vs Stage 2 (CE within 5%).

**Implementation:** TRL `GRPOTrainer` — DO NOT reimplement GRPO from scratch.

**Reward functions (implement in this order):**
1. Format: +0.1 if `<think>...</think>` tags correctly open/close
2. Correctness: +1.0 if answer verified (SymPy for math, E2B sandbox for code)
3. Length penalty (optional): mild negative for traces > 2× median length

**Compute (T4, 16GB):**
- 1.5B model in 4-bit (bitsandbytes): ~3GB
- G=4 GRPO rollouts: ~6GB additional
- Use unsloth + gradient checkpointing. Drop to G=2 if OOM.

---

### Stage 5 — Quantization ⬜ NOT STARTED

**Gate:** Stage 4 complete (or Stage 3 if skipping GRPO).

**Method:** Quamba (Hadamard + DLS). Standard 4-bit breaks Mamba models due to
activation outliers in the linear recurrence. Quamba (2024) resolves this.

**Target:** INT4/INT8 artifact for Jetson or similar edge device.

---

## Part 6 — Checkpoint Format (canonical across all stages)

Every checkpoint directory must contain exactly:
```
checkpoint-NNNNNNN/
  training_state.pt             ← full training state for resume
  resolved_backbone_config.json ← BaselineConfig as JSON for inference
```

**training_state.pt required keys:**
```python
{
    "step":                    int,
    "epoch":                   int,
    "chunks_in_epoch":         int,   # Stage 1 only — for data-position resume
    "tokens_processed":        int,   # Stage 1 only
    "model_state_dict":        dict,
    "ema_backbone_state_dict": dict,  # EMA shadow + lm_head.weight alias
    "optimizer":               dict,
    "scheduler":               dict,
    "scaler":                  dict or None,
    "ema":                     dict,
    "backbone_config":         dict,  # asdict(BaselineConfig)
    "val_ce":                  float or None,
}
```

**lm_head.weight alias rule:** When `tie_embeddings=True`, `named_parameters()` yields
one tensor for both embedding and lm_head. `ema_backbone_state_dict` MUST explicitly
contain `"lm_head.weight"` as an alias to `"token_embedding.weight"`. Verified in
pretrain.py's `save_checkpoint`.

**Hub push protocol (corrected — see Bug 5):**
1. Write fully to `checkpoint-NNNNNNN.tmp`
2. Rename `.tmp` → `checkpoint-NNNNNNN` immediately (local finalize — always happens)
3. Prune old local checkpoints
4. Attempt Hub push from `checkpoint-NNNNNNN` — fire-and-forget, warn on failure, never block

---

## Part 7 — Known Bugs and Issues

### Bug 3 — HIGH: `train_sft.py::collate` applies loss to all tokens including the prompt
 
**Status:** 🔴 NOT FIXED. Apply before any Stage 2 run.
 
**Why this also blocks Stage 3:**
The Stage 3 gate check compares answer val_ce with the Stage 2 baseline.
If the Stage 2 baseline was measured with prompt tokens in the loss, it is
artificially inflated and the gate threshold is meaningless.
Fix Bug 3 first, record a clean answer-only Stage 2 val_ce, then use that
as `--stage2_val_ce` in `recursive_finetune.py`.
 
**Fix — two changes required:**
 
Change 1 — in `load_and_tokenize`, record prompt length:
```python
# Before (current — broken):
samples.append({"input_ids": torch.tensor(ids[:max_seq_len], dtype=torch.long)})
 
# After (fixed):
prefix_text = f"User: {q}\n\nAssistant: "
if r:
    prefix_text += f"<think>\n"
prefix_ids  = tokenizer.encode(prefix_text, add_special_tokens=False)
prompt_len  = len(prefix_ids)
samples.append({
    "input_ids":  torch.tensor(ids[:max_seq_len], dtype=torch.long),
    "prompt_len": prompt_len,
})
```
 
Change 2 — in `collate`, mask prompt tokens in labels:
```python
# Before (current — broken):
labels[idx, :length] = ids   # all tokens supervised
 
# After (fixed):
pl = min(sample.get("prompt_len", 0), length)
labels[idx, pl:length] = ids[pl:]   # only response tokens supervised
```
 
Also apply the same fix in `load_mixed_dataset` for the MetaMathQA and
OpenHermes extractors (each has its own prefix format).

---

### Bug 6 — MEDIUM: Stage 2 val_ce measures full-sequence CE, not answer-only CE
 
**Symptom (potential):** If Bug 3 is fixed mid-training and val_ce is recorded
both before and after, the values are not comparable. The pre-fix val_ce
includes prompt supervision; the post-fix val_ce does not.
 
**Impact on Stage 3:** The `--stage2_val_ce` argument to `recursive_finetune.py`
must be the answer-only val_ce from a Stage 2 run with Bug 3 fixed.
Do NOT use val_ce from the Session 3 dry-run (which used the buggy collate).
 
**Fix:** Record Stage 2 final val_ce only after Bug 3 is confirmed fixed.
Add a note in terminal_log.md at the start of each Stage 2 run indicating
whether Bug 3 is fixed.
 
**Status:** 🟡 PENDING — becomes relevant when Stage 2 produces a checkpoint.

---

## Part 8 — Compute Plan

| Stage | Platform | Estimate | Notes |
|---|---|---|---|
| 0 | Colab free / local | ~15 min | ✅ Done |
| 1 | Kaggle T4 | ~18.5h (nano, 2B tokens) | Apply TRC immediately (async) |
| TRC application | — | 5 min | sites.research.google/trc |
| 2 | Kaggle T4 or TRC | ~2–4h | 3 epochs × ~50k samples |
| 3 | TRC preferred | ~4–8h | Incremental, gated per n_loops |
| 4 | TRC + unsloth | ~8–12h | GRPO G=4 rollouts |
| 5 | Local / Jetson | ~2h | Post-training quantization |

---

## Part 9 — References

- **Mamba** (Gu & Dao, 2023) — linear-time SSM design rationale
- **Jamba** (AI21 Labs, 2024) — 1:7 TRM-Mamba at 52B; our 1B is a miniaturized Jamba
- **GPT-2** (Radford et al., 2019) — residual scaling formula source
- **RoPE** (Su et al., 2021) — reference in LLaMA 2 codebase
- **SwiGLU** (Shazeer, 2020) — `ceil((8/3 × d_model) / 64) × 64` hidden_dim
- **GQA** (Ainslie et al., 2023) — grouped query attention
- **FineWeb-Edu** (HuggingFaceTB, 2024) — https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **DeepSeek-R1** (2025) — GRPO reward functions and implementation (Appendix)
- **Quamba** (2024) — post-training quantization for Mamba models
- **build-nanoGPT** (Karpathy) — https://github.com/karpathy/build-nanogpt
- **TRL GRPOTrainer** — https://huggingface.co/docs/trl

---

## Appendix B — Stage 3 Agent Prompt

> The original Appendix B (EMA hidden-state loop) is **superseded and incorrect**.
> It is retained below as an archived note. The correct implementation is in
> `stage3_agent_prompt.md` (a separate file created during the 2026-04-03 session).
 
### Why the original Appendix B approach was wrong
 
The Samsung TRM paper (arXiv:2510.04871) describes an **encoder** model for
fixed-size grid tasks (ARC-AGI, Sudoku, Mazes). Its key properties:
 
- **Encoder architecture**: Bidirectional, processes the entire grid simultaneously.
  Not an autoregressive decoder.
- **Fixed-size I/O**: Input and output have the same shape. Recursion refines
  the whole answer grid in-place.
- **Ground-truth at every loop**: Deep supervision works because the correct
  grid is known at every recursion step.
- **EMA of model weights** (not hidden states): TRM uses weight EMA for
  generalisation, not for the recursion mechanism itself.
 
Applying the "EMA-loop hidden states" approach to an autoregressive LLM:
- Has no gradient signal — the model gets no feedback saying "use these loops".
- Does not create new information — averaging hidden states across loops
  is a linear operation on the same computation.
- The deep supervision via gamma weights requires ground-truth at each step,
  which text generation does not have.
 
### Correct approach: Coconut-Ouroboros
 
Based on Meta's Coconut paper (arXiv:2412.06769), adapted for TRM-Mamba.
 
**Core mechanism**: Replace `<think>...</think>` reasoning tokens with K
*latent thought positions* where the last hidden state is injected directly
as the next position's input embedding (bypassing the token embedding table).
 
**Mamba SSM advantage**: During each latent pass, the Mamba SSM recurrent
state propagates a compressed O(d_state) summary of all previous positions.
Pure Transformer-Coconut only has attention at the latent position; our
Mamba-Coconut accumulates a persistent scratch-memory in the SSM state
across all K passes. This is a genuine architectural advantage.
 
**Curriculum (3 sub-stages)**:
 
| Sub-stage | K | Resume from | Output dir | Gate |
|---|---|---|---|---|
| 3.1 | 1 | Stage 2 final ckpt | runs/stage3_k1 | answer val_ce ≤ stage2 × 1.05 |
| 3.2 | 4 | Stage 3.1 final ckpt | runs/stage3_k4 | answer val_ce ≤ stage2 × 1.05 |
| 3.3 | 16 | Stage 3.2 final ckpt | runs/stage3_k16 | answer val_ce ≤ stage2 × 1.05 |
 
**Key implementation files**:
- `stage3_agent_prompt.md` — complete coding agent prompt (created 2026-04-03)
- `recursive_finetune.py` — to be generated by coding agent from above prompt
- Requires two new methods in `baseline_trm_mamba.py`:
  `forward_with_hidden()` and `forward_from_embeddings()`
 
**Success criterion**: answer val_ce at K=16 ≤ stage2 answer val_ce × 1.05,
with coherent generation on GEN_PROMPTS_STAGE3. Print banner and record in terminal_log.md.
 
### Archived: Original Appendix B (for reference only — do not implement)
 
The original approach proposed:
- `RecursiveSERF` wrapper with n_loops forward passes
- EMA of hidden states: `ema = decay*ema + (1-decay)*hidden`
- Deep supervision with gamma=0.8 weights across loops
- Two new backbone methods: `forward_with_hidden` and `forward_from_embeddings`
 
The `forward_with_hidden` and `forward_from_embeddings` methods ARE still
needed (for Coconut-Ouroboros), so those signatures are retained.
The `RecursiveSERF` wrapper and EMA-loop mechanism are discarded.

---

*End of Appendix B. Appendix C (Stage 4 GRPO) will be written when Stage 3 gate is passed.*
