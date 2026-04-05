# Project Ouroboros — Master Blueprint

> **Thread-resume header.** In any new session: read Part 0 first, then jump to
> the current stage. This file is the single source of truth for architecture,
> status, bugs, and next actions. Append new findings — never rewrite history.

---

## Part 0 — Quick-Resume Context

### What this project is
Novel hybrid Transformer-Mamba language model ("TRM-Mamba", 1:7 ratio) pre-trained
from scratch on FineWeb-Edu, then fine-tuned for chain-of-thought reasoning using
R1-distilled datasets. Target: prove the architecture outperforms comparably-sized
models at nano scale (92.5M), then scale to medium/large on TRC. Test-time compute
scaling via Coconut-Ouroboros recursive inference (SERF). Zero budget — Kaggle Dual
T4 + Google TRC application.

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
| 1 | Pre-training | 🟡 IN PROGRESS — step ~15900 / 61,036 (~26.1%) | val_ce < 3.0 + UWR > 0.05 |
| 2 | SFT | ✅ READY — blocked only on Stage 1 gate | Fix confirmed; see Stage 2 |
| 3 | Recursive Inference (Coconut-Ouroboros) | ⬜ NOT STARTED | Stage 2 answer val_ce < 1.5 |
| 4 | GRPO | ⬜ NOT STARTED | Stage 3 gate |
| 5 | Quantization | ⬜ NOT STARTED | Stage 4 or 3 gate |

### Immediate next actions

Stage 1 is running. While it runs:

**Action 1 — Stage 2 is ready.** No code changes needed. `train_sft.py` is fully
verified. Launch immediately after Stage 1 gate is met.

**Action 2 — shuffle_buffer 20000 already active** in current Session 6 run.
No change needed on resume.

**Action 3 — Hard decision point: step 30000.**
Val CE flat at 5.28–5.29 since step ~4500. Cosine LR does its primary work in the
50–80% range (steps 30000–48000). Do not intervene before step 30000.
Decision rule: *if val CE is still declining at step 30000, extend token budget to
4-6B tokens (increase `--token_budget`) rather than stopping at 2B. The empirical
sweet spot for a 92.5M model is 4-6B high-quality tokens. Do not extend beyond 6B —
diminishing returns dominate past that point at this model size.*

```bash
# Stage 1 resume (next session) — shuffle_buffer already set, just resume:
python pretrain.py \
  --preset nano \
  --resume_from runs/stage1 \
  --shuffle_buffer 20000 \
  --session_timeout_hours 12.0

# Stage 1 extended budget (if val_ce still declining at step 30000):
python pretrain.py \
  --preset nano \
  --resume_from runs/stage1 \
  --token_budget 6_000_000_000 \
  --shuffle_buffer 20000 \
  --session_timeout_hours 12.0

# Stage 2 launch (after Stage 1 banner: val_ce < 3.0 AND mean_uwr > 0.05):
python train_sft.py \
  --preset nano \
  --resume_from runs/stage1/checkpoint-XXXXXXX \
  --max_seq_len 1024 \
  --ema_decay 0.995 \
  --dataset_mix full

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
| Stage 1 token budget | 4–6B tokens. Evaluate at step 30000: if val_ce still declining, extend. Hard cap at 6B (diminishing returns dominate past this for 92.5M params). Do NOT go to 10B at nano scale. |
| Stage 2 dataset | Bespoke-Stratos-17k + MetaMathQA + OpenHermes-2.5 (existing mix) + OpenR1-Math-220k + OpenR1-Code (new additions). Filter/truncate at 1024 tokens — never train on mid-chain truncated samples. |
| Stage 2 max_seq_len | **1024** tokens. Required for R1 reasoning traces (most are 2000–8000 tokens; 512 truncates mid-chain). Fits comfortably on Dual T4 at batch_size=2, grad_accum=8. |
| Stage 2 epochs | 2–3 passes maximum over combined dataset. Beyond 3 epochs → memorization, not generalization. |
| SFT vs pretrain tokens | Not interchangeable. Pretraining supervises every token position (dense signal). SFT supervises answer tokens only (sparse signal, structurally narrow). Cannot substitute 8B SFT tokens for 8B pretrain tokens — they teach fundamentally different things. |
| Width scaling | Cannot increase d_model without abandoning checkpoints (every weight tensor shape changes). Net2Wider is too complex for hybrid architecture. Decision: prove architecture at nano, start medium/large from scratch on TRC. |
| Architecture scaling path | Nano (92.5M) is the proof-of-concept. If architecture outperforms at this scale → start small (270M) or medium (760M) from scratch on TRC. Nano checkpoints are not transferable to wider configs. |
| Depth stacking (DUS) | Viable in principle (see SOLAR 10.7B). Only meaningful after Stage 2 SFT, requires asymmetric initialization + residual re-scaling. Filed as Appendix C stretch goal — not on primary path. |
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
| `baseline_trm_mamba.py` | 0 | ✅ COMPLETE | All smoke checks passed. No changes needed until Stage 3. |
| `viability_gate.py` | 0 | ✅ COMPLETE | All 4 gates passed. Imports from training_utils. |
| `training_utils.py` | All | ✅ COMPLETE | Shared infrastructure module. See Part 4a. |
| `pretrain.py` | 1 | 🟡 RUNNING on Kaggle | Session 6, step ~15900. Imports from training_utils. |
| `train_sft.py` | 2 | ✅ READY TO RUN | All bugs fixed. Blocked only on Stage 1 gate. |
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

## Part 4a — `training_utils.py` Module Reference

Shared infrastructure imported by `pretrain.py`, `train_sft.py`, and `viability_gate.py`.
Do not duplicate any of these in per-script code.

| Export | Purpose | Notes |
|---|---|---|
| `ModelEMA` | EMA weight tracking | `load_state_dict` shape-checks before copying |
| `ema_scope(model, ema)` | Context manager: swap EMA weights in, restore live weights on exit | Handles device/dtype transfer. Replaces the manual `live_backup` pattern from Bug 1. |
| `autocast_context(device, dtype)` | Centralized `torch.autocast` | No-op on CPU |
| `build_adamw_optimizer(model, ...)` | AdamW with decay/no-decay groups | Returns `(optimizer, fused_enabled)` |
| `cosine_with_warmup(optimizer, ...)` | Linear warmup + cosine decay to `min_lr_ratio` | |
| `checkpoint_step_from_name(name)` | Parse step integer from `checkpoint-NNNNNNN` | Returns -1 on failure |
| `list_local_checkpoints(output_dir)` | Sorted list of finalized local checkpoint dirs | Newest first, excludes `.tmp` |
| `try_load_state(path, device)` | Load `training_state.pt`, return `None` on corruption | |
| `list_remote_checkpoint_names(repo_id, token)` | Hub checkpoint names, newest first | |
| `download_checkpoint_from_hub(name, ...)` | Download one Hub checkpoint to disk | |
| `sync_checkpoint_to_hub(dir, ...)` | Upload checkpoint dir with timeout, fire-and-forget | Bug 5 fix pattern |
| `cleanup_temporary_checkpoints(output_dir)` | Remove stale `.tmp` dirs | Call at startup |
| `set_seed(seed)` | Seed Python + PyTorch RNGs | |
| `vram_gb(device)` | Current allocated VRAM in GiB | Returns 0.0 on CPU |
| `resolve_hf_token(cli_value)` | HF token from CLI or env | Never reads from disk |
| `pad_vocab_size(actual, multiple)` | Round vocab size up to multiple of 128 | |

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
**Status:** Running — step ~15900 / 61,036 (~26.1% complete). See terminal_log.md for full output.

**Checkpoint status (Session 6):**
- Local (keep_last=3): checkpoint-0013000 pruned; checkpoint-0015000 saved and Hub-uploaded (commit=cc7891dc) ✅
- Hub: checkpoint-0015000 confirmed

**Key hyperparameters (nano):**
```
batch_size=8, grad_accum=4, chunk_size=1024 → 32,768 tokens/step
lr=6e-4, warmup=200 steps, cosine decay to 6e-5
weight_decay=0.1, max_grad_norm=1.0, ema_decay=0.995
shuffle_buffer=20000 (active since Session 6)
save_every=1000, val_every=500, gen_every=500
DDP throughput: ~5,800 tok/s
```

**Token budget (revised):**
- Original plan: 2B tokens (61,036 steps). Now superseded.
- Target: **4–6B tokens**. Modern over-training practice for small models (LLaMA 3.2 1B: 9T tokens; Chinchilla is a training-compute optimum, not inference-quality optimum).
- Hard cap: **6B tokens** (~183,000 steps). Diminishing returns dominate past this at 92.5M params.
- Decision at step 30000: if val_ce still declining → extend to 6B. If plateau is total → accept and proceed to Stage 2.

**Live observations (step ~15900):**
- Val CE: 5.289 (resume) → 5.2792 (step 15000) → 5.2799 (step 15500). Flat since step ~4500. Train CE declining steadily (≈4.11 at step 15900). Expected at 26% — cosine LR hasn't entered primary decay phase yet (steps 30000–48000).
- VRAM flat at 2.035 GB — no graph retention ✅
- Spike clusters recurring (15030, 15215, 15767–15787, 15880). shuffle_buffer 20000 now active.
- Tokenizer length warning (133809 > 131072) — harmless, from streaming raw docs.

**Success criteria:**
- [ ] `val_ce < 3.0` AND `mean_uwr > 0.05` (script prints banner when met)
- [ ] Val CE showing clear decline in steps 30000–48000 (cosine decay phase)

**Loss curve summary (all sessions):**

| Step | Train CE | Val CE | Tokens | Notes |
|---|---|---|---|---|
| 1 | 11.98 | — | 32k | Random init |
| 500 | 5.46 | 6.38 | 16.4M | Phrases forming |
| 1000 | 4.97 | 5.85 | 32.8M | Real sentences |
| 2000 | — | 5.56 | 65.5M | Resumed (ckpt-2000) |
| 3000 | 4.48 | 5.42 | 98.3M | Hub sync working |
| 5000 | 4.47 | 5.30 | 163.8M | Val plateau begins ⚠ |
| 8000 | 4.92 | 5.29 | 262.1M | Spike cluster (7971/7987/8000) |
| 9000 | 4.19 | 5.29 | 295M | Plateau continues |
| 14902 | — | 5.290 | 488.3M | Resumed (ckpt-14902) |
| 15000 | 4.34 | 5.279 | 491.5M | Session 6 starts |
| 15500 | 4.16 | 5.280 | 507.9M | Flat; spikes at 15767–15787 |
| 15900 | 4.11 | 5.280 | 521M | Session 6 captured log end |

---

### Stage 2 — SFT ✅ READY (blocked only on Stage 1 gate)

**Gate:** Stage 1 `val_ce < 3.0` AND `mean_uwr > 0.05`

**Script:** `train_sft.py` — all bugs fixed, all features live.

**Key implementation details (verified in current codebase):**

`_build_sft_sample` — central tokenization helper used by all dataset loaders. Computes
`prompt_len` and clamps it to `min(prompt_len, len(ids))` so truncated samples are safe.

`_format_training_text` / `_build_prompt_prefix` — enforce the canonical output format
for all sources without duplication.

`collate` — masks labels with `labels[idx, pl:length] = ids[pl:]`. Only response tokens
are supervised. `prompt_len` defaults to 0 if missing (safe fallback).

`compute_val_ce` + `run_generation_callback` — both use `ema_scope` from `training_utils`.
Sequence: `model.eval()` → EMA swap → work → EMA restore → `model.train()`.

`load_latest_checkpoint` — handles: direct `.pt` file, direct checkpoint dir, parent dir
scan, Hub fallback. `_looks_like_pretrain_checkpoint` detects Stage 1 checkpoints and
resets optimizer/scheduler while keeping model weights. Smart preference logic: if Stage 2
checkpoints already exist in `output_dir`, they take priority over an external
`--resume_from` path, preventing accidental Stage 1 re-load mid-Stage-2 run.

`load_mixed_dataset` — calls `_build_sft_sample` for all three sources (Stratos,
MetaMathQA, OpenHermes). Bug 3 fix propagates automatically; no per-extractor patch needed.

**Dataset mix (expanded):**

| Source | Type | Approx size | Notes |
|---|---|---|---|
| Bespoke-Stratos-17k | Math + reasoning | ~25M tokens | Core dataset, already in `load_mixed_dataset` |
| MetaMathQA | Math | ~400M tokens | Already in mix |
| OpenHermes-2.5 | General instruction | ~1B tokens | Already in mix |
| OpenR1-Math-220k | R1-distilled math traces | ~800M tokens | **New — add extractor** |
| OpenR1-Code | R1-distilled code traces | ~300M tokens | **New — add extractor** |

Filter rule: keep only samples where full sequence (question + think + answer) fits
within 1024 tokens. Never train on mid-chain truncated reasoning traces.

**Target format:**
```
User: {question}
Assistant: <think>
{reasoning_chain}
</think>
{final_answer}{eos}
```

**Hyperparameters:**
```
lr=1e-4 (default; use 3e-5 if loss spikes early)
warmup=100 steps, cosine decay to 1e-5
num_epochs=2–3 max (beyond 3 → memorization, not generalization)
batch_size=2, grad_accum=8 → effective batch=16
max_seq_len=1024  ← increased from 512; required for R1 trace compatibility
ema_decay=0.995
```

**Code changes needed before Stage 2 launch:**
- Add extractors for `open-r1/OpenR1-Math-220k` and `open-r1/OpenR1-Code` in `train_sft.py::load_mixed_dataset`
- Update mix ratios to accommodate new sources
- Update `--max_seq_len` default or pass explicitly at launch

**Success criteria:**
- [ ] `val_ce < 1.5` (answer tokens only — guaranteed clean since Bug 3 fixed before first run)
- [ ] Model generates `<think>` blocks before answering
- [ ] Semantically coherent answers on GEN_PROMPTS

---

### Stage 3 — Incremental Recursion (Coconut-Ouroboros) ⬜ NOT STARTED

**Gate:** Stage 2 `val_ce < 1.5`

**Agent prompt:** `stage3_agent_prompt.md` (complete, ready to feed to coding agent).
**Script to create:** `recursive_finetune.py`

**Curriculum:**

| Sub-stage | K | Resume from | Output dir | Gate |
|---|---|---|---|---|
| 3.1 | 1 | Stage 2 final ckpt | runs/stage3_k1 | answer val_ce ≤ stage2 × 1.05 |
| 3.2 | 4 | Stage 3.1 final ckpt | runs/stage3_k4 | answer val_ce ≤ stage2 × 1.05 |
| 3.3 | 16 | Stage 3.2 final ckpt | runs/stage3_k16 | answer val_ce ≤ stage2 × 1.05 |

Each sub-stage: `num_epochs=2`, `lr=1e-5`, `warmup_steps=50`.

---

### Stage 4 — GRPO ⬜ NOT STARTED

**Gate:** Stage 3 quality does not degrade vs Stage 2 (CE within 5%).

**Implementation:** TRL `GRPOTrainer` — DO NOT reimplement from scratch.

**Reward functions:**
1. Format: +0.1 if `<think>...</think>` tags correctly open/close
2. Correctness: +1.0 if answer verified (SymPy for math, E2B sandbox for code)
3. Length penalty (optional): mild negative for traces > 2× median length

---

### Stage 5 — Quantization ⬜ NOT STARTED

**Gate:** Stage 4 complete (or Stage 3 if skipping GRPO).

**Method:** Quamba (Hadamard + DLS). Standard 4-bit breaks Mamba models.

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
    "chunks_in_epoch":         int,   # Stage 1 only
    "tokens_processed":        int,   # Stage 1 only
    "model_state_dict":        dict,
    "ema_backbone_state_dict": dict,  # EMA shadow + lm_head.weight alias
    "optimizer":               dict,
    "scheduler":               dict,
    "scaler":                  dict or None,
    "ema":                     dict,
    "backbone_config":         dict,
    "val_ce":                  float or None,
}
```

**lm_head.weight alias rule:** When `tie_embeddings=True`, `ema_backbone_state_dict`
MUST explicitly contain `"lm_head.weight"` as an alias to `"token_embedding.weight"`.
Verified in pretrain.py's `save_checkpoint`.

**Hub push protocol:**
1. Write fully to `checkpoint-NNNNNNN.tmp`
2. Rename `.tmp` → `checkpoint-NNNNNNN` immediately (local finalize — always happens)
3. Prune old local checkpoints
4. Attempt Hub push — fire-and-forget, warn on failure, never block

---

## Part 7 — Bug Tracker

### Bug 1 — compute_val_ce did not restore live weights after EMA eval
**Status:** ✅ FIXED — `compute_val_ce` and `run_generation_callback` in `train_sft.py`
both use `ema_scope` from `training_utils.py`. The context manager handles weight swap,
device/dtype transfer, and guaranteed restore on exit (including on exception). Also fixed
identically in `pretrain.py::compute_val_ce`.

### Bug 2 — load_latest_checkpoint did not handle direct checkpoint paths
**Status:** ✅ FIXED — `load_latest_checkpoint` in `train_sft.py` handles: direct
`training_state.pt` file, direct checkpoint directory, parent directory scan, Hub fallback.
Additional smart logic: if Stage 2 checkpoints already exist in `output_dir`, they take
priority over an external `--resume_from` to prevent accidental Stage 1 re-load.

### Bug 3 — collate applied loss to all tokens including prompt
**Status:** ✅ FIXED — `_build_sft_sample` computes `prompt_len` (clamped to sequence
length after truncation) for every sample. `collate` uses `labels[idx, pl:length] = ids[pl:]`.
Fix propagates to all dataset sources via `_build_sft_sample` — no per-extractor patch needed.
Bug 3 was fixed before any real Stage 2 run, so all Stage 2 val_ce values will be
answer-only from step 1. No mixed baseline risk.

### Bug 5 — Hub upload failure corrupted local checkpoint save
**Status:** ✅ FIXED — `sync_checkpoint_to_hub` in `training_utils.py` uses
`run_as_future=True` with timeout. Local finalization (`.tmp` → final rename) always
happens first and is never blocked by Hub failures. Verified working in Session 5+.

### Bug 6 — Stage 2 val_ce might mix prompt-supervised and answer-only measurements
**Status:** ✅ MOOT — Bug 3 fixed before any real Stage 2 run. All Stage 2 checkpoints
will have clean answer-only val_ce from step 1. Close this bug.

---

## Part 8 — Compute Plan

**Stage 1 token budget reference (Dual T4, ~5,800 tok/s):**

| Token Budget | Steps | Sessions (12h) | Wall Time |
|---|---|---|---|
| 2B (original) | 61,036 | ~11 | ~5 days |
| 4B | 122,072 | ~23 | ~11 days |
| 6B (target cap) | 183,108 | ~34 | ~17 days |

| Stage | Platform | Estimate | Notes |
|---|---|---|---|
| 0 | Colab free / local | ~15 min | ✅ Done |
| 1 | Kaggle T4 | ~17–34 sessions (4–6B tokens) | ~26% complete; budget decision at step 30000 |
| TRC application | — | 5 min | sites.research.google/trc |
| 2 | Kaggle T4 or TRC | ~4–8h | 2–3 epochs over expanded R1 dataset mix |
| 3 | TRC preferred | ~4–8h | Incremental, gated per n_latent |
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
- **Chinchilla** (Hoffmann et al., 2022) — compute-optimal scaling laws (training optimum, NOT inference optimum)
- **FineWeb-Edu** (HuggingFaceTB, 2024) — https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **Phi-1/Phi-2** (Microsoft, 2023) — synthetic "textbook quality" data at small scale
- **SOLAR 10.7B** (Upstage, 2023) — Depth Up-Scaling (DUS) via pretrained weight stacking
- **OpenR1 datasets** (open-r1 collection) — https://huggingface.co/collections/open-r1/reasoning-datasets
- **Coconut** (Meta, arXiv:2412.06769) — latent thought injection, basis for Stage 3
- **TRM** (Samsung, arXiv:2510.04871) — architecture inspiration only (encoder model)
- **DeepSeek-R1** (2025) — GRPO reward functions + R1-distilled datasets
- **Quamba** (2024) — post-training quantization for Mamba models
- **TRL GRPOTrainer** — https://huggingface.co/docs/trl

---

## Appendix B — Stage 3: Coconut-Ouroboros

> The original Appendix B (EMA hidden-state loop) is **superseded and incorrect**.
> The correct implementation spec is in `stage3_agent_prompt.md`.

### Why the original approach was wrong

The Samsung TRM paper describes an **encoder** for fixed-size grids (ARC-AGI, Sudoku).
Its EMA-loop recursion has ground-truth supervision at every step and is architecturally
incompatible with an autoregressive decoder. Applying it to an LLM gives no gradient
signal — the model learns to ignore the loop structure.

### Correct approach: Coconut-Ouroboros

Based on Meta's Coconut paper (arXiv:2412.06769). Replace `<think>...</think>` reasoning
tokens with K *latent thought positions* where the last hidden state is injected directly
as the next position's input embedding (bypassing the token embedding table).

**Mamba SSM advantage:** During each latent pass, the Mamba SSM recurrent state propagates
a compressed O(d_state) summary of all previous positions. This accumulates persistent
scratch-memory across K passes — a genuine advantage over pure Transformer-Coconut.

**Required additions to `baseline_trm_mamba.py`** (two methods, do not rewrite file):
- `forward_with_hidden()` — returns logits + final hidden states
- `forward_from_embeddings()` — forward pass starting from pre-computed embeddings

See `stage3_agent_prompt.md` for complete implementation specification.

---

## Appendix C — Stretch Goals (post-Stage 3, no-TRC path)

> These are explicitly NOT on the primary path. Document here so ideas are not lost.
> Re-evaluate only after Stage 3 (Coconut-Ouroboros) is complete.

### C.1 — Depth Up-Scaling (DUS) for a Deeper Nano

**What:** Stack two Stage-2-fine-tuned nano checkpoints to get n_groups=2 at d_model=512
(107M params, 18 residual layers). Based on SOLAR 10.7B technique.

**When it makes sense:** No TRC access, Stage 3 complete, want to push capability
ceiling before committing to a full medium-scale retraining run.

**Requirements — do not attempt without all of these:**
1. **Asymmetric initialization:** Group 1 from final Stage 2 checkpoint; Group 2 from
   an earlier Stage 2 checkpoint (different step → different learned representations).
   Never use identical weights for both groups — symmetric init collapses gradients.
2. **Residual re-scaling:** After loading, multiply all residual writer weights
   (`o_proj`, `mlp.down_proj`, `mamba.out_proj`) by `1/sqrt(2) ≈ 0.707` to correct
   for the doubling of depth without re-scaling.
3. **Healing run:** ~300–500M tokens of continued pretraining at lr=1e-4 to let both
   groups learn to cooperate.
4. **Re-SFT:** Full Stage 2 SFT pass on the stacked model after healing.

**What you do NOT get:** wider representations (d_model stays 512), any parameter
efficiency gain (15% more params for 2× compute), compatibility with existing nano
checkpoints in later stages.

### C.2 — Sparse MoE at MLP Layer

**What:** Replace SwiGLU with top-2-of-N sparse MoE. Doubles MLP capacity at constant
FLOPs per token.

**Why not now:** Requires rewriting `SwiGLU` and `TRMBlock`, incompatible with all
existing checkpoints, introduces load-balancing instability. This is a Stage 0
architectural decision — cannot be retrofitted.

**File for:** next full run if architecture proves out at nano scale.

### C.3 — Synthetic Data Augmentation for SFT (Phi-style)

**What:** Use a frontier model (DeepSeek-R1, Qwen2.5-72B via API) to generate
reasoning traces targeting the nano's demonstrated weaknesses from Stage 2 generation
callbacks. Keep traces short (< 1024 tokens) since longer chains are truncated.

**Cost:** API credits, not GPU time. Low implementation risk.

**When:** Before Stage 2 launch, as an additive data source alongside the open-r1
collection. Highest ROI per dollar at small model scales (Phi-1 validated this).

### C.4 — Architecture Scaling (if TRC never arrives)

Primary path: nano proves architecture → medium/large from scratch on TRC.

If TRC never arrives: nano + SERF K=16 is still a valid publishable result. A 92.5M
model with 16 latent thought passes is doing 16× the inference compute of a single
forward pass — competitive with much larger single-pass models on structured reasoning
tasks where depth matters more than breadth.
