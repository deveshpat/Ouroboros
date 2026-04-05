# Project Ouroboros — Master Blueprint

> **Thread-resume header.** Read Part 0 first in any new session, then jump to the current stage. Append new findings — never rewrite history.

---

## Part 0 — Quick-Resume Context

### What this project is
Novel hybrid Transformer-Mamba language model ("TRM-Mamba", 1:7 ratio) pre-trained from scratch on FineWeb-Edu, then fine-tuned for chain-of-thought reasoning using R1-distilled datasets. Target: prove the architecture outperforms comparably-sized models at nano scale (92.5M), then scale to medium/large on TRC. Test-time compute scaling via Coconut-Ouroboros recursive inference (SERF).

### Root failure on record
`checkpoint-950` (archived): randomly-initialized model trained directly on SFT data → comma-loops and "the the the". Root causes: no pre-training, EMA initialized to zeros, novel components stacked without per-component validation.

### Current status

| Stage | Name | Status | Gate |
|---|---|---|---|
| 0 | Architecture & Viability | ✅ COMPLETE | All 4 gates passed |
| 1 | Pre-training | ✅ BYPASSED — SFT launched from ckpt-0021000 | val_ce < 3.0 + UWR > 0.05 |
| 2 | SFT | 🟡 RUNNING — step ~1020/2979 (34%); stratos-only; val_ce=4.95 | answer val_ce < 1.5 |
| 3 | Recursive Inference (Coconut-Ouroboros) | ⬜ NOT STARTED | Stage 2 gate |
| 4 | GRPO | ⬜ NOT STARTED | Stage 3 gate |
| 5 | Quantization | ⬜ NOT STARTED | Stage 4/3 gate |

### Immediate next actions

**Stage 2 (current run — stratos-only, 3 epochs):**
Monitor val CE at steps 1500 and 2000:
- If val_ce < 3.0 by step 2000 → format inflection occurred, on track
- If val_ce > 4.0 by step 2000 → extend with `--dataset_mix full --num_epochs 5 --resume_from runs/stage2/checkpoint-XXXXXXX`
- If val_ce < 1.5 by step 2979 → Stage 2 gate passed; record val_ce and launch Stage 3

**Stage 2 (next session, if current run didn't reach gate):**
```bash
python train_sft.py \
  --preset nano \
  --resume_from runs/stage2/checkpoint-XXXXXXX \
  --max_seq_len 1024 \
  --ema_decay 0.995 \
  --dataset_mix full \
  --num_epochs 5
```

**Stage 3 (after Stage 2 answer val_ce < 1.5):**
```bash
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
| No pre-training | Model must learn language + reasoning simultaneously from 17k examples | Pre-train on 2B+ tokens first |
| EMA init = zeros | Bias-correction `ema/(1-decay^t)` blows up at t=1 | `pretrain.py`: EMA init from live weights |
| Too many new components | Cannot isolate failure | One change at a time, gate each |
| No non-recursive baseline | No reference to compare against | `baseline_trm_mamba.py` is the baseline |

**Loss=54.67 explained (not a bug):** `DeepSupervisedSERF` summed CE across 8 recursive steps with gamma=0.8 → weight sum ≈4.57. Random-init CE = ln(151665) ≈ 11.93. So 4.57 × 11.93 ≈ 54.5.

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

### Configuration presets

| Preset | d_model | n_groups | n_heads | n_kv_heads | mlp_hidden | Params |
|---|---|---|---|---|---|---|
| nano | 512 | 1 | 8 | 4 | 1408 | 92.5M |
| small | 1024 | 2 | 16 | 8 | 2816 | ~270M |
| medium | 2048 | 2 | 16 | 8 | 5632 | ~760M |

**Nano checksum (verified Stage 0):**
```
parameters          = 92,477,440
mlp_hidden          = 1408   = ceil((8/3×512)/64)×64
total_residual_layers = 9    = 1×(2+7)
initial_loss        = 11.9904
grad_norm_total     = 6.4242
```

**Fixed config values (all presets):**
```python
vocab_size            = 151_680       # ceil(151_665/128)×128
rope_theta            = 1_000_000.0
mamba_d_state         = 16
mamba_d_conv          = 4
mamba_expand          = 2
max_seq_len           = 2048
dropout               = 0.0
rms_norm_eps          = 1e-5
tie_embeddings        = True
mlp_hidden            = ceil((8/3 × d_model) / 64) × 64
total_residual_layers = n_groups × (2 + 7)
```

### Key implementation notes

**RoPE** — Non-interleaved. Cache rebuilt lazily on device/dtype change. Theta=1e6.

**GQA** — `repeat_interleave` applied after RoPE on compact KV tensors. `n_repeats = n_heads / n_kv_heads`.

**Padding + causal attention** — SDPA requires `is_causal=False` + combined causal+padding additive bias when `attention_mask` is provided.

**Mamba padding** — Padded positions zeroed: `h = h * mask` before AND after Mamba forward. Mask convention: `True` = valid token.

**Residual scaling (GPT-2 style)** — Residual writers (`o_proj`, `mlp.down_proj`, `mamba.out_proj`) scaled by `0.02 / sqrt(total_residual_layers)` in a second pass after base init.

---

## Part 3 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Tokenizer | Qwen2.5-0.5B; vocab_size=151_680 everywhere |
| Mamba version | `mamba_ssm.Mamba` (original). Mamba2 deferred to Stage 3+ |
| Stage 1 dataset | HuggingFaceFW/fineweb-edu sample-10BT, streamed |
| Stage 1 outcome | Bypassed gate (val_ce 5.32 at step 21501). SFT launched from ckpt-0021000 at 705M tokens. Viable starting point — SFT signal dominates pretrain quality past basic language structure. |
| Stage 2 max_seq_len | **1024** tokens — required for R1 reasoning traces |
| Stage 2 epochs | 2–3 passes on stratos-only; up to 5 on full mix. Beyond 5 → memorization risk. |
| Stage 2 dataset | Stratos-17k for initial run; expand to full mix (+ MetaMathQA + OpenHermes-2.5 + OpenR1-Math-220k + OpenR1-Code) if stratos-only doesn't reach gate. Filter: full sequence must fit within 1024 tokens. |
| Stage 2 target format | `User: {q}\n\nAssistant: <think>\n{reasoning}\n</think>\n{answer}{eos}` |
| Stage 3 recursion | Coconut-Ouroboros (latent hidden-state injection, K=1→4→16 curriculum). NOT TRM EMA-loop. |
| Stage 3 Mamba advantage | SSM state accumulates across K latent passes → richer scratch-memory vs Transformer-Coconut |
| GRPO | TRL GRPOTrainer. Never reimplement from scratch |
| Width scaling | Cannot increase d_model without abandoning checkpoints. Prove at nano, start medium from scratch on TRC. |
| Hub repo | WeirdRunner/Ouroboros (private) |

---

## Part 4 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `baseline_trm_mamba.py` | 0 | ✅ COMPLETE | Stage 3 requires two method additions (`forward_with_hidden`, `forward_from_embeddings`) — surgical only |
| `viability_gate.py` | 0 | ✅ COMPLETE | All 4 gates passed |
| `training_utils.py` | All | ✅ COMPLETE | Shared infrastructure; see Part 4a |
| `pretrain.py` | 1 | ✅ COMPLETE (session-ended) | Last checkpoint: ckpt-0021501 (local), ckpt-0021000 (Hub) |
| `train_sft.py` | 2 | 🟡 RUNNING | Session 1, step ~1020/2979 |
| `stage3_agent_prompt.md` | 3 | ✅ COMPLETE | Full Coconut-Ouroboros spec; feed to coding agent |
| `recursive_finetune.py` | 3 | ⬜ NOT CREATED | Generate from `stage3_agent_prompt.md` after Stage 2 gate |
| `BLUEPRINT.md` | — | Living document | This file |
| `terminal_log.md` | — | Verified outputs | Append only |

**Deprecated (do not use):** `recursive_trm_mamba.py`, `train_sft_deep_supervision.py`, `inference_trm_mamba.py`, `architecture.py`

---

## Part 4a — `training_utils.py` Module Reference

| Export | Purpose |
|---|---|
| `ModelEMA` | EMA weight tracking; `load_state_dict` shape-checks before copying |
| `ema_scope(model, ema)` | Context manager: swap EMA weights in, restore live on exit (handles device/dtype) |
| `autocast_context(device, dtype)` | Centralized `torch.autocast`; no-op on CPU |
| `build_adamw_optimizer(model, ...)` | AdamW with decay/no-decay groups; returns `(optimizer, fused_enabled)` |
| `cosine_with_warmup(optimizer, ...)` | Linear warmup + cosine decay to `min_lr_ratio` |
| `checkpoint_step_from_name(name)` | Parse step integer from `checkpoint-NNNNNNN`; returns -1 on failure |
| `list_local_checkpoints(output_dir)` | Sorted list of finalized local checkpoint dirs, newest first |
| `try_load_state(path, device)` | Load `training_state.pt`; return `None` on corruption |
| `list_remote_checkpoint_names(repo_id, token)` | Hub checkpoint names, newest first |
| `download_checkpoint_from_hub(name, ...)` | Download one Hub checkpoint to disk |
| `sync_checkpoint_to_hub(dir, ...)` | Upload with timeout, fire-and-forget (Bug 5 fix) |
| `cleanup_temporary_checkpoints(output_dir)` | Remove stale `.tmp` dirs on startup |
| `set_seed(seed)` | Seed Python + PyTorch RNGs |
| `vram_gb(device)` | Current allocated VRAM in GiB; 0.0 on CPU |
| `resolve_hf_token(cli_value)` | HF token from CLI or env; never reads disk |
| `pad_vocab_size(actual, multiple)` | Round vocab size up to multiple of 128 |

---

## Part 5 — Stage Definitions

### Stage 0 — Architecture & Viability Gate ✅ COMPLETE

Completed checks: model instantiates correctly; initial loss = 11.9904; backward with no NaN/missing gradients; param_count = 92,477,440; G1–G4 all passed (CE 12.04→2.00, UWR=0.573, max gnorm=4.03, VRAM Δ=0.000 GB).

---

### Stage 1 — Pre-training ✅ BYPASSED (SFT launched)

**Final state:** step 21501, tokens 705M, val_ce 5.3241 (rising). SFT launched from Hub checkpoint-0021000 (step 21000, val_ce 5.319). Gate not formally met but starting point is viable.

**Loss curve summary:**

| Step | Val CE | Tokens | Notes |
|---|---|---|---|
| 1000 | 5.85 | 32.8M | — |
| 5000 | 5.30 | 163.8M | Plateau begins |
| 9000 | 5.29 | 295M | — |
| 14902 | 5.290 | 488.3M | Resumed ckpt-14902 |
| 16000 | 5.277 | 524M | Only improvement in Session 6 |
| 18000 | 5.305 | 590M | Val rising + dense spikes |
| 21000 | 5.319 | 688M | Hub ckpt-0021000 ← **used for SFT** |
| 21501 | 5.324 | 705M | Session end; local-only ckpt |

---

### Stage 2 — SFT 🟡 RUNNING

**Script:** `train_sft.py` | **Hardware:** Kaggle T4 (single GPU)

**Current run (Session 1):**
- Dataset: stratos-only (16,710 samples; train=15,875, val=835)
- Schedule: 3 epochs × 993 steps = 2979 total; cosine with warmup=100
- Step 1020/2979 (34%); train_ce ~2.82, val_ce 4.9480, val_acc 0.2606
- VRAM flat at 2.440 GB ✅ | Hub syncing ✅
- Generation still degenerate (repetitive loops, no `<think>` tags) — expected at epoch 1

**Val CE trajectory:**

| Step | Val CE | Val Acc | Notes |
|---|---|---|---|
| 250 | 5.2199 | 0.2367 | LR at peak |
| 500 | 5.0644 | 0.2514 | — |
| 750 | 4.9821 | 0.2578 | — |
| 1000 | 4.9480 | 0.2606 | ~0.27 nat drop over 1000 steps |

**Decision rules:**
- step 1500–2000: if val_ce < 3.0 → format inflection occurred, on track for gate by step 2979
- step 1500–2000: if val_ce > 4.0 → extend training with full mix (see Part 0)
- step 2979 (end): if val_ce < 1.5 → Stage 2 gate passed; record and launch Stage 3

**Hyperparameters:**
```
lr=1e-4, warmup=100, cosine to 1e-5
batch_size=2, grad_accum=8 → effective batch=16
max_seq_len=1024, ema_decay=0.995
```

**Code changes needed before full-mix run:**
- Add extractors for `open-r1/OpenR1-Math-220k` and `open-r1/OpenR1-Code` in `load_mixed_dataset`
- Update mix ratios accordingly

**Success criteria:**
- [ ] `val_ce < 1.5` (answer tokens only)
- [ ] Model generates `<think>` blocks before answering
- [ ] Coherent answers on GEN_PROMPTS

---

### Stage 3 — Coconut-Ouroboros ⬜ NOT STARTED

**Script to create:** `recursive_finetune.py` | **Gate:** Stage 2 answer val_ce < 1.5

**Required additions to `baseline_trm_mamba.py`** (insert after `forward` method, surgical only):
- `forward_with_hidden()` — returns `(logits [B,T,V], hidden [B,T,D])`
- `forward_from_embeddings()` — forward from pre-computed embeddings, bypasses token embedding table

**Curriculum:**

| Sub-stage | K | Resume from | Output dir | Gate |
|---|---|---|---|---|
| 3.1 | 1 | Stage 2 final | runs/stage3_k1 | answer val_ce ≤ stage2_val_ce × 1.05 |
| 3.2 | 4 | Stage 3.1 final | runs/stage3_k4 | same |
| 3.3 | 16 | Stage 3.2 final | runs/stage3_k16 | same |

Each sub-stage: `num_epochs=2`, `lr=1e-5`, `warmup_steps=50`. K=16 on seq_len=512 uses ~3.5 GB VRAM for nano.

**Mamba SSM advantage over Transformer-Coconut:** SSM recurrent state propagates compressed O(d_state) scratch-memory across all K latent passes — each pass genuinely refines context rather than just re-attending to the same positions.

Full implementation spec: `stage3_agent_prompt.md`.

---

### Stage 4 — GRPO ⬜ NOT STARTED

**Gate:** Stage 3 quality does not degrade vs Stage 2 (CE within 5%). **Implementation:** TRL `GRPOTrainer` only.

**Reward functions:**
1. Format: +0.1 if `<think>...</think>` tags correctly open/close
2. Correctness: +1.0 if answer verified (SymPy for math, E2B sandbox for code)
3. Length penalty (optional): mild negative for traces > 2× median length

---

### Stage 5 — Quantization ⬜ NOT STARTED

**Gate:** Stage 4 complete (or Stage 3 if skipping GRPO). **Method:** Quamba (Hadamard + DLS). Standard 4-bit breaks Mamba models.

---

## Part 6 — Checkpoint Format

Every checkpoint directory contains:
```
checkpoint-NNNNNNN/
  training_state.pt
  resolved_backbone_config.json
```

**`training_state.pt` required keys:**
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

**lm_head.weight alias rule:** When `tie_embeddings=True`, `ema_backbone_state_dict` MUST explicitly contain `"lm_head.weight"` as an alias to `"token_embedding.weight"`.

**Hub push protocol:** Write to `.tmp` → rename to final (always local-first) → prune old → attempt Hub push (fire-and-forget, never blocks).

---

## Part 7 — Bug Tracker

### Bug 1 — compute_val_ce did not restore live weights after EMA eval
**Status:** ✅ FIXED — both `compute_val_ce` and `run_generation_callback` use `ema_scope` from `training_utils.py`.

### Bug 2 — load_latest_checkpoint did not handle direct checkpoint paths
**Status:** ✅ FIXED — handles: direct `.pt` file, direct checkpoint dir, parent dir scan, Hub fallback, Stage 1→2 transfer.

### Bug 3 — collate applied loss to all tokens including prompt
**Status:** ✅ FIXED — `_build_sft_sample` computes `prompt_len` for every sample; `collate` supervises response tokens only. Fixed before any real Stage 2 run.

### Bug 5 — Hub upload failure corrupted local checkpoint save
**Status:** ✅ FIXED — `sync_checkpoint_to_hub` uses `run_as_future=True` with timeout; local finalization always happens first.

---

## Part 8 — Compute Plan

| Stage | Platform | Estimate | Notes |
|---|---|---|---|
| 1 | Kaggle T4 | ✅ Done (705M tokens) | Bypassed gate; SFT launched |
| 2 | Kaggle T4 | ~4–12h | 3–5 epochs; may need full mix |
| 3 | TRC preferred | ~4–8h | Gated per K sub-stage |
| 4 | TRC + unsloth | ~8–12h | GRPO G=4 rollouts |
| 5 | Local / Jetson | ~2h | Post-training quantization |

---

## Part 9 — References

- **Mamba** (Gu & Dao, 2023); **Jamba** (AI21 Labs, 2024) — 1:7 TRM-Mamba at 52B
- **GPT-2** (Radford et al., 2019) — residual scaling formula
- **RoPE** (Su et al., 2021); **SwiGLU** (Shazeer, 2020); **GQA** (Ainslie et al., 2023)
- **Chinchilla** (Hoffmann et al., 2022) — compute-optimal scaling (training optimum, NOT inference optimum)
- **FineWeb-Edu** (HuggingFaceTB, 2024)
- **SOLAR 10.7B** (Upstage, 2023) — Depth Up-Scaling
- **Coconut** (Meta, arXiv:2412.06769) — latent thought injection, basis for Stage 3
- **TRM** (Samsung, arXiv:2510.04871) — architecture inspiration only (encoder model; EMA loop NOT applicable to autoregressive LLMs)
- **DeepSeek-R1** (2025) — GRPO reward functions + R1-distilled datasets
- **Quamba** (2024) — post-training quantization for Mamba
- **TRL GRPOTrainer** — https://huggingface.co/docs/trl
- **OpenR1 datasets** — https://huggingface.co/collections/open-r1/reasoning-datasets

---

## Appendix A — Stretch Goals (post-Stage 3)

### A.1 — Depth Up-Scaling (DUS)
Stack two Stage-2 nano checkpoints → n_groups=2 at d_model=512 (~107M). Requirements: asymmetric init (different steps for group 1 and 2), residual re-scaling (×0.707), healing run (~300–500M tokens at lr=1e-4), re-SFT. Only attempt post-Stage 3.

### A.2 — Sparse MoE at MLP Layer
Replace SwiGLU with top-2-of-N sparse MoE. Requires full architectural restart — file for next clean run.

### A.3 — Synthetic Data Augmentation (Phi-style)
Use frontier model API to generate reasoning traces targeting nano's demonstrated SFT weaknesses. Keep traces < 1024 tokens. Highest ROI per dollar at small model scales.

### A.4 — Architecture Scaling
Nano + SERF K=16 is a valid publishable result without TRC. If TRC arrives: start small or medium from scratch; nano checkpoints are not transferable to wider configs.
