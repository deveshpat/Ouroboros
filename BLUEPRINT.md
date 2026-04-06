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
| 2 | SFT | 🔴 NEEDS PATCH — Session 2 failed; bugs fixed in `stage2_patch_prompt.md`; Hub has checkpoint-0002979 | answer val_ce < 1.5 |
| 3 | Recursive Inference (Coconut-Ouroboros) | ⬜ NOT STARTED | Stage 2 gate |
| 4 | GRPO | ⬜ NOT STARTED | Stage 3 gate |
| 5 | Quantization | ⬜ NOT STARTED | Stage 4/3 gate |

### Immediate next actions

**Stage 2 (next session):**
1. Apply `stage2_patch_prompt.md` patches to `train_sft.py`
2. Dry-run to verify fixes (see patch prompt checklist)
3. Launch full-mix training — it will auto-download checkpoint-0002979 from Hub, detect data_changed, reset optimizer, and train fresh:

```bash
python train_sft.py \
  --preset nano \
  --resume_from runs/stage2 \
  --max_seq_len 2048 \
  --dataset_mix full \
  --num_epochs 3 \
  --batch_size 2 \
  --grad_accum 16 \
  --lr 1e-4 \
  --warmup_steps 100 \
  --ema_decay 0.995 \
  --output_dir runs/stage2 \
  --push_to_hub \
  --session_timeout_hours 11.5 \
  --graceful_exit_buffer_minutes 7
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
| Stage 2 Session 1: answer-only training | Stratos reasoning chains excluded by max_seq_len=1024; model never learned `<think>` format | Increase to max_seq_len=2048 |
| Stage 2 Session 2: resume bug cascade | Hub downloads landed in output_dir; prune deleted all Stage 2 checkpoints; optimizer not reset on data change | See Bug Tracker (Bugs 6–10) |

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
max_seq_len           = 2048          # ← raised from 1024 for Stage 2
dropout               = 0.0
rms_norm_eps          = 1e-5
tie_embeddings        = True
mlp_hidden            = ceil((8/3 × d_model) / 64) × 64
total_residual_layers = n_groups × (2 + 7)
```

### Key implementation notes

**RoPE** — Non-interleaved. Cache rebuilt lazily on device/dtype change. Theta=1e6. Safe to change max_seq_len between checkpoint and training run (no positional tensors in state_dict).

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
| Stage 1 outcome | Bypassed gate (val_ce 5.32 at step 21501). SFT launched from ckpt-0021000 at 705M tokens. |
| Stage 2 max_seq_len | **2048** — 1024 filtered 97% of Stratos reasoning chains; Session 1 effectively trained answer-only |
| Stage 2 epochs | 2–3 passes; beyond 3 → memorization risk |
| Stage 2 dataset | Full mix: Stratos + MetaMathQA + OpenHermes + OpenR1-Math + OpenR1-Code |
| Stage 2 target format | `User: {q}\n\nAssistant: <think>\n{reasoning}\n</think>\n{answer}{eos}` |
| Stage 2 starting checkpoint | Hub checkpoint-0002979 (commit=8981b950) — trained on stratos answer-only, but reasonable weights |
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
| `train_sft.py` | 2 | 🔴 NEEDS PATCH | See `stage2_patch_prompt.md` — 5 changes required |
| `stage2_patch_prompt.md` | 2 | ✅ COMPLETE | Feed to coding agent before next Stage 2 session |
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

Completed checks: model instantiates correctly; initial loss = 11.9904; backward with no NaN/missing gradients; param_count = 92,477,440; G1–G4 all passed.

---

### Stage 1 — Pre-training ✅ BYPASSED (SFT launched)

**Final state:** step 21501, tokens 705M, val_ce 5.3241 (rising). SFT launched from Hub checkpoint-0021000.

**Loss curve summary:**

| Step | Val CE | Tokens | Notes |
|---|---|---|---|
| 1000 | 5.85 | 32.8M | — |
| 5000 | 5.30 | 163.8M | Plateau begins |
| 16000 | 5.277 | 524M | Only improvement in Session 6 |
| 18000 | 5.305 | 590M | Val rising + dense spikes |
| 21000 | 5.319 | 688M | Hub ckpt-0021000 ← **used for SFT** |
| 21501 | 5.324 | 705M | Session end; local-only ckpt |

---

### Stage 2 — SFT 🔴 NEEDS PATCH

**Script:** `train_sft.py` | **Hardware:** Kaggle T4 or Dual T4

**Session 1 (complete, gate not met):**
- stratos-only, max_seq_len=1024, 3 epochs, 2979 steps
- val_ce plateaued at 4.9153 (gate requires < 1.5)
- NO reasoning chains learned — model never produced `<think>` tags
- Root cause: max_seq_len=1024 filtered all long reasoning traces; model trained answer-only
- Hub checkpoint-0002979 saved (commit=8981b950) ✅

**Session 2 (catastrophic failure — see Bug Tracker):**
- full-mix with DDP, 717 steps
- Bugs 6–10 caused: OOM during resume, pruning of all Stage 2 checkpoints, number-token generation loop
- val_ce=5.7135 (worse than Session 1 start)
- No Hub checkpoints from this session saved
- Patch prompt: `stage2_patch_prompt.md`

**Expected data counts at max_seq_len=2048 (after patch):**
- Stratos: ~10000-12000 samples (~70%)
- MetaMathQA: ~11000
- OpenHermes: ~8000
- OpenR1-Math: ~3000-5000
- OpenR1-Code: ~500-1500
- Total: ~35000-38000 samples; 3 epochs ≈ ~3500-3600 steps

**Hyperparameters (post-patch):**
```
lr=1e-4, warmup=100, cosine to 1e-5
batch_size=2, grad_accum=16 → effective batch=32
max_seq_len=2048, ema_decay=0.995
dataset_mix=full
```

**Code changes needed (see `stage2_patch_prompt.md`):**
- [ ] Fix `load_latest_checkpoint`: local Stage 2 checkpoints tried first; Hub downloads go to `.hub_resume` temp dir
- [ ] Fix data stream change: reset optimizer + scheduler + step when `data_changed=True`
- [ ] `load_latest_checkpoint` returns 4-tuple with `reset_optimizer: bool`
- [ ] `run_training` rebuilds optimizer/scheduler when `need_opt_reset=True`
- [ ] Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- [ ] Change default `max_seq_len` to 2048
- [ ] Fix prune to skip `.hub_*` subdirectories

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

### Bug 1 — compute_val_ce did not restore live weights after EMA eval ✅ FIXED
### Bug 2 — load_latest_checkpoint did not handle direct checkpoint paths ✅ FIXED
### Bug 3 — collate applied loss to all tokens including prompt ✅ FIXED
### Bug 5 — Hub upload failure corrupted local checkpoint save ✅ FIXED

### Bug 6 — Resume downloads Hub checkpoints even when local Stage 2 exists 🔴 OPEN
Sort key `(step, priority)` causes Hub step=21000 to outrank local step=2979. Fix: try all local Stage 2 checkpoints before any Hub download. See `stage2_patch_prompt.md` Change 1.

### Bug 7 — Hub-downloaded checkpoints placed inside output_dir 🔴 OPEN
`download_checkpoint_from_hub` used `local_root` (= output_dir) as target. Hub Stage 1 checkpoints (steps 3000–21000) were written to `runs/stage2/`. Fix: use `output_dir / ".hub_resume"` temp dir. See Change 1.

### Bug 8 — Prune deleted all Stage 2 checkpoints (cascaded from Bug 7) 🔴 OPEN
Prune saw Hub-downloaded steps 3000–21000 mixed with Stage 2 steps 2000–2979. `keep_last=3` kept steps 19000/20000/21000 (Stage 1). ALL Stage 2 checkpoints deleted. Fix: Hub downloads in separate dir (Change 1) + skip `.hub_*` in prune (Change 5).

### Bug 9 — Optimizer NOT reset when data stream changes 🔴 OPEN
Code reset epoch/samples_seen but kept optimizer momentum and LR at cosine minimum (~1e-5). New data distribution → 75 spikes in 717 steps. Fix: reset optimizer + scheduler + step=0 on data_changed. See Change 2.

### Bug 10 — max_seq_len=1024 excludes nearly all reasoning traces 🔴 OPEN
Stratos: 97% filtered; OpenR1-Math: 92% filtered; OpenR1-Code: 99.9% filtered. Session 1 effectively trained answer-only (no `<think>` learning). Fix: default max_seq_len=2048. See Change 4.

---

## Part 8 — Compute Plan

| Stage | Platform | Estimate | Notes |
|---|---|---|---|
| 1 | Kaggle T4 | ✅ Done (705M tokens) | Bypassed gate; SFT launched |
| 2 | Kaggle T4/Dual | ~4–8h per session | Patch required first; checkpoint-0002979 on Hub |
| 3 | TRC preferred | ~4–8h | Gated per K sub-stage |
| 4 | TRC + unsloth | ~8–12h | GRPO G=4 rollouts |
| 5 | Local / Jetson | ~2h | Post-training quantization |

---

## Part 9 — References

- **Mamba** (Gu & Dao, 2023); **Jamba** (AI21 Labs, 2024) — 1:7 TRM-Mamba at 52B
- **GPT-2** (Radford et al., 2019) — residual scaling formula
- **RoPE** (Su et al., 2021); **SwiGLU** (Shazeer, 2020); **GQA** (Ainslie et al., 2023)
- **Chinchilla** (Hoffmann et al., 2022) — compute-optimal scaling
- **FineWeb-Edu** (HuggingFaceTB, 2024)
- **Coconut** (Meta, arXiv:2412.06769) — latent thought injection, basis for Stage 3
- **TRM** (Samsung, arXiv:2510.04871) — architecture inspiration only
- **DeepSeek-R1** (2025) — GRPO reward functions + R1-distilled datasets
- **Quamba** (2024) — post-training quantization for Mamba
- **TRL GRPOTrainer** — https://huggingface.co/docs/trl
- **OpenR1 datasets** — https://huggingface.co/collections/open-r1/reasoning-datasets

---

## Appendix A — Stretch Goals (post-Stage 3)

### A.1 — Depth Up-Scaling (DUS)
Stack two Stage-2 nano checkpoints → n_groups=2 at d_model=512 (~107M). Requirements: asymmetric init, residual re-scaling (×0.707), healing run (~300–500M tokens), re-SFT. Only attempt post-Stage 3.

### A.2 — Sparse MoE at MLP Layer
Requires full architectural restart — file for next clean run.

### A.3 — Synthetic Data Augmentation (Phi-style)
Use frontier model API to generate reasoning traces targeting nano's demonstrated SFT weaknesses. Keep traces < 2048 tokens. Highest ROI per dollar at small model scales.

### A.4 — Architecture Scaling
Nano + SERF K=16 is a valid publishable result without TRC. If TRC arrives: start small or medium from scratch; nano checkpoints are not transferable to wider configs.
