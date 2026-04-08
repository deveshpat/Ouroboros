# Project Ouroboros — Master Blueprint

> **Thread-resume header.** Read Part 0 first in any new session, then jump to the current stage.

---

## Part 0 — Quick-Resume Context

### What this project is
Novel hybrid Transformer-Mamba language model ("TRM-Mamba", 1:7 ratio) pre-trained on FineWeb-Edu, then fine-tuned for chain-of-thought reasoning via R1-distilled datasets. Test-time compute scaling via Coconut-Ouroboros recursive inference (SERF).

### Current status

| Stage | Name | Status | Gate |
|---|---|---|---|
| 0 | Architecture & Viability | ✅ COMPLETE | All 4 gates passed |
| 1 | Pre-training | ✅ BYPASSED — SFT from ckpt-0021000 | val_ce < 3.0 + UWR > 0.05 |
| 2 | SFT | 🔴 NEEDS REWRITE — S5/S6 both killed by NCCL watchdog (val+gen > 600s under DDP) | answer val_ce < 1.5 |
| 3 | Recursive Inference (Coconut-Ouroboros) | ⬜ NOT STARTED | Stage 2 gate |
| 4 | GRPO | ⬜ NOT STARTED | Stage 3 gate |
| 5 | Quantization | ⬜ NOT STARTED | Stage 4/3 gate |

### Immediate next actions

**Apply the `stage2_rewrite_prompt.md` patch to `train_sft.py`**, then run the DDP dry-run check before launching S7.

**S7 run command (after patch applied and dry-run passes):**
```bash
python train_sft.py \
  --preset nano \
  --max_seq_len 2048 \
  --dataset_mix full \
  --num_epochs 3 \
  --batch_size 2 \
  --grad_accum 16 \
  --lr 3e-4 \
  --warmup_steps 50 \
  --ema_decay 0.99 \
  --val_max_samples 500 \
  --val_batch_size 16 \
  --output_dir runs/stage2 \
  --push_to_hub \
  --hf_token $HF_TOKEN \
  --wandb_project ouroboros-stage2 \
  --save_every 500 \
  --val_every 250 \
  --gen_every 500 \
  --session_timeout_hours 11.5 \
  --graceful_exit_buffer_minutes 15
```

---

## Part 1 — Root Failure Record

| Session | Cause | Effect | Status |
|---|---|---|---|
| Pre-S1 | No pre-training | Comma-loops from random init | ✅ Fixed: pretrain first |
| S1 | max_seq_len=1024 | Filtered all reasoning; no `<think>` | ✅ Fixed: 2048 + truncation |
| S2 | Bugs 6–10 cascade | Hub downloads in output_dir; prune deleted all Stage 2 ckpts; no optimizer reset | ✅ Fixed |
| S5 | Bugs 13–19 cascade | val+gen exceeded NCCL 600s watchdog; SIGABRT; val_ce frozen at 5.62 | 🔴 Needs stage2_rewrite_prompt.md applied |
| S6 | Same as S5 (rewrite not yet applied) | Same NCCL crash at step 3750 | 🔴 Confirmed same bug |

---

## Part 2 — Architecture Specification

### Block structure
```
TokenEmbedding(vocab_size=151_680, d_model)
└─ n_groups × TRMMambaGroup:
   ├─ 1 × TRMBlock: RMSNorm + GQA(RoPE) + residual; RMSNorm + SwiGLU + residual
   └─ 7 × MambaLayer: RMSNorm + mamba_ssm.Mamba + residual   ← 1:7 ratio, fixed
FinalRMSNorm → LM Head (weight-tied)
```

### Configuration presets

| Preset | d_model | n_groups | n_heads | n_kv_heads | Params |
|---|---|---|---|---|---|
| nano | 512 | 1 | 8 | 4 | 92.5M |
| small | 1024 | 2 | 16 | 8 | ~270M |
| medium | 2048 | 2 | 16 | 8 | ~760M |

**Fixed values:** vocab_size=151_680, rope_theta=1e6, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, max_seq_len=2048, dropout=0.0, rms_norm_eps=1e-5, tie_embeddings=True.

---

## Part 3 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Tokenizer | Qwen2.5-0.5B; vocab_size=151_680 |
| Stage 1 outcome | Bypassed gate (val_ce 5.32 at step 21501); SFT from ckpt-0021000 |
| Stage 2 max_seq_len | 2048 — 1024 filtered 97% of Stratos reasoning |
| Stage 2 dataset | Full mix: Stratos + MetaMathQA + OpenHermes + OpenR1-Math + OpenR1-Code (~55k samples) |
| Stage 2 target format | `User: {q}\n\nAssistant: <think>\n{reasoning}\n</think>\n{answer}{eos}` |
| Stage 2 starting checkpoint | Hub ckpt-0002979 (stratos-only weights; optimizer reset due to data change) |
| Stage 2 DDP | Works correctly on Dual T4. Post-training SIGABRT is benign NCCL teardown race. |
| Stage 2 data loading | Rank 0 loads + tokenizes; `broadcast_object_list` to rank 1. Eliminates double disk I/O. |
| Stage 2 val speed | 500 samples capped, batch_size=16 → ~10s per val run (was 2761 ÷ 2 = 557s) |
| Stage 2 NCCL timeout | `NCCL_TIMEOUT=1800` (30 min). Kills genuine hangs; 18× headroom vs 100s worst-case wait. |
| Stage 3 recursion | Coconut-Ouroboros, K=1→4→16. NOT TRM EMA-loop. See stage3_agent_prompt.md. |
| Hub repo | WeirdRunner/Ouroboros (private) |

---

## Part 4 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `baseline_trm_mamba.py` | 0 | ✅ COMPLETE | Stage 3 needs `forward_with_hidden` + `forward_from_embeddings` (surgical, see stage3_agent_prompt.md) |
| `viability_gate.py` | 0 | ✅ COMPLETE | |
| `training_utils.py` | All | ✅ COMPLETE | Canonical Hub/checkpoint utilities. Do not duplicate. |
| `pretrain.py` | 1 | ✅ COMPLETE | Last Hub ckpt: ckpt-0021000 |
| `train_sft.py` | 2 | 🔴 NEEDS PATCH | Apply `stage2_rewrite_prompt.md` (8 changes) |
| `stage2_rewrite_prompt.md` | 2 | ✅ COMPLETE | Feed to coding agent; fixes Bugs 13–19 |
| `stage3_agent_prompt.md` | 3 | ✅ COMPLETE | Feed to coding agent after Stage 2 gate |
| `recursive_finetune.py` | 3 | ⬜ NOT CREATED | Generate from `stage3_agent_prompt.md` |

---

## Part 5 — Stage Definitions

### Stage 2 — SFT 🔴 NEEDS PATCH BEFORE S7

**Sessions:**
- S1 (✅): stratos-only, max_seq_len=1024, val_ce=4.9153, no `<think>` learning. Hub: ckpt-0002979.
- S2 (❌): full-mix DDP, Bugs 6–10, val_ce=5.7135. All local Stage 2 ckpts deleted.
- S3–S4 (✅): dry-runs; all patches from stage2_patch_prompt.md verified.
- S5 (🔴): full-scale DDP, hard-killed at 43200.6s (~step 3700). val_ce plateaued at 5.62. Generation frozen (number loops). Exit code 137 (SIGKILL). Root cause: val+gen > NCCL 600s watchdog under DDP. Hub: ckpts 500–3700.
- S6 (🔴): Same crash at step 3750. NCCL watchdog confirmed. `WorkNCCL(ALLREDUCE, NumelIn=1, Timeout=600000ms)` ran for 600003ms → SIGABRT. Confirmed val took 557s + gen 90s = 647s > 600s.
- S7 (⬜): **NEXT** — after applying `stage2_rewrite_prompt.md`.

**Root cause of S5/S6 NCCL crash (confirmed from log):**
```
Step 3750 hits val_every=250 AND gen_every=250 simultaneously.
Rank 0: compute_val_metrics(2761 samples, batch_size=2) = 1380 passes ≈ 557s
        + run_generation_callback() ≈ 90s
        = 647s total on rank 0 while rank 1 waits at dist.barrier()
NCCL watchdog fires at 600s → SIGABRT
Fix: val capped to 500 samples at batch_size=16 (≈10s), gen_every=500,
     NCCL_TIMEOUT=1800s.
```

**Note on "streaming dataset" hypothesis:** Investigated and ruled out. Dataset
download does not cause NCCL desync — NCCL's watchdog only activates on pending
collectives, which are not called during data loading. The 55k-sample dataset
fits in ~225 MB RAM and is cached by HuggingFace after first download. Streaming
would break deterministic epoch_permutation sampling with no training-time benefit.

**All patches applied to current `train_sft.py` (from stage2_patch_prompt.md):**
- [x] Local Stage 2 checkpoints tried first; Hub downloads to `.hub_resume/` temp dir
- [x] `need_opt_reset=True` rebuilds optimizer/scheduler on data stream change
- [x] `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- [x] Default `max_seq_len=2048`
- [x] Prune skips `.hub_*` subdirs
- [x] `_build_sft_sample_truncated` — truncate reasoning instead of skipping

**Pending patches (from stage2_rewrite_prompt.md):**
- [ ] Bug 13: Save order `save→val→gen`
- [ ] Bug 14: Emergency save `push_to_hub=False`
- [ ] Bug 15: Remove `dist.barrier()` from timeout break path
- [ ] Bug 16: Default `lr=3e-4`, `warmup_steps=50`, `ema_decay=0.99`
- [ ] Bug 17: Delete 3 locally re-implemented Hub utility functions
- [ ] Bug 18: Default `gen_every=500`
- [ ] Bug 19A: `NCCL_TIMEOUT=1800` before `init_process_group`
- [ ] Bug 19B: Val capped to 500 samples at `val_batch_size=16`; new CLI args
- [ ] Data broadcast: Rank 0 loads, `broadcast_object_list` to rank 1

**Hyperparameters (S5–S6 / S7 planned):**
```
S5/S6: lr=1e-4, warmup=100, ema_decay=0.995, gen_every=250  ← NCCL crash
S7:    lr=3e-4, warmup=50,  ema_decay=0.99,  gen_every=500  ← after rewrite
batch_size=2, grad_accum=16 → effective batch=32 (global, DDP)
max_seq_len=2048, dataset_mix=full, num_epochs=3, total_steps=4920
val_max_samples=500, val_batch_size=16
```

**Success criteria:**
- val_ce < 1.5 (answer tokens only, using EMA weights)
- Generated text contains `<think>` blocks
- Coherent answers on GEN_PROMPTS

---

### Stage 3 — Coconut-Ouroboros ⬜ NOT STARTED

| Sub-stage | K | Resume from | Output dir |
|---|---|---|---|
| 3.1 | 1 | Stage 2 final | runs/stage3_k1 |
| 3.2 | 4 | Stage 3.1 final | runs/stage3_k4 |
| 3.3 | 16 | Stage 3.2 final | runs/stage3_k16 |

Full spec: `stage3_agent_prompt.md`. Required additions to `baseline_trm_mamba.py`: `forward_with_hidden()` and `forward_from_embeddings()`.

---

### Stage 4 — GRPO ⬜  |  Stage 5 — Quantization ⬜

---

## Part 6 — File Audits & Engineering Principles

### 6.1 train_sft.py — Full Audit

**CONFIRMED BUGS — all addressed in `stage2_rewrite_prompt.md`:**

**Bug 13 — Save order is wrong (CRITICAL)**
Current order per optimizer step: `val_every → gen_every → save_every`. A timeout during generation loses the checkpoint for that step.
Fix: reorder to `save_every → val_every → gen_every`.

**Bug 14 — Emergency save includes Hub push (CRITICAL)**
The timeout handler calls `save_checkpoint(..., push_to_hub=args.push_to_hub)`. Hub upload inside the graceful-exit buffer risks SIGKILL before clean exit.
Fix: pass `push_to_hub=False` in the emergency save call.

**Bug 15 — DDP barrier deadlock on graceful exit (CRITICAL)**
After the emergency save, `dist.barrier()` is called before breaking. Rank 0 may be mid-save while rank 1 is already at the next micro-step. The `broadcast_timeout()` flag handles rank synchronization; the explicit barrier here is redundant and dangerous.
Fix: delete `dist.barrier()` from the timeout break path entirely.

**Bug 16 — Number-loop attractor not broken by lr=1e-4 (FUNCTIONAL)**
The model inherited a strong FineWeb-Edu number prior. lr=1e-4 is too conservative to overwrite it within the first epoch.
Fix: `lr=3e-4`, `warmup_steps=50`.

**Bug 17 — DRY violation: utility functions redefined (MEDIUM)**
`list_remote_checkpoint_names`, `download_checkpoint_from_hub`, and `sync_checkpoint_to_hub` are all re-implemented inside `train_sft.py`. Any Hub fix must be applied in two places.
Fix: delete local re-definitions; use canonical imports from `training_utils.py`.

**Bug 18 — gen_every=250 excessive overhead (MINOR)**
Generation takes ~90s. At gen_every=250 and 9.5s/step, that is 90s of non-checkpointed time every ~40 min.
Fix: `gen_every=500`.

**Bug 19 — NCCL watchdog killed by combined val+gen (CRITICAL — new)**
At step 3750 (divisible by both val_every=250 and gen_every=250), rank 0 spends 647s (557s val + 90s gen) while rank 1 waits at `dist.barrier()`. NCCL's default 600s watchdog fires.
Three sub-fixes: (A) `NCCL_TIMEOUT=1800`; (B) cap val to 500 samples at batch_size=16 (≈10s); (C) reorder callbacks (Bug 13 fix).

**DESIGN PRINCIPLES (proven by trial-and-error):**

- **Checkpoint before callbacks.** save → val → gen. Callbacks are interruptible; the checkpoint is not.
- **Emergency saves never push to Hub.** Hub push has unpredictable latency. Emergency saves write to local disk only.
- **Remove `dist.barrier()` from graceful exit.** `broadcast_timeout()` already propagates the flag. A barrier here risks NCCL deadlock when rank 0 is mid-save.
- **Val batch size is independent of train batch size.** No gradients needed; use `val_batch_size=16` regardless of training `batch_size=2`.
- **Rank 0 loads data; rank 1 receives via `broadcast_object_list`.** Eliminates double dataset download/tokenization. ~225 MB broadcast for 55k samples is fast and well within T4 RAM.
- **NCCL watchdog must be set tighter than the slowest barrier wait, but looser than expected worst case.** At 10s (val) + 90s (gen) = 100s max, 1800s is 18× headroom while still catching genuine hangs.
- **Optimizer/scheduler reset on data stream change.** Detected via `build_data_fingerprint()` comparison on resume. Keep model weights and EMA; reset everything else.
- **EMA decay governs val/gen responsiveness.** decay=0.99 at ~5000 steps gives half-life ~69 steps. Use 0.99 for Stage 2, 0.995 for Stage 1 (62,500 steps).
- **Local Stage 2 checkpoints always take priority over Hub.** Hub downloads go to `.hub_resume/` (temp), never into `output_dir`.
- **Prune logic must filter non-training directories.** Skip `.hub_*`, `.tmp`, and anything failing `checkpoint_step_from_name()`.
- **The cosine schedule total_steps is recomputed on optimizer reset.** After data-stream change resume (step=0), total_steps = steps_per_epoch × num_epochs using the new dataset size.

---

### 6.2 pretrain.py — Notes
*(unchanged from previous BLUEPRINT version)*

Architecture is sound. Plateau at val_ce=5.32 is a data/compute limitation (705M of ~1.85B compute-optimal tokens). No bugs found.

---

### 6.3 baseline_trm_mamba.py — Notes
*(unchanged from previous BLUEPRINT version)*

Architecture is sound. Stage 3 additions (`forward_with_hidden`, `forward_from_embeddings`) are additive-only — no existing code changes.

---

### 6.4 training_utils.py — Notes

Canonical implementations for all Hub/checkpoint/EMA utilities. **Do not re-implement in training scripts.** Bug 17 documents the consequences of doing so.

---

## Part 7 — Checkpoint Format

```python
# Stage 1 (pretrain.py)
{
    "step", "epoch", "chunks_in_epoch", "tokens_processed",
    "model_state_dict", "ema_backbone_state_dict",  # includes lm_head.weight alias
    "optimizer", "scheduler", "scaler", "ema",
    "backbone_config", "pretrain_config", "val_ce",
}

# Stage 2 (train_sft.py)
{
    "stage": "sft",
    "step", "epoch", "samples_seen", "val_ce",
    "model_state_dict", "ema_backbone_state_dict",  # includes lm_head.weight alias
    "optimizer", "scheduler", "scaler", "ema",
    "backbone_config", "sft_config", "data_fingerprint",
}

# Stage 3 (recursive_finetune.py — to be created)
{
    "stage": "coconut",
    "n_latent", "lat_token_id", "vocab_size",
    # + all Stage 2 keys
}
```

**Classification:** `"sft_config" in state` → Stage 2. `"pretrain_config" in state` (without `"sft_config"`) → Stage 1. Stage 2 loader handles Stage 1 checkpoints as cold-start (load weights + EMA, reset optimizer/scheduler, return step=0, reset_optimizer=True).

---

## Part 8 — Bug Tracker

| # | Description | Status |
|---|---|---|
| 1–5 | Various early-stage bugs | ✅ FIXED |
| 6 | Resume downloads Hub ckpts even when local Stage 2 exists | ✅ FIXED |
| 7 | Hub downloads placed inside output_dir | ✅ FIXED |
| 8 | Prune deleted all Stage 2 checkpoints | ✅ FIXED |
| 9 | Optimizer not reset on data stream change | ✅ FIXED |
| 10 | max_seq_len=1024 filtered 97% of reasoning datasets | ✅ FIXED |
| 11 | NCCL teardown race: post-training SIGABRT on Dual T4 | ✅ BENIGN |
| 12 | EMA decay=0.995 causes severe lag | ✅ MITIGATED (reduce to 0.99 in S7) |
| 13 | Save order wrong: val→gen→save instead of save→val→gen | 🔴 OPEN — stage2_rewrite_prompt.md |
| 14 | Emergency save includes Hub push | 🔴 OPEN — stage2_rewrite_prompt.md |
| 15 | DDP barrier deadlock on graceful timeout exit | 🔴 OPEN — stage2_rewrite_prompt.md |
| 16 | lr=1e-4 insufficient to break number-loop attractor | 🔴 OPEN — stage2_rewrite_prompt.md |
| 17 | DRY: utility functions re-implemented in train_sft.py | 🔴 OPEN — stage2_rewrite_prompt.md |
| 18 | gen_every=250 adds ~9% overhead per gen window | 🔴 OPEN — stage2_rewrite_prompt.md |
| 19 | NCCL watchdog (600s) killed by val(557s)+gen(90s)=647s under DDP | 🔴 OPEN — stage2_rewrite_prompt.md |

---

## Part 9 — Compute Plan

| Stage | Platform | Estimate | Notes |
|---|---|---|---|
| 1 | Kaggle Dual T4 | ✅ Done (705M tokens) | Would benefit from 5B tokens on TRC |
| 2 | Kaggle Dual T4 | 1–2 sessions after rewrite | S7 with all Bug 13–19 fixes applied |
| 3 | TRC preferred | ~4–8h per K sub-stage | Coconut K=1→4→16 |
| 4 | TRC + unsloth | ~8–12h | GRPO G=4 rollouts |
| 5 | Local / Jetson | ~2h | Post-training quantization |

**TRC note:** Application submitted 2026-04-07. Awaiting email.

---

## Part 10 — References

- **Mamba** (Gu & Dao, 2023); **Jamba** (AI21 Labs, 2024); **Coconut** (Meta, arXiv:2412.06769)
- **TRM** (Samsung, arXiv:2510.04871); **DeepSeek-R1** (2025); **Quamba** (2024)
- **GPT-2** residual scaling; **RoPE** (Su et al., 2021); **SwiGLU** (Shazeer, 2020); **GQA** (Ainslie et al, 2023)
- **Chinchilla scaling laws** (Hoffmann et al., 2022) — compute-optimal: ~20 tokens per param
- **TRL GRPOTrainer** — https://huggingface.co/docs/trl
- **OpenR1 datasets** — https://huggingface.co/collections/open-r1/reasoning-datasets
