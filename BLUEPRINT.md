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
| 2 | SFT | 🔴 NEEDS FIX — S5 hard timeout (43200.6s), val_ce frozen at 5.62 | answer val_ce < 1.5 |
| 3 | Recursive Inference (Coconut-Ouroboros) | ⬜ NOT STARTED | Stage 2 gate |
| 4 | GRPO | ⬜ NOT STARTED | Stage 3 gate |
| 5 | Quantization | ⬜ NOT STARTED | Stage 4/3 gate |

### Immediate next actions

**Before re-running Stage 2:** Rewrite `train_sft.py` (see Part 6 — File Audits). The current implementation has multiple confirmed bugs causing the plateau and the hard-kill timeout. Do NOT re-run S6 on the current `train_sft.py`.

**After rewrite, S6 run command:**
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
Key changes vs S5: `--lr 3e-4` (burst to escape number-loop attractor), `--warmup_steps 50`, `--ema_decay 0.99`, `--gen_every 500` (reduce EMA-swap overhead), `--graceful_exit_buffer_minutes 15` (larger buffer given Hub push + DDP barrier latency).

---

## Part 1 — Root Failure Record

| Session | Cause | Effect | Status |
|---|---|---|---|
| Pre-S1 | No pre-training | Comma-loops from random init | ✅ Fixed: pretrain first |
| S1 | max_seq_len=1024 | Filtered all reasoning; no `<think>` | ✅ Fixed: 2048 + truncation |
| S2 | Bugs 6–10 cascade | Hub downloads in output_dir; prune deleted all Stage 2 ckpts; no optimizer reset | ✅ Fixed |
| S5 | Number-loop attractor + save ordering + DDP barrier | val_ce frozen 5.62; SIGKILL before clean exit | 🔴 Needs train_sft.py rewrite |

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
| Stage 3 recursion | Coconut-Ouroboros, K=1→4→16. NOT TRM EMA-loop. See stage3_agent_prompt.md. |
| Hub repo | WeirdRunner/Ouroboros (private) |

---

## Part 4 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `baseline_trm_mamba.py` | 0 | ✅ COMPLETE | Stage 3 needs `forward_with_hidden` + `forward_from_embeddings` (surgical, see stage3_agent_prompt.md) |
| `viability_gate.py` | 0 | ✅ COMPLETE | |
| `training_utils.py` | All | ✅ COMPLETE | Core utilities; do not duplicate in training scripts |
| `pretrain.py` | 1 | ✅ COMPLETE | Last Hub ckpt: ckpt-0021000. See Part 6 for plateau analysis. |
| `train_sft.py` | 2 | 🔴 NEEDS REWRITE | See Part 6 for full audit and required changes |
| `stage2_patch_prompt.md` | 2 | ✅ APPLIED | All 6 changes in current `train_sft.py` |
| `stage3_agent_prompt.md` | 3 | ✅ COMPLETE | Feed to coding agent after Stage 2 gate |
| `recursive_finetune.py` | 3 | ⬜ NOT CREATED | Generate from `stage3_agent_prompt.md` |

---

## Part 5 — Stage Definitions

### Stage 2 — SFT 🔴 NEEDS REWRITE BEFORE S6

**Sessions:**
- S1 (✅): stratos-only, max_seq_len=1024, val_ce=4.9153, no `<think>` learning. Hub: ckpt-0002979.
- S2 (❌): full-mix DDP, Bugs 6–10, val_ce=5.7135. All local Stage 2 ckpts deleted.
- S3–S4 (✅): dry-runs; all patches verified.
- S5 (🔴): full-scale DDP, hard-killed at 43200.6s (~step 3700). val_ce plateaued at 5.62 from step 2500 onward. Generation frozen (number loops) throughout. Exit code 137 (SIGKILL). Root cause: number-loop attractor + save ordering bug + DDP barrier deadlock on timeout. Hub: ckpts 500–3700.
- S6 (⬜): **NEXT** — after rewriting train_sft.py. See Part 6.

**All patches applied to current `train_sft.py` (from stage2_patch_prompt.md):**
- [x] Local Stage 2 checkpoints tried first; Hub downloads to `.hub_resume/` temp dir
- [x] `need_opt_reset=True` rebuilds optimizer/scheduler on data stream change
- [x] `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- [x] Default `max_seq_len=2048`
- [x] Prune skips `.hub_*` subdirs
- [x] `_build_sft_sample_truncated` — truncate reasoning instead of skipping

**Hyperparameters (S5 / S6 planned):**
```
S5: lr=1e-4, warmup=100, cosine to 1e-5, ema_decay=0.995, gen_every=250  ← plateau, SIGKILL
S6: lr=3e-4, warmup=50, cosine,           ema_decay=0.99,  gen_every=500  ← after rewrite
batch_size=2, grad_accum=16 → effective batch=32 (global, DDP)
max_seq_len=2048, dataset_mix=full, num_epochs=3, total_steps=4920
```

**Data counts at max_seq_len=2048 (with truncation):**
55,230 total | train: 52,469 | val: 2,761

**S5 val_ce analysis:**
```
Steps 250–2500: 5.7453 → 5.6245  (improvement)
Steps 2500–3500: 5.6245 → 5.6257 (REVERSAL — model is diverging or overfitting)
Train/val gap at step 3500: ~2.63 CE units — not EMA lag, genuine distribution mismatch
Root cause: model memorizes integer-answer patterns from MetaMathQA/OpenR1-Math
            but does not generalize to sentence-level structure
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

This section captures every design insight that took trial-and-error to discover. It is the authoritative reference for writing, debugging, or reviewing any training script in this project.

---

### 6.1 train_sft.py — Full Audit

**CONFIRMED BUGS (must fix before S6):**

**Bug 13 — Save order is wrong (CRITICAL)**  
Current order per optimizer step: `val_every → gen_every → save_every`. Generation with EMA weights requires swapping ~740MB of parameters twice and running greedy decode on 5 prompts — takes ~90 seconds total. When a Kaggle session timeout fires during generation, the checkpoint for that step is **never saved**. This is why the S5 emergency save may have fired at step ~3700 rather than at a regular save_every boundary.  
Fix: reorder to `save_every → val_every → gen_every`. Checkpoint is written before any slow callbacks. This is the exact order used in `pretrain.py` which has never lost a checkpoint to timeout.

**Bug 14 — Emergency save includes Hub push (CRITICAL)**  
The timeout handler calls `save_checkpoint(..., push_to_hub=args.push_to_hub)`. With `push_to_hub=True`, a 740MB Hub upload runs inside the 7-minute graceful-exit buffer. The upload is nominally fast (~10s) but Hub API latency varies. If it hangs, the buffer is exhausted and Kaggle SIGKILL fires before `dist.barrier()` completes. In S5 this caused exit code 137.  
Fix: pass `push_to_hub=False` in the emergency save call. The local checkpoint is safe; Hub sync can happen on the next regular save.

**Bug 15 — DDP barrier placement on graceful exit (CRITICAL)**  
After the emergency save, the code does `dist.barrier()` before breaking. If rank 0 is mid-Hub-push and rank 1 already advanced to the next micro-step, the barrier deadlocks until Kaggle SIGKILL fires. The `broadcast_timeout()` function is only called once per optimizer step, so rank 1 may be several seconds ahead.  
Fix: increase `graceful_exit_buffer_minutes` to at least 15 to absorb this slack, AND skip Hub push in emergency saves (Bug 14 fix). Optionally, replace the final `dist.barrier()` in the timeout path with a `dist.barrier(timeout=timedelta(minutes=2))` call to fail gracefully instead of deadlocking.

**Bug 16 — Number-loop attractor not broken by lr=1e-4 (FUNCTIONAL)**  
The model inherited a strong FineWeb-Edu number prior from Stage 1. After 3500 SFT steps at lr=1e-4 on a mix where answer tokens are dominated by integers, val_ce reversed and generation remained frozen. lr=1e-4 is too conservative to overcome the attractor in the first epoch.  
Fix: start with lr=3e-4 (3× higher) for the first 50 warmup steps, then cosine decay. The higher learning rate is justified because we're trying to overwrite an existing prior, not learn from scratch. If spikes become excessive (>30% of steps), reduce to 2e-4.

**Bug 17 — DRY violation: utility functions redefined (MEDIUM)**  
`list_remote_checkpoint_names`, `download_checkpoint_from_hub`, and `sync_checkpoint_to_hub` are all re-implemented inside `train_sft.py` with modified signatures. This shadows the versions in `training_utils.py` and creates drift. Any future fix to the Hub logic must be applied in both files.  
Fix: delete the local re-definitions and import the canonical versions from `training_utils.py`. Adjust call sites to match the imported signatures.

**Bug 18 — gen_every=250 causes excessive EMA-swap overhead (MINOR)**  
At 9.5s/step and gen_every=250, generation runs every ~40 minutes. Each generation call takes ~90s (EMA swap + 5 prompts × 120 tokens greedy decode). This represents ~225s/2500s = ~9% overhead per gen window. More importantly, it adds 90s of non-checkpointed time to every val window.  
Fix: set gen_every=500. Generation is for human monitoring only; running half as often is sufficient.

**DESIGN PRINCIPLES proven by trial-and-error (preserve in all future scripts):**

- **Checkpoint before callbacks.** The save must happen before val CE computation and generation. Callbacks are slow and can be interrupted. The checkpoint is the only thing that must survive.
- **Emergency saves never push to Hub.** Hub push is a network call with unpredictable latency. Emergency saves exist to protect against hard session limits; their only goal is writing a valid file to local disk.
- **Optimizer/scheduler reset on data stream change.** When the dataset fingerprint changes on resume (different mix, different max_seq_len, different seed), reset step=0, optimizer, scheduler, and scaler. Keep model weights and EMA. The code detects this via `build_data_fingerprint()` comparison.
- **Local Stage 2 checkpoints always take priority over Hub.** Never download a Hub checkpoint when a valid local Stage 2 checkpoint exists in output_dir. Hub downloads go to `.hub_resume/` (a temp directory), never into output_dir itself, to prevent contaminating the prune logic.
- **Prune logic must filter non-training directories.** The prune loop must skip `.hub_*`, `.tmp`, and any directory that doesn't pass `checkpoint_step_from_name()`. Without this filter, Hub-downloaded checkpoints get pruned into as regular training checkpoints and delete legitimate saves.
- **EMA decay governs val/gen responsiveness.** With decay=0.995 and <5000 steps, the EMA is a heavily smoothed average of the full training trajectory. val_ce using EMA weights will appear higher than the live model and move slowly. Use decay=0.99 for Stage 2 (half-life ~69 steps vs ~139 at 0.995). For Stage 1 pretrain with 62,500 total steps, 0.995 is appropriate.
- **DDP world_size=2 requires every synchronization point to be symmetric.** Both ranks must call `dist.barrier()` the same number of times. Any conditional that skips a barrier on rank 0 (e.g., inside `if is_main_process(rank)`) must have a matching unconditional barrier or a paired dummy barrier on rank 1.
- **The generation uses EMA weights via `ema_scope`.** This context manager swaps live weights for EMA shadow, runs generation, then restores. It requires two full-model copies in memory (~1.5GB for nano). This is acceptable but means generation cannot be concurrent with forward passes.
- **Spike monitor bias correction matters early.** Without bias correction, the EMA-based spike detector will flag nearly every step in the first 100 steps (EMA starts near 0 or the first loss value). The SpikeMonitor in train_sft.py correctly applies bias correction (matches pretrain.py behavior).
- **The cosine schedule total_steps is computed from scratch on optimizer reset.** When resuming with a new dataset (need_opt_reset=True), total_steps = steps_per_epoch * num_epochs is recalculated using the new dataset's size. This is correct because we're training fresh from step 0.

---

### 6.2 pretrain.py — Plateau Analysis

**Observed:** val_ce plateaued at ~5.32 at step 21,501 (705M tokens, nano 92M).

**Is it a code bug?** No. The pretrain.py code has been carefully reviewed and is architecturally sound. The optimizer, data pipeline, scheduler, and checkpoint handling all function correctly. The plateau is a data/compute limitation.

**Explanation:**
- Chinchilla scaling laws suggest compute-optimal training of 92M params requires ~1.85B tokens. We trained on 705M = 38% of compute-optimal.
- The LR schedule was designed for a 2B token budget (~62,500 steps). Session timeout at step 21,501 means the LR had decayed to ~3.1e-4 (still not too low, not the cause).
- A 92M model should reach ~3.5–4.0 val CE on educational web text with sufficient compute. 5.32 reflects primarily insufficient training, not architecture failure.
- The number-bias in FineWeb-Edu (financial articles, STEM content) creates a strong integer-sequence prior that 705M tokens wasn't enough to balance with richer language structure.

**Implications for future pre-training (TRC hardware):**
- Use `token_budget=5_000_000_000` (5B tokens) for compute-optimal nano training.
- Use `chunk_size=2048` instead of 1024 to leverage Mamba SSM's long-context advantage. The SSM recurrent state accumulates O(d_state) = O(16) compressed history per token; longer chunks give more context for each supervision signal.
- Consider including instruction-formatted text (10–20% of budget) to pre-load the QA response format, reducing the SFT sample efficiency problem.

---

### 6.3 baseline_trm_mamba.py — Investigation Notes

**Architecture is sound.** No bugs found. The following are observations for future reference:

**BF16 on T4:** `torch.cuda.is_bf16_supported()` returns True on T4 (Turing, sm_75) due to PyTorch software emulation. Hardware BF16 requires Ampere (sm_80+). In practice, BF16 on T4 runs at approximately the same speed as FP16 because neither has native BF16 tensor cores. The `dtype=bfloat16` path in training is not wrong, but on T4 `float16` would be equally fast and has better-characterized numerics. On A100/H100 (TRC), BF16 is preferable.

**No gradient checkpointing:** For sequences >2048 or model sizes >nano, activation memory grows linearly with sequence length × number of layers. The `TRMMambaGroup.forward()` method is the natural insertion point for `torch.utils.checkpoint.checkpoint()`. Not needed for nano at seq_len=2048 (VRAM ~6.6GB measured), but will be necessary for medium preset.

**Mamba masking overhead:** `MambaLayer.forward()` applies `h * mask` before and after the Mamba call when `attention_mask is not None`. This is correct behavior (SSM processes padded zeros rather than garbage) but adds two elementwise ops per Mamba layer per forward pass. Not a bottleneck at current scale.

**Tied embeddings and EMA shadow:** The EMA shadow dict must include `lm_head.weight` as an alias for `token_embedding.weight` when `tie_embeddings=True`. The checkpoint save code in both pretrain.py and train_sft.py correctly adds this alias. Any new training script must replicate this behavior or the checkpoint will fail to load into an EMA-swapped inference context.

**Stage 3 additions (surgical only):** Two methods must be added to `BaselineTRMMamba` immediately after the existing `forward` method: `forward_with_hidden()` and `forward_from_embeddings()`. See stage3_agent_prompt.md for the exact code. These are additive only; nothing in the existing forward path changes.

---

### 6.4 training_utils.py — Notes

Functions in this module are the canonical implementations. Training scripts must import from here rather than re-implementing:
- `ModelEMA` — EMA tracking and state dict I/O
- `ema_scope` — context manager for EMA-weight swap during val/gen
- `cosine_with_warmup` — LR schedule
- `build_adamw_optimizer` — parameter grouping (decay/no-decay)
- `list_local_checkpoints`, `try_load_state`, `checkpoint_step_from_name` — checkpoint discovery
- `list_remote_checkpoint_names`, `download_checkpoint_from_hub`, `sync_checkpoint_to_hub` — Hub I/O
- `cleanup_temporary_checkpoints` — removes `.tmp` dirs on startup
- `vram_gb`, `set_seed`, `pad_vocab_size` — utilities

**Do not re-implement any of these in training scripts.** The DRY violations in `train_sft.py` are Bug 17 and must be cleaned up.

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

**Classification logic:** `"sft_config" in state` → Stage 2. `"pretrain_config" in state` (or `"tokens_processed"` without `"sft_config"`) → Stage 1. Stage 2 loader must handle Stage 1 checkpoints as cold-start (load weights + EMA, reset optimizer/scheduler, return step=0, reset_optimizer=True).

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
| 12 | EMA decay=0.995 causes severe lag | ✅ MITIGATED (reduce to 0.99 in S6) |
| 13 | Save order wrong: val→gen→save instead of save→val→gen | 🔴 OPEN — train_sft.py rewrite |
| 14 | Emergency save includes Hub push | 🔴 OPEN — train_sft.py rewrite |
| 15 | DDP barrier deadlock on graceful timeout exit | 🔴 OPEN — train_sft.py rewrite |
| 16 | lr=1e-4 insufficient to break number-loop attractor | 🔴 OPEN — use 3e-4 in S6 |
| 17 | DRY: utility functions re-implemented in train_sft.py | 🔴 OPEN — train_sft.py rewrite |
| 18 | gen_every=250 adds ~9% overhead per gen window | 🟡 MINOR — set gen_every=500 |

---

## Part 9 — Compute Plan

| Stage | Platform | Estimate | Notes |
|---|---|---|---|
| 1 | Kaggle Dual T4 | ✅ Done (705M tokens) | Would benefit from 5B tokens on TRC for future runs |
| 2 | Kaggle Dual T4 | 2–3 more sessions after rewrite | S6 with lr=3e-4, ema=0.99, fixed save order |
| 3 | TRC preferred | ~4–8h per K sub-stage | Coconut K=1→4→16 |
| 4 | TRC + unsloth | ~8–12h | GRPO G=4 rollouts |
| 5 | Local / Jetson | ~2h | Post-training quantization |

**TRC note:** Application submitted 2026-04-07. Awaiting email. TRC provides free TPU/GPU compute via Google. Expected allocation: v4-8 TPU or A100 GPU. When confirmed, re-evaluate whether Stage 2 should be re-run on TRC (A100 BF16 would be 3–4× faster).

---

## Part 10 — References

- **Mamba** (Gu & Dao, 2023); **Jamba** (AI21 Labs, 2024); **Coconut** (Meta, arXiv:2412.06769)
- **TRM** (Samsung, arXiv:2510.04871); **DeepSeek-R1** (2025); **Quamba** (2024)
- **GPT-2** residual scaling; **RoPE** (Su et al., 2021); **SwiGLU** (Shazeer, 2020); **GQA** (Ainslie et al., 2023)
- **Chinchilla scaling laws** (Hoffmann et al., 2022) — compute-optimal: ~20 tokens per param
- **TRL GRPOTrainer** — https://huggingface.co/docs/trl
- **OpenR1 datasets** — https://huggingface.co/collections/open-r1/reasoning-datasets
