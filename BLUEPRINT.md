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
| 2 | SFT | 🟡 IN PROGRESS — full-scale run launched | answer val_ce < 1.5 |
| 3 | Recursive Inference (Coconut-Ouroboros) | ⬜ NOT STARTED | Stage 2 gate |
| 4 | GRPO | ⬜ NOT STARTED | Stage 3 gate |
| 5 | Quantization | ⬜ NOT STARTED | Stage 4/3 gate |

### Immediate next actions

**Stage 2 — currently running.** Expected: ~8–11h on Dual T4 DDP.

Post-run checklist:
- [ ] Confirm Hub has checkpoint(s) from this session
- [ ] Check val_ce trajectory for `<think>` tag emergence (expect by step 500–1000)
- [ ] If val_ce < 1.5: proceed to Stage 3
- [ ] If val_ce plateaus above 1.5: consider additional epochs or LR adjustment

**Stage 3 (after Stage 2 gate):**
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
| No pre-training | Comma-loops from random init | Pre-train first |
| Stage 2 S1: max_seq_len=1024 | Filtered all reasoning; no `<think>` learning | Raised to 2048 + truncation |
| Stage 2 S2: resume bug cascade | Hub downloads in output_dir; prune deleted all Stage 2 ckpts; optimizer not reset | Bugs 6–10 fixed |

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

**Fixed values:** vocab_size=151_680, rope_theta=1e6, mamba_d_state=16, mamba_d_conv=4,
mamba_expand=2, max_seq_len=2048, dropout=0.0, rms_norm_eps=1e-5, tie_embeddings=True.

---

## Part 3 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Tokenizer | Qwen2.5-0.5B; vocab_size=151_680 |
| Stage 1 outcome | Bypassed gate (val_ce 5.32 at step 21501); SFT from ckpt-0021000 |
| Stage 2 max_seq_len | 2048 — 1024 filtered 97% of Stratos reasoning |
| Stage 2 dataset | Full mix: Stratos + MetaMathQA + OpenHermes + OpenR1-Math + OpenR1-Code |
| Stage 2 target format | `User: {q}\n\nAssistant: <think>\n{reasoning}\n</think>\n{answer}{eos}` |
| Stage 2 starting checkpoint | Hub ckpt-0002979 (commit=8981b950) |
| Stage 2 DDP | Works correctly on Dual T4. SIGABRT is benign post-training teardown race (Bug 11). |
| Stage 3 recursion | Coconut-Ouroboros, K=1→4→16. NOT TRM EMA-loop. |
| Hub repo | WeirdRunner/Ouroboros (private) |

---

## Part 4 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `baseline_trm_mamba.py` | 0 | ✅ COMPLETE | Stage 3 needs `forward_with_hidden` + `forward_from_embeddings` (surgical) |
| `viability_gate.py` | 0 | ✅ COMPLETE | |
| `training_utils.py` | All | ✅ COMPLETE | |
| `pretrain.py` | 1 | ✅ COMPLETE | Last Hub ckpt: ckpt-0021000 |
| `train_sft.py` | 2 | ✅ IN USE | All bugs fixed; DDP works |
| `stage2_patch_prompt.md` | 2 | ✅ APPLIED | Changes 1–6 in current `train_sft.py` |
| `stage3_agent_prompt.md` | 3 | ✅ COMPLETE | Feed to coding agent after Stage 2 gate |
| `recursive_finetune.py` | 3 | ⬜ NOT CREATED | Generate from `stage3_agent_prompt.md` |

---

## Part 5 — Stage Definitions

### Stage 2 — SFT 🟡 IN PROGRESS

**Sessions:**
- S1 (✅): stratos-only, max_seq_len=1024, val_ce=4.9153, no `<think>` learning. Hub: ckpt-0002979.
- S2 (❌): full-mix DDP, Bugs 6–10, val_ce=5.7135. All local Stage 2 ckpts deleted.
- S3 (✅ training, teardown crash): dry-run max_steps=10, patches verified.
- S4 (✅ training, teardown crash): dry-run confirmed S3 diagnosis. Full-scale launched.

**All patches applied to `train_sft.py`:**
- [x] Local Stage 2 checkpoints tried first; Hub downloads to `.hub_resume/` temp dir
- [x] `need_opt_reset=True` rebuilds optimizer/scheduler on data stream change
- [x] `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- [x] Default `max_seq_len=2048`
- [x] Prune skips `.hub_*` subdirs
- [x] `_build_sft_sample_truncated` — truncate reasoning instead of skipping

**Hyperparameters (active run):**
```
lr=1e-4, warmup=100, cosine to 1e-5
batch_size=2, grad_accum=16 → effective batch=32 (global, DDP)
max_seq_len=2048, ema_decay=0.995
dataset_mix=full, num_epochs=3
```

**Expected data counts (at max_seq_len=2048 with truncation):**
~35000–38000 total samples; 3 epochs ≈ ~3300–3600 steps.

**Success criteria:**
- val_ce < 1.5 (answer tokens only)
- Generated text contains `<think>` blocks
- Coherent answers on GEN_PROMPTS

---

### Stage 3 — Coconut-Ouroboros ⬜ NOT STARTED

| Sub-stage | K | Resume from | Output dir |
|---|---|---|---|
| 3.1 | 1 | Stage 2 final | runs/stage3_k1 |
| 3.2 | 4 | Stage 3.1 final | runs/stage3_k4 |
| 3.3 | 16 | Stage 3.2 final | runs/stage3_k16 |

Full spec: `stage3_agent_prompt.md`. Required additions to `baseline_trm_mamba.py`:
`forward_with_hidden()` and `forward_from_embeddings()`.

---

### Stage 4 — GRPO ⬜  |  Stage 5 — Quantization ⬜

---

## Part 6 — Checkpoint Format

```python
{
    "step", "epoch", "model_state_dict",
    "ema_backbone_state_dict",   # includes lm_head.weight alias when tie_embeddings=True
    "optimizer", "scheduler", "scaler", "ema",
    "backbone_config", "val_ce",
    # Stage 2 additions:
    "stage": "sft", "sft_config", "data_fingerprint", "samples_seen",
}
```

---

## Part 7 — Bug Tracker

### Bugs 1–5 ✅ FIXED
### Bug 6 — Resume downloads Hub ckpts even when local Stage 2 exists ✅ FIXED
### Bug 7 — Hub downloads placed inside output_dir ✅ FIXED
### Bug 8 — Prune deleted all Stage 2 checkpoints ✅ FIXED
### Bug 9 — Optimizer not reset on data stream change ✅ FIXED
### Bug 10 — max_seq_len=1024 filtered 97% of reasoning datasets ✅ FIXED

### Bug 11 — NCCL teardown race: post-training SIGABRT on Dual T4 ✅ BENIGN (no fix needed)
**Symptom:** `process 1 terminated with signal SIGABRT` after training completes.
**Root cause:** After the training loop exits, rank 0 and rank 1 reach teardown at slightly
different times. One rank enters a final scalar ALLREDUCE (e.g. `distributed_mean` or
`dist.barrier()`) while the other has already moved past it. After the 600s NCCL watchdog
timeout, the lagging rank is killed with SIGABRT.
**Impact:** Zero. All checkpoints, Hub uploads, and EMA weights are written inside the training
loop before this point. The crash is pure cleanup noise.
**Prior incorrect diagnosis:** "mamba_ssm CUDA kernels fail in mp.spawn subprocess" — WRONG.
DDP + mamba_ssm works correctly on Kaggle Dual T4. The crash is timing-only, not architectural.
**Optional future fix:** Add a final `dist.barrier()` inside `run_training` before
`dist.destroy_process_group()` to synchronize both ranks cleanly. Low priority.

---

## Part 8 — Compute Plan

| Stage | Platform | Estimate | Notes |
|---|---|---|---|
| 1 | Kaggle Dual T4 | ✅ Done (705M tokens) | |
| 2 | Kaggle Dual T4 | ~8–11h per session | DDP works; teardown crash is benign |
| 3 | TRC preferred | ~4–8h | Per K sub-stage |
| 4 | TRC + unsloth | ~8–12h | GRPO G=4 rollouts |
| 5 | Local / Jetson | ~2h | Post-training quantization |

---

## Part 9 — References

- **Mamba** (Gu & Dao, 2023); **Jamba** (AI21 Labs, 2024); **Coconut** (Meta, arXiv:2412.06769)
- **TRM** (Samsung, arXiv:2510.04871); **DeepSeek-R1** (2025); **Quamba** (2024)
- **GPT-2** residual scaling; **RoPE** (Su et al., 2021); **SwiGLU** (Shazeer, 2020); **GQA** (Ainslie et al., 2023)
- **TRL GRPOTrainer** — https://huggingface.co/docs/trl
- **OpenR1 datasets** — https://huggingface.co/collections/open-r1/reasoning-datasets
