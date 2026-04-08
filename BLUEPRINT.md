# Project Ouroboros — Master Blueprint

> **Thread-resume header.** Read Part 0 first in any new session.

---

## Part 0 — Quick-Resume Context

### What this project is
Novel hybrid Transformer-Mamba language model ("TRM-Mamba", 1:7 ratio) pre-trained on FineWeb-Edu, fine-tuned for chain-of-thought reasoning, with test-time compute scaling via Coconut-Ouroboros recursive inference (SERF).

### Current status

| Stage | Name | Status | Gate |
|---|---|---|---|
| 0 | Architecture & Viability | ✅ COMPLETE | All 4 gates passed |
| 1 | Pre-training | ✅ COMPLETE (bypassed gate) | val_ce=5.32 @ step 21501, Hub: ckpt-0021000 |
| 2 | SFT | 🔴 NEEDS REWRITE — DDP killed every session; removing DDP entirely | answer val_ce < 1.5 |
| 3 | Recursive Inference (Coconut-Ouroboros) | ⬜ NOT STARTED | Stage 2 gate |
| 4 | GRPO | ⬜ NOT STARTED | Stage 3 gate |
| 5 | Quantization | ⬜ NOT STARTED | Stage 4 gate |

### Immediate next actions

1. **Apply `train_sft_simplify_prompt.md`** to `train_sft.py` — strips DDP, adds `--dataset_mix=cached`.
2. **Run `prepare_sft_dataset.py`** once (Kaggle or Colab) to process + upload the ~55k sample dataset to `WeirdRunner/Ouroboros` (dataset repo, config `sft-mix-v1`). This is a one-time operation.
3. **Run S7** with the simplified single-GPU script.

**S7 run command (after both steps above):**
```bash
python train_sft.py \
  --preset nano \
  --max_seq_len 2048 \
  --dataset_mix cached \
  --num_epochs 3 \
  --batch_size 4 \
  --grad_accum 8 \
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

## Part 1 — Architecture

```
TokenEmbedding(vocab_size=151_680, d_model)
└─ n_groups × TRMMambaGroup:
   ├─ 1 × TRMBlock: RMSNorm + GQA(RoPE) + residual; RMSNorm + SwiGLU + residual
   └─ 7 × MambaLayer: RMSNorm + mamba_ssm.Mamba + residual   ← 1:7 ratio, fixed
FinalRMSNorm → LM Head (weight-tied)
```

| Preset | d_model | n_groups | n_heads | n_kv_heads | Params |
|---|---|---|---|---|---|
| nano | 512 | 1 | 8 | 4 | 92.5M |
| small | 1024 | 2 | 16 | 8 | ~270M |
| medium | 2048 | 2 | 16 | 8 | ~760M |

**Fixed:** vocab_size=151_680, rope_theta=1e6, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, max_seq_len=2048, dropout=0.0, rms_norm_eps=1e-5, tie_embeddings=True.

---

## Part 2 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Tokenizer | Qwen2.5-0.5B; vocab_size=151_680 |
| Stage 1 | Bypassed gate (val_ce 5.32 at step 21501). SFT starts from Hub ckpt-0021000. |
| Stage 2 max_seq_len | 2048 — 1024 filtered 97% of reasoning chains |
| Stage 2 dataset | Full mix: Stratos + MetaMathQA + OpenHermes + OpenR1-Math + OpenR1-Code (~55k samples). Pre-processed and cached at `WeirdRunner/Ouroboros` (dataset repo, config `sft-mix-v1`). |
| Stage 2 target format | `User: {q}\n\nAssistant: <think>\n{reasoning}\n</think>\n{answer}{eos}` |
| Stage 2 starting checkpoint | Hub ckpt-0002979 (stratos-only weights; optimizer reset on data change) |
| Stage 2 DDP | **REMOVED.** 3 consecutive NCCL watchdog kills (steps 3522, 3750, ~3700). Single GPU only. |
| Stage 2 data loading | Pre-processed cached HF dataset — load in <60s vs 25-45min from scratch |
| Stage 2 val speed | 500 samples capped, batch_size=16 → ~10s per val run |
| Stage 3 recursion | Coconut-Ouroboros, K=1→4→16. See `stage3_agent_prompt.md`. |
| Hub repo | WeirdRunner/Ouroboros (private model repo + dataset repo) |

---

## Part 3 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `baseline_trm_mamba.py` | 0 | ✅ COMPLETE | Stage 3 needs `forward_with_hidden` + `forward_from_embeddings` (surgical, see `stage3_agent_prompt.md`) |
| `viability_gate.py` | 0 | ✅ COMPLETE | |
| `training_utils.py` | All | ✅ COMPLETE | Canonical Hub/checkpoint/EMA utilities. Do not duplicate. |
| `pretrain.py` | 1 | ✅ COMPLETE | Last Hub ckpt: ckpt-0021000 |
| `prepare_sft_dataset.py` | 2 | ✅ READY — run once | Builds + uploads cached SFT dataset to HF |
| `train_sft.py` | 2 | 🔴 NEEDS PATCH | Apply `train_sft_simplify_prompt.md` (strip DDP, add cached mode) |
| `train_sft_simplify_prompt.md` | 2 | ✅ COMPLETE | Feed to coding agent |
| `stage3_agent_prompt.md` | 3 | ✅ COMPLETE | Feed to coding agent after Stage 2 gate |
| `recursive_finetune.py` | 3 | ⬜ NOT CREATED | Generate from `stage3_agent_prompt.md` |

---

## Part 4 — Stage 2 History & Root Cause

| Session | Root Cause | Outcome |
|---|---|---|
| S1 | max_seq_len=1024 filtered 97% of reasoning; no `<think>` learning | val_ce=4.92 plateau. Hub: ckpt-0002979. |
| S2 | Bugs 6–10 cascade (Hub downloads in output_dir, prune deleted all local ckpts) | val_ce=5.71. All local Stage 2 ckpts deleted. |
| S3–S4 | Dry-runs | Patches from `stage2_patch_prompt.md` verified. |
| S5 | NCCL watchdog: val(557s)+gen(90s)=647s > 600s timeout at step ~3700 | val_ce plateau 5.62. Exit code 137 (SIGKILL). |
| S6 | Same as S5 (rewrite not yet applied) | NCCL crash at step 3750. SIGABRT confirmed. |
| S7 (this run) | **Same NCCL crash at step 3522** — `stage2_rewrite_prompt.md` was NOT applied before run | val_ce still plateau ~3.5 (only saw 3 steps). DDP removed permanently. |
| S8 | ⬜ NEXT — after `train_sft_simplify_prompt.md` applied and cached dataset uploaded | — |

**Why DDP was removed (not just patched):** Every patch attempt introduced new failure modes (barrier deadlocks, checkpoint broadcast races, val timing overruns). The throughput gain (~1.8×) does not justify the engineering cost for a 92M-parameter model on a ~55k sample dataset where a single T4 can complete 3 epochs in ~14 hours.

**S7 training snippet (from log):**
```
step 3520  ce=3.2366  acc=0.4122  — training was running, then NCCL watchdog at step 3522
[rank1]: Terminating the process after attempting to dump debug info, due to ProcessGroupNCCL watchdog hang.
process 1 terminated with signal SIGABRT
```

---

## Part 5 — Stage 2 Hyperparameters (S8 target)

```
batch_size=4, grad_accum=8  → effective batch=32
max_seq_len=2048, lr=3e-4, warmup_steps=50, ema_decay=0.99
val_max_samples=500, val_batch_size=16
dataset_mix=cached (load from WeirdRunner/Ouroboros, sft-mix-v1)
num_epochs=3, total_steps≈4920 (depends on dataset size after split)
~9-10s/step on single T4 → ~14h for full 3 epochs
```

---

## Part 6 — Stage 3 Coconut-Ouroboros

| Sub-stage | K | Resume from | Output dir |
|---|---|---|---|
| 3.1 | 1 | Stage 2 final | runs/stage3_k1 |
| 3.2 | 4 | Stage 3.1 final | runs/stage3_k4 |
| 3.3 | 16 | Stage 3.2 final | runs/stage3_k16 |

Full spec: `stage3_agent_prompt.md`. Required additions to `baseline_trm_mamba.py`: `forward_with_hidden()` and `forward_from_embeddings()`. Gate: answer val_ce ≤ stage2_val_ce × 1.05.

---

## Part 7 — Checkpoint Format

```python
# Stage 2 (train_sft.py)
{
    "stage": "sft",
    "step", "epoch", "samples_seen", "val_ce",
    "model_state_dict", "ema_backbone_state_dict",
    "optimizer", "scheduler", "scaler", "ema",
    "backbone_config", "sft_config", "data_fingerprint",
}
# Stage 3 adds: "stage": "coconut", "n_latent", "lat_token_id", "vocab_size"
```

Stage 2 loader handles Stage 1 checkpoints as cold-start (load weights + EMA, reset optimizer, return step=0).

---

## Part 8 — Compute Plan

| Stage | Platform | Estimate |
|---|---|---|
| 2 | Kaggle single T4 | ~14h/session for 3 epochs. 1–2 sessions. |
| 3 | TRC preferred | ~4–8h per K sub-stage |
| 4 | TRC + unsloth | ~8–12h |
| 5 | Local / Jetson | ~2h |

TRC application submitted 2026-04-07. Awaiting email.
