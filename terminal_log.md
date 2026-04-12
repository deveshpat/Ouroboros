# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**
**Purpose: session outcomes, key metrics, bugs found/fixed, generation samples.**
**Blueprint holds decisions/status. This file holds evidence.**

---

## Stage 3 — Smoke Test (Kaggle Dual T4, Latest Container Image)
**Script:** `jamba_coconut_finetune.py`  **Date:** 2026-04-11  **Hardware:** Kaggle Dual T4 (single GPU used)
**Status:** ✅ COMPLETED — all 3 stages ran without crashing. Two bugs found.

**Command:**
```
python jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit \
  --epochs_per_stage 1 --max_stage 2 --max_samples 200 \
  --batch_size 2 --grad_accum 4 \
  --session_timeout_hours 1.5 --graceful_exit_buffer_minutes 10 \
  --wandb_mode disabled --output_dir runs/smoke
```

**Environment:** Latest Container Image (NOT pinned) — mamba CUDA kernels unavailable, slow path used (~520s/step)

**Key metrics (verbatim):**
```
trainable params: 26,851,328 || all params: 3,056,191,360 || trainable%: 0.8786
d_model=2560  layers=28
<|lat|> token id: 65536  vocab: 65537
embed_tokens and lm_head paths verified after PEFT wrap.

Stage 0/2  (CoT warmup)   Epochs: 1  Steps/epoch: 24  Total: 24
Stage 1/2  1 latent pass  Epochs: 1  Steps/epoch: 1   Total: 1
  S1E0: 100%  ce=0.872  gn=0.954
  [val] s=1 ep=0 val_ce=0.0000 val_acc=0.0000   ← BUG

Stage 2/2  2 latent pass  Epochs: 1  Steps/epoch: 1   Total: 1
  S2E0: 100%  ce=1.464  gn=36.926               ← EXPLODING GRAD
  [val] s=2 ep=0 val_ce=0.0000 val_acc=0.0000   ← BUG
```

**Generation samples (verbatim):**
```
-- Generation @ step 2 stage=1 --
Q: What is 15 + 27?
A: The user is asking for the sum of 15 and 27...15 plus 20 is 35, then plus 7 more  [k_actual=1 uwr=0.613]
Q: What is the capital of Japan?
A: The user is asking for the capital of Japan. The answer is straightforward: the capital is Tokyo.  [k_actual=1 uwr=0.506]
Q: Solve for x: 3x + 6 = 21.
A: The user wants to solve the equation...x = 15/3 = 5. Wait, but let me check  [k_actual=1 uwr=0.716]
Mean UWR: 0.592

-- Generation @ step 3 stage=2 --
Q: What is the capital of Japan?
A: ,aemic for Japan. </think> about the question...Wait is a term for a type of furniture.  [k_actual=2 uwr=0.296]
Q: Explain what a neural network is in simple terms.
A: , </think>  Okay </think>  <think>  <think>  <think>  the  the  the  the  [k_actual=2 uwr=0.074]
Mean UWR: 0.290
```

**Checklist against Part 15:**
- [x] No import errors; trainable parameters printed
- [x] attn_implementation: eager fallback (flash-attn absent)
- [x] `<|lat|>` token added; embed_tokens resized
- [x] embed_tokens and lm_head paths verified after PEFT wrap
- [x] Stage 0 forward: last_hidden_state assert passes; loss finite
- [x] Stage 1 forward: prefix pass works; CE finite
- [x] Stage 2 forward: two prefix passes; no shape errors
- [x] stage_0/best/ created with adapter_model/ + training_state.pt
- [x] Stage advancement loads best_ckpt before next stage
- [x] Curriculum complete banner printed
- [ ] val_ce / val_acc correct — **FAIL: always 0.0**
- [ ] Gradient norm stable — **FAIL: gn=36.926 at stage 2**

**Root causes:**
1. **val=0.0**: `build_sample_at_stage` returns None for all val samples because sequences exceed `--max_seq_len 512`. `n_valid_total` stays 0, function returns `torch.zeros` fallback silently.
2. **gn=36.926**: Latent injection at k=2 with only 1 training step destabilises gradients. Real run needs `--max_grad_norm 0.3` or lower for k≥2.

**Fixes before real run:**
- `--max_seq_len 1024` (512 too tight for Jamba reasoning traces)
- `--max_grad_norm 0.3`
- Pin Kaggle environment to get mamba CUDA kernels working

**Duration:** 3768.7s (~62 min). Stage 0 truncated in log (ran ~24 steps before log capture). Stages 1 and 2 each ran 1 step.

---

## Stage 2 SFT — Session 9 (Single GPU, 3 epochs attempt)
**Script:** `train_sft_single_gpu.py`  **Date:** 2026-04-09  **Hardware:** Kaggle Single T4
**wandb run:** `comic-planet-8`  **Status:** 🔴 DEGENERATE — val_acc collapsed, no `<think>` tags

**Key metrics (verbatim from wandb):**
```
train/ce:         2.17731
train/ce_smooth:  2.04761
train/accuracy:   0.51889
gen/mean_uwr:     0.08077   ← BELOW viability threshold (0.10)
val/accuracy:     █▅▄▂▁▁    ← DECLINING over 6 val steps
train/lr:         ██▇▇▇▆▆▅▅▄▄▃▃▂▂▁▁  ← cosine ran to near-completion
world_size:       1
```

**Generation samples (verbatim):**
```
Q: What is 15 + 27?                     A: "1000 + 1000 + 1000 +..."
Q: What is the capital of Japan?         A: "1000 (1000) (1000) ..."
Q: Solve for x: 3x + 6 = 21.            A: "2012. The answer is ..."
Q: Explain what a neural network is...  A: "1. The answer is 1. ..."
Q: Write a Python function...            A: "1. The first two num..."
```

**Duration:** 14,472s (~4 hours). ~800/9,840 steps before timeout.

---

## Stage 2 SFT — Session 8 (DDP, OOM at step 250)
**Script:** `train_sft.py` (DDP v2)  **Date:** 2026-04-08  **Hardware:** Kaggle Dual T4
**Status:** 🔴 OOM at first val step

**Training snippet (verbatim):**
```
      1     3.6784      0.3993          -         -   1.2656   1.20e-05    1.527     4599
    100     3.1499      0.4089          -         -   1.4531   3.00e-04    4.345     4362
    200     2.6602      0.4589          -         -   0.9648   2.99e-04    4.345     4236
    240     2.6970      0.4450          -         -   1.0547   2.99e-04    1.546     4165
```

**Crash:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 9.25 GiB.
GPU 1: 14.56 GiB total, 433.81 MiB free
```

---

## Stage 2 SFT — Sessions 5–7 (DDP, NCCL watchdog)
**Status:** 🔴 All killed by NCCL watchdog

| Session | Steps | Last val_ce | Root cause |
|---|---|---|---|
| S5 | ~3700 | 5.62 | val(557s)+gen(90s) > NCCL 600s timeout |
| S6 | 3750 | 5.63 | Same (SIGABRT) |
| S7 | 3522 | ~3.24 | Same — rewrite not applied before run |

---

## Stage 2 SFT — Sessions 1–4 (Single GPU, early runs)

| Session | Steps | val_ce | Outcome |
|---|---|---|---|
| S1 | 2979 | 4.92 | Plateau. max_seq_len=1024 filtered 97% of reasoning chains. |
| S2 | ~1500 | 5.71 | Bugs 6–10 cascade; prune bug deleted all local checkpoints. |
| S3–S4 | <100 | — | Dry-runs only; patches verified. |

---

## Stage 1 Pre-training — Session 6 (Final)
**Hardware:** Kaggle Dual T4  **Status:** ✅ COMPLETE (graceful timeout)

```
Tokens processed: 704,544,768   Last val CE: 5.324
Hub: checkpoint-0021000 (commit=d70d2c49)
```

---

## Stage 0 — Viability Gate ✅ ALL GATES PASSED
```
G1 CE < 3.5        final CE = 2.0034   PASS ✓
G2 UWR > 0.1       mean UWR = 0.573    PASS ✓
G3 gnorm < 10.0    max = 4.0312        PASS ✓
G4 VRAM Δ < 1.0GB  Δ = 0.000 GB       PASS ✓
Total time: 3.4 min   Peak VRAM: 2.07 GB
```
