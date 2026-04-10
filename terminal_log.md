# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**
**Purpose: session outcomes, key metrics, bugs found/fixed, generation samples.**
**Blueprint holds decisions/status. This file holds evidence.**

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

**Root cause:**
- "1000" loop = greedy-decode number attractor at low training signal. Model hasn't seen enough answer tokens to break it.
- val_acc declining while train_ce falling → overfitting signal. lr=3e-4 too aggressive for fine-tuning from plain-text pretrain.
- No `<think>` tags → model hasn't learned CoT format entry point; only ~8% of the schedule completed.
- Greedy decode masks true quality; temperature sampling required to diagnose properly.

**Fixes for S10:** lr=1e-4, warmup=100, dropout=0.1, ema_decay=0.995, num_epochs=5, temp=0.8/top_p=0.9 generation.

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
Root cause: `val_batch_size=16`, seq=2048, vocab=151,680 → logit tensor = 9.25 GiB. Fix: `val_batch_size=2` + `torch.cuda.empty_cache()`.

**Resume fingerprint (correct behavior):**
```
[resume] checkpoint-0003500: loaded model weights (step=3500),
         but data stream changed — resetting step/optimizer/scheduler.
[resume] saved_data={'dataset_mix': 'full', ...}
[resume] new_data  ={'dataset_mix': 'cached', ...}
```

---

## Stage 2 SFT — Sessions 5–7 (DDP, NCCL watchdog)
**Status:** 🔴 All killed by NCCL watchdog

| Session | Steps | Last val_ce | Root cause |
|---|---|---|---|
| S5 | ~3700 | 5.62 | val(557s)+gen(90s) > NCCL 600s timeout (SIGKILL exit 137) |
| S6 | 3750 | 5.63 | Same (SIGABRT) |
| S7 | 3522 | ~3.24 | Same — rewrite not applied before run |

Fix: `stage2_rewrite_prompt.md` → val capped (500 samples, batch=2, ~10s), `gen_every=500`, NCCL timeout raised to 1800s.

---

## Stage 2 SFT — Sessions 1–4 (Single GPU, early runs)

| Session | Steps | val_ce | Outcome |
|---|---|---|---|
| S1 | 2979 | 4.92 | Plateau. max_seq_len=1024 filtered 97% of reasoning chains. |
| S2 | ~1500 | 5.71 | Bugs 6–10 cascade; prune bug deleted all local checkpoints. |
| S3–S4 | <100 | — | Dry-runs only; patches verified. |

Fix: max_seq_len raised to 2048.

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

## Stage 0 — Baseline Smoke Test ✅ PASSED
```
parameters: 92,477,440 (92.5M)   initial loss: 11.9904   All checks passed.
```
