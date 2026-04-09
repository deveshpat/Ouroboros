# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**
**Purpose: session outcomes, key metrics, bugs found/fixed, generation samples.**
**Blueprint holds decisions/status. This file holds evidence.**

---

## Stage 2 SFT — Session 9 (Single GPU, 3 epochs attempt)
**Script:** `train_sft_single_gpu.py`  **Date:** 2026-04-09  **Hardware:** Kaggle Single T4
**wandb run:** `comic-planet-8`  **Status:** 🔴 DEGENERATE — val_acc collapsed, no `<think>` tags

**Key metrics (from wandb run summary):**
```
train/ce:         2.17731  (final)
train/ce_smooth:  2.04761
train/accuracy:   0.51889
gen/mean_uwr:     0.08077   ← BELOW viability threshold (0.10)
val/accuracy:     █▅▄▂▁▁    ← DECLINING over 6 val steps (critical failure signal)
train/lr:         ██▇▇▇▆▆▅▅▄▄▃▃▂▂▁▁  ← cosine schedule ran to ~completion
world_size:       1 (single GPU confirmed)
```

**Generation samples (verbatim from wandb):**
```
Q: What is 15 + 27?                     A: "1000 + 1000 + 1000 +..."   [number loop]
Q: What is the capital of Japan?         A: "1000 (1000) (1000) ..."    [number loop]
Q: Solve for x: 3x + 6 = 21.            A: "2012. The answer is ..."    [wrong + loop]
Q: Explain what a neural network is...  A: "1. The answer is 1. ..."    [degenerate]
Q: Write a Python function...            A: "1. The first two num..."    [degenerate]
```

**Duration:** ~4 hours (14,472s). Ran ~800 steps before timeout (out of 9,840 total for 3 epochs).

**Root cause analysis:**
- `val/accuracy` declining to near 0 → strong overfitting OR catastrophic distribution shift signal
- No `<think>` tags generated → model hasn't learned the CoT format entry point
- "1000" loop = classic greedy-decode number attractor; insufficient training signal to break it
- LR 3e-4 may be too aggressive for fine-tuning from a stage-1 checkpoint that only saw plain text
- Greedy decoding in generation callback masks real quality; temperature sampling needed

**Fixes needed for S10:**
- Lower LR to `1e-4`, increase warmup to 100 steps
- Add `dropout=0.1` to combat overfitting
- Add temperature sampling (temp=0.8, top_p=0.9) to generation callback
- More epochs (5) to push past the plateau
- Verify val/ce (not just val/acc) on wandb before next run

---

## Stage 2 SFT — Session 8 (DDP, OOM at step 250)
**Script:** `train_sft.py` (DDP v2)  **Date:** 2026-04-08  **Hardware:** Kaggle Dual T4
**Status:** 🔴 OOM at first val step

**Training snippet:**
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
Root cause: `val_batch_size=16`, seq=2048, vocab=151,680 → logit tensor = 9.25 GiB.
Fix: `val_batch_size=2` + `torch.cuda.empty_cache()` before val.

**Resume fingerprint (correct behavior):**
```
[resume] runs/stage2/checkpoint-0003500 loaded model weights (step=3500),
         but data stream changed — resetting step/optimizer/scheduler for new data.
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

Fix applied: `stage2_rewrite_prompt.md` → val capped (500 samples, batch=16, ~10s), `gen_every=500`, NCCL timeout raised to 1800s.

---

## Stage 2 SFT — Sessions 1–4 (Single GPU, early runs)

| Session | Steps | val_ce | Outcome |
|---|---|---|---|
| S1 | 2979 | 4.92 | Plateau. max_seq_len=1024 filtered 97% of reasoning chains. |
| S2 | ~1500 | 5.71 | Bugs 6–10 cascade; prune bug deleted all local checkpoints. |
| S3–S4 | <100 | — | Dry-runs only; patches verified. |

Fix: max_seq_len raised to 2048. All S1-S2 bugs resolved in `stage2_rewrite_prompt.md`.

---

## Stage 1 Pre-training — Session 6 (Final, Kaggle Dual T4)
**Status:** ✅ COMPLETE (graceful timeout exit)

```
Tokens processed: 704,544,768   Last val CE: 5.324
Hub: checkpoint-0021000 (commit=d70d2c49)  ← SFT starting point
```

Note: val_ce=5.32 did not hit the < 3.0 gate, but Stage 1 was intentionally cut short to preserve GPU time for SFT. Architecture was proven healthy.

---

## Stage 0 — Viability Gate  ✅ ALL GATES PASSED
```
G1 CE < 3.5        final CE = 2.0034   PASS ✓
G2 UWR > 0.1       mean UWR = 0.573    PASS ✓
G3 gnorm < 10.0    max = 4.0312        PASS ✓
G4 VRAM Δ < 1.0GB  Δ = 0.000 GB       PASS ✓
Total time: 3.4 min   Peak VRAM: 2.07 GB
```

## Stage 0 — Baseline Smoke Test  ✅ PASSED
```
parameters: 92,477,440 (92.5M)   initial loss: 11.9904   All checks passed.
```
