# terminal_log.md — Project Ouroboros
**All verified terminal outputs, newest first.**

---

> **Naming note:** `phase1_viability_gate.py` → `viability_gate.py` | `train_sft_phase2.py` → `train_sft.py` | `Ouroboros_Blueprint_v3.md` → `BLUEPRINT.md`

---

## Stage 2 SFT — Session 5 (full-scale, DDP, HARD TIMEOUT at 43200.6s)
**Script:** `train_sft.py`  **Date:** 2026-04-06 → 2026-04-07  **Hardware:** Kaggle Dual T4 (world_size=2)  
**Status:** 🔴 HARD TIMEOUT (exit code 137, SIGKILL) — val_ce plateau confirmed, gate NOT met  
**Kaggle output:** 2.23 GB (confirms ≥3 checkpoints saved locally before kill)

**Dataset counts (with truncation at max_seq_len=2048):**
```
Bespoke-Stratos-17k=16594, MetaMathQA=11140, OpenHermes-2.5=8355,
OpenR1-Math-220k=11140, OpenR1-Code=8001
Total: 55,230  |  train: 52,469  |  val: 2,761
```

**Resume:**
```
[resume] local Stage 2 candidates: 0 (newest step=none)
[resume] no local Stage 2 found; checking Hub for Stage 2 checkpoints...
[hub]  downloading runs/stage2/checkpoint-0002979 ...
[resume] runs/stage2/checkpoint-0002979  loaded model weights (step=2979),
         but data stream changed — resetting step/optimizer/scheduler for new data.
[resume] saved_data={'dataset_mix': 'stratos', 'max_seq_len': 1024, ...}
[resume] new_data  ={'dataset_mix': 'full',    'max_seq_len': 2048, ...}
[resume] optimizer/scheduler reset; training starts fresh from step 0.
```

**Val CE trajectory (answer-only, EMA weights):**
```
Step    Val CE    Val Acc    Note
  250   5.7453    0.2170
  500   5.7019    0.2191     ckpt saved + Hub uploaded
  750   5.6747    0.2203
 1000   5.6532    0.2207     ckpt saved + Hub uploaded
 1250   5.6447    0.2209
 1500   5.6372    0.2211     ckpt saved + Hub uploaded
 1750   5.6306    0.2209
 2000   5.6288    0.2208     ckpt saved + Hub uploaded; ckpt-0000500 pruned
 2250   5.6251    0.2207
 2500   5.6245    0.2206     ckpt saved + Hub uploaded; ckpt-0001000 pruned
 2750   5.6245    0.2205
 3000   5.6248    0.2203     ckpt saved + Hub uploaded; ckpt-0001500 pruned
 3250   5.6254    0.2203
 3500   5.6257    0.2203     ckpt saved + Hub uploaded; ckpt-0002000 pruned
 ~3700  ~5.625    ~0.220     timeout emergency checkpoint (Hub uploaded)
```
Δval_ce over 3500 steps = **0.12** → effectively zero progress. val_ce REVERSED after step 2500.

**Train CE (selected):**
```
    1   4.5404      980   3.2757    2000   3.1677    3000   3.3297
  240   3.3992     1500   3.0564    2380   2.9425    3380   2.6127
  500   3.5602     1280   3.3913*   2500   3.4573    3500   2.9457
                   (* gn=8.1875, largest observed)
```
Train/val gap at step 3500: train_ce ≈ 3.0, val_ce ≈ 5.63 → **Δ = 2.63**

**Spike count:** 20 spikes over steps 1–2500 (threshold=0.5)

**Generation (completely frozen steps 250–3500, representative sample):**
```
Step 250 / 1000 / 2000 / 3000:
  Q: What is 15 + 27?
  A: 1000 - 1000 - 100 - 100 - 100 ... [or pure 100000000...]  uwr≈0.064
  Q: Write a Python function...
  A: 100000000000000000000000000000...   uwr=1.000
  Q: What is the capital of Japan?
  A: 100000000000000000000000000000...   uwr=1.000   [occasionally: "2000 The first two years..."]
  Q: Explain what a neural network is...
  A: 100000000000000000000000000000...   uwr=1.000
  Q: Solve for x: 3x + 6 = 21.
  A: 2012 = 2012 = 2012 = ...            uwr≈0.125
  Mean UWR: ~0.45–0.49 (inflated by uwr=1.0 on single-token number loops)
```
No `<think>` tags observed at any step.

**WandB charts confirm (wandb.ai/devesh-patel0922-weirdrunner, run cardassian-frontier-2):**
- val/ce: plateau between 5.62–5.63 from step 1000 onward
- val/accuracy: peaks at 0.221, then gently decays to 0.218
- gen/mean_uwr: high at step 250 (0.87, due to all-ones UWR on number loops), drops to 0.45 and stays flat
- train/ce_smooth: clear downward trend 4.4 → 3.4
- train/grad_norm: mostly 1–3, occasional spikes to 8–9
- train/lr: cosine decay from 1e-4 to ~3e-5 by step 3500

**Hub checkpoints from Session 5 (all retained on Hub):**
```
WeirdRunner/Ouroboros/runs/stage2/checkpoint-0000500  (pruned locally at step 2000)
WeirdRunner/Ouroboros/runs/stage2/checkpoint-0001000  (pruned locally at step 2500)
WeirdRunner/Ouroboros/runs/stage2/checkpoint-0001500  (pruned locally at step 3000)
WeirdRunner/Ouroboros/runs/stage2/checkpoint-0002000  (pruned locally at step 3500)
WeirdRunner/Ouroboros/runs/stage2/checkpoint-0002500  (local: retained, keep_last=3)
WeirdRunner/Ouroboros/runs/stage2/checkpoint-0003000  (local: retained)
WeirdRunner/Ouroboros/runs/stage2/checkpoint-0003500  (local: retained)
WeirdRunner/Ouroboros/runs/stage2/checkpoint-~003700  (timeout emergency save)
```

**Root cause analysis:**
```
Primary:  The model is stuck in a number-loop attractor inherited from Stage 1
          FineWeb-Edu pre-training. Answer tokens in the full mix are dominated
          by integers (MetaMathQA, OpenR1-Math), which reinforces this prior
          instead of breaking it. The model learns to minimize CE on
          numeric answer tokens while never learning sentence-level structure.

Secondary: EMA@0.995 is not the root cause (by step 3500, EMA^3500 ≈ 0,
           initial weights fully forgotten). The val/train gap of 2.63 is
           genuine overfitting to numeric patterns, not EMA lag.

Third:     LR decayed from 1e-4 to ~3e-5 by step 3500 (cosine over 4920 steps).
           The model may need a higher-LR burst to escape the number attractor.
```

**Kaggle session terminated:** 43200.6s (hard limit), exit code 137 (SIGKILL). The Python graceful exit (code 0 path) did not complete. Most likely cause: DDP dist.barrier() deadlock after emergency save + Hub push consumed the 7-minute buffer, and rank 1 was already in a different state when rank 0 tried to synchronize.

---

## Stage 2 SFT — Session 4 (dry-run, DDP, training succeeded / teardown SIGABRT)
**Date:** 2026-04-07 | **Status:** ✅ TRAINING COMPLETE — SIGABRT is cosmetic post-training NCCL teardown only (Bug 11, benign)

---

## Stage 2 SFT — Session 3 (dry-run, DDP, SIGABRT — mis-diagnosed initially)
**Date:** 2026-04-07 | **Status:** ✅ TRAINING COMPLETE (see Session 4 for corrected diagnosis)

---

## Stage 2 SFT — Session 2 (full dataset mix, DDP, FAILED — Bugs 6–10 cascade)
**Date:** 2026-04-06 | **Status:** ❌ FAILED
```
val_ce=5.7135 at timeout. All local Stage 2 checkpoints deleted by prune bug.
75 loss spikes in 717 steps. Generation degraded to pure number loops.
Hub: checkpoint-0002979 (only valid Stage 2 checkpoint, kept from Session 1).
```

---

## Stage 2 SFT — Session 1 (stratos-only, single GPU)
**Date:** 2026-04-05 | **Status:** ✅ COMPLETE — step 2979, val_ce=4.9153, gate NOT met
```
step 2979  val_ce=4.9153  ← plateau
Total time: 290.2 min   Peak VRAM: 6.59 GB
[hub] uploaded  checkpoint-0002979 (commit=8981b950)
⚠ No <think> tags ever appeared — max_seq_len=1024 filtered all reasoning chains.
```

---

## Stage 1 Pre-training — Session 6 (Kaggle Dual T4)
**Status:** ✅ COMPLETE (graceful timeout) — steps 14902→21501, tokens 488M→705M
```
Tokens processed: 704,544,768   Last val CE: 5.324081295402125
```
Hub: checkpoint-0021000 (commit=d70d2c49) ← SFT starting point.

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
