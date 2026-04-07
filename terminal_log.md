# terminal_log.md — Project Ouroboros
**All verified terminal outputs, newest first.**

---

> **Naming note:** `phase1_viability_gate.py` → `viability_gate.py` | `train_sft_phase2.py` → `train_sft.py` | `Ouroboros_Blueprint_v3.md` → `BLUEPRINT.md`

---

## Stage 2 SFT — Session 5 (full-scale, DDP, TIMED OUT ~step 3900)
**Script:** `train_sft.py`
**Date:** 2026-04-06 → 2026-04-07
**Hardware:** Kaggle Dual T4 (world_size=2, DDP)
**Status:** 🔴 TIMED OUT — val_ce plateau, gate NOT met

**Dataset counts (with truncation at max_seq_len=2048):**
```
Bespoke-Stratos-17k=16594, MetaMathQA=11140, OpenHermes-2.5=8355,
OpenR1-Math-220k=11140, OpenR1-Code=8001
Total: 55,230  |  train: 52,469  val: 2,761
```

**Resume:** Hub `checkpoint-0002979` → data_changed detected → optimizer/scheduler reset (step=0).

**Val CE trajectory (answer-only):**
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
```
Δval_ce over 2000 steps = **0.12** — catastrophically slow. Projected to reach <1.5: ~34,000 more steps.

**Train CE trajectory (selected steps):**
```
    1   4.5404      980   3.2757    1900   3.7181
   20   3.8257     1000   3.3384    1940   3.3890
  240   3.3992     1100   3.2742    2000   3.1677
  500   3.5602     1200   3.4535    2100   3.4084
  740   3.2355     1400   3.3492    2200   3.5414
  900   3.7585     1500   3.0564    2380   2.9425
```

**Spike count (steps 1–2380):** 20 spikes (threshold=0.5)
Notable spike at step 1280: `gn=8.1875` — largest observed gradient norm.

**Generation (frozen across all checkpoints 250–2250):**
```
Q: What is 15 + 27?
A: 1000 - 1000 - 100 - 100 - ...      uwr=0.064  ⚠ DEGENERATE
Q: Write a Python function...
A: 100000000000000000000000000000...   uwr=1.000  ⚠ number loop
Q: What is the capital of Japan?
A: 2000 The first two years of...      uwr=0.107  ⚠ DEGENERATE
Q: Explain what a neural network is...
A: 100000000000000000000000000000...   uwr=1.000  ⚠ number loop
Q: Solve for x: 3x + 6 = 21.
A: 2012 = 2012 = 2012 = 1012 = ...    uwr=0.125  ⚠ DEGENERATE
Mean UWR: ~0.46–0.49 (deceptively high due to UWR=1.0 on number loops)
```
No `<think>` tags observed. Generation pattern completely unchanged from step 250 to step 2250.

**Hub checkpoints from Session 5:**
- `WeirdRunner/Ouroboros/runs/stage2/checkpoint-0000500` (pruned locally, Hub copy retained)
- `WeirdRunner/Ouroboros/runs/stage2/checkpoint-0001000`
- `WeirdRunner/Ouroboros/runs/stage2/checkpoint-0001500`
- `WeirdRunner/Ouroboros/runs/stage2/checkpoint-0002000`
- `WeirdRunner/Ouroboros/runs/stage2/checkpoint-~0003900` (timeout emergency save, step TBD)

**Diagnosis — why val_ce is plateauing:**
- EMA@0.995 at step 2000 reflects a ~200-step-old model snapshot. Live train CE (3.0–3.2) has improved meaningfully but EMA validation barely tracks it.
- The model inherited a strong "number loop" prior from Stage 1 FineWeb-Edu pre-training. Answer tokens in the SFT mix are dominated by math answers (numbers), reinforcing this pattern.
- Val acc completely flat at 0.2207–0.2211 — model has not learned to predict the next answer token correctly despite 2000 steps.
- Root cause: **EMA decay too high (0.995)** → EMA lags too far behind live weights at this step count and learning rate.

---

## Stage 2 SFT — Session 4 (dry-run, DDP, training succeeded / teardown crash)
**Date:** 2026-04-07 | **Status:** ✅ TRAINING COMPLETE — SIGABRT is cosmetic post-training NCCL teardown only

---

## Stage 2 SFT — Session 3 (dry-run, DDP SIGABRT — benign post-training teardown)
**Date:** 2026-04-07 | **Status:** ✅ TRAINING COMPLETE (see Session 4 for corrected diagnosis)

---

## Stage 2 SFT — Session 2 (full dataset mix, DDP, FAILED)
**Date:** 2026-04-06 | **Status:** ❌ FAILED — Bugs 6–10 cascading

Critical failures: Hub downloads contaminated output_dir → prune deleted all Stage 2 checkpoints → optimizer not reset on data change → 75 spikes → val_ce=5.7135. Hub: `checkpoint-0002979` (only valid Stage 2 checkpoint).

---

## Stage 2 SFT — Session 1 (stratos-only, single GPU)
**Date:** 2026-04-05 | **Status:** ✅ COMPLETE — step 2979, val_ce=4.9153, gate NOT met

⚠ No `<think>` tags ever appeared — max_seq_len=1024 filtered all reasoning chains.
```
step 2979  val_ce=4.9153  ← plateau
Total time: 290.2 min   Peak VRAM: 6.59 GB
[hub] uploaded  checkpoint-0002979 (commit=8981b950)
```

---

## Stage 1 — Pre-training, Session 6 (Kaggle Dual T4)
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
