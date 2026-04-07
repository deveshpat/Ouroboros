# terminal_log.md — Project Ouroboros
**All verified terminal outputs, newest first.**

---

> **Naming note:** `phase1_viability_gate.py` → `viability_gate.py` | `train_sft_phase2.py` → `train_sft.py` | `Ouroboros_Blueprint_v3.md` → `BLUEPRINT.md`

---

## Stage 2 SFT — Session 5 (full-scale, DDP, IN PROGRESS)
**Script:** `train_sft.py`
**Date:** 2026-04-06 → 2026-04-07
**Hardware:** Kaggle Dual T4 (world_size=2, DDP)
**Status:** 🟡 IN PROGRESS — log captured to step 1520 / 4920

**Dataset counts (with truncation at max_seq_len=2048):**
```
Bespoke-Stratos-17k=16594, MetaMathQA=11140, OpenHermes-2.5=8355,
OpenR1-Math-220k=11140, OpenR1-Code=8001
Total: 55,230  |  train: 52,469  val: 2,761
```

**Resume:** Hub `checkpoint-0002979` → data_changed detected → optimizer/scheduler reset (step=0).

**Training trajectory:**
```
Step     Train CE   Val CE (answer-only)   Val Acc
   1     4.5404     —
 250     3.3992     5.7453                 0.2170
 500     3.5602     5.7019                 0.2191   [ckpt saved + Hub uploaded]
 750     3.2355     5.6747                 0.2203
1000     3.3384     5.6532                 0.2207   [ckpt saved + Hub uploaded]
1250     3.2777     5.6447                 0.2209
1500     3.0564     5.6372                 0.2211   [ckpt saved + Hub uploaded]
1520     3.2845     5.6372                 0.2211   (last captured)
```

**Spike count (steps 1–1520):** 13 spikes (threshold=0.5)

**Generation @ step 1500 (EMA weights):**
```
Q: What is 15 + 27?
A: 1000 - 1000 - 100 - 100 - 100 - ...   uwr=0.064  ⚠ DEGENERATE

Q: Write a Python function that returns the factorial of n.
A: 100000000000000000000000000000000...   uwr=1.000  ⚠ number loop

Q: What is the capital of Japan?
A: 2000 The first two years of the year...  uwr=0.107  ⚠ DEGENERATE

Q: Explain what a neural network is in simple terms.
A: 100000000000000000000000000000000...   uwr=1.000  ⚠ number loop

Q: Solve for x: 3x + 6 = 21.
A: 2012 = 2012 = 2012 = 2012 = 1012...   uwr=0.295  ⚠ DEGENERATE

Mean UWR: 0.493
```

**⚠ Diagnosis — generation still degenerate at step 1500:**
- Train CE has dropped meaningfully (4.54 → 3.06) — the model IS learning from training data.
- Val CE (answer-only) drops slowly: 5.74 → 5.64 over 1500 steps (Δ = 0.10).
- EMA lags live weights at decay=0.995; at step 1500 the EMA model is approximately a 300-step-old snapshot.
- Generation patterns suggest the EMA model inherited the "number loop" bias from S1 (stratos-only, no `<think>` learned).
- Expected: generation should improve noticeably between steps 1500–3000 as EMA catches up.
- If val_ce is not below 5.0 by step 2500, consider reducing ema_decay to 0.99 in the next session.

**Hub checkpoints confirmed:**
- `WeirdRunner/Ouroboros/runs/stage2/checkpoint-0000500`
- `WeirdRunner/Ouroboros/runs/stage2/checkpoint-0001000`
- `WeirdRunner/Ouroboros/runs/stage2/checkpoint-0001500`

---

## Stage 2 SFT — Session 4 (dry-run, DDP, training succeeded / teardown crash)
**Date:** 2026-04-07 | **Status:** ✅ TRAINING COMPLETE — SIGABRT is cosmetic post-training NCCL teardown only

Training ran successfully for all intended steps. The SIGABRT occurred AFTER training completed,
during NCCL process group teardown (single-scalar ALLREDUCE race, NCCL 600s watchdog).

**Impact on full-scale run:** None. All patches verified working with DDP.

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
