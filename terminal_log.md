# terminal_log.md — Project Ouroboros
**All verified terminal outputs, newest first.**

---

> **Naming note:** `phase1_viability_gate.py` → `viability_gate.py` | `train_sft_phase2.py` → `train_sft.py` | `Ouroboros_Blueprint_v3.md` → `BLUEPRINT.md`

---

## Stage 2 SFT — Session 6 (full-scale, DDP, NCCL watchdog crash at step 3750)
**Script:** `train_sft.py`  **Date:** 2026-04-07  **Hardware:** Kaggle Dual T4 (world_size=2)  
**Status:** 🔴 NCCL WATCHDOG KILL — same root cause as S5, now confirmed from log  

**Root cause (confirmed verbatim from log):**
```
4557.9s  [val] step=3750  val_ce=5.6254  val_acc=0.2203
4600.0s  WorkNCCL(SeqNum=13274, OpType=ALLREDUCE, NumelIn=1, NumelOut=1,
          Timeout(ms)=600000) ran for 600003 milliseconds before timing out.
4600.0s  [Rank 1] Watchdog caught collective operation timeout
4600.0s  terminate called after throwing an instance of 'c10::DistBackendError'
4600.0s  process 1 terminated with signal SIGABRT
```

**Kill chain:**
Step 3750 hits both `val_every=250` AND `gen_every=250` simultaneously.  
Rank 0: `compute_val_metrics(2761 samples, batch_size=2)` = 1380 forward passes ≈ 557s  
Rank 0: `run_generation_callback()` ≈ 90s  
Total rank 0 time: 647s. Rank 1 waits at `dist.barrier()` the entire time.  
NCCL watchdog (600s default) fires at 4600s → rank 1 SIGABRT.

**Val CE at crash:**
```
step=3750  val_ce=5.6254  val_acc=0.2203  (same plateau as S5)
```

**Note:** "streaming dataset" hypothesis was investigated and ruled out. Dataset
download does not trigger NCCL collectives; the watchdog only fires on pending
collectives. The crash is 100% caused by val(557s)+gen(90s)=647s > 600s NCCL timeout.

**Fix:** Apply `stage2_rewrite_prompt.md` (8 changes: NCCL_TIMEOUT=1800, val capped
to 500 samples at batch_size=16 ≈10s, gen_every=500, save→val→gen order,
emergency save no Hub push, remove barrier from timeout break, lr=3e-4,
rank 0 loads data + broadcast_object_list to rank 1).

---

## Stage 2 SFT — Session 5 (full-scale, DDP, HARD TIMEOUT at 43200.6s)
**Script:** `train_sft.py`  **Date:** 2026-04-06 → 2026-04-07  **Hardware:** Kaggle Dual T4 (world_size=2)  
**Status:** 🔴 HARD TIMEOUT (exit code 137, SIGKILL) — val_ce plateau confirmed, gate NOT met  

**Val CE trajectory (answer-only, EMA weights):**
```
Step    Val CE    Val Acc
  250   5.7453    0.2170
  500   5.7019    0.2191
 1000   5.6532    0.2207
 1500   5.6372    0.2211
 2000   5.6288    0.2208
 2500   5.6245    0.2206
 3000   5.6248    0.2203
 3500   5.6257    0.2203
 ~3700  ~5.625    ~0.220   (timeout emergency checkpoint)
```
Δval_ce over 3500 steps = 0.12 → effectively zero progress after step 2500.

**Generation (frozen throughout, representative):**
```
Q: What is 15 + 27?
A: 1000 - 1000 - 100 - 100 ...  uwr≈0.064
Q: Write a Python function...
A: 100000000000000000000000000...  uwr=1.000
Mean UWR: ~0.45–0.49 (inflated by single-token number loops)
```
No `<think>` tags observed at any step.

**Hub checkpoints (retained):**
```
WeirdRunner/Ouroboros/runs/stage2/checkpoint-0000500 through checkpoint-~003700
```

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
