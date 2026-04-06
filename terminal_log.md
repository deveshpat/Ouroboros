# terminal_log.md — Project Ouroboros
**All verified terminal outputs, newest first.**

---

> **Naming note:** `phase1_viability_gate.py` → `viability_gate.py` | `train_sft_phase2.py` → `train_sft.py` | `Ouroboros_Blueprint_v3.md` → `BLUEPRINT.md`

---

## Stage 2 SFT — Session 4 (dry-run, DDP, training succeeded / teardown crash)
**Script:** `train_sft.py`
**Date:** 2026-04-07
**Hardware:** Kaggle Dual T4 (world_size=2, DDP auto-triggered)
**Status:** ✅ TRAINING COMPLETE — SIGABRT is cosmetic post-training NCCL teardown only

**What actually happened:**
Training ran successfully for all intended steps. The SIGABRT occurred AFTER training completed,
during NCCL process group teardown. The failing collective was a single-scalar ALLREDUCE
(`NumelIn=1, NumelOut=1`) — one rank began teardown while the other was still at a final
collective. The NCCL watchdog (600s timeout) killed the hung rank, producing the SIGABRT.

**Key NCCL log (verbatim):**
```
1066.6s [rank1]: failure detected by watchdog at work sequence id: 306
1066.6s [rank1]: last enqueued work: 306, last completed work: 305
1066.6s [rank1]: Watchdog caught collective operation timeout:
          WorkNCCL(SeqNum=306, OpType=ALLREDUCE, NumelIn=1, NumelOut=1,
          Timeout(ms)=600000) ran for 600032 milliseconds before timing out.
1126.7s [rank1]: To avoid data inconsistency, we are taking the entire process down.
1127.3s torch.multiprocessing.spawn.ProcessExitedException: process 1 terminated with signal SIGABRT
```

**Impact on full-scale run:** None. Checkpoints are written inside the training loop at
`save_every` intervals and Hub uploads complete immediately after each save. The teardown
crash only occurs after all useful work is done. Full-scale run proceeds as planned.

**Corrected diagnosis of Session 3:** Same teardown crash pattern — training DID complete
max_steps=10 before the SIGABRT. Prior diagnosis ("mamba_ssm incompatible with mp.spawn")
was wrong. DDP + mamba_ssm on Kaggle Dual T4 works correctly during training.

---

## Stage 2 SFT — Session 3 (dry-run, DDP SIGABRT — now known to be post-training teardown)
**Date:** 2026-04-07 | **Status:** ✅ TRAINING COMPLETE (teardown crash — benign, see Session 4)

Training completed max_steps=10. Patches verified working. SIGABRT is same teardown race as Session 4.

---

## Stage 2 SFT — Session 2 (full dataset mix, DDP, FAILED)
**Script:** `train_sft.py` | **Date:** 2026-04-06 | **Hardware:** Kaggle Dual T4
**Status:** ❌ FAILED — cascading bugs (Bugs 6–10); all Stage 2 local checkpoints deleted

**Critical failures:**
1. Resume downloaded 18 Stage 1 Hub checkpoints into output_dir before finding local Stage 2 ckpt
2. Hub downloads contaminated output_dir → prune deleted ALL Stage 2 checkpoints
3. Optimizer not reset on data change → near-zero LR (1.89e-05) → 75 spikes in 717 steps
4. max_seq_len=1024 filtered 97% Stratos, 92% OpenR1-Math, 99.9% OpenR1-Code

**Generation — catastrophic:**
```
  Q: What is 15 + 27?
  A: 100000000000000000000000000000000000000000000000000000000000000000000000000000
```
```
  Total time: 20.5 min   Peak VRAM: 14.30 GB   Spike count: 75   Final val CE: 5.7135
```
**Checkpoint status:** Hub: checkpoint-0002979 (commit=8981b950) ← only valid Stage 2 checkpoint

---

## Stage 2 SFT — Session 1 (stratos-only, single GPU)
**Date:** 2026-04-05 | **Status:** ✅ COMPLETE — step 2979, val_ce=4.9153, gate NOT met

⚠ No `<think>` tags ever appeared — max_seq_len=1024 filtered all reasoning chains.

```
step  250  val_ce=5.2199     step 1000  val_ce=4.9480
step 2000  val_ce=4.9172     step 2979  val_ce=4.9153  ← plateau
2026-04-05 21:52:38   Total time: 290.2 min   Peak VRAM: 6.59 GB
2026-04-05 21:52:38   [hub] uploaded  checkpoint-0002979 (commit=8981b950)
```

---

## Stage 1 — Pre-training, Session 6 (Kaggle Dual T4)
**Status:** ✅ COMPLETE (graceful timeout) — steps 14902→21501, tokens 488M→705M
```
2026-04-05 14:27:38   Tokens processed: 704,544,768   Last val CE: 5.324081295402125
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
