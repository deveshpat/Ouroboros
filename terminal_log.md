# terminal_log.md — Project Ouroboros
**All verified terminal outputs, newest first.**

---

## Stage 2 SFT — Session 7 (single GPU attempt, DDP still active, NCCL crash step 3522)
**Script:** `train_sft.py`  **Date:** 2026-04-08  **Hardware:** Kaggle Dual T4  
**Status:** 🔴 NCCL WATCHDOG KILL — `stage2_rewrite_prompt.md` was NOT applied before this run

**Root cause:** `maybe_launch_multi_gpu()` still in script → DDP auto-launched → same NCCL hang.

**Last training step visible before crash:**
```
step 3520  ce=3.2366  acc=0.4122  gn=1.5312  lr=2.71e-05  vram=1.499  tok/s=4594
```

**Crash (verbatim):**
```
[rank1]: Terminating the process after attempting to dump debug info,
         due to ProcessGroupNCCL watchdog hang.
W0408 12:01:02.579000  torch/multiprocessing/spawn.py:165
  Terminating process 96 via signal SIGTERM
torch.multiprocessing.spawn.ProcessExitedException:
  process 1 terminated with signal SIGABRT
```

**Resolution:** DDP removed entirely via `train_sft_simplify_prompt.md`. Single GPU only for all future Stage 2 runs.

---

## Stage 2 SFT — Session 6 (full-scale, DDP, NCCL watchdog crash at step 3750)
**Date:** 2026-04-07  **Hardware:** Kaggle Dual T4  **Status:** 🔴 Same NCCL crash

**Crash (verbatim):**
```
WorkNCCL(SeqNum=13274, OpType=ALLREDUCE, NumelIn=1, NumelOut=1,
  Timeout(ms)=600000) ran for 600003 milliseconds before timing out.
[Rank 1] Watchdog caught collective operation timeout
process 1 terminated with signal SIGABRT
```
Val CE at crash: `step=3750  val_ce=5.6254  val_acc=0.2203`

---

## Stage 2 SFT — Session 5 (full-scale, DDP, HARD TIMEOUT at 43200.6s)
**Date:** 2026-04-06→07  **Hardware:** Kaggle Dual T4  **Status:** 🔴 SIGKILL (exit 137)

**Val CE trajectory:**
```
Step    Val CE    Val Acc
  250   5.7453    0.2170
 1000   5.6532    0.2207
 2500   5.6245    0.2206
 3500   5.6257    0.2203   ← plateau, effectively no progress after step 2500
```
Generation: number loops throughout. No `<think>` tags ever appeared.

Hub checkpoints retained: `checkpoint-0000500` through `checkpoint-~003700`

---

## Stage 2 SFT — Sessions 1–4

| Session | Outcome |
|---|---|
| S1 | stratos-only, max_seq_len=1024, val_ce=4.92 plateau. No `<think>` (1024 filtered all reasoning). Hub: ckpt-0002979. |
| S2 | full-mix DDP, Bugs 6–10, val_ce=5.71. All local Stage 2 ckpts deleted by prune bug. |
| S3–S4 | Dry-runs only. Patches from `stage2_patch_prompt.md` verified. |

---

## Stage 1 Pre-training — Session 6 (Kaggle Dual T4)
**Status:** ✅ COMPLETE (graceful timeout)
```
Tokens processed: 704,544,768   Last val CE: 5.324
Hub: checkpoint-0021000 (commit=d70d2c49)  ← SFT starting point
```

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
