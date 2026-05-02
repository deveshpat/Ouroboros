# Session Log
> Curated record of key coordinator runs and root cause resolutions.
> Raw verbatim output stays in `terminal_log.md` (last session only).
> Oldest entries are dropped when this file exceeds ~150 lines.

---

## Session 21 — kaggle>=1.8.4 + T4 GPU fix (2026-04-22) ✅ VERIFIED WORKING

**Root cause 1:** `kaggle==1.6.17` predates `--accelerator` (added v1.8.4 PR #907). `KernelPushRequest` had no GPU-type field — `"accelerator"` in JSON silently discarded. Kaggle assigned P100 (default).

**Root cause 2:** `"nvidiaTeslaT4"` (lowercase n) is invalid. Correct value: `"NvidiaTeslaT4"`.

**Fix (3 files):**
- `.github/workflows/diloco_coordinator.yml`: `kaggle>=1.8.4`
- `diloco_coordinator.py`: `"NvidiaTeslaT4"` in JSON + `--accelerator NvidiaTeslaT4` in push_args
- `jamba_coconut_finetune.py`: `cc < (7,5)` guard → `_diloco_reset_triggered_at()` + signal + exit

**Status:** No P100 assignment observed since deployment.

---

## Session 20 — W&B round tracking fix (2026-04-22) ✅ VERIFIED WORKING

**Root cause:** `wandb==0.25.0` with `resume="allow"` on a cleanly finished run creates a new ephemeral run (discards specified `id=`). Stage 2 (6 rounds) showed exactly 1 "Worker A | Stage 2" entry in W&B.

**Fix:** Per-round unique run IDs + `group=` parameter:
- `id = diloco-{worker_lower}-s{stage_k}-r{round_n}`
- `group = diloco-{worker_lower}-s{stage_k}`
- Remove `resume="allow"`

---

## Session 19 — triggered_at=0 recovery (2026-04-22) ✅ VERIFIED WORKING

**Problem:** Stage 3 round 1 deadlock. Workers A+B had `status.json` done for round 0, but coordinator read `stage_k=3, round_n=1` with `triggered_at=0` (unconfirmed dispatch). Coordinator printed "Waiting for workers to finish this round" and returned.

**Fix:** Added `triggered_at <= 0` branch in normal-mode missing-worker check → immediate re-dispatch.

**Coordinator output (key lines):**
```
[coordinator] Round 1: ['A', 'B'] marked triggered but triggered_at=0. Re-dispatching now.
[coordinator] Triggered Worker A: Kernel version 14 successfully pushed.
[coordinator] Triggered Worker B: Kernel version 69 successfully pushed.
[coordinator] Triggered Worker C: Maximum weekly GPU quota of 30.00 hours reached.
[coordinator] Done (re-dispatch unconfirmed round 1).
```
C quota exhausted → attendance mechanism activated correctly.

---

## Session 17 — kernels push trigger verified (2026-04-21) ✅

First successful end-to-end `kernels push` flow (replacing the broken `kernels pull` path):
```
[coordinator] Triggered Worker A: weirdrunner/kaggle-utils  (Kernel version 9 successfully pushed.)
[coordinator] Triggered Worker B: weirdrunner007/kaggle-utils  (Kernel version 11 successfully pushed.)
[coordinator] WARNING: kernels push failed for Worker C: [quota/creds issue]
```

---

## Session 15 — Coordinator Run #2 (2026-04-20) ✅

403 root cause identified: `kaggle==2.0.1` uses gRPC. `KernelsApiService/GetKernel` (pull endpoint) blocked. Fix: pin `kaggle==1.6.17` (later superseded by `>=1.8.4`).
