# Session Log
> Curated record of key coordinator runs and root cause resolutions.
> Raw verbatim output stays in `terminal_log.md` (last session only).
> Oldest entries are dropped when this file exceeds ~150 lines.

---

## Session 25 — docs retirement + Kaggle dispatch hardening (2026-05-04) ✅

**Decision:** retire obsolete PRDs/plans only after their durable decisions are documented in `wiki/`. Keep the runtime signal mechanism; generated `signals/*.json` files are disposable GitHub Actions doorbells, while `signals/.gitkeep` keeps the directory present.

**Bug found:** Kaggle can print `Kernel push error: Maximum weekly GPU quota...` while the CLI invocation does not behave like a hard process failure. Treating any zero exit code as success leaves workers in `triggered_workers`, making the coordinator wait for the long timeout.

**Fix:** dispatch success now requires a `successfully pushed` marker and rejects quota/error markers. Failed dispatches flow through post-dispatch reconciliation immediately, demoting workers to attendance instead of waiting ~13h.

**Guardrails:** dispatch tests cover success marker classification and zero-exit quota output; source-of-truth tests keep the signal trigger mechanism present while ignoring generated signal JSONs.

---

## Session 24 — source-of-truth docs + obsolete plan retirement (2026-05-04) ✅

**Decision:** completed PRDs/plans are not durable source-of-truth files. Once a phase is implemented, durable decisions move into `wiki/`, then obsolete files under `plans/` are deleted.

**Fix:** added `wiki/Architecture-Extraction.md` and `wiki/Engineering-Workflow.md`, updated `BLUEPRINT.md`/`STATUS.md`, and retired completed training/coordinator PRD/plan files.

**Guardrail:** `tests/test_source_of_truth_contract.py` now checks that completed extraction decisions live in wiki docs and that obsolete plan files are absent.

---

## Session 23 — coordinator zero-drift extraction and adapter thinning (2026-05-04) ✅

**Goal:** extract `diloco_coordinator.py` without changing CLI, Hub state, Kaggle dispatch, aggregation math, or recovery behavior.

**Result:** root `diloco_coordinator.py` became a thin compatibility adapter. Coordinator behavior now lives in:
- `ouroboros.diloco.aggregation`
- `ouroboros.diloco.state`
- `ouroboros.diloco.dispatch`
- `ouroboros.diloco.coordinator`

**Tests added:** aggregation, state, dispatch, orchestration, adapter, and source-of-truth guardrails.

**Follow-up fix:** dispatch runtime-env test was made hermetic after a developer shell leaked a real `GITHUB_TOKEN`; tests now clear ambient token/runtime env before asserting fake payloads.

---

## Session 22 — training monolith extraction and Kaggle launch seam (2026-05-04) ✅

**Goal:** extract the training monolith while preserving runtime behavior, then thin `jamba_coconut_finetune.py` into a compatibility adapter.

**Result:** reusable behavior moved under `ouroboros/*`; the root training script now owns process startup and delegates to `ouroboros.train.run_cli`.

**Kaggle contract:** `kaggle-utils.ipynb` remains a thin adapter and preserves the `!torchrun` shell-magic launch seam instead of using Python `subprocess.run`.

**Artifact hygiene:** generated `signals/*.json` files are runtime artifacts and should not be tracked; `signals/.gitkeep` keeps the directory present.

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
