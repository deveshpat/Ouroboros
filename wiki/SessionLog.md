# Session Log
> Curated record of key coordinator runs and root cause resolutions.
> Raw verbatim output stays in `terminal_log.md` (last session only).
> Oldest entries are dropped when this file exceeds ~150 lines.

---

## Session 31 — Azure H100 DGAC checkpoint promoted to canonical anchor (2026-05-15) 🟡

**Evidence:** Promotion metadata recorded `promoted_at=2026-05-15T11:24:25+00:00`, `repo_id=WeirdRunner/Ouroboros`, `source_revision=374a9a32d81242224465b786d62aaef7564639e6`, and `source_checkpoint=runs/azure_h100_dgac/stage_10/checkpoint-0001154`.

**Promotion:** Copied the checkpoint adapter weights/config and `halt_gate.pt` into `diloco_state/anchor`, set `terminal_stage=10`, recorded `total_train_samples=36906`, and marked `mark_dgac_complete=true`.

**Decision:** Treat `diloco_state/anchor` as the active DGAC-complete model. The next quality gate is `dgac-anchor-eval` on the canonical anchor; direct Azure checkpoint eval is now optional forensic comparison or a resume source if another H100 continuation is needed.

---

## Session 30 — Azure H100 corrected DGAC epoch-0 checkpoint (2026-05-15) 🟡

**Evidence:** W&B run `Azure H100 SCUS DGAC full budgeted` finished after loading 36,906 train / 1,940 val samples on `NVIDIA H100 NVL` (`sm90`, 100GB, BF16). Bootstrap verified `flash_attention_2` and the Mamba CUDA fast path. The run loaded `diloco_state/anchor` and restored `diloco_state/anchor/halt_gate.pt`, then completed Stage 10 epoch 0 with logged CE/grad-norm samples through step 1150.

**Checkpoint:** Saved and uploaded `runs/azure_h100_dgac/stage_10/checkpoint-0001154` with `training_state.pt`, adapter weights, and `halt_gate.pt`.

**Caveat:** This is not a final quality pass. Epoch-end val/gen were skipped because only 299 min remained and the Azure script used `--val_skip_buffer_minutes 720`; checkpoint metadata has `acc=None` and `ce=None`.

**Decision:** Evaluate the H100 checkpoint or resume from it explicitly via normal checkpoint resume. Do not use `--resume_from_diloco_anchor` for checkpoint evaluation/resume, because that flag intentionally reloads `diloco_state/anchor` and starts fresh from the anchor.

---

## Session 29 — DGAC HaltGate objective corrected (2026-05-10) ✅

**Problem:** DGAC diagnostics could show `k_actual=1` across visible samples even when the base anchor quality was usable. The old training objective let HaltGate behavior be driven by ponder/diversity pressure instead of a correctness-aware stop target.

**Fix:** DGAC now constructs supervised halt targets from forced-depth CE probes (`1,2,4,stage_k` by default): choose the smallest latent depth within CE tolerance of full stage-`k`, train `continue` before that depth and `halt` at that depth, and keep ponder/diversity as separate regularizers.

**Metrics:** Training logs now expose `train/dgac_halt_loss`, `train/dgac_ponder`, `train/dgac_diversity`, `train/dgac_halt_step_mean`, plus target/support diagnostics. Existing eval-only forced/gated/full CE diagnostics remain unchanged.

**Validation:** CPU fake tests cover target construction, clipped probe depths, no early-halt reward when CE is worse, HaltGate gradients from supervised halt loss, CLI defaults, and existing DGAC diagnostics.

---

## Session 28 — DGAC complete; anchor eval loader bug found (2026-05-10) ✅

**Evidence:** Coordinator printed `DGAC dedicated round COMPLETE (36906/36906 samples)` and uploaded `DiLoCo anchor: stage 10 round 1 (3 workers, 1386 samples, mode=diloco)`.

**Bug found before post-DGAC eval:** `--resume_from_diloco_anchor --eval_only` loaded adapter weights from `diloco_state/anchor` but did not pass the live `HaltGate` into `diloco_download_anchor`, so eval logs said `HaltGate at zero-init` and could not measure the aggregated post-DGAC halt gate.

**Decision:** Patch eval/training anchor load so the current `diloco_state/anchor/halt_gate.pt` is restored when present. Rerun `kaggle_run_mode=dgac-anchor-eval` after the patch; trust the result only if logs include `Loaded halt gate from diloco_state/anchor/halt_gate.pt`.

---
## Session 27 — Stage 10 terminal anchor eval-only → DGAC cleared (2026-05-09) ✅

**Evidence:** Kaggle GPU eval-only run loaded `WeirdRunner/Ouroboros/diloco_state/anchor` after the terminal DiLoCo aggregation and reported `stage=10 val_ce=0.4863 val_acc=0.0889`.

**Generation check:** arithmetic/code/factual/explanatory prompts were coherent, with `Mean UWR=0.733`. All samples used `k_actual=10`, which is expected before DGAC because HaltGate starts at zero-init and has not learned adaptive halting yet.

**Decision:** quality gate passes. Add workflow-dispatched `dgac-train` mode so Phase 3.4 can launch from GitHub Actions without manually editing the Kaggle notebook.

**Next action:** run GitHub Actions → `coordinate` with `kaggle_run_mode=dgac-train`, `force_worker_ids=A`, `skip_trigger=false`, `dry_run=false`, and empty `workflow_validate`; monitor W&B train/val/gen metrics and Hub checkpoints under `runs/stage3_dgac`.

---

## Session 26 — Stage 10 terminal aggregation → DGAC manual gate (2026-05-09) ✅

**Evidence:** GitHub Actions coordinator log from 2026-05-09T17:35Z read `stage=10 round=2 mode=diloco`, projected final shards A=8,665, B=8,665, C=8,664, and found all three worker outputs ready.

**Result:** coordinator loaded the anchor and worker weights, aggregated on CPU, logged `coordinator/workers_aggregated=3`, `coordinator/samples_this_round=25994`, `coordinator/total_samples_stage=36906`, `coordinator/pct_stage_done=100`, and uploaded `DiLoCo anchor: stage 10 round 2 (3 workers, 25994 samples, mode=diloco)`.

**Gate:** coordinator printed `Stage 10 COMPLETE (36906/36906 samples). Entering DGAC manual gate.`, then `Stage 10 is terminal for DiLoCo`, then `Done (DGAC manual gate)`. No stage-11 DiLoCo dispatch should run from cron.

**Next action:** superseded by Session 27; terminal anchor quality passed and DGAC is cleared for explicit launch via `kaggle_run_mode=dgac-train`.

**Artifact hygiene:** generated `signals/*.json` files remain disposable runtime doorbells. Keep only `signals/.gitkeep` in source.

---

## Session 25 — docs retirement + Kaggle dispatch hardening (2026-05-04) ✅

**Decision:** retire obsolete PRDs/plans only after their durable decisions are documented in `wiki/`. Keep the runtime signal mechanism; generated `signals/*.json` files are disposable GitHub Actions doorbells, while `signals/.gitkeep` keeps the directory present.

**Bug found:** Kaggle can print `Kernel push error: Maximum weekly GPU quota...` while the CLI invocation does not behave like a hard process failure. Treating any zero exit code as success leaves workers in `triggered_workers`, making the coordinator wait for the long timeout.

**Fix:** dispatch success now requires a `successfully pushed` marker and rejects quota/error markers. Failed dispatches flow through post-dispatch reconciliation immediately, demoting workers to attendance instead of waiting ~13h.

**Guardrails:** dispatch tests cover success marker classification and zero-exit quota output; source-of-truth tests keep the signal trigger mechanism present while ignoring generated signal JSONs.

---

## Session 24 — source-of-truth docs + obsolete plan retirement (2026-05-04) ✅

**Decision:** completed PRDs/plans are not durable source-of-truth files. Once a phase is implemented, durable decisions move into `wiki/`, then obsolete files under `prds/` and `plans/` are deleted.

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

**Kaggle contract:** `kaggle-utils.ipynb` remains a thin adapter and preserves the IPython shell-magic launch seam (`!{shell_command}`) instead of using Python `subprocess.run`; command argv comes from `ouroboros.kaggle_launch_matrix`.

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
