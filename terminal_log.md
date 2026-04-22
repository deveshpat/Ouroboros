# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**

---

## Session 20 — P100 GPU Assignment & W&B Round Tracking (2026-04-22) ⚠️ PATCH PENDING

**Context:** Session 19's `triggered_at=0` fix is confirmed working (see coordinator log below). Stage 3 round 1 was re-dispatched. However, Worker B's kernel ran on **GPU P100** (not T4) and was manually cancelled after 28 minutes. Worker A's status unknown. Two new issues diagnosed and fixed.

---

### Session 19 Coordinator Run — CONFIRMED WORKING (verbatim key lines)

```
[coordinator] stage=3 round=1 mode=diloco
[coordinator] Attendance workers: ['C']
[coordinator] Remaining samples for stage 3: 12302
[coordinator] Projected shards: {'A': 4101, 'B': 4101, 'C': 4100}
[coordinator] Next round mode: diloco  active workers: ['A', 'B']
[coordinator] Worker A: not ready (status={'worker_id': 'A', 'stage_k': 3, 'round_n': 0, ...})
[coordinator] Worker B: not ready (status={'worker_id': 'B', 'stage_k': 3, 'round_n': 0, ...})
[coordinator] Worker C: not ready (status=None)
[coordinator] Round 1: ['A', 'B'] marked triggered but triggered_at=0 (unconfirmed dispatch). Re-dispatching now.
[coordinator] Triggered Worker A: ***/kaggle-utils  (Warning: Looks like you're using an outdated API Version, please consider updating (server 2.0.1 / client 1.6.17)
Kernel version 14 successfully pushed. ...)
[coordinator] Triggered Worker B: ***/kaggle-utils  (Warning: ...
Kernel version 69 successfully pushed. ...)
[coordinator] Triggered Worker C: ***/kaggle-utils  (Warning: ...
Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.)
[coordinator] Done (re-dispatch unconfirmed round 1).
```

✅ `triggered_at=0` recovery path works as designed. C attendance (quota exhausted) works as designed.

---

### Issue 1 — P100 GPU: Root Cause

`_build_kaggle_kernel_metadata()` generates only `"enable_gpu": true` with no GPU type. Kaggle assigned P100 to Worker B (version 69), which was cancelled after 28 minutes. Worker A (version 14) GPU type unknown.

**Root cause:** `kernel-metadata.json` is authoritative over the notebook's own `metadata.kaggle.accelerator: "nvidiaTeslaT4"`. Our generated JSON has no accelerator type → Kaggle picks freely.

**Fix:** Add `"accelerator": "nvidiaTeslaT4"` to `_build_kaggle_kernel_metadata()`. This is a metadata JSON field; distinct from the old `--accelerator` CLI flag.

W&B coordinator run version: 0.26.0 (GitHub Actions).

---

### Issue 2 — W&B Missing Rounds: Root Cause

Workers use `wandb==0.25.0` (Kaggle pre-installed). In this version, `resume="allow"` on a **cleanly finished** run does not reopen the existing run; it creates a new run with an auto-generated ID (discarding the specified `id=`). So rounds 1..N of each stage logged to invisible ephemeral runs.

Evidence: Stage 2 (6 rounds) shows exactly 1 "Worker A | Stage 2" and 1 "Worker B | Stage 2" entry in W&B. Only round 0 data is visible.

Contrast: coordinator uses wandb 0.26.0 where `resume="allow"` on finished runs works correctly (`wandb: Resuming run Coordinator | Stage 3` confirmed in log).

**Fix:** Per-round unique run IDs + `group=` parameter:
- `id = diloco-{worker}-s{stage}-r{round}` (unique per round, no resume needed)
- `group = diloco-{worker}-s{stage}` (groups all rounds for a stage in W&B dashboard)
- `name = Worker {worker} | Stage {stage} | Round {round}`
- Remove `resume="allow"`

---

## Session 19 — Stage 3 Round 1 Deadlock: Root Cause & Fix (2026-04-22) ✅ VERIFIED

**Context:** Stage 3 Round 0 completed successfully (A: 12302 samples, B: 12302 samples = 24604 total, 66.7% of stage). Round 1 deadlocked — workers A and B marked triggered but never started.

### Failing Coordinator Run — verbatim (Stage 3, Round 1 check)

```
[coordinator] stage=3 round=1 mode=diloco
[coordinator] Attendance workers: ['C']
[coordinator] Remaining samples for stage 3: 12302
[coordinator] Projected shards: {'A': 4101, 'B': 4101, 'C': 4100}
[coordinator] Next round mode: diloco  active workers: ['A', 'B']
[coordinator] Worker A: not ready (status={'worker_id': 'A', 'stage_k': 3, 'round_n': 0, 'samples_seen': 12302, 'status': 'done', ...})
[coordinator] Worker B: not ready (status={'worker_id': 'B', 'stage_k': 3, 'round_n': 0, 'samples_seen': 12302, 'status': 'done', ...})
[coordinator] Worker C: not ready (status=None)
[coordinator] Waiting for workers to finish this round: ['A', 'B']
```

W&B coordinator summary (round 0 aggregate):
```
coordinator/round:                0
coordinator/samples_this_round:   24604
coordinator/total_samples_stage:  24604
coordinator/workers_aggregated:   2
coordinator/pct_stage_done:       66.7
```

**Root cause:** Pre-patch coordinator wrote `triggered_at=<April 21 22:30>` but Kaggle push for round 1 silently failed. Post-patch coordinator saw `is_round_timed_out=False` (only ~8h elapsed) and returned "Waiting" on every run. The `triggered_at <= 0` recovery branch existed only in waiting-mode path, not in normal-mode path.

**Fix applied:** Added `triggered_at <= 0` branch to normal-mode missing-worker check → immediate re-dispatch. Manual state fix: set `triggered_at=0` in `round_state.json`.

**Verification (Session 20):** ✅ Coordinator log confirms `Round 1: ['A', 'B'] marked triggered but triggered_at=0 (unconfirmed dispatch). Re-dispatching now.`

---

## Session 18 — Round 4 Worker C Deadlock Diagnosed (2026-04-21) ✅ RESOLVED

```
[coordinator] Worker A: not ready (status={'worker_id': 'A', 'stage_k': 2, 'round_n': 3, 'samples_seen': 188, 'status': 'done', ...})
[coordinator] Worker B: not ready (status={'worker_id': 'B', 'stage_k': 2, 'round_n': 3, 'samples_seen': 187, 'status': 'done', ...})
[coordinator] Worker C: not ready (status=None)
[coordinator] Waiting for workers to finish this round: ['A', 'B', 'C']
```

Root cause: C in `triggered_workers` with exhausted quota; no deadline logic. Fixed by attendance mechanism.

---

## Session 17 — Coordinator Run #3 SUCCESS + `kernels push` trigger patch tested (2026-04-21) ✅

```
[coordinator] stage=2 round=3
[coordinator] Worker A: 188 samples ready
[coordinator] Worker B: 187 samples ready
[coordinator] New anchor uploaded: DiLoCo anchor: stage 2 round 3 (2 workers, 375 samples, mode=diloco)
[coordinator] Stage 2 progress: 36531/36906 samples seen
[coordinator] round_state.json updated: stage=2 round=4
[coordinator] Triggered Worker A: weirdrunner/kaggle-utils  (Kernel version 9 successfully pushed.)
[coordinator] Triggered Worker B: weirdrunner007/kaggle-utils  (Kernel version 11 successfully pushed.)
[coordinator] WARNING: kernels push failed for Worker C (weirdrunner008/kaggle-utils): [quota/creds issue]
```

**`kernels push` verdict: WORKING ✅**

---

## Session 16 — Previous `kernels push --accelerator` failure diagnosed (2026-04-21)

```
kaggle: error: unrecognized arguments: --accelerator NvidiaTeslaT4
```

Root cause: coordinator was generating metadata that passed `--accelerator` as a CLI flag. Fixed by removing the CLI flag (GPU still enabled via `enable_gpu: true` in JSON).

---

## Session 15 — Coordinator Run #2 SUCCESS (2026-04-20) ✅

```
[coordinator] stage=2 round=0
[coordinator] Worker A: 5060 samples ready
[coordinator] Worker B: 5059 samples ready
[coordinator] New anchor uploaded: DiLoCo anchor: stage 2 round 0 (2 workers, 10119 samples)
[coordinator] Stage 2 progress: 31847/36906 samples seen
[coordinator] round_state.json updated: stage=2 round=1
[coordinator] WARNING: Failed to trigger ***/kaggle-utils: 403 Client Error: Forbidden for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernel
```

Root cause of 403: `kaggle==2.0.1` uses gRPC. Fix: pin `kaggle==1.6.17`.

---

## Session 14 — DiLoCo Stage 2 Round 0 Training Complete (2026-04-20) ✅

Worker A — 159/159 steps, ~48s/step, ce=0.593, gn=0.874
Worker B — 159/159 steps, ~53s/step, ce=0.462, gn=0.828

GitHub signal push failed on both (bad PAT scope). Coordinator triggered manually.
Coordinator crashed at save step — `numpy` not installed. Fix deployed.

---

## Session 13 — Stage 1 Complete, Stage 2 59% Timeout (2026-04-19) ✅

```
S1E0: 100%|█████████████████| 15/15 [10:21<00:00, 41.46s/it, ce=0.372, gn=0.575]
  [val] s=1 ep=0 val_ce=0.4912 val_acc=0.0444
  [best] stage=1 new best acc=0.0444
S2E0:  59%|█████▉    | 679/1154 [9:43:06<6:55:29, 52.48s/it, ce=0.636, gn=1.069]
  [timeout] 10.68h elapsed - 19.5 min remaining (< 20 min buffer).
  [ckpt] saved -> runs/stage3_curriculum/stage_2/checkpoint-0002987
```

---

## Session 12 — Stage 0 Val Complete, Stage 1 Started (2026-04-17 → 2026-04-18) ✅

```
  [val] s=0 ep=0 val_ce=0.4041 val_acc=0.0222
  [ckpt] best -> runs/stage3_curriculum/stage_0/best
```

---

## Sessions 4–11 — Bootstrap / Kernel / FP16 Debugging (2026-04-14 → 2026-04-17)

- Session 12: Stage 0 training complete (260/260 steps, 137s/it). NCCL watchdog killed val → `timedelta(hours=4)` fix.
- Session 10: Dual T4 DDP profile. Mamba fast path ACTIVE. val s=0 acc=0.4000 (sanity check).
- Session 9: First successful Stage 0 smoke test, ~113s/step single T4.
- Sessions 6–8: `causal_conv1d_fn` shape fix, `selective_state_update` import path, 10-alias generation shim.
