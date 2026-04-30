# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**

---

## Session 21 — Full Root Cause: kaggle==1.6.17 Predates `--accelerator` Feature (2026-04-22) ⚠️ PATCH PENDING

**Context:** Session 20's `"accelerator": "nvidiaTeslaT4"` patch confirmed non-functional.
Version #70 (Worker B) still assigned P100. Full investigation conducted; correct fix designed.

**Evidence:** `github.com/Kaggle/kaggle-cli/blob/main/CHANGELOG.md` confirms:
> v1.8.4: "Add `--acc` to set accelerator for: `kaggle kernels push` ... (#907)"

**Evidence:** `github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md` confirms:
> `--accelerator <ACCELERATOR_ID>`: "NvidiaTeslaP100" (aka default GPU), "NvidiaTeslaT4", "TpuV6E8"

### Root Cause (complete)

**Root cause 1 — Feature didn't exist in our pinned version:**
`kaggle==1.6.17` predates `--accelerator` by at least two major versions. The `KernelPushRequest`
object in v1.6.17's REST/Swagger API has no GPU-type field. `"accelerator": "nvidiaTeslaT4"` in
`kernel-metadata.json` is silently discarded. Kaggle assigns "NvidiaTeslaP100" (documented default).

**Root cause 2 — Wrong capitalisation (secondary):**
Even if the JSON field were read, `"nvidiaTeslaT4"` (lowercase n) does not match the official
valid value `"NvidiaTeslaT4"` (capital N) as documented.

**Why upgrading is now safe:**
The Session 15 403 error was on `KernelsApiService/GetKernel` — the **pull** endpoint. The old
flow tried to GET another account's kernel. We already fixed this by switching to push-only.
`kernels push` uses a different gRPC method in `kaggle>=1.8.3` that is not blocked.

### Complete Fix (three files)

**`.github/workflows/diloco_coordinator.yml`:**
- `"kaggle==1.6.17"` → `"kaggle>=1.8.4"`
- Update comment (no longer pinning to 1.6.17 for REST API reasons)

**`diloco_coordinator.py`:**
- `_build_kaggle_kernel_metadata()`: `"nvidiaTeslaT4"` → `"NvidiaTeslaT4"` (correct capitalisation)
- `_trigger_single_worker()`: `push_args` → add `"--accelerator", "NvidiaTeslaT4"`

**`jamba_coconut_finetune.py`** (safety net for P100 silent fallback):
- Add `"sm60"` to `_KNOWN_ARCH_SUFFIXES`
- Add `_diloco_reset_triggered_at()` helper
- Add GPU guard in `main()`: if `diloco_mode` and `cc < (7,5)` → reset `triggered_at=0` + push signal + exit

---

## Session 20 — P100 GPU Assignment & W&B Round Tracking (2026-04-22) ⚠️ PATCH PENDING

**Context:** Session 19's `triggered_at=0` fix confirmed working. Stage 3 round 1
re-dispatched. Worker B's kernel ran on **GPU P100** (Version #70) and was manually
cancelled after 28 minutes. Worker A status unknown.

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

`_build_kaggle_kernel_metadata()` generates only `"enable_gpu": true` with `"accelerator": "nvidiaTeslaT4"`.
Kaggle assigned P100 to Worker B (version 69, then 70). **Root cause: `kaggle==1.6.17` REST API
`KernelPushRequest` has no GPU-type field. `"accelerator"` key silently dropped.**

**Fix:** Runtime fast-fail in `main()` (see Session 21 above).

---

### Issue 2 — W&B Missing Rounds: Root Cause

Workers use `wandb==0.25.0`. `resume="allow"` on a cleanly finished run does not reopen
it — creates a new run with auto-generated ID (discarding specified `id=`). Rounds 1..N
logged to invisible ephemeral runs.

Evidence: Stage 2 (6 rounds) shows exactly 1 "Worker A | Stage 2" entry in W&B.

**Fix:** Per-round unique run IDs + `group=` parameter:
- `id = diloco-{worker}-s{stage}-r{round}` (unique per round, no resume needed)
- `group = diloco-{worker}-s{stage}` (groups all rounds for a stage in W&B dashboard)
- `name = Worker {worker} | Stage {stage} | Round {round}`
- Remove `resume="allow"`

---

## Session 19 — Stage 3 Round 1 Deadlock: Root Cause & Fix (2026-04-22) ✅ VERIFIED

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

**Fix applied:** Added `triggered_at <= 0` branch to normal-mode missing-worker check.
Manual state fix: set `triggered_at=0` in `round_state.json`. **Verification (Session 20): ✅**

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

Root cause: coordinator was generating metadata that passed `--accelerator` as a CLI flag. Fixed by removing CLI flag.

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
