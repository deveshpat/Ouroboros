---
title: Coordinator Retry Flow
type: pattern
sources:
  - BLUEPRINT.md
  - ouroboros/diloco/protocol.py
  - ouroboros/diloco/coordinator_runner_logic.py
  - ouroboros/diloco/coordinator_dispatch.py
  - terminal_log.md
updated: 2026-04-30
---

# Coordinator Retry Flow

## What This Pattern Is

The coordinator is a single-pass GitHub Actions job (no polling loop, no `while True`,
no `time.sleep`). It runs every 30 minutes via cron and on every `signals/*.json`
push. Its retry behaviour is entirely encoded in `round_state.json` — specifically
in the `triggered_at` field and the `triggered_workers` list.

This pattern documents how the coordinator handles each failure mode it encounters
and what state it leaves behind for the next run.

---

## Decision Tree (one coordinator pass)

```
Read round_state.json from Hub
│
├─ triggered_at == 0 OR no expected workers
│   └─ Re-dispatch immediately → write new round_state with triggered_at = now
│
├─ triggered_at > 0 AND expected workers present
│   ├─ Download status.json for each expected worker
│   ├─ All workers ready (status = "done" for this stage/round)?
│   │   └─ Aggregate → write next round_state → dispatch next round
│   │
│   ├─ Some workers ready, some missing AND elapsed > 13h (timeout)
│   │   ├─ Demote missing → attendance_workers
│   │   ├─ Aggregate survivors
│   │   └─ Dispatch survivors + attendance
│   │
│   ├─ No workers ready AND elapsed > 13h
│   │   ├─ All missing → mode = "waiting"
│   │   ├─ triggered_at = 0
│   │   └─ Dispatch attendance pings
│   │
│   └─ Workers missing AND elapsed ≤ 13h
│       └─ Print "Waiting..." → exit (next cron run will check again)
```

---

## The `triggered_at=0` Sentinel — When and Why

Set `triggered_at=0` in `round_state.json` whenever a dispatch is structurally
unconfirmed — meaning: the coordinator wrote `triggered_workers` before knowing
whether Kaggle actually accepted the kernel push.

**Canonical cases where it should be 0:**
- Fresh round state (no workers dispatched yet)
- Kaggle push failed for all workers
- Manual state reset to force re-dispatch
- GPU fast-fail: worker detects P100, exits, sets `triggered_at=0` before exiting

**Effect on coordinator:** the `triggered_at <= 0` branch is checked first,
before any timeout calculation. This means re-dispatch happens within 30 minutes
(next cron) regardless of how long the stale state has been sitting there.

**The Stage 3 round 1 deadlock:** workers A and B had completed round 0 (signals
present), but round 1 was triggered with `triggered_at` set to a non-zero value
before Kaggle confirmed the push. Workers never ran. The coordinator saw
`triggered_at > 0` and waited 13h, then waited another 13h, indefinitely.
Fix: manually set `triggered_at=0` in Hub `round_state.json`. Next coordinator
run re-dispatched immediately.

---

## Attendance Worker Lifecycle

```
Worker joins attendance_workers when:
  - Quota exhausted (push stderr contains "quota" or "exceeded")
  - Timed out after 13h with no status upload

Attendance worker behaviour:
  - Downloads current anchor
  - Uploads status(samples=0, status="done")
  - Pushes signal file
  - Returns immediately

Worker exits attendance when:
  - Quota renews
  - Coordinator detects it uploaded a non-zero-sample status
  - Manual --force_worker_ids overrides attendance list
```

Worker C was in attendance from stage 2 through stage 3 round 4. It rejoined at
stage 3 round 5 when its weekly quota renewed.

---

## Dispatch Failure Recovery

`reconcile_after_dispatch()` in `coordinator_dispatch.py` handles partial push
failures. If some workers' `kernels push` fails:

- Successfully pushed workers → stay in `triggered_workers`, `triggered_at = now`
- Failed workers → move to `attendance_workers`, effectively demoted
- All failed → `mode = "waiting"`, `triggered_at = 0`

This state is written back to Hub before the coordinator exits. Next run sees
the corrected state and acts accordingly.

---

## Why There's No Polling Loop

Kaggle kernel execution is asynchronous and can take up to 12 hours. Polling
would require either a long-running process (wasteful in GHA free tier) or
complex webhook infrastructure. Instead, workers self-report by pushing signal
files to the GitHub repo, which triggers the coordinator via the `push` event.

The cron every 30 minutes is the fallback: if a signal push fails or the
GHA trigger is missed, the coordinator still runs and either waits (if within
timeout), re-dispatches (if `triggered_at=0`), or promotes attendance workers.

This design is verified by the test `test_main_contains_no_polling_loop_or_sleep`
in `tests/test_coordinator_runner.py`.

---

## Key Invariants

1. **`triggered_at=0` means re-dispatch, not timeout.** This is checked before
   any elapsed-time calculation.
2. **`triggered_workers` is always the list of workers who were successfully pushed,**
   not the list who were attempted. Failed pushes land in `attendance_workers`.
3. **Round number never advances without aggregation.** The coordinator only writes
   `round_n + 1` after a successful `_aggregate_ready_workers()` call.
4. **Solo mode skips outer update.** Direct weight promotion — outer_lr has no effect.
5. **Stage never closes until `remaining < min_shard_samples` per active worker.**
   Geometric remainder is handled gracefully — no rounding error can leave workers
   stuck with a shard they can't process.
