---
title: DiLoCo Protocol
type: concept
sources:
  - BLUEPRINT.md
  - ouroboros/diloco/protocol.py
  - ouroboros/diloco/coordinator_runner_logic.py
  - ouroboros/diloco/coordinator_dispatch.py
  - ouroboros/diloco/hub_state.py
  - signals/worker_A_stage_7_round_0.json
updated: 2026-04-30
---

# DiLoCo Protocol

## Overview

Three Kaggle accounts (A, B, C) each own a 30h/week GPU quota. They train
independently on non-overlapping shards of the dataset each round, then a GitHub
Actions coordinator aggregates their weights (outer DiLoCo update) and dispatches
the next round. No direct communication between workers — all coordination is through
Hugging Face Hub JSON state + GitHub signal files.

---

## Round State Machine

The coordinator reads `diloco_state/round_state.json` from the Hub at the start of
every run and executes exactly one pass (no polling loop).

### Modes

| Mode | Condition | Behaviour |
|---|---|---|
| `diloco` | ≥2 active workers with sufficient shard | Weighted outer update across workers |
| `solo` | Exactly 1 active worker | Direct weight promotion — no outer update (outer_lr irrelevant) |
| `waiting` | All credentialed workers in attendance | Round frozen, no training dispatched |
| `complete` | Remaining samples < `min_shard_samples` per worker | Stage advance triggered |

### Mode Determination

`determine_round_mode()` in `ouroboros/diloco/protocol.py`:
1. Compute `projected_shards` — always a 3-way split (A/B/C), even if some are inactive. This preserves shard determinism.
2. Filter to workers whose projected shard ≥ `min_shard_samples` (32 samples = 1 optimizer step).
3. Count survivors → `solo` or `diloco`. Zero survivors → `complete`.

`force_worker_ids` bypasses shard math entirely — useful for manual dispatch.

---

## The `triggered_at` Sentinel

**Most important invariant in the system:**

`triggered_at` in `round_state.json` is set to the Unix timestamp when workers were
last dispatched. The coordinator uses it to determine whether to wait or re-dispatch.

| `triggered_at` value | Interpretation | Coordinator action |
|---|---|---|
| `0` or missing | Dispatch unconfirmed / never sent | Re-dispatch immediately, no timeout check |
| `> 0` | Dispatch confirmed at that time | Wait up to `worker_timeout_hours` (13h) |

**Why `triggered_at=0` exists:** Kaggle push can fail silently (notebook stages fine
but execution never starts). Setting `triggered_at=0` is the canonical way to signal
"needs re-dispatch" without losing the worker list. The coordinator detects this on
its very next run and re-dispatches without waiting 13h.

This was the fix for the Stage 3 round 1 deadlock (Session 19). Workers A and B had
signals from round 0 but round 1 was never dispatched because `triggered_at` was set
non-zero before the push confirmed. Manual reset to 0 unblocked it immediately.

---

## Attendance Workers

A worker enters `attendance_workers` when it cannot participate in training:
- Quota exhausted (detected via `kernels push` error output containing "quota")
- Timed out (13h elapsed since dispatch, no status uploaded)

**Attendance behaviour:** worker downloads the current anchor, uploads
`status(samples=0)`, pushes its signal file, and returns. This keeps the worker
"alive" in the system so it can rejoin when quota renews, without blocking
aggregation.

**Waiting mode:** all credentialed workers are in attendance. Round number is frozen.
The coordinator keeps dispatching attendance pings every 30 minutes (via cron) until
at least one worker has quota.

Worker C was in attendance from stage 2 through stage 3 round 4 due to quota
exhaustion. It rejoined at stage 3 round 5 and has been active since.

---

## Shard Partitioning

`compute_projected_shards()` always partitions into exactly 3 shards (A/B/C index
positions are fixed), regardless of how many workers are active. This is intentional:
workers use the same shard logic on their side, so A always trains on the first ~⅓
of remaining samples, B on the middle ~⅓, C on the last ~⅓.

`partition_stage_shard()` in `curriculum.py` mirrors this math on the worker side.
The two implementations must stay in sync.

---

## Aggregation (DiLoCo Outer Update)

```
pseudo_grad[key] = anchor[key] - worker[key]   (for each worker)
new_anchor[key]  = anchor[key] - outer_lr × Σ(pseudo_grad[key] × n_i / N)
```

`outer_lr = 0.7` for diloco mode. In solo mode, worker weights are promoted directly
(equivalent to `outer_lr = 1.0` with a single worker, but skips the pseudo-gradient
math entirely to avoid blending a stale anchor).

If no prior anchor exists (bootstrap), a weighted average of worker weights is used
instead of crashing.

---

## Hub State Layout

```
WeirdRunner/Ouroboros/
  diloco_state/
    round_state.json                    ← coordinator reads/writes every run
    anchor/stage_{k}/round_{n}/         ← aggregated weights after each round
      anchor.safetensors
    workers/{A,B,C}/
      status.json                       ← worker uploads after training
      stage_{k}/round_{n}/              ← per-round adapter weights
```

`worker_status_path()` and `worker_weights_prefix()` in `hub_state.py` are the
canonical path generators. Use them — never hardcode paths.

---

## Signal Files

GitHub signal files (`signals/worker_{id}_stage_{k}_round_{n}.json`) are the
coordinator trigger mechanism. A worker pushes one after uploading its status to Hub.
The GitHub Actions workflow fires on `push` to `signals/*.json`.

Signal files do **not** contain training output — they only carry `worker_id`,
`stage_k`, `round_n`, and `timestamp`. The actual training results live in Hub state.

**Current state (from signal files, 2026-04-30):**
Stage 7 round 0 complete for all three workers. BLUEPRINT.md was one stage behind —
signals are ground truth.

---

## Worker Timeout Flow

1. Coordinator sees `triggered_at > 0` and workers listed in `triggered_workers`
2. Downloads `status.json` for each worker
3. If missing and `now - triggered_at > 13h`: worker is timed out
4. Timed-out workers → demoted to `attendance_workers`
5. Surviving workers → `triggered_workers` updated, `mode` recalculated
6. If no survivors → `mode = "waiting"`, `triggered_at = 0`
7. Attendance workers are dispatched for their ping-only run

See `plan_next_round()` in `protocol.py` for the full decision tree as pure logic,
and `coordinator_runner_logic.py` for the Hub I/O layer around it.
