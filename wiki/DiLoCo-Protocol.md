# DiLoCo Protocol
> Load this page when debugging round state, attendance/waiting mode, or shard assignment.

---

## Round State Machine

```
                    ┌─────────────────────────────────────────────┐
                    │              COORDINATOR RUN                │
                    └─────────────────────────────────────────────┘
                                         │
                    ┌────────────────────▼──────────────────────┐
                    │  Read round_state.json from Hub            │
                    │  stage_k, round_n, triggered_workers,      │
                    │  attendance_workers, triggered_at, mode    │
                    └────────────────────┬──────────────────────┘
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │                    mode == "waiting"?                │
              └───────┬──────────────────────────────────┬──────────┘
                   YES│                               NO │
                      ▼                                  ▼
           [Waiting Mode Path]              [Normal Aggregation Path]
           Check attendance_workers         Check triggered_workers
           for status.json "done"           for status.json "done"
                      │                                  │
              All absent → re-dispatch        Any missing?
              Some responded → wait grace     ├─ force_worker_ids → additive repair
              Grace/all done → promote        ├─ triggered_at=0 → re-dispatch NOW
                                              ├─ timeout (<13h) → wait
                                              └─ timeout (>13h) → demote to attendance
```

---

## triggered_at Semantics

| Value | Meaning | Coordinator Action |
|---|---|---|
| `0.0` | Dispatch unconfirmed — Kaggle push may have failed | **Immediate re-dispatch** on next run (≤30 min). Verified Session 19. |
| `> 0` and `< 13h ago` | Normal — workers are running | Wait |
| `> 0` and `> 13h ago` | Timeout — Kaggle 12h wall hit | Demote non-responsive workers to `attendance_workers` |

**Manual reset:** Edit `round_state.json` on Hub → set `triggered_at: 0`. Next coordinator run re-dispatches immediately.

---

## Mode Definitions

| Mode | Meaning |
|---|---|
| `"diloco"` | ≥2 active workers training in parallel, outer update applied |
| `"solo"` | 1 active worker, weights promoted directly (no outer update, effective LR = 1.0) |
| `"complete"` | No workers have enough samples → advance stage |
| `"waiting"` | All credentialed workers in `attendance_workers`. `round_n` frozen. Coordinator re-dispatches attendance pings until someone responds. |
| `"terminal"` | Stage 10 DiLoCo is complete. DGAC is manually gated; coordinator must not dispatch more DiLoCo workers. |

---

## Shard Assignment (deterministic)

```python
# diloco_get_shard() — same logic in coordinator and worker
rng = random.Random(seed + stage_k * 100_003 + round_n * 7)
indices = list(range(n))
rng.shuffle(indices)
remaining = indices[samples_already_seen_this_stage:]
start, end = _partition_contiguous_range(len(remaining), 3, worker_idx)
shard = remaining[start:end]
```

Key invariants:
- Seed is fixed per stage+round → same shard assigned regardless of who calls it
- `samples_already_seen` trims the prefix so partial stages resume correctly
- Shard ≥ `min_shard_samples` (32) = worker gets triggered; below = attendance/skip

---

## Attendance Mechanism

Workers enter `attendance_workers` when:
1. Their Kaggle quota is exhausted (push succeeds, kernel immediately errors)
2. They fail to respond within 13h (coordinator demotes them)
3. Their shard is below `min_shard_samples`

Attendance behavior (worker side):
- Download anchor weights
- Upload `status.json` with `samples_seen=0`
- Push GitHub signal
- Exit without training

Attendance promotion: coordinator records responders, waits for all attendance
workers or the `attendance_join_grace_minutes` barrier (default 5 minutes), then
promotes all collected responders together. The first responder must not create
an accidental solo round before the grace window resolves.

`samples_seen=0` is attendance-only. It is not valid active-round training output
and must not be aggregated or used to mark an active worker complete.

---

## Manual Force Repair

`--force_worker_ids` means: “make sure these workers are active for the current
round if available and not already active/done.” It is not a filter.

Rules:
- existing `triggered_workers` keep running and remain counted;
- completed valid current-round work is aggregated even if the worker was not in
  `force_worker_ids`;
- only forced workers that are missing, credentialed, and not already done are
  dispatched;
- failed forced pushes are reconciled back to attendance/demoted state.

---

## Stage Close Conditions

Stage advances when ANY of:
- `total_samples_seen[stage_k] >= total_train_samples` (36 906)
- All projected shards below `min_shard_samples` (32) — "geometric remainder"
- `completion_mode == "complete"` from `_determine_round_mode()`

On advance before stage 10: `stage_k += 1`, `round_n = 0`, `total_samples_seen[new_stage] = 0`.

Stage 10 is terminal for DiLoCo. On stage-10 completion the coordinator writes a
manual gate state (`mode="terminal"`, `dgac_manual_gate=true`, no active or
attendance workers, `triggered_at=0`) and exits successfully without launching
DGAC or dispatching stage 11.
