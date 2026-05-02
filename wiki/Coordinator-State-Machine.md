# Coordinator State Machine
> Load this page when debugging coordinator runs, dispatch failures, or aggregation logic.

---

## Entry Points (GitHub Actions triggers)

| Trigger | Path |
|---|---|
| `push` to `signals/*.json` | Normal round completion signal |
| `schedule` (every 30 min) | Watchdog: timeout demotion, waiting mode re-dispatch |
| `workflow_dispatch` | Manual: `force_worker_ids`, `skip_trigger`, `dry_run` |

Concurrency: `group: diloco-coordinate`, `cancel-in-progress: false` — runs serialize, never race.

---

## Normal Mode Code Path

```
1. Read round_state.json
2. Compute projected_shards for current stage/round
3. collect_ready_workers(triggered_workers + attendance_workers)
4. Partition: active_ready vs attendance_ready
5. Check missing triggered workers:
   a. triggered_at == 0  → re-dispatch immediately (unconfirmed path)
   b. elapsed < 13h      → print "waiting", return
   c. elapsed > 13h      → demote to attendance_workers
6. Load anchor weights (CPU)
7. Load worker weights for contributing workers (samples_seen > 0)
8. Aggregate:
   - solo mode  → direct weight promotion (worker weights become new anchor)
   - diloco mode → weighted_average_deltas(anchor, workers, samples, outer_lr=0.7)
9. Upload new anchor to diloco_state/anchor/
10. Update total_samples_seen[stage_k]
11. Check stage_complete
12. Determine next_mode, next_active_workers, next_attendance_workers
13. Write round_state.json
14. Trigger Kaggle workers (kernels push)
15. _reconcile_post_dispatch_state if any push failed
```

---

## triggered_at=0 Recovery Path (verified Session 19)

Triggered when: `expected_workers` present in `round_state.triggered_workers` but `triggered_at == 0`.

Interpretation: coordinator wrote the state but the Kaggle push was never confirmed (network timeout, auth failure, etc.).

Action:
```
new_state = {...state, triggered_at: time.time()}
hub_upload(round_state.json)
trigger_kaggle_workers(expected_workers + attendance_workers)
_reconcile_post_dispatch_state(...)  # fix if some pushes fail
return  # no aggregation this run
```

Next coordinator run sees workers as expected, waits normally.

---

## Dispatch Reconciliation

After `trigger_kaggle_workers()`, for any worker with status `"failed"`:
- Remove from `triggered_workers` → move to `attendance_workers`
- Update `triggered_at = 0` if no workers actually dispatched
- Re-upload corrected `round_state.json`

This prevents the coordinator from waiting 13h for a worker that was never launched.

---

## Outer Update Formula (DiLoCo)

```
pseudo_grad_i = anchor_weights - worker_i_weights
outer_grad    = Σ(pseudo_grad_i × samples_i / total_samples)
new_anchor    = anchor_weights - outer_lr × outer_grad
```

`outer_lr = 0.7` (DiLoCo paper default).
Solo mode: `new_anchor = worker_weights` directly (outer_lr effectively 1.0).

---

## W&B Coordinator Logging

Run ID: `diloco-coordinator-s{stage_k}` (one run per stage, resumed across rounds).

Logged per round:
- `coordinator/round`, `coordinator/workers_aggregated`
- `coordinator/samples_this_round`, `coordinator/total_samples_stage`
- `coordinator/mode`, `coordinator/pct_stage_done`
- `coordinator/stage_complete` (logged once on stage advance)

---

## Common Failure Patterns

| Symptom | Cause | Fix |
|---|---|---|
| Coordinator loops "Waiting for workers" indefinitely | triggered_at > 0 but workers never ran | Manual: set `triggered_at: 0` in round_state.json |
| Worker status "done" but coordinator ignores it | Worker not in `triggered_workers` list | Use `--force_worker_ids` or check attendance_workers |
| `kernels push` returns 0 but worker assigned P100 | kaggle < 1.8.4 or wrong capitalisation | Verify `kaggle>=1.8.4`, `"NvidiaTeslaT4"` in metadata, `--accelerator NvidiaTeslaT4` in push_args |
| Stage never closes | Geometric remainder < min_shard_samples | Normal — coordinator declares stage complete automatically |
