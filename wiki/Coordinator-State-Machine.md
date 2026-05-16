# Coordinator State Machine
> Load this page when debugging coordinator runs, dispatch failures, or aggregation logic.

---

## Entry Points (GitHub Actions triggers)

| Trigger | Path |
|---|---|
| `push` to `signals/*.json` | Normal round completion signal |
| `workflow_dispatch` | Manual: `force_worker_ids`, `skip_trigger`, `dry_run`, `workflow_validate=cpu-smoke`, `kaggle_run_mode`, and DGAC anchor eval resume controls (`dgac_anchor_eval_resume_mode`, `dgac_diagnostics_forced_kmax_ce`) |

Concurrency: `group: diloco-coordinate`, `cancel-in-progress: false` — runs serialize, never race.

The scheduled watchdog is intentionally disabled while strict local Mac DGAC
fallback is available. Signal pushes and manual dispatch remain available, but
GitHub Actions must not wake itself on a timer and race a Mac fallback that has
claimed the Hub round.

---

## Normal Mode Code Path

```
1. Read round_state.json
2. Compute projected_shards for current stage/round
3. collect_ready_workers(triggered_workers + attendance_workers)
4. Partition: active_ready vs attendance_ready
5. Check missing triggered workers:
   a0. `force_worker_ids` present → add only missing available/done forced workers to active set
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

`force_worker_ids` is additive manual repair. It does **not** replace
`triggered_workers`, does **not** filter aggregation, and does **not** discard
already-running or already-completed work. In the common stuck state
`triggered_workers=["B"]`, `attendance_workers=["A", "C"]`, running
`--force_worker_ids A,C` writes `triggered_workers=["B", "A", "C"]` and only
dispatches A/C.

Active worker statuses with `samples_seen=0` are not useful training output and
do not satisfy active-round completion. Zero-sample statuses remain valid as
attendance check-ins.

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

Force-repair dispatch reconciliation preserves already-active workers that were
not re-triggered in the repair run. Example: if B was already running and force
repair dispatches A/C, a C push failure leaves `triggered_workers=["B", "A"]`
and moves C back to `attendance_workers`.

---

## Stage 10 Terminal Gate

Stage 10 is terminal for DiLoCo. When stage 10 completes, the coordinator:

- keeps `stage_k=10` and sets `round_n=0`;
- sets `mode="terminal"` and `dgac_manual_gate=true`;
- clears `triggered_workers` and `attendance_workers`;
- sets `triggered_at=0`;
- preserves the final stage-10 anchor;
- prints the DGAC manual-gate reminder;
- exits without pushing Kaggle kernels.

DGAC is launched manually after quality review. Cron must never auto-dispatch a
stage-11 DiLoCo round.

---

## Strict Mac DGAC Fallback

The local Mac path is `python -m ouroboros.coordinator.mac_dgac_fallback`. It is allowed to
mutate Hub state only after all of these pass:

- live `round_state.json` still matches the expected DGAC waiting round
  (`stage_k=10`, `round_n=3`, `mode=waiting`, `total_samples_seen[10]=23481`,
  projected A/B/C shards of 4,475 each);
- Apple Silicon MPS is available and CUDA is unavailable;
- `mamba-ssm-macos` imports and its probe passes;
- a Jamba FP16/MPS forward/backward probe passes;
- `diloco_state/anchor` contains the adapter and `halt_gate.pt`;
- `--use_4bit` is not requested.

After preflight, the runner writes
`diloco_state/locks/mac_dgac_fallback.json` with a short-lived claim id, converts
the waiting round back to `mode="dgac-diloco"` with `triggered_workers=["A","B","C"]`,
runs workers sequentially on the Mac with no GitHub signal token, then runs the
local coordinator with `--skip_trigger --mac_claim_id <claim>`.

Any GitHub Actions coordinator run that sees an active foreign Mac claim exits
before reading or mutating `round_state.json`, so manual dispatch and signal
doorbells cannot create a competing Kaggle session. A local coordinator process
with the matching `--mac_claim_id` may aggregate the worker artifacts and still
skips the next trigger.

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
| Force-trigger appears to drop a running worker | Bug — force must be additive | Preserve existing active workers and dispatch only missing forced workers |
| `kernels push` returns 0 but worker assigned P100 | kaggle < 1.8.4 or wrong capitalisation | Verify `kaggle>=1.8.4`, `"NvidiaTeslaT4"` in metadata, `--accelerator NvidiaTeslaT4` in push_args |
| Stage never closes | Geometric remainder < min_shard_samples | Normal — coordinator declares stage complete automatically |
