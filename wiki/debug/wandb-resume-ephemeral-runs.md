---
title: W&B Resume Creates Ephemeral Runs
type: debug
sources:
  - terminal_log.md
  - BLUEPRINT.md
  - ouroboros/diloco/observability.py
updated: 2026-04-30
---

# W&B Resume Creates Ephemeral Runs (wandb==0.25.0)

## Symptom

Stage 2 completed 6 rounds across workers A and B. W&B dashboard showed exactly
**1 run** labelled "Worker A | Stage 2" — all subsequent rounds were invisible.
The training had happened (weights were uploaded, signals pushed) but the W&B
metrics for rounds 1–5 were gone.

## Root Cause

`wandb==0.25.0` with `resume="allow"`: when a run with the given `id=` has
already *finished*, `resume="allow"` does not reopen it. Instead it silently
creates a new run with an **auto-generated ID**, discarding the specified `id=`.
The new run is created as an ephemeral run attached to the same project but
with no stable identity — it doesn't show up under the expected run name.

This meant every round after round 0 was logged to a throwaway run that
disappeared or was impossible to find in the dashboard.

## Fix

Remove `resume="allow"`. Use unique run IDs per round instead:

```python
run_id  = f"diloco-{worker_lower}-s{stage_k}-r{round_n}"   # unique per round
group   = f"diloco-{worker_lower}-s{stage_k}"               # groups all rounds in UI
name    = f"Worker {worker} | Stage {stage_k} | Round {round_n}"
```

No `resume` parameter at all — each round starts a fresh, stable run with its
own ID. The `group=` parameter clusters all rounds for a stage together in the
W&B UI without needing resume.

## Implementation

`worker_wandb_identity()` in `ouroboros/diloco/observability.py` generates these
values deterministically from `(worker_id, stage_k, round_n)`. Workers call this
at run init. The coordinator uses `coordinator_wandb_identity()` separately.

## W&B Step Axis

Without resume, step counts reset to 0 each round. The fix: a monotonic step offset.

```python
step_offset = round_n × (shard_step_estimate + 1)
```

`shard_step_estimate = ceil(36906 / 3 / (batch_size × grad_accum)) = 385`

The `+1` gap between rounds prevents step collisions at round boundaries.
This is computed in `worker_wandb_identity()` as `step_offset` and added to each
logged step in the worker training loop.

## Verification

Dashboard showing separate, named runs per round from stage 4 onward. Each run
is findable by its deterministic ID. Group view clusters them per stage.
