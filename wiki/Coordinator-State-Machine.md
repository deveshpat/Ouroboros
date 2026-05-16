# Coordinator State Machine

Load when dispatch/aggregation/promotion weird.

## Trigger

`signals/*.json` push -> coordinate.
manual `workflow_dispatch` -> force/skip/dry-run/mode controls.
concurrency -> one coordinator run at a time.

## Normal path

```text
read round_state
-> compute projected shards
-> collect ready workers
-> split active vs attendance
-> missing worker?
   -> triggered_at=0 -> re-dispatch
   -> age < timeout -> wait
   -> age >= timeout -> demote attendance
-> load anchor
-> load contributing worker weights
-> aggregate
   -> solo -> direct promotion
   -> multi -> weighted delta update
-> upload anchor
-> update total_samples_seen
-> decide next mode/workers
-> write round_state
-> push Kaggle kernels
-> reconcile failed dispatch
```

## Worker sets

`triggered_workers` -> must train/report.
`attendance_workers` -> report presence only.
`force_worker_ids` -> manual additive repair, not replacement.

## Dispatch rules

local notebook + generated metadata -> `kaggle kernels push`.
No pull.
Success -> output contains `successfully pushed`.
Quota/error text -> failed dispatch even if process code looks ok.

## Stage close

remaining samples < `min_shard_samples` -> close stage.
empty shard -> no fake work.
solo worker -> direct promotion, no stale-anchor blend.

