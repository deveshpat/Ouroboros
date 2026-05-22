# DiLoCo Protocol

Load when round math, worker state, or promotion path unclear.

## Objects

anchor -> `diloco_state/anchor`.
round state -> `diloco_state/round_state.json`.
worker status -> `diloco_state/workers/{id}/status.json`.
worker weights -> `diloco_state/workers/{id}/round_{n}_stage_{k}/`.

## Worker

download anchor -> train shard -> upload adapter -> upload status -> push signal.

## Coordinator

read statuses -> load anchor + worker weights -> aggregate -> promote new anchor -> write next round.

Dispatch is explicit: `launch_worker_ids=A,B` pushes selected staged notebooks; empty `launch_worker_ids` performs aggregate/check only. The staged notebook injection remains the transport seam for worker id, tokens, repo/ref/commit, state/signal repos, and W&B metadata.

## Aggregation

one contributor -> direct promotion.
multiple contributors -> weighted deltas by `samples_seen`, outer LR default `0.7`.

## Attendance

no shard/quota repair -> attendance status only.
Coordinator needs presence, not fake samples.

## DGAC DiLoCo

terminal anchor -> explicit `launch_worker_ids` DGAC worker round -> adapter + HaltGate aggregation.
DGAC start must load `diloco_state/anchor` with `halt_gate.pt` when present. The visible Kaggle command in `kaggle-utils.ipynb` owns the DGAC/DiLoCo launch args.
