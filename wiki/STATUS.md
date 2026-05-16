# Status

Current truth -> seven-package runtime.

## Anchor

canonical anchor -> `WeirdRunner/Ouroboros/diloco_state/anchor`.
source checkpoint -> `runs/azure_h100_dgac/stage_10/checkpoint-0001154`.
adapter + config + `halt_gate.pt` -> promoted.

## Caveat

H100 run skipped val/gen near timeout.
So anchor = training evidence, not quality proof.
Next gate -> `kaggle_run_mode=dgac-anchor-eval` -> compare val CE/acc/gen/diagnostics.

## Workflow

Coordinator -> dispatch/aggregate/promote.
Eval -> quality gates + lm-eval.
Coconut -> training/DGAC.
Bootstrap -> runtime guardrails.
Models -> HF CausalLM compatibility.
Utils -> provider IO.

## Dispatch controls

manual inputs -> `force_worker_ids`, `skip_trigger`, `dry_run`, `kaggle_run_mode`, `dgac_anchor_eval_resume_mode`, `dgac_diagnostics_forced_kmax_ce`.

## Active risks

quota exhaustion -> attendance/timeout path.
Kaggle push false-success -> strict output parser.
wrong GPU -> accelerator + runtime fast-fail.
DGAC eval OOM -> inference-mode guard.
