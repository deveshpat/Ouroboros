# Status

Current truth -> alpha research runtime with healthy DGAC anchor signal, public CLI/eval artifact shell implemented, generated-answer comparison artifacts pending.

## Anchor

canonical anchor -> `WeirdRunner/Ouroboros/diloco_state/anchor`.
source checkpoint -> `runs/azure_h100_dgac/stage_10/checkpoint-0001154`.
adapter + config + `halt_gate.pt` -> promoted.

## Latest teacher-forced health signal

```text
dataset -> 36,906 train / 1,940 val
stage -> 10
gpu -> Tesla T4 16GB fp16
mamba fast path -> active
anchor restore -> adapter + halt_gate.pt restored from canonical anchor
mode -> eval-only
teacher_forced_ce -> 0.4114
teacher_forced_token_acc -> 0.8693
result -> healthy checkpoint signal, not generated-answer progress
```

## Caveat

Healthy anchor != benchmark win.

The latest eval-only run proves the canonical anchor can be restored for teacher-forced training-health validation. It does not prove generated-answer progress, does not prove Ouroboros beats `ai21labs/AI21-Jamba-Reasoning-3B`, and does not prove broad benchmark superiority.

Next gate -> ID-backed in-domain holdout comparison on the canonical Coconut validation split:

```text
baseline -> ai21labs/AI21-Jamba-Reasoning-3B
candidate -> same base + Ouroboros adapter + <|lat|> + DGAC HaltGate + latent runtime
```

## Release-readiness workflow

```text
docs alignment
-> public CLI smoke repair [done]
-> dry-run/inspect artifact shell [done]
-> sampled ID-backed Coconut generated-answer comparison [pending real weights/data]
-> full Coconut validation after sampled artifact inspection
-> research README + HF model card metrics from artifacts
-> faithful cloud demo
-> optional lm-eval bridge later
-> optimization/edge experiments
```

## Current docs/release artifacts

```text
README.md -> research-style alpha overview
plans/public-alpha-release.md -> implementation plan for CLI repair, Coconut val artifacts, demo, lm-eval bridge
docs/release/HF_MODEL_CARD_DRAFT.md -> Hugging Face model card draft
```

## Runtime/package truth

Implemented package roots:

```text
Bootstrap -> runtime guardrails
Coconut -> training/DGAC/eval-only
Models -> HF CausalLM compatibility
Inference -> package API + `python -m ouroboros.inference --help`
Coordinator -> dispatch/aggregate/promote
Eval -> Coconut validation inspection/dry-run artifacts + generated-answer comparison CLI; lm-eval bridge pending
Utils -> provider IO
```

## Dispatch controls

manual inputs -> `force_worker_ids`, `skip_trigger`, `dry_run`, `kaggle_run_mode`, `benchmark_suite`, optional `benchmark_tasks`, `dgac_anchor_eval_resume_mode`.

## Active risks

```text
validation claim boundary -> Coconut val is ID-backed in-domain holdout, not external benchmark
missing real generated-answer artifacts -> docs/model card must not publish wins
quota exhaustion -> attendance/timeout path
Kaggle push false-success -> strict output parser
wrong GPU -> accelerator + runtime fast-fail
DGAC eval OOM -> inference-mode guard
optimization drift -> quantized/edge path must be compared against faithful runtime
```
