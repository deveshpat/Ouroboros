# Status

Current truth -> alpha research runtime with healthy teacher-forced DGAC anchor signal, generated-answer comparison harness implemented, latest sampled generated-answer run failed the release gate.

## Anchor

canonical anchor -> `WeirdRunner/Ouroboros/diloco_state/anchor`.
source checkpoint -> `runs/azure_h100_dgac/stage_10/checkpoint-0001154`.
adapter + config + `halt_gate.pt` -> promoted, but not generated-answer validated.

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

## Latest generated-answer gate

```text
mode -> compare-coconut-val
sample -> 25 scorable validation rows
baseline -> ai21labs/AI21-Jamba-Reasoning-3B
candidate -> WeirdRunner/Ouroboros/diloco_state/anchor
stage_k -> 10
halt_threshold -> 0.5
baseline generated_answer_exact_match -> 0.08
candidate generated_answer_exact_match -> 0.04
candidate actual_latents -> 19 rows at 1 latent, 6 rows at 10 latents
result -> failed_candidate_regression; do not promote or claim win
```

Next diagnostic -> fixed-depth ablation with `--disable_candidate_halt_gate` while keeping `--candidate_requires_halt_gate`; only run full validation after sampled generated-answer comparison passes.

## Caveat

Healthy anchor != benchmark win.

The latest eval-only run proves the canonical anchor can be restored for teacher-forced training-health validation. It does not prove generated-answer progress, does not prove Ouroboros beats `ai21labs/AI21-Jamba-Reasoning-3B`, and does not prove broad benchmark superiority.

## Release-readiness workflow

```text
docs alignment
-> public CLI smoke repair [done]
-> dry-run/inspect artifact shell [done]
-> sampled ID-backed Coconut generated-answer comparison [failed diagnostic]
-> fixed-depth HaltGate ablation [next]
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

manual inputs -> `launch_worker_ids`, `dry_run`, `attendance_join_grace_minutes`.

Kaggle launch model -> edit visible command in `kaggle-utils.ipynb`; run coordinator with `launch_worker_ids=A,B` to push selected workers. Empty `launch_worker_ids` means aggregate/check only.

## Active risks

```text
latest generated-answer sample regressed vs base
HaltGate may be stopping too early at threshold 0.5
canonical anchor may not be the best generated-answer checkpoint
PEFT config compatibility warnings need version alignment before public claims
```
