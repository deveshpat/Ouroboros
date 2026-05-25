# Status

Current truth -> alpha research runtime with healthy teacher-forced DGAC anchor signal and a working generated-answer comparison harness. The canonical anchor is **not release-safe with HaltGate enabled**: sampled generated-answer eval shows the HaltGate over-stops at one latent for most rows. The same adapter improves over baseline only in fixed-depth diagnostic mode with the HaltGate bypassed.

## Anchor

canonical anchor -> `WeirdRunner/Ouroboros/diloco_state/anchor`.
source checkpoint -> `runs/azure_h100_dgac/stage_10/checkpoint-0001154`.
adapter + config + `halt_gate.pt` -> promoted to anchor path, but HaltGate behavior is not generated-answer validated.

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

## Latest generated-answer evidence

### HaltGate-enabled release path

```text
mode -> compare-coconut-val
sample -> 25 scorable validation rows
baseline -> ai21labs/AI21-Jamba-Reasoning-3B
candidate -> WeirdRunner/Ouroboros/diloco_state/anchor
stage_k -> 10
halt_threshold -> 0.5
baseline generated_answer_exact_match -> 0.08
candidate generated_answer_exact_match -> 0.04
candidate actual_latents_mean -> 3.16
candidate actual_latents histogram -> 1:19, 10:6
one_latent_fraction -> 0.76
result -> failed_candidate_regression; HaltGate over-stop suspected/confirmed by ablation
```

### Fixed-depth diagnostic ablation

```text
mode -> compare-coconut-val --disable_candidate_halt_gate --candidate_requires_halt_gate
sample -> same 25 scorable validation rows
baseline generated_answer_exact_match -> 0.08
candidate generated_answer_exact_match -> 0.12
candidate actual_latents_mean -> 10.0
candidate actual_latents histogram -> 10:25
result -> passed diagnostic only; adapter/runtime path can produce coherent gains when HaltGate decisions are bypassed
```

Interpretation -> tokenizer/adapter loading is unlikely to be the root cause. The current blocker is HaltGate training/calibration. Fixed-depth pass is not a release claim because the declared DGAC/HaltGate-backed runtime did not use learned HaltGate decisions.

Next work -> train/calibrate HaltGate, then rerun sampled HaltGate-enabled `compare-coconut-val`. Do not run full validation, promote claims, or publish benchmark tables until the HaltGate-enabled sampled gate passes.

## Caveat

Healthy anchor != benchmark win.

The eval-only run proves the canonical anchor can be restored for teacher-forced training-health validation. The fixed-depth ablation proves the adapter/runtime path is not obviously broken. Neither proves that the DGAC/HaltGate-backed model beats `ai21labs/AI21-Jamba-Reasoning-3B`, and neither proves broad benchmark superiority.

## Release-readiness workflow

```text
docs alignment
-> public CLI smoke repair [done]
-> dry-run/inspect artifact shell [done]
-> sampled ID-backed Coconut generated-answer comparison, HaltGate enabled [failed: over-stop]
-> fixed-depth HaltGate ablation [done: passed diagnostic]
-> HaltGate training/calibration [next]
-> sampled HaltGate-enabled comparison [required before full validation]
-> full Coconut validation after sampled HaltGate-enabled artifact inspection
-> research README + HF model card metrics from release-valid artifacts
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
HaltGate-enabled generated-answer sample regressed vs base
HaltGate over-stops at threshold 0.5: 19/25 rows used one latent
fixed-depth ablation passed sample-25 but is diagnostic-only, not release-valid
canonical anchor may not be the best generated-answer checkpoint
PEFT config compatibility warnings need version alignment before public claims
```
