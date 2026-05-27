# Status

Current truth -> alpha research runtime with a healthy teacher-forced DGAC anchor signal and a working generated-answer comparison harness, but **no release-valid model-quality claim**. The HaltGate path produced degenerate/over-stopped generations. The latest fixed-depth longest-25 run completed after memory fixes, but candidate regressed against baseline. The project is now in **evaluation/generation-harness sanity** before more training, release framing, or JEPA work.

## Anchor

canonical anchor -> `WeirdRunner/Ouroboros/diloco_state/anchor`.
source checkpoint -> `runs/azure_h100_dgac/stage_10/checkpoint-0001154`.
adapter + config + `halt_gate.pt` -> promoted to anchor path, but generated-answer behavior is not release validated.

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

### HaltGate-enabled path

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
result -> failed_candidate_regression; HaltGate over-stopped and generated degenerate answers
```

### Earlier fixed-depth diagnostic slice

```text
mode -> compare-coconut-val --disable_candidate_halt_gate --candidate_requires_halt_gate
sample -> first 25 scorable validation rows
baseline generated_answer_exact_match -> 0.08
candidate generated_answer_exact_match -> 0.12
candidate actual_latents_mean -> 10.0
candidate actual_latents histogram -> 10:25
result -> diagnostic-only pass; not release-valid because learned HaltGate decisions were bypassed
```

### Latest fixed-depth longest-25 OOM/memory stress check

```text
mode -> compare-coconut-val --disable_candidate_halt_gate --candidate_requires_halt_gate
sample_strategy -> longest
sample -> 25 longest scorable validation rows
max_seq_len -> 8192
baseline generated_answer_exact_match -> 0.12
candidate generated_answer_exact_match -> 0.08
candidate actual_latents_mean -> 10.0
candidate actual_latents histogram -> 10:25
result -> completed after memory fixes, but failed candidate gate; use as OOM/harness evidence, not quality evidence
```

## Interpretation

```text
HaltGate quality -> not release-ready
fixed-depth quality -> not established; latest hardest-slice diagnostic regressed
OOM/memory fixes -> improved enough for longest-25 run completion
current blocker -> generation/eval harness fidelity, PEFT version compatibility, raw output quality, and decoding/extraction sanity
```

Do not infer that more HaltGate epochs alone will fix the issue. Do not infer that JEPA/curriculum work should start before eval sanity. The next milestone is to prove that the harness is letting both baseline and candidate generate answers under sane settings.

## Next work

```text
1. resolve or reproduce PEFT config compatibility warnings with the training-time PEFT version
2. inspect raw `results.jsonl` generations for empty answers, degenerate tokens, prompt leakage, and answer-extraction misses
3. run small decoding sweeps with explicit max_new_tokens, stop criteria, temperature/do_sample, and answer extraction rules
4. add/use lm-eval-compatible generation harness for standardized benchmark configuration
5. run tiny lm-eval generation sanity before MC/loglikelihood suites
6. only after sane generation: choose between full in-domain eval, public benchmark phase, HaltGate objective repair, or JEPA/curriculum branch
```

## Caveat

Healthy anchor != benchmark win.

The eval-only run proves the canonical anchor can be restored for teacher-forced training-health validation. The fixed-depth earlier pass proved the path could sometimes produce useful answers when HaltGate was bypassed. The latest longest-25 fixed-depth result shows the hardest-slice eval now completes but does not show a candidate win.

## Release-readiness workflow

```text
docs alignment [done]
-> public CLI smoke repair [done]
-> dry-run/inspect artifact shell [done]
-> sampled ID-backed Coconut generated-answer comparison, HaltGate enabled [failed: degenerate/over-stop]
-> fixed-depth OOM/memory stress run on longest-25 [done: completed but regressed]
-> PEFT/runtime fidelity check [next]
-> raw generation and decoding sanity [next]
-> lm-eval generation sanity bridge [next]
-> full Coconut validation only after sample-level generation is sane
-> research README + HF model card metrics from release-valid artifacts
-> faithful cloud demo only after release-valid evidence
-> HaltGate/JEPA curriculum branch only after eval failure modes are separated from model failure modes
```

## Current docs/release artifacts

```text
README.md -> research-style alpha overview and current pivot
plans/public-alpha-release.md -> implementation/eval plan and current pivot
docs/release/HF_MODEL_CARD_DRAFT.md -> Hugging Face model card draft with no public metrics yet
wiki/Future-JEPA-Multimodal-Latent.md -> JEPA/curriculum branch guardrails
```

## Runtime/package truth

Implemented package roots:

```text
Bootstrap -> runtime guardrails
Coconut -> training/DGAC/eval-only
Models -> HF CausalLM compatibility
Inference -> package API + `python -m ouroboros.inference --help`
Coordinator -> dispatch/aggregate/promote
Eval -> Coconut validation inspection/dry-run artifacts + generated-answer comparison CLI; lm-eval bridge/sanity is now a priority
Utils -> provider IO
```

## Dispatch controls

manual inputs -> `launch_worker_ids`, `dry_run`, `attendance_join_grace_minutes`.

Kaggle launch model -> edit visible command in `kaggle-utils.ipynb`; run coordinator with `launch_worker_ids=A,B` to push selected workers. Empty `launch_worker_ids` means aggregate/check only.

## Active risks

```text
HaltGate-enabled generation regressed and produced degenerate/over-stopped behavior
fixed-depth longest-25 completed but regressed vs baseline
canonical anchor may be undertrained or not the best generated-answer checkpoint
answer extraction/decoding may be suppressing correct answers
PEFT config compatibility warnings need version alignment before public claims
lm-eval/loglikelihood support must be faithful to the latent runtime before external claims
```
