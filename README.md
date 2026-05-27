# Ouroboros

Ouroboros is an alpha research runtime for training and evaluating a Coconut/DGAC-style latent reasoning adapter on top of `ai21labs/AI21-Jamba-Reasoning-3B`.

The project goal is not to publish another thin model wrapper. The goal is to test whether a lightweight latent-reasoning adapter, a dedicated latent token, and a learned compute controller can improve reasoning behavior while keeping the runtime small enough to evaluate, deploy, and eventually optimize for personal/local use.

## Current status

```text
base model      -> ai21labs/AI21-Jamba-Reasoning-3B
adapter target  -> WeirdRunner/Ouroboros/diloco_state/anchor
method          -> PEFT adapter + <|lat|> token + Coconut latent passes + experimental DGAC HaltGate
runtime state   -> package-based runtime extracted from notebook/root-script shape
release state   -> alpha, pre-claim; no public benchmark or superiority claim
current phase   -> evaluation/generation-harness sanity before more training or release framing
```

Latest anchor health signal:

```text
dataset loaded      -> 36,906 train / 1,940 validation
stage               -> 10
eval mode           -> eval-only
teacher-forced CE  -> 0.4114
teacher-forced token acc -> 0.8693
mamba fast path     -> active
anchor restored     -> adapter + halt_gate.pt restored
status              -> healthy checkpoint signal, not a benchmark claim
```

Generated-answer evidence now has three separate meanings:

```text
HaltGate enabled sample-25         -> degenerate/over-stopped generated tokens; not release-valid
fixed-depth earlier diagnostic     -> candidate could beat baseline on one small sampled slice when HaltGate was bypassed
fixed-depth longest-25 OOM check   -> baseline 0.12, candidate 0.08, actual_latents 10:25, failed_candidate_regression
current interpretation             -> runtime can run without OOM on the hardest slice, but generation quality and scoring are not yet trusted enough for release claims
```

The latest longest-25 fixed-depth run was an OOM/memory-stability stress check, not a promotion signal. It used `--disable_candidate_halt_gate`, selected the 25 longest scorable rows, and failed the generated-answer gate. Treat it as evidence that the harness can complete after memory fixes, not evidence that the anchor is ready.

## Current decision

Do **not** publish the anchor as a fixed-depth preview yet. Do **not** restart HaltGate workers blindly. Do **not** jump straight to JEPA/curriculum work before proving the generation/evaluation path is measuring the model correctly.

Current next step:

```text
1. fix/verify PEFT/runtime fidelity warnings
2. run a small generation sanity sweep with explicit decoding settings and raw output inspection
3. add/use an lm-eval-compatible benchmark harness for proper eval configuration and artifact discipline
4. only then decide whether the next investment is benchmark phase, HaltGate retraining, or JEPA/curriculum research
```

## Why this exists

Most model experiments fail to become useful because they stop at one of two incomplete states:

1. a training log with no reproducible comparison, or
2. a demo that hides the actual research runtime behind an unrelated serving path.

Ouroboros is being structured so the research path and the release path stay connected:

```text
train anchor
-> restore exact adapter/HaltGate/runtime artifacts
-> run generated-answer sanity checks with raw output inspection
-> run ID-backed Coconut validation comparison only when generation is sane
-> run lm-eval-compatible benchmark harnesses with faithful candidate and baseline paths
-> diagnose whether failures are eval setup, PEFT/runtime fidelity, undertraining, HaltGate objective, or latent curriculum
-> then train/calibrate HaltGate or start JEPA/curriculum branches from evidence
-> publish model card + results table only from release-valid artifacts
-> deploy faithful demo only after behavior is preserved
```

## Runtime map

| Package | Owns | Public surface |
|---|---|---|
| `ouroboros.bootstrap` | runtime setup, device/dtype guardrails, known-failure triage | imported before heavy runtime |
| `ouroboros.coconut` | curriculum, latent passes, DGAC/HaltGate, train/checkpoint/resume | `python -m ouroboros.coconut ...` |
| `ouroboros.models` | Hugging Face model/tokenizer loading, PEFT adapter loading, quant/memory policy | `ouroboros.models` |
| `ouroboros.inference` | prompt formatting, latent decode, text generation | `python -m ouroboros.inference ...` |
| `ouroboros.coordinator` | DiLoCo/solo/DDP dispatch, aggregation, promotion, repair | `python -m ouroboros.coordinator ...` |
| `ouroboros.eval` | Coconut validation inspection/dry-run artifacts, generated-answer comparison, and lm-eval bridge work | `python -m ouroboros.eval ...` |
| `ouroboros.utils` | provider IO helpers for Hub, W&B, Kaggle, local runtime | helper layer only |

## What works today

```bash
python -m ouroboros.coconut --help
python -m ouroboros.coordinator --help
python -m ouroboros.inference --help
python -m ouroboros.eval --help
python -m ouroboros.eval dry-run-coconut-val ...
python -m ouroboros.eval inspect-coconut-val ...
python -m ouroboros.eval compare-coconut-val ...
```

Recent eval-only anchor validation completed successfully through the Coconut runtime as a teacher-forced training-health signal. The public package compiles successfully:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m compileall -q ouroboros
```

The `coordinate` GitHub Actions workflow can dispatch generated-answer eval to a Kaggle worker notebook:

```text
eval_mode=sample-25   -> CPU token-budget preflight, then compare-coconut-val on the first 25 scorable rows with max_seq_len=512 unless overridden
eval_mode=longest-25  -> CPU token-budget preflight, then compare-coconut-val on the 25 longest prompt-budget rows with max_seq_len=8192 unless overridden
eval_mode=full        -> CPU token-budget preflight over the full validation split, then compare-coconut-val with max_seq_len=8192 unless overridden
fixed-depth ablation  -> workflow input eval_disable_candidate_halt_gate=true or set OUROBOROS_EVAL_DISABLE_CANDIDATE_HALT_GATE=1
inspect artifacts     -> runs/eval/coconut_val_compare_<mode>
multi-GPU eval        -> eval_model_device_map=balanced_low_0 by default; use single to reproduce old cuda:0 pinning
truncation policy     -> token_budget.preflight.json records tokenizer-only counts before model loading; truncation exits non-zero unless --allow_truncated_eval is set
memory guardrail      -> eval cleanup runs every 25 samples by default; this does not replace bounded context
training note         -> training remains DDP/torchrun; eval model sharding is not a drop-in replacement for gradient-synchronized training
```

## Known release blockers

These are intentional blockers before public claims or a world-facing deployment:

```text
1. HaltGate-enabled generation is degenerate/over-stopped; do not claim dynamic stopping works
2. fixed-depth longest-25 now completes after memory fixes but regresses against baseline; do not claim fixed-depth quality yet
3. PEFT config compatibility warnings must be resolved or explicitly reproduced with the training-time PEFT version
4. raw generated outputs must be inspected to confirm answer extraction/decoding is not suppressing correct answers
5. lm-eval-compatible benchmark harness must be added or used for stable decoding/configuration and external benchmark discipline
6. full in-domain validation should wait until sample-level generation is sane and memory-stable
7. public metric tables must come only from release-valid artifacts, never from teacher-forced health metrics or fixed-depth smoke checks
8. JEPA/curriculum/HaltGate retraining should start only after the eval harness can distinguish model failure from harness failure
```

## Evaluation standard

A result is release-worthy only if it answers all of these:

```text
what model was evaluated?
what checkpoint/adapter was used?
what PEFT/transformers/runtime versions loaded the adapter?
what prompt template was used?
what dataset or benchmark split was used?
what split/revision was used, and what is its contamination/claim boundary?
what decoding settings were used?
what exact scoring script produced generated-answer exact match?
were raw generations inspected for degenerate tokens, empty answers, or answer-extraction failures?
can the base model run through the same harness?
```

Generated-answer comparison writes artifacts before enforcing the release gate. By default it exits non-zero when candidate exact match is below baseline exact match. Use `--allow_candidate_regression` only for diagnostics, not promotion. Use `--disable_candidate_halt_gate` only for fixed-depth ablations and OOM/memory-stability checks.

Current sampled evidence says the project is not ready to choose between release and new training. The HaltGate path is not generated-answer valid, and the latest fixed-depth longest-25 stress run completed but regressed. The next valid milestone is **evaluation-harness sanity**, including PEFT version fidelity and lm-eval-compatible benchmark configuration.

The first comparison target remains:

```text
baseline  -> ai21labs/AI21-Jamba-Reasoning-3B
candidate -> same base + Ouroboros adapter + <|lat|> + Coconut latent runtime
controller -> HaltGate only when explicitly evaluating dynamic stopping; otherwise fixed-depth is diagnostic-only
```

## Documentation map

```text
BLUEPRINT.md                         -> package ownership and public command map
wiki/STATUS.md                       -> current project truth and next gates
wiki/Engineering-Workflow.md          -> repo-change workflow
plans/public-alpha-release.md          -> implementation/eval plan and current pivot
docs/release/HF_MODEL_CARD_DRAFT.md   -> Hugging Face model card draft, no public metrics yet
wiki/Future-JEPA-Multimodal-Latent.md  -> JEPA/curriculum parking lot and guardrails
terminal_log.md                       -> latest relevant run evidence
```

## Non-claims

Until generation/eval sanity and release-valid artifacts exist, this project does **not** claim:

```text
Ouroboros beats Jamba
Ouroboros fixed-depth latent inference is better than baseline
DGAC/HaltGate dynamic stopping works
HaltGate knows the optimal loop count
JEPA/curriculum improvements are implemented
Ouroboros is production-ready
Ouroboros is safety-aligned beyond the base model
Ouroboros is fully edge-compatible
quantized/merged exports preserve latent/HaltGate behavior
```

## License

See `LICENSE`.
