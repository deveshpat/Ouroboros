# Ouroboros

Ouroboros is an alpha research runtime for training and evaluating a Coconut/DGAC reasoning adapter on top of `ai21labs/AI21-Jamba-Reasoning-3B`.

The project goal is not to publish another thin model wrapper. The goal is to test whether a lightweight latent-reasoning adapter, a dedicated latent token, and a DGAC HaltGate can improve reasoning behavior while keeping the runtime small enough to evaluate, deploy, and eventually optimize for personal/local use.

## Current status

```text
base model      -> ai21labs/AI21-Jamba-Reasoning-3B
adapter target  -> WeirdRunner/Ouroboros/diloco_state/anchor
method          -> PEFT adapter + <|lat|> token + DGAC HaltGate
runtime state   -> package-based runtime extracted from notebook/root-script shape
release state   -> alpha, pre-claim, generated-answer comparison harness implemented; real artifacts pending
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

This result is a training-health side metric, not real generated-answer progress. The next release gate is an unbiased generated-answer exact-match comparison where the true base model and Ouroboros are evaluated on the same validation IDs, prompt policy, decoding settings, normalization, and scoring scripts.

## Why this exists

Most model experiments fail to become useful because they stop at one of two incomplete states:

1. a training log with no reproducible comparison, or
2. a demo that hides the actual research runtime behind an unrelated serving path.

Ouroboros is being structured so the research path and the release path stay connected:

```text
train anchor
-> restore exact adapter/HaltGate runtime
-> run ID-backed Coconut validation comparison
-> then run unbiased external benchmark evals
-> publish model card + results table
-> deploy faithful demo
-> then optimize/quantize only after behavior is preserved
```

## Runtime map

| Package | Owns | Public surface |
|---|---|---|
| `ouroboros.bootstrap` | runtime setup, device/dtype guardrails, known-failure triage | imported before heavy runtime |
| `ouroboros.coconut` | curriculum, latent passes, DGAC/HaltGate, train/checkpoint/resume | `python -m ouroboros.coconut ...` |
| `ouroboros.models` | Hugging Face model/tokenizer loading, PEFT adapter loading, quant/memory policy | `ouroboros.models` |
| `ouroboros.inference` | prompt formatting, latent decode, text generation | `python -m ouroboros.inference ...` |
| `ouroboros.coordinator` | DiLoCo/solo/DDP dispatch, aggregation, promotion, repair | `python -m ouroboros.coordinator ...` |
| `ouroboros.eval` | Coconut validation inspection/dry-run artifacts and generated-answer comparison; lm-eval bridge later | `python -m ouroboros.eval ...` |
| `ouroboros.utils` | provider IO helpers for Hub, W&B, Kaggle, local runtime | helper layer only |

## What works today

```bash
python -m ouroboros.coconut --help
python -m ouroboros.coordinator --help
python -m ouroboros.inference --help
python -m ouroboros.eval --help
python -m ouroboros.eval dry-run-coconut-val ...
python -m ouroboros.eval inspect-coconut-val ...
```

Recent eval-only anchor validation completed successfully through the Coconut runtime as a teacher-forced training-health signal. The public package compiles successfully:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m compileall -q ouroboros
```

## Known release blockers

These are intentional blockers before public claims or a world-facing deployment:

```text
1. run sampled `compare-coconut-val` with real model weights and local validation data
2. inspect generated `run_config.json`, `summary.json`, and `results.jsonl`
3. run the full Coconut validation split only after sampled artifacts pass inspection
4. copy/generate public metric tables only from real artifacts
5. deploy a faithful demo that uses the actual Ouroboros latent/HaltGate runtime
6. add optional lm-eval bridge later, after latent-aware loglikelihood is implemented
```

## Evaluation standard

A result is release-worthy only if it answers all of these:

```text
what model was evaluated?
what checkpoint/adapter was used?
what prompt template was used?
what dataset or benchmark split was used?
what split/revision was used, and what is its contamination/claim boundary?
what decoding settings were used?
what exact scoring script produced generated-answer exact match?
can the base model run through the same harness?
```

The first comparison target is:

```text
baseline  -> ai21labs/AI21-Jamba-Reasoning-3B
candidate -> same base + Ouroboros adapter + <|lat|> + DGAC HaltGate + latent runtime
```

## Documentation map

```text
BLUEPRINT.md                         -> package ownership and public command map
wiki/STATUS.md                       -> current project truth and next gates
wiki/Engineering-Workflow.md          -> repo-change workflow
plans/public-alpha-release.md          -> implementation plan for CLI, eval artifacts, demo, lm-eval bridge
docs/release/HF_MODEL_CARD_DRAFT.md   -> Hugging Face model card draft, metric tables pending real artifacts
terminal_log.md                       -> latest relevant run evidence
```

## Non-claims

Until the comparison eval is complete, this project does **not** claim:

```text
Ouroboros beats Jamba
Ouroboros is production-ready
Ouroboros is safety-aligned beyond the base model
Ouroboros is fully edge-compatible
quantized/merged exports preserve the latent/HaltGate behavior
```

## License

See `LICENSE`.
