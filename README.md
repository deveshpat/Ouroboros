# Ouroboros

Ouroboros is an alpha research runtime for training and evaluating a Coconut/DGAC reasoning adapter on top of `ai21labs/AI21-Jamba-Reasoning-3B`.

The project goal is not to publish another thin model wrapper. The goal is to test whether a lightweight latent-reasoning adapter, a dedicated latent token, and a DGAC HaltGate can improve reasoning behavior while keeping the runtime small enough to evaluate, deploy, and eventually optimize for personal/local use.

## Current status

```text
base model      -> ai21labs/AI21-Jamba-Reasoning-3B
adapter target  -> WeirdRunner/Ouroboros/diloco_state/anchor
method          -> PEFT adapter + <|lat|> token + DGAC HaltGate
runtime state   -> package-based runtime extracted from notebook/root-script shape
release state   -> alpha, pre-claim; HaltGate-enabled sample failed, fixed-depth diagnostic passed
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

This result is a training-health side metric, not real generated-answer progress. Generated-answer artifacts now show a split diagnosis:

```text
HaltGate enabled sample-25   -> baseline 0.08, candidate 0.04, actual_latents 1:19 / 10:6, failed_candidate_regression
fixed-depth diagnostic       -> baseline 0.08, candidate 0.12, actual_latents 10:25, passed diagnostic-only gate
current interpretation       -> adapter/tokenizer path is likely sound; HaltGate is over-stopping and needs more training/calibration
```

The fixed-depth ablation is not a public win claim because it bypasses learned HaltGate decisions. A release-valid DGAC/HaltGate-backed claim requires the HaltGate-enabled generated-answer gate to pass.

## Why this exists

Most model experiments fail to become useful because they stop at one of two incomplete states:

1. a training log with no reproducible comparison, or
2. a demo that hides the actual research runtime behind an unrelated serving path.

Ouroboros is being structured so the research path and the release path stay connected:

```text
train anchor
-> restore exact adapter/HaltGate runtime
-> run ID-backed Coconut validation comparison
-> diagnose HaltGate vs fixed-depth behavior before promotion
-> train/calibrate HaltGate until sampled release gate passes
-> then run full in-domain validation and unbiased external benchmark evals
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

The `coordinate` GitHub Actions workflow can now dispatch the generated-answer eval to a Kaggle worker notebook:

```text
eval_mode=sample-25   -> CPU token-budget preflight, then compare-coconut-val on the first 25 scorable rows with max_seq_len=512 unless overridden
eval_mode=longest-25  -> CPU token-budget preflight, then compare-coconut-val on the 25 longest prompt-budget rows with max_seq_len=8192 unless overridden
inspect artifacts     -> runs/eval/coconut_val_compare_<mode>
fixed-depth ablation  -> workflow input eval_disable_candidate_halt_gate=true or set OUROBOROS_EVAL_DISABLE_CANDIDATE_HALT_GATE=1
eval_mode=full       -> CPU token-budget preflight over the full validation split, then compare-coconut-val with max_seq_len=8192 unless overridden
multi-GPU eval       -> eval_model_device_map=balanced_low_0 by default; use single to reproduce old cuda:0 pinning
truncation policy    -> token_budget.preflight.json records tokenizer-only counts before model loading; truncation exits non-zero unless --allow_truncated_eval is set
memory guardrail     -> eval cleanup runs every 25 samples by default; this does not replace bounded context
training note        -> training remains DDP/torchrun; eval model sharding is not a drop-in replacement for gradient-synchronized training
```

## Known release blockers

These are intentional blockers before public claims or a world-facing deployment:

```text
1. keep HaltGate-enabled sample-25 failure as the current release blocker
2. treat fixed-depth sample-25 pass as diagnostic evidence only, not a public metric claim
3. train/calibrate HaltGate until it no longer over-stops at one latent on sampled generated-answer eval
4. rerun sampled `compare-coconut-val` with HaltGate enabled and inspect `run_config.json`, `summary.json`, and `results.jsonl`
5. run the full Coconut validation split only after HaltGate-enabled sampled artifacts pass inspection
6. copy/generate public metric tables only from release-valid artifacts
7. deploy a faithful demo that uses the actual Ouroboros latent/HaltGate runtime
8. add optional lm-eval bridge later, after latent-aware loglikelihood is implemented
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

Generated-answer comparison writes artifacts before enforcing the release gate. By default it exits non-zero when candidate exact match is below baseline exact match. Use `--allow_candidate_regression` only for diagnostics, not promotion. Use `--disable_candidate_halt_gate` to run a fixed-depth latent ablation while still comparing the same rows and scoring path. When paired with `--candidate_requires_halt_gate`, the eval still verifies that `halt_gate.pt` exists; it just bypasses HaltGate decisions for the ablation.

Current sampled evidence says the learned HaltGate is the release blocker: with HaltGate enabled, 19/25 rows stopped at one latent and the candidate regressed to 0.04 exact match; with fixed depth, 25/25 rows used 10 latents and the candidate reached 0.12 exact match. Therefore, fixed-depth results may guide debugging/training, but only HaltGate-enabled generated-answer results are release-valid for DGAC/HaltGate claims.

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

Until the HaltGate-enabled generated-answer comparison passes, this project does **not** claim:

```text
Ouroboros beats Jamba in the declared DGAC/HaltGate runtime
Ouroboros is production-ready
Ouroboros is safety-aligned beyond the base model
Ouroboros is fully edge-compatible
quantized/merged exports preserve the latent/HaltGate behavior
```

## License

See `LICENSE`.
