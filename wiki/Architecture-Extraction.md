# Architecture Extraction Record

This page now records the completed extraction history only. The current source of truth is the seven-package runtime model in `BLUEPRINT.md`.

## Current public model

Ouroboros exposes seven public package roots:

| Public package | Current owner |
|---|---|
| `ouroboros.bootstrap` | Runtime setup, device/dtype readiness, fast-path checks, hard-lesson guardrails |
| `ouroboros.coconut` | Training CLI, stage loop, Coconut data, DGAC/HaltGate, latent execution, checkpoint/resume |
| `ouroboros.models` | HF CausalLM-compatible model/tokenizer loading, adapter, memory, quantization, family quirks |
| `ouroboros.inference` | Prompt-to-output generation and inference smoke |
| `ouroboros.eval` | Eval smoke, anchor checks, benchmark/lm-eval helpers, diagnostics-as-quality-gates |
| `ouroboros.coordinator` | DiLoCo/solo orchestration, dispatch, workers, aggregation, promotion, repair |
| `ouroboros.utils` | Provider-neutral env, Hub, W&B, Kaggle runtime, Azure/cost helper glue |

Root workflow scripts are retired. Operator entrypoints now run through package modules, for example `python -m ouroboros.coconut`, `python -m ouroboros.coordinator`, and `python -m ouroboros.eval`.

## Durable extraction lessons

- Keep public imports package-root oriented unless a documented internal seam is required.
- Keep wrappers deleted once package entrypoints are proven by tests; do not regrow root workflow monoliths.
- Store hard runtime lessons as executable guardrails, smoke checks, known-error classifiers, or behavior tests.
- Keep provider helpers in `ouroboros.utils`; orchestration decisions belong in `ouroboros.coordinator`.
- Keep quality gates in `ouroboros.eval`; external lm-eval is one backend, not the whole concern.
- Keep Coconut internally separated, but make the package root the normal public interface.

## Validation ownership

| Validation area | Test owner |
|---|---|
| Seven-package public surface and retired root scripts | `tests/test_minimal_runtime_public_architecture.py` |
| Coconut train/stage/DGAC/latent/checkpoint contracts | Coconut package tests |
| Coordinator state/dispatch/aggregation/worker contracts | Coordinator package tests |
| Eval smoke/benchmark/lm-eval contracts | Eval package tests |
| Bootstrap runtime and guardrail contracts | Bootstrap package tests |
| Provider/env/Hub/W&B/Kaggle helpers | Utils package tests |

Historical extraction details should not be used as a current operator guide. If a historical detail is still useful, promote it to the owning package test or to a concise hard-lesson row backed by executable behavior.
