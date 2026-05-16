# Project Ouroboros — Minimal Runtime Blueprint

Ouroboros is now described through seven public package roots. Load this file first, then inspect the owning package for implementation details.

## Public runtime map

| Package | Owns | Normal operator surface |
|---|---|---|
| Bootstrap | runtime readiness, dependency bootstrap, device/dtype checks, CUDA/MPS/CPU guardrails, hard-lesson failure triage | `python -m ouroboros.coconut --help` stays bootstrap-safe; runtime setup is called once before training |
| Coconut | stage curriculum, latent execution, DGAC/HaltGate, training loop, checkpoints, resume, Coconut dataset shaping | `python -m ouroboros.coconut ...` |
| Models | HF CausalLM-compatible model/tokenizer loading, adapters, LoRA/PEFT, quantization, memory policy, model-family quirks | imported through `ouroboros.models` |
| Inference | prompt resolution, latent-pass inference, decode, generation result contract | `python -m ouroboros.inference ...` |
| Eval | anchor eval/generation checks, lm-eval, benchmark sharding, diagnostics, smoke quality gates | `python -m ouroboros.eval ...` |
| Coordinator | DiLoCo/solo/DDP orchestration, workers, dispatch, aggregation, promotion, repair, Kaggle launch decisions | `python -m ouroboros.coordinator ...` |
| Utils | provider-neutral env, Hub IO, W&B, Kaggle/Azure/Mac utility glue, command formatting helpers | imported through `ouroboros.utils` or explicit utility submodules |

## Package ownership rules

1. Runtime readiness, device, dtype, CUDA/MPS/CPU, guardrail → Bootstrap.
2. Stage, latent execution, DGAC, HaltGate, train loop, checkpoint → Coconut.
3. HF model/tokenizer/adapter/quantization/memory behavior → Models.
4. Prompt-to-generation path → Inference.
5. Anchor eval/gen, diagnostics, smoke, lm-eval, benchmark → Eval.
6. DiLoCo, solo, worker, DDP, launch, aggregation, promotion, repair → Coordinator.
7. Provider/env/Hub/W&B/Kaggle/Azure/Mac helper with no orchestration decision → Utils.

## Current command surfaces

```bash
python -m ouroboros.coconut --help
python -m ouroboros.coconut --use_halt_gate --resume_from_diloco_anchor --eval_only
python -m ouroboros.coordinator --help
python -m ouroboros.eval --help
python -m ouroboros.inference --help
```

Root workflow scripts were removed. Public imports should use package roots unless a submodule is explicitly part of a package's documented internal implementation.

## Validation gates

Run dependency-light checks locally first:

```bash
python -m compileall -q ouroboros tests
python -m pytest tests/test_minimal_runtime_public_architecture.py -q
```

When optional ML dependencies are available, run the full package contract suite. Tests should protect public behavior and hard lessons, not historical file locations.

## Current operational status

The canonical project state remains the promoted DGAC anchor under `WeirdRunner/Ouroboros/diloco_state/anchor`. Coordinator still owns dispatch/aggregation/promotion. Eval owns quality gates before further promotion or benchmark claims.

## Hard-lesson policy

Hard lessons must survive as executable guardrails, smoke checks, known-error classifiers, or behavior tests. They should not exist only as passive historical prose.
