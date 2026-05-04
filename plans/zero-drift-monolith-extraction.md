# Plan: Zero-Drift Monolith Extraction

## Goal

Extract reusable behavior from the training monolith into `ouroboros/*` modules without changing runtime behavior.

The original training entrypoint remained authoritative during extraction so tests could compare the extracted implementation against the monolith behavior before adapter thinning.

## Phase 1: Validation/OOM regression tracer bullet

- Preserve evaluation and generation behavior.
- Preserve `torch.no_grad` / inference-mode behavior.
- Preserve train/eval mode restoration.
- Keep CPU fake tests as the fast confidence gate.
- Use Kaggle GPU runs only after local tests are green.

Kaggle GPU runs are final confidence validation only, not the primary development loop.

## Phase 2: CLI and bootstrap contract

- Keep `--help` bootstrap-free.
- Keep critical environment variables set before torch/NCCL imports.
- Move parser ownership into a stdlib-only package seam.
- Keep the root script runnable during migration.

## Phase 3: Adapter thinning

- Convert `jamba_coconut_finetune.py` into a compatibility adapter.
- Move reusable training behavior into `ouroboros.train`.
- Keep source-of-truth fixture coverage for high-risk extracted functions.
- Defer coordinator monolith thinning to a separate phase.

## Deferred

- `diloco_coordinator.py` extraction.
- Kaggle API CPU workflow validation.
- Benchmarking.
- Quantization.
- Edge-device inference work.
