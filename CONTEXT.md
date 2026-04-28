# Ouroboros Context

## Domain language

- **Coconut Curriculum** — progressive replacement of visible reasoning steps with latent passes across stage `k`.
- **Latent Reasoner** — the module that owns execution of latent passes and model-specific hidden-state plumbing.
- **DGAC Objective** — the adaptive halt-gate objective, including task, ponder, diversity, and halt metrics.
- **DiLoCo Round Protocol** — the state machine for `stage_k`, `round_n`, active workers, attendance workers, timeouts, and stage advancement.
- **Hub State Store** — the seam that owns Hugging Face Hub path naming and JSON state shape for anchors, worker status, and round state.
- **Kaggle Dispatcher** — the seam that owns notebook staging, runtime-env injection, kernel metadata, accelerator choice, and `kaggle kernels push` outcomes.
- **Observability Identity** — the W&B naming, grouping, step-offset, and redaction policy for coordinator and workers.

## Architecture language

Use **module**, **interface**, **implementation**, **depth**, **seam**, **adapter**, **leverage**, and **locality** consistently when discussing architecture changes.
