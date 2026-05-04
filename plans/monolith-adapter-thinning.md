# Plan: Monolith Adapter Thinning and Kaggle Entrypoint Consolidation

## Decision

The zero-drift training extraction has graduated from source-preserving duplication to adapter thinning. The training root file is a compatibility adapter: it preserves launch compatibility, bootstrap ordering, and CLI behavior while delegating all reusable training behavior to `ouroboros/*`.

The coordinator root file remains monolithic until a separate coordinator extraction decision is made.

## Contract

- `jamba_coconut_finetune.py` owns only startup process concerns:
  - critical env vars before any torch/NCCL import;
  - bootstrap-free `--help`;
  - dependency/environment bootstrap;
  - delegation to `ouroboros.cli.parse_args` and `ouroboros.train.run_cli`.
- `ouroboros.cli` is stdlib-only and safe before bootstrap.
- `ouroboros.train` owns the training CLI orchestration after bootstrap.
- Kaggle notebook becomes the final thin adapter after the root adapter is stable.
- No optimization may change training behavior until adapter contract tests are green.

## Local gate

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python jamba_coconut_finetune.py --help
```

## Deferred optimization track

After adapter contract tests are green, Kaggle consolidation may remove duplicated notebook cells and invoke the package/root adapter directly. CPU workflow validation through the Kaggle API, benchmarking, quantization, and edge-device inference remain later tracks.
