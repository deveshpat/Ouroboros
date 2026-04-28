# Ouroboros entrypoint-thinning overlay

This overlay turns the two large repo-root scripts into thin adapters while preserving the old invocation paths.

Run from the repo root:

```bash
python tools/apply_entrypoint_thinning.py
```

What it does:

- Moves `diloco_coordinator.py` to `ouroboros/diloco/coordinator_runtime.py`.
- Moves `jamba_coconut_finetune.py` to `ouroboros/coconut/finetune_runtime.py`.
- Replaces both repo-root scripts with thin adapters.
- Adds `tests/test_entrypoint_adapters.py` to lock the entrypoint line-count regression.
- Runs `py_compile` and `pytest -q` unless `--skip-tests` is passed.
- Saves backups under `.refactor_backups/entrypoint_thinning/<timestamp>/`.

The Kaggle/GitHub Actions paths stay stable:

```bash
python diloco_coordinator.py ...
python jamba_coconut_finetune.py ...
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py ...
```
