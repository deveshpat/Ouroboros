# Deep-module architecture refactor

This overlay keeps the existing CLI entrypoints intact while introducing deep modules behind them.

## Added package structure

```text
ouroboros/
  diloco/
    protocol.py          # round planning, attendance, timeout, dispatch reconciliation
    hub_state.py         # Hub path/schema seam and in-memory/HF adapters
    aggregation.py       # DiLoCo weighted outer update
    kaggle_dispatch.py   # notebook staging, dispatch-cell injection, T4 metadata, CLI dispatch
    observability.py     # W&B run identity, grouping, step offsets, redaction
  coconut/
    bootstrap.py         # runtime inspection seam; heavy bootstrap remains in entrypoint for now
    curriculum.py        # stage sample construction and deterministic shard math
    latent.py            # latent reasoning seam
    dgac.py              # halt-gate objective seam
    training_runtime.py  # orchestration seam
```

## Behavior-preserving choices

- `diloco_coordinator.py` remains the GitHub Actions entrypoint.
- `jamba_coconut_finetune.py` remains the Kaggle training entrypoint.
- The coordinator delegates pure helper behavior to the new modules where the function names are present in the live checkout.
- Heavy training logic is not blindly rewritten. The new Coconut modules create seams and test surfaces first.
- `kaggle-utils.ipynb` now syncs the whole repo into `/kaggle/working/Ouroboros` and changes cwd there, so package imports work without copying individual files.

## Verification included

Pure tests cover:

- deterministic three-way shard projection
- `complete`, `solo`, and `diloco` mode selection
- attendance workers not blocking active aggregation
- `triggered_at == 0` retry semantics
- timeout demotion to attendance
- final-stage remainder advancement
- dispatch failure reconciliation
- Kaggle T4 metadata and dispatch-cell injection
- W&B per-round identity and per-stage grouping
- Hub path canonicalization
- Coconut stage sample splitting

Run:

```bash
pytest -q
python -m py_compile diloco_coordinator.py jamba_coconut_finetune.py
```

## Apply in Codespaces

```bash
cd /workspaces/Ouroboros
unzip -o /workspaces/Ouroboros/ouroboros_refactor_payload.zip -d /tmp/ouroboros-refactor
python /tmp/ouroboros-refactor/apply_refactor.py .
python -m pip install -q pytest
pytest -q
python -m py_compile diloco_coordinator.py jamba_coconut_finetune.py

git status
git diff --stat
```
