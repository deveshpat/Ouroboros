# Implementation Notes

Applied the DiLoCo and DRY changes requested in `AGENT_PROMPT_diloco.md` and `BLUEPRINT.md`.

## What changed

### `jamba_coconut_finetune.py`
- Added DiLoCo CLI flags:
  - `--diloco_mode`
  - `--diloco_worker_id`
  - `--diloco_outer_lr`
  - `--diloco_min_workers`
  - `--diloco_state_repo`
  - `--diloco_signal_repo`
  - `--diloco_run_val`
- Added DiLoCo worker helpers:
  - `diloco_get_shard()`
  - `diloco_read_round_state()`
  - `diloco_upload_worker_state()`
  - `diloco_download_anchor()`
  - `diloco_push_signal()`
- Refactored the core training loop into `run_training_stages()` so the sequential curriculum path and the DiLoCo worker path share the same implementation.
- Added `run_diloco_worker()` to wrap the existing stage training logic in a clean opt-in path.
- Applied DRY refactors from the blueprint:
  - merged token resolution into `_resolve_hf_token_common()`
  - added `_run_latent_passes()` and reused it in training/eval/generation
  - cached `_get_backbone()`, `_get_embed_tokens()`, `_get_lm_head()` lookups on the model object
  - collapsed stage-0 forward into the latent forward path
  - added `_ddp_sum()` for repeated all-reduce summation

### `diloco_coordinator.py`
- New CPU-only coordinator for GitHub Actions.
- Aggregates worker adapter deltas on CPU and uploads the new anchor.
- Updates `diloco_state/round_state.json`.
- Triggers Kaggle worker notebooks.
- Includes the requested fallback rule: after 3 consecutive single-worker rounds, it warns and aggregates with one worker.

### `bootstrap_diloco.py`
- New bootstrap script to initialize `diloco_state/anchor/` and `diloco_state/round_state.json` from the existing Stage 2 checkpoint.
- Defaults to the blueprint’s current anchor source: `runs/stage3/checkpoint-0002987`.

### `.github/workflows/diloco_coordinator.yml`
- New workflow to run the coordinator when workers push signal files.
- Added workflow concurrency protection so overlapping signal pushes do not race the same round update.

### `signals/.gitkeep`
- Added placeholder file so the trigger directory exists in git.

## Intentional implementation details
- In DiLoCo mode, regular stage checkpoint Hub sync is disabled; worker uploads go to `diloco_state/...` only.
- Automatic pre-round validation is deterministic: Worker **A** runs it when `round_n == 0` (or when `--diloco_run_val` is passed explicitly). This avoids all three workers redundantly running validation on the same new stage.
- `--diloco_mode` is guarded against `--use_halt_gate`. The prompt’s DiLoCo design syncs LoRA adapters only, while DGAC adds extra trainable state that is not part of the worker aggregation contract.

## Validation performed
- `python -m py_compile jamba_coconut_finetune.py diloco_coordinator.py bootstrap_diloco.py`
- All three Python files compile successfully.
