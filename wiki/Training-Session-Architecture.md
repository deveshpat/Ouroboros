# Training Session Architecture

Durable record for RFC 002: deep training session and stage-execution ownership.

## Goal

Move checkpointing, evaluation/generation, stage execution, and post-CLI session orchestration out of `ouroboros.train` while preserving zero-drift behavior, CLI compatibility, checkpoint layout, Hub paths, DiLoCo schemas, and Kaggle launch behavior.

## Current ownership

| Concern | Owner | Compatibility path |
|---|---|---|
| Public training command | `jamba_coconut_finetune.py` | unchanged root adapter |
| CLI parsing/help | `ouroboros.cli` | root adapter imports parser after bootstrap-free help guard |
| Compatibility imports | `ouroboros.train` | re-exports moved public training symbols |
| Post-CLI session orchestration | `ouroboros.training.session` | `ouroboros.train.run_cli` delegates to `run_training_session` |
| Stage-loop execution | `ouroboros.training.stage_runner` | `ouroboros.train.run_training_stages` re-export |
| Checkpoint save/load/prune and Hub resume sync | `ouroboros.training.checkpointing` | `ouroboros.train.save_checkpoint`, `load_checkpoint`, `prune_epoch_checkpoints` re-exports |
| Evaluation, generation callback, DGAC diagnostics, eval-only flow | `ouroboros.training.evaluation` | `ouroboros.train.evaluate_stage`, `run_generation_callback`, `run_eval_only` re-exports |
| DiLoCo worker execution | `ouroboros.diloco.worker` | imports `stage_runner` / `evaluation` directly, not `ouroboros.train` |

## RFC 005 latent-forward seam

`ouroboros.latent` now owns generic Coconut latent execution: runtime handle resolution, question-context construction, fixed/gated latent passes, latent-state injection into `<|lat|>` positions, CE batch accounting, and greedy decode from latent context.

`ouroboros.dgac` remains the DGAC policy owner: `HaltGate`, lambda scheduling, halt-probe depth/target policy, supervised HaltGate loss, ponder/diversity composition, and the public `coconut_forward(...) -> (loss, metrics)` training contract.

`ouroboros.training.evaluation` should orchestrate validation, generation, and diagnostics only. It must call `ouroboros.latent` for latent execution and must not import model-private `_get_backbone`, `_get_embed_tokens`, `_get_lm_head`, `_autocast_ctx`, `_extract_last_hidden_state`, or DGAC's compatibility `_run_latent_passes`.

## Dependency direction

```text
jamba_coconut_finetune.py
  └── ouroboros.train.run_cli
        └── ouroboros.training.session.run_training_session
              ├── ouroboros.training.stage_runner
              ├── ouroboros.training.evaluation
              ├── ouroboros.training.checkpointing
              ├── ouroboros.training_plan
              ├── ouroboros.model / data / dgac / latent / hub
              └── ouroboros.diloco.worker only inside DiLoCo-specific branches

ouroboros.diloco.worker
  ├── ouroboros.training.stage_runner
  └── ouroboros.training.evaluation
```

`ouroboros.training.stage_runner` must not import `ouroboros.train`, `ouroboros.diloco.worker`, or the coordinator.

## Preserved behavior

- Existing imports from `ouroboros.train` still work.
- `jamba_coconut_finetune.py` remains the public runnable training entrypoint.
- `run_training_stages`, checkpointing, evaluation, and generation were moved without behavioral rewrites.
- AST/no-drift tests now compare moved owners against `tests/fixtures/training_monolith_source.py` instead of protecting the old file shape.
- DiLoCo worker no longer imports `ouroboros.train`; it imports stage/eval seams from the new owners.
- The session layer defers DiLoCo worker-helper imports until DiLoCo-specific branches so ordinary sequential/eval usage does not import worker internals.

## Guardrails

| Test file | Contract protected |
|---|---|
| `tests/test_training_deep_module_contracts.py` | ownership of new training modules and `ouroboros.latent`, `ouroboros.train` façade compatibility, worker not importing `ouroboros.train`, stage runner not depending on DiLoCo/worker/coordinator |
| `tests/test_checkpoint_smoke_training.py` | checkpoint/stage-runner AST parity plus fake CPU smoke training through compatibility imports |
| `tests/test_validation_no_drift.py` | eval/generation no-grad and AST parity at the new owner |
| `tests/test_latent_execution_contract.py` | fixed/gated latent passes, latent CE batch accounting, hidden-sequence exposure, and latent-context decode contract |
| `tests/test_diloco_worker_fake.py` | worker fake path patches the new stage/eval owners and still uploads compatible status |
| `tests/test_training_entrypoint_adapter.py` | root adapter bootstrap/import ordering and `run_cli` delegation |
| `tests/test_source_of_truth_contract.py` | root adapter and `ouroboros.train` façade stay thin |

## Operational notes

- New internal code should import from `ouroboros.training.*`, not `ouroboros.train`.
- Keep `ouroboros.train` as a compatibility façade unless all external callers have been migrated.
- When moving future training code, preserve zero-drift first; update ownership tests to follow the new owner instead of weakening AST checks.
- Keep latent execution mechanics in `ouroboros.latent`; keep DGAC/HaltGate policy in `ouroboros.dgac`.
- Future latent objectives should extend `ouroboros.latent` instead of reaching into private DGAC helpers.

## Validation evidence

RFC 002 local validation after implementation:

- Targeted RFC set: `42 passed in 18.09s`.
- Current chunked full-suite coverage across all 24 test files: `158 passed` total (`36 passed`, `84 passed`, `38 passed`).
- Single-command `python -m pytest -q` can still time out locally before final summary, so chunked validation remains the reported final evidence for this snapshot.
