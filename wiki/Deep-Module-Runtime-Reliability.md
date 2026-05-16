# Deep-Module Runtime Reliability

Durable record for the DiLoCo/DGAC runtime reliability pass completed on 2026-05-10.

## Goal

Reduce active-run fragility by moving decision-heavy behavior behind pure, CPU-testable seams while keeping root entrypoints, CLI defaults, Hub paths, Kaggle dispatch semantics, and worker status schemas compatible.

## Final ownership

| Runtime concern | Deep module | Adapter that still performs side effects |
|---|---|---|
| Coordinator start-of-run planning, force repair, timeout/unconfirmed-dispatch classification | `ouroboros.coordinator_decision` | `ouroboros.diloco.coordinator` writes Hub state, polls Hub, dispatches Kaggle, logs W&B |
| Kaggle mode policy, CPU/GPU intent, worker/coordinator mode flags, mutation/validation/training intent, expected env/CLI keys | `ouroboros.kaggle_contract` | `ouroboros.kaggle_launch_matrix`, `ouroboros.kaggle`, `ouroboros.diloco.dispatch`, `kaggle-utils.ipynb`, GitHub Actions workflow |
| Kaggle launch matrix: per-mode command argv, output env key, worker-id requirement, workflow label, and launch-time env defaults | `ouroboros.kaggle_launch_matrix` | `ouroboros.kaggle` compatibility builders, notebook shell-magic adapter, GitHub Actions workflow YAML, dispatch metadata |
| Training branch classification before heavy model/dataset execution | `ouroboros.training_plan` | `ouroboros.training.session.run_training_session` loads model/data, initializes distributed state, and dispatches training/eval; `ouroboros.train.run_cli` is a façade |
| Worker round classification | `ouroboros.worker_lifecycle` | `ouroboros.diloco.worker` still downloads anchors, trains, uploads worker artifacts, and pushes signals |
| Runtime env aliases and safe parsing | `ouroboros.runtime_env` | `ouroboros.bootstrap`, `ouroboros.kaggle`, `ouroboros.workflow_validation`, and `ouroboros.diloco.dispatch` consume centralized aliases |

## Preserved behavior

- `diloco_coordinator.py` and `jamba_coconut_finetune.py` remain runnable compatibility adapters.
- Force-trigger is additive repair: it does not discard already-valid worker work.
- `triggered_at=0` remains the unconfirmed-dispatch marker and still triggers immediate re-dispatch.
- Kaggle push failure/quota output still reconciles round state immediately instead of waiting for the long worker timeout.
- `ouroboros.kaggle_launch_matrix` owns launch behavior above the policy layer; `ouroboros.kaggle_contract` remains stdlib-safe policy, while notebook/workflow/dispatch remain adapters.
- Notebook launch now uses a single IPython `!{shell_command}` seam; tests assert the command is built from `ouroboros.kaggle_launch_matrix` instead of duplicated as literal notebook `torchrun` branches.
- CPU-smoke mode stays GPU-free.
- DGAC DiLoCo workers skip duplicate pre-validation because anchor eval/gen already covers that validation path.
- Attendance-only and empty-shard workers upload coordinator-compatible zero-sample statuses and push signals without local training.
- Env alias resolution remains stdlib-safe and importable before torch or network SDKs.

## Test coverage added

| Test file | New contract protected |
|---|---|
| `tests/test_runtime_env_aliases.py` | stdlib-safe env alias normalization, token alias resolution, force-worker parsing, Kaggle credential lookup, bool/int parsing |
| `tests/test_deep_module_contracts.py` | launch-mode contracts, training session plans, worker lifecycle classifications, coordinator force-repair/round-start decisions |
| `tests/test_kaggle_launch_matrix.py` | matrix coverage for every launch mode, compatibility-builder parity, GPU/CPU intent, notebook single-command seam, workflow exposure |

Existing coordinator, worker, Kaggle, bootstrap, notebook, CPU-smoke, training-session, Kaggle-runtime, and latent-execution tests still cover adapters end-to-end with fakes. Current chunked local CPU validation covers all 24 test files and passes `158` tests (`36 passed`, `84 passed`, `38 passed`). A single `pytest -q` run can still time out locally before final summary, so chunked validation remains the reported local gate.

## Operational notes

- For launch-mode policy questions, read `ouroboros.kaggle_contract` first. For launch behavior, read `ouroboros.kaggle_launch_matrix`; notebook code should call the matrix instead of mirroring command strings.
- For credentials and aliases, use `ouroboros.runtime_env`; avoid adding new hand-rolled `os.environ.get(...) or ...` chains.
- For coordinator behavior, keep new state-machine branches in `ouroboros.coordinator_decision` unless they require Hub/Kaggle/W&B side effects.
- For worker behavior, classify with `ouroboros.worker_lifecycle` before adding execution branches.
- For training CLI branching, extend `ouroboros.training_plan` before loading real model/data so branch behavior remains CPU-testable.
- For training stage/session ownership, read `ouroboros.training.*` first; `ouroboros.train` should remain a compatibility façade only.
