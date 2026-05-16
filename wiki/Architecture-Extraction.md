# Architecture Extraction Record
> Load this page when you need the completed refactor history, retired PRD/plan decisions, or the current adapter/module ownership map.

---

## Current Architecture Checkpoint

The repository has completed both planned zero-drift extraction tracks:

| Area | Public entrypoint | Package owner | Status |
|---|---|---|---|
| Training worker | `jamba_coconut_finetune.py` | `ouroboros.train` compatibility façade over `ouroboros.training.*`, plus `ouroboros.cli`, `bootstrap`, `data`, `model`, `dgac`, `hub`, `kaggle` | Thin compatibility adapter + deep training package |
| DiLoCo coordinator | `diloco_coordinator.py` | `ouroboros.diloco.coordinator`, `aggregation`, `state`, `dispatch`, `shared` | Thin compatibility adapter |
| Kaggle worker notebook | `kaggle-utils.ipynb` | `ouroboros.kaggle_launch_matrix` + repo package + notebook shell-magic launch seam | Thin runtime adapter, launch magic preserved |

Root entrypoints remain supported so existing local, GitHub Actions, and Kaggle launch paths do not drift while the code behind them becomes testable.

---

## Completed Track: Training Monolith Extraction

### Goal

Extract reusable training behavior from the original worker monolith into package modules without changing runtime behavior, then thin the root script into a compatibility adapter.

### Preserved contracts

- `python jamba_coconut_finetune.py --help` stays bootstrap-free.
- Critical NCCL/CUDA/process environment variables are set before heavyweight imports.
- Training CLI behavior delegates through the package without changing launch flags.
- Kaggle launch remains IPython shell magic (`!{shell_command}`), not a Python subprocess, with argv built only from `ouroboros.kaggle_launch_matrix`.
- The notebook remains a runtime adapter that resolves worker identity and launches the root training entrypoint.

### Final ownership

| Responsibility | Owner |
|---|---|
| CLI parser | `ouroboros.cli` |
| Dependency/bootstrap checks | `ouroboros.bootstrap` |
| Dataset/shard helpers | `ouroboros.data` |
| Model construction/runtime helpers | `ouroboros.model` |
| DGAC/HaltGate policy, halt supervision, and `coconut_forward` training loss contract | `ouroboros.dgac` |
| Coconut/DGAC latent execution mechanics, latent context decode, latent CE batch forward | `ouroboros.latent` |
| Hub I/O helpers | `ouroboros.hub` |
| Kaggle runtime helpers | `ouroboros.kaggle` |
| Compatibility training imports | `ouroboros.train` |
| Checkpoint save/load/prune and Hub resume sync | `ouroboros.training.checkpointing` |
| Evaluation, generation, DGAC diagnostics, eval-only flow | `ouroboros.training.evaluation` |
| Stage-loop execution | `ouroboros.training.stage_runner` |
| Post-CLI training session orchestration | `ouroboros.training.session` |
| Public worker command | `jamba_coconut_finetune.py` |


### Deep latent-forward ownership update

RFC 005 moved generic Coconut/DGAC latent execution into `ouroboros.latent` without changing the public training call shape. `ouroboros.dgac` now owns DGAC/HaltGate policy, supervised halt targets, ponder/diversity loss composition, and `coconut_forward(...)`; it delegates raw latent mechanics to `ouroboros.latent`. `ouroboros.training.evaluation` uses the latent seam for latent passes and decode, so it no longer imports model-private backbone/embed/lm-head/autocast helpers or `_run_latent_passes`.

Current dependency shape:

```text
training.stage_runner
  -> dgac.coconut_forward
      -> latent.forward_latent_batch

training.evaluation
  -> latent.prepare_latent_runtime
  -> latent.run_latent_passes
  -> latent.decode_from_latent_context

dgac
  -> HaltGate + DGAC loss/target policy

latent
  -> Coconut latent execution mechanics
```

### Deep training-session update

RFC 002 moved the package-depth training implementation behind `ouroboros.training.*` without changing the public root adapter or `ouroboros.train` imports. `ouroboros.train` is now a compatibility façade; new code should use `ouroboros.training.checkpointing`, `evaluation`, `stage_runner`, and `session` directly. The DiLoCo worker imports the new stage/eval seams directly and no longer imports `ouroboros.train`. See [Training-Session-Architecture](Training-Session-Architecture.md).

### Deferred work from this track

- Kaggle CPU/API workflow validation.
- Benchmarking.
- Quantization.
- CPU/edge portability.
- Removal of legacy wrapper duplication that is not on a tested seam.

---

## Completed Track: Coordinator Zero-Drift Extraction

### Goal

Extract the operational coordinator behind testable seams while preserving CLI compatibility, Hub state semantics, Kaggle dispatch behavior, aggregation math, and recovery behavior.

### Preserved contracts

- `python diloco_coordinator.py ...` remains the coordinator command.
- `diloco_state/round_state.json` field names, defaults, and semantics remain compatible.
- `triggered_at=0` remains the canonical unconfirmed-dispatch marker and triggers immediate re-dispatch.
- Kaggle dispatch stays push-only and preserves both `NvidiaTeslaT4` metadata and `--accelerator NvidiaTeslaT4` CLI flag.
- Kaggle push success is strict: a zero exit code is not enough; output must contain `successfully pushed` and must not contain quota/error markers.
- Runtime `signals/*.json` files are GitHub Actions doorbells only. The signal mechanism remains active, but generated signal JSONs are ignored and should not be durable source files.
- Worker runtime env injection preserves aliases for worker ID, HF token, W&B token, GitHub token, repo URL/ref/commit, state repo, signal repo, outer LR, and output directory.
- Solo mode promotes the single worker weights directly.
- Multi-worker mode preserves weighted pseudo-gradient aggregation.

### Final ownership

| Responsibility | Owner |
|---|---|
| CPU weight loading, weighted deltas, solo promotion, anchor upload | `ouroboros.diloco.aggregation` |
| Projected shard math, worker ordering, active/attendance partitioning, post-dispatch reconciliation | `ouroboros.diloco.state` |
| Runtime env injection, dispatch-cell generation, notebook staging, Kaggle metadata, `kernels push` execution | `ouroboros.diloco.dispatch` |
| End-to-end coordinator orchestration, Hub state I/O, W&B logging, CLI parser | `ouroboros.diloco.coordinator` |
| Shared worker-side DiLoCo primitives | `ouroboros.diloco.shared`, `ouroboros.diloco.worker` |
| Public coordinator command | `diloco_coordinator.py` |

### Characterization coverage

| Test file | Contract protected |
|---|---|
| `tests/test_diloco_coordinator_aggregation.py` | weighted delta math, sample weighting, missing keys, zero-sample failure, solo promotion |
| `tests/test_diloco_coordinator_state.py` | shard projection, worker ordering, ready/attendance partitioning, post-dispatch reconciliation |
| `tests/test_diloco_coordinator_dispatch.py` | metadata, accelerator flag, runtime env payload, dispatch-cell replacement/insertion, strict Kaggle push classification, fake Kaggle push results |
| `tests/test_diloco_coordinator_orchestration.py` | dry-run, unconfirmed dispatch, failed-dispatch reconciliation, orchestration seams |
| `tests/test_diloco_coordinator_adapter.py` | root coordinator remains a thin runnable adapter |
| `tests/test_source_of_truth_contract.py` | training/coordinator adapter guardrails and documentation source-of-truth guardrails |
| `tests/test_training_deep_module_contracts.py` | `ouroboros.training.*` ownership, `ouroboros.latent` ownership, `ouroboros.train` façade compatibility, worker/train dependency direction |

---

## Completed Track: Deep-Module Runtime Reliability

### Goal

Move DiLoCo/DGAC runtime decisions behind small pure interfaces so active Kaggle runs spend GPU quota on intended work rather than duplicate validation, stale waiting, or stringly-coupled launch paths.

### Final ownership

| Responsibility | Owner |
|---|---|
| Coordinator start-of-run decision planning and additive force repair | `ouroboros.coordinator_decision` |
| Kaggle launch mode policy, CPU/GPU intent, mutation/training/validation flags | `ouroboros.kaggle_contract` |
| Kaggle launch-mode matrix: command argv, notebook shell templates, output env keys, workflow labels, worker-id requirement | `ouroboros.kaggle_launch_matrix` |
| CLI-derived training branch planning | `ouroboros.training_plan` |
| Worker active/attendance/empty-shard/DGAC classification | `ouroboros.worker_lifecycle` |
| Stdlib-safe env aliases for worker IDs, tokens, credentials, bools, ints | `ouroboros.runtime_env` |

### Validation

- New contracts: `tests/test_runtime_env_aliases.py`, `tests/test_deep_module_contracts.py`.
- Existing fake/integration coverage still protects coordinator orchestration, worker fake runs, Kaggle dispatch, notebook launch, bootstrap, and CPU-smoke workflow validation.
- Full local CPU suite after the runtime reliability pass originally passed `140` tests; current post-RFC chunked local CPU coverage now passes `158` tests across all 24 test files.
- After RFC 002–005 deepening: chunked validation covers every test file and totals `158 passed`; a single `pytest -q` run can still time out locally before final summary.
- RFC 003 deepened Kaggle launch behavior behind `ouroboros.kaggle_launch_matrix`; notebook `!torchrun` commands are now tested against matrix shell templates while dispatch reads GPU intent through the matrix.
- RFC 005 deepened Coconut/DGAC latent execution behind `ouroboros.latent`; `tests/test_latent_execution_contract.py` and `tests/test_training_deep_module_contracts.py` protect the new ownership seam.

Canonical runtime details: [Deep-Module-Runtime-Reliability](Deep-Module-Runtime-Reliability.md).

---

## Retired Planning Artifacts

The following files were intentionally retired after their decisions were promoted into the wiki:

| Retired file | Canonical replacement |
|---|---|
| `plans/zero-drift-monolith-extraction.md` | This page: Completed Track: Training Monolith Extraction |
| `plans/monolith-adapter-thinning.md` | This page: Completed Track: Training Monolith Extraction + adapter ownership map |
| `plans/diloco-coordinator-zero-drift-extraction-prd.md` | This page: Completed Track: Coordinator Zero-Drift Extraction |
| `plans/diloco-coordinator-zero-drift-extraction-plan.md` | This page + `wiki/Engineering-Workflow.md` |
| `prds/dgac-readiness-cpu-smoke.md` | `wiki/Kaggle-CPU-API-Workflow-Validation.md` + `wiki/STATUS.md` |
| `plans/dgac-readiness-cpu-smoke.md` | `wiki/Kaggle-CPU-API-Workflow-Validation.md` + `wiki/STATUS.md` |

Future PRDs and plans are temporary execution artifacts. Once a phase is complete, durable decisions move into `wiki/`, and obsolete files under `prds/` and `plans/` should be deleted rather than kept as stale source-of-truth duplicates.

---

## Current Architectural Rule

Adapters are allowed; monoliths are not.

A root script or notebook may own process startup concerns such as environment setup, argument parsing handoff, shell-magic launch, and delegation. It must not regain business logic that already belongs to package modules.
