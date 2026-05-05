# Architecture Extraction Record
> Load this page when you need the completed refactor history, retired PRD/plan decisions, or the current adapter/module ownership map.

---

## Current Architecture Checkpoint

The repository has completed both planned zero-drift extraction tracks:

| Area | Public entrypoint | Package owner | Status |
|---|---|---|---|
| Training worker | `jamba_coconut_finetune.py` | `ouroboros.train`, `ouroboros.cli`, `ouroboros.bootstrap`, `ouroboros.data`, `ouroboros.model`, `ouroboros.dgac`, `ouroboros.hub`, `ouroboros.kaggle` | Thin compatibility adapter |
| DiLoCo coordinator | `diloco_coordinator.py` | `ouroboros.diloco.coordinator`, `aggregation`, `state`, `dispatch`, `shared` | Thin compatibility adapter |
| Kaggle worker notebook | `kaggle-utils.ipynb` | Repo package + notebook shell-magic launch seam | Thin runtime adapter, launch magic preserved |

Root entrypoints remain supported so existing local, GitHub Actions, and Kaggle launch paths do not drift while the code behind them becomes testable.

---

## Completed Track: Training Monolith Extraction

### Goal

Extract reusable training behavior from the original worker monolith into package modules without changing runtime behavior, then thin the root script into a compatibility adapter.

### Preserved contracts

- `python jamba_coconut_finetune.py --help` stays bootstrap-free.
- Critical NCCL/CUDA/process environment variables are set before heavyweight imports.
- Training CLI behavior delegates through the package without changing launch flags.
- Kaggle launch remains an IPython `!torchrun` shell-magic command, not a Python subprocess.
- The notebook remains a runtime adapter that resolves worker identity and launches the root training entrypoint.

### Final ownership

| Responsibility | Owner |
|---|---|
| CLI parser | `ouroboros.cli` |
| Dependency/bootstrap checks | `ouroboros.bootstrap` |
| Dataset/shard helpers | `ouroboros.data` |
| Model construction/checkpoint helpers | `ouroboros.model` |
| DGAC/halt gate helpers | `ouroboros.dgac` |
| Hub I/O helpers | `ouroboros.hub` |
| Kaggle runtime helpers | `ouroboros.kaggle` |
| Training orchestration | `ouroboros.train` |
| Public worker command | `jamba_coconut_finetune.py` |

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
