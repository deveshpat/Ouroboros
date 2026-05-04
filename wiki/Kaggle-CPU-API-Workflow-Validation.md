# Kaggle CPU/API Workflow Validation
> Durable record for the completed PRD → plan → tracer-bullet implementation loop.

---

## Problem Statement

Ouroboros can now keep the training script, coordinator script, and Kaggle notebook thin, but the riskiest execution path is still the live Kaggle worker workflow. Before this track, validating notebook staging, worker identity injection, repo/ref checkout, Kaggle push behavior, and coordinator reconciliation still risked consuming scarce GPU quota or depending on live secrets.

## Solution

Add a CPU-safe validation seam that exercises Kaggle workflow plumbing without importing training dependencies, launching `torchrun`, requesting a Kaggle accelerator, pushing checkpoints, or consuming GPU quota. The real GPU path remains unchanged: the Kaggle notebook still keeps its IPython `!torchrun` shell magic, and coordinator dispatch still requests `NvidiaTeslaT4` unless explicit CPU-smoke validation mode is active.

## User Stories

1. As the project operator, I want to validate Kaggle notebook staging and runtime env injection without GPU quota, so that orchestration bugs are caught before real worker dispatch.
2. As the project operator, I want CPU-smoke validation to exit before the real training launch, so that no Jamba/Mamba/CUDA path is accidentally exercised.
3. As the project operator, I want fake coordinator/worker tests to cover dispatch failure reconciliation, so that quota/push failures do not regress into 13-hour waits.
4. As the project operator, I want manual Kaggle API validation to be opt-in, so that normal CI stays hermetic and secret-free.

## Implementation Decisions

- `ouroboros.kaggle_runtime` owns testable repo/ref/commit checkout helpers for the Kaggle runtime contract.
- `ouroboros.workflow_validation` owns the CPU-smoke validation branch and emits a coordinator-compatible local status JSON plus a report JSON.
- `ouroboros.workflow_validation_worker` is a tiny stdlib-only fake worker command used as a command-construction contract; it does not import training dependencies.
- `OUROBOROS_WORKFLOW_VALIDATE=cpu-smoke` is the validation switch.
- Coordinator dispatch copies `OUROBOROS_*` env vars into the staged notebook payload. When the payload contains `OUROBOROS_WORKFLOW_VALIDATE=cpu-smoke`, dispatch writes CPU metadata (`enable_gpu=false`) and omits `--accelerator NvidiaTeslaT4` from `kaggle kernels push`.
- The Kaggle notebook checks CPU-smoke mode after repo sync and worker ID resolution, runs the validation branch, then exits before the real `!torchrun` line.
- The real GPU path remains T4-pinned by default through both metadata and CLI accelerator flag.
- Signals remain the coordinator doorbell; this track does not replace or remove the signal mechanism.

## Implementation Plan Executed

### Phase 1 — Characterize Kaggle workflow seams

Covered repo URL/ref/commit resolution, GitHub auth env construction, fetch/checkout fallback behavior, runtime file copying, dispatch cell staging, metadata generation, runtime env payload encoding, worker identity resolution, and real launch command contracts.

### Phase 2 — Add CPU-safe validation mode

Added a CPU-smoke validation branch that writes local status/report artifacts and proves no GPU, `torchrun`, or Hub push is requested. The notebook exits before the real launch when validation mode is active.

### Phase 3 — Simulate coordinator/worker loop

Extended fake coordinator orchestration coverage so a fake completed worker status can advance planning, fake Kaggle results can include success/manual/failure, and post-dispatch reconciliation demotes failed workers immediately.

### Phase 4 — Promote durable documentation

This page is the canonical durable record. No completed PRD/plan files need to remain in `plans/` for this track.

## Manual Kaggle API Validation

Use this only on demand before risky Kaggle runtime changes.

1. Push the branch/ref you want Kaggle workers to validate.
2. Set these coordinator-side env vars before dispatch:

```bash
export OUROBOROS_WORKFLOW_VALIDATE=cpu-smoke
export OUROBOROS_REPO_URL=https://github.com/deveshpat/Ouroboros.git
export OUROBOROS_REPO_REF=<branch-or-tag>
# Optional exact pin:
export OUROBOROS_REPO_COMMIT=<commit-sha>
```

3. Run the coordinator through `workflow_dispatch` with the intended worker credentials and, if needed, `force_worker_ids=A` for a single-account smoke.
4. Expected dispatch behavior:
   - staged `kernel-metadata.json` has `enable_gpu=false`;
   - `kaggle kernels push` omits `--accelerator NvidiaTeslaT4`;
   - the generated dispatch cell injects `OUROBOROS_WORKFLOW_VALIDATE=cpu-smoke`;
   - the notebook prints `[workflow-validate] CPU smoke validation complete`;
   - the notebook exits before the real `!torchrun` line.

## Validation Commands

Targeted validation:

```bash
python -m pytest \
  tests/test_kaggle_runtime_contract.py \
  tests/test_kaggle_cpu_api_workflow_validation.py \
  tests/test_diloco_coordinator_dispatch.py \
  tests/test_diloco_coordinator_orchestration.py
```

Regression chunks used during implementation:

```bash
python -m pytest tests/test_bootstrap_cli_contract.py tests/test_checkpoint_smoke_training.py
python -m pytest tests/test_data_coconut_fake.py tests/test_dgac_anchor_cli_contract.py tests/test_diloco_coordinator_adapter.py tests/test_diloco_coordinator_aggregation.py
python -m pytest tests/test_diloco_coordinator_dispatch.py tests/test_diloco_coordinator_orchestration.py tests/test_diloco_coordinator_state.py tests/test_diloco_worker_fake.py
python -m pytest tests/test_kaggle_cpu_api_workflow_validation.py tests/test_kaggle_launch_contract.py tests/test_kaggle_notebook_contract.py tests/test_kaggle_runtime_contract.py tests/test_source_of_truth_contract.py tests/test_training_entrypoint_adapter.py tests/test_validation_no_drift.py
```

## Out of Scope

- model quality benchmarking;
- quantization;
- CPU/Mamba portability experiments;
- edge-device inference optimization;
- deleting the signal mechanism;
- replacing the real notebook `!torchrun` launch with Python `subprocess.run`.

## Notes

A latent dependency-fallback issue was fixed while running the regression subset: `diloco_download_anchor()` now keeps optional Hub/PEFT/safetensors imports inside the retry/download path so a missing optional dependency is swallowed the same way as a missing anchor. This preserves the fake-worker fallback contract in dependency-light CPU test environments.
