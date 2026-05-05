# Plan: DGAC Readiness CPU-Smoke Gate

> Source PRD: `prds/dgac-readiness-cpu-smoke.md`

## Architectural decisions

- **Coordinator mode**: `workflow_validate=cpu-smoke` runs before normal DiLoCo state handling and is read-only with respect to training state.
- **Default worker**: CPU smoke defaults to Worker A when `force_worker_ids` is empty.
- **Remote artifact namespace**: `diloco_state/workflow_validation/<run_id>/worker_<id>_{status,report}.json` in the Hub state repo.
- **Notebook safety**: CPU smoke exits before `torchrun` and never requests Kaggle GPU metadata or CLI accelerator flags.
- **Validation source of truth**: GitHub Actions passes only when the coordinator observes matching remote Hub validation status from Kaggle.

---

## Phase 1: Remote validation artifact contract

**User stories**: 3, 5

### What to build

Extend CPU-smoke validation so the Kaggle notebook can publish a status/report pair to the Hub validation namespace while local smoke tests remain network-free by default.

### Acceptance criteria

- [x] Local CPU-smoke report includes run ID, state repo, remote status path, remote report path, publish-requested flag, and published flag.
- [x] Publish is explicit via `OUROBOROS_WORKFLOW_VALIDATION_PUBLISH=1`.
- [x] Tests verify remote path construction and fake publishing without network access.

---

## Phase 2: Coordinator validation mode

**User stories**: 1, 2, 3, 4

### What to build

Add a coordinator path that dispatches CPU-smoke notebooks, defaults to Worker A, polls Hub validation artifacts, and bypasses live round-state mutation.

### Acceptance criteria

- [x] `workflow_validate=cpu-smoke` returns before normal round-state read/write logic.
- [x] Empty `force_worker_ids` dispatches only Worker A.
- [x] Explicit `force_worker_ids` are honored.
- [x] Validation fails if Kaggle dispatch fails or if matching remote status does not appear before timeout.
- [x] Tests prove validation mode does not read or write `round_state.json`.

---

## Phase 3: Workflow-dispatch operator surface

**User stories**: 1, 2

### What to build

Expose CPU-smoke validation in GitHub Actions and pass stable run metadata into the coordinator and staged Kaggle notebook.

### Acceptance criteria

- [x] `workflow_dispatch` exposes `workflow_validate` with `cpu-smoke` guidance.
- [x] GitHub run ID/attempt is passed as `OUROBOROS_WORKFLOW_VALIDATION_RUN_ID`.
- [x] Coordinator CLI receives `--workflow_validate` and `--workflow_validation_run_id`.
- [x] Source-of-truth tests cover the workflow YAML contract.

---

## Phase 4: Documentation and DGAC readiness positioning

**User stories**: 1, 4

### What to build

Update durable docs so CPU smoke is treated as a hard DGAC readiness gate, and reconcile status docs with observed Stage 10 state.

### Acceptance criteria

- [x] CPU/API workflow validation docs describe remote Hub artifact verification.
- [x] STATUS records Stage 10 round 1 waiting/quota state and CPU-smoke gate.
- [x] terminal log reflects the latest observed coordinator run instead of stale Stage 7/8 notes.
