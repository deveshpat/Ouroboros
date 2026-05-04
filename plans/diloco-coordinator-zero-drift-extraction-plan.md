# Plan: DiLoCo Coordinator Zero-Drift Extraction and Adapter Thinning

> Source PRD: `plans/diloco-coordinator-zero-drift-extraction-prd.md`

## Architectural decisions

Durable decisions that apply across all phases:

- **Public entrypoint**: `python diloco_coordinator.py ...` remains the supported coordinator command until the final adapter-thinning slice. The root file may delegate earlier, but it must remain runnable throughout.
- **Commit policy**: do not commit the PRD or this plan alone. The first worthwhile commit should bundle both planning files with the first behavior-preserving aggregation extraction slice.
- **Module layout**: coordinator behavior moves under `ouroboros/diloco/` behind deep modules:
  - `aggregation.py` for CPU tensor loading, solo promotion, weighted delta aggregation, and anchor upload;
  - `state.py` for pure round-state normalization, worker ordering, shard projection, mode selection, waiting/timeout planning, and stage advancement decisions;
  - `dispatch.py` for runtime env injection, Kaggle metadata, notebook staging, push execution, and post-dispatch reconciliation;
  - `coordinator.py` for orchestration after the pure seams are proven.
- **State schema**: `diloco_state/round_state.json` remains field-compatible. Unknown fields must round-trip. Existing meanings of `mode`, `triggered_workers`, `attendance_workers`, `triggered_at`, `total_samples_seen`, `completed_stages`, and `seed` are preserved.
- **Worker identity**: valid worker IDs remain `A`, `B`, and `C`. Ordering remains canonical and stable across trigger, attendance, reconciliation, and ready-worker collection.
- **Kaggle dispatch contract**: dispatch remains push-only, keeps `NvidiaTeslaT4` accelerator metadata and CLI flag, stages the checked-in `kaggle-utils.ipynb`, and injects runtime env through a generated dispatch cell without changing notebook launch semantics.
- **Remote services**: Hugging Face Hub, Kaggle CLI, and W&B must sit behind fakeable seams before orchestration moves. No local test should require network, GPU, Hugging Face credentials, Kaggle credentials, or W&B credentials.
- **Behavior rule**: characterize first, extract second. If current behavior is awkward but operationally relied upon, preserve it in this phase and defer cleanup to the later optimization phase.

---

## Execution protocol

Implementation follows the continuous tracer-bullet loop agreed after the initial plan:

```text
for each phase:
    while the phase is not complete:
        choose one behavior-preserving tracer bullet
        characterize the current public behavior
        extract or rewire only the minimum needed for that tracer bullet
        run the targeted tests for that seam
        prune only local redundancy discovered by the slice
    move to the next phase
exit only after every phase is complete
```

This means the work must not devolve into a bulk blind refactor, but it also must not stop after each tiny red/green step. Commits remain significant checkpoints, not per-tracer-bullet snapshots.

---

## Phase 1: Aggregation characterization and extraction

**User stories**: 1, 2, 6

### What to build

Create the first vertical slice around CPU aggregation. Add characterization tests that compare the current root coordinator aggregation behavior against the extracted package seam, then move the aggregation path into `ouroboros/diloco/aggregation.py` while keeping `diloco_coordinator.py` runnable.

This slice should include the PRD and plan files in the same eventual commit because aggregation extraction is the first significant repo improvement.

### Acceptance criteria

- [ ] Add `tests/test_diloco_coordinator_aggregation.py` with CPU tensor fixtures covering weighted delta aggregation, sample weighting, dtype preservation, missing worker keys, and zero/invalid sample behavior.
- [ ] Add solo-promotion coverage proving a single contributing worker is promoted directly rather than averaged.
- [ ] Extract aggregation helpers into `ouroboros/diloco/aggregation.py` without importing heavyweight training dependencies at module import time.
- [ ] Keep `diloco_coordinator.py` behavior-compatible by importing/delegating to the extracted aggregation helpers.
- [ ] Keep the root coordinator executable and CLI-compatible.
- [ ] Run the targeted aggregation tests plus the existing coordinator-adjacent/training contract tests.

---

## Phase 2: Round planning and state-transition seam

**User stories**: 1, 3, 6

### What to build

Extract pure state-planning logic into `ouroboros/diloco/state.py`. The root coordinator should still own orchestration, but calculations for projected shards, mode selection, worker deduplication, ready/attendance partitioning, timeout demotion, waiting-mode decisions, and stage advancement should become testable pure functions.

### Acceptance criteria

- [ ] Add `tests/test_diloco_coordinator_state.py` with fixtures for existing `round_state.json` dictionaries, including unknown fields.
- [ ] Prove worker ordering and deduplication match the monolith behavior for triggered and attendance workers.
- [ ] Prove `triggered_at=0` remains an unconfirmed dispatch marker and selects immediate re-dispatch behavior.
- [ ] Prove `triggered_at>0` inside the timeout window waits without demotion.
- [ ] Prove timed-out missing active workers are demoted into attendance.
- [ ] Prove attendance workers that respond are promoted back into eligibility for the next round.
- [ ] Prove all-workers-absent behavior enters waiting mode without incorrectly advancing the round.
- [ ] Prove stage completion advances `stage_k`, resets `round_n`, and preserves/updates `completed_stages` compatibly.
- [ ] Keep all state helpers stdlib-only at import time.

---

## Phase 3: Kaggle dispatch staging seam

**User stories**: 1, 4, 6

### What to build

Extract runtime environment construction, payload encoding, notebook dispatch-cell generation, Kaggle kernel metadata generation, local notebook staging, and single-worker/multi-worker trigger behavior into `ouroboros/diloco/dispatch.py`. All tests should use temporary notebooks and fake subprocess results.

### Acceptance criteria

- [ ] Add `tests/test_diloco_coordinator_dispatch.py` with notebook-staging fixtures that do not call Kaggle.
- [ ] Prove generated `kernel-metadata.json` preserves the expected slug, notebook filename, kernel type, internet setting, and `NvidiaTeslaT4` accelerator value.
- [ ] Prove Kaggle push arguments preserve `--accelerator NvidiaTeslaT4`.
- [ ] Prove generated dispatch cells replace an existing `diloco-dispatch` cell rather than duplicating it.
- [ ] Prove generated dispatch cells insert correctly when the notebook starts with markdown or code.
- [ ] Prove runtime env payload includes worker identity, state repo, signal repo, outer LR, repo URL/ref/commit, and token aliases without asserting raw secret values.
- [ ] Prove missing credentials return `manual`, failed pushes return `failed`, and successful fake pushes return `success`.
- [ ] Keep dispatch import safe when Kaggle CLI is absent.

---

## Phase 4: Dispatch reconciliation and recovery behavior

**User stories**: 1, 3, 4, 5

### What to build

Move post-dispatch reconciliation and recovery decisions behind package seams after dispatch staging is covered. This slice locks behavior around failed active launches, failed attendance launches, manual dispatch, `triggered_at` correction, attendance preservation, and waiting-mode re-dispatch.

### Acceptance criteria

- [ ] Add tests proving `_reconcile_post_dispatch_state` behavior is preserved when all workers fail, some active workers fail, only attendance workers fail, and no workers fail.
- [ ] Prove failed active workers move to attendance while successful/manual active workers remain triggered.
- [ ] Prove `triggered_at` is reset to `0.0` when no worker was actually dispatched.
- [ ] Prove `triggered_at` is refreshed when at least one worker was successfully or manually dispatched.
- [ ] Prove dispatch failures are recorded in `dispatch_failures` without dropping unrelated state fields.
- [ ] Prove waiting-mode attendance dispatch and reconciliation preserve the current root coordinator behavior using fake Hub upload/download calls.

---

## Phase 5: Coordinator orchestration seam

**User stories**: 1, 3, 4, 5, 6

### What to build

Move the main orchestration flow into `ouroboros/diloco/coordinator.py` behind fakeable service boundaries. The root script should still parse the same CLI and can still expose compatibility wrappers, but the real workflow should be callable from package code for local fake/API validation later.

### Acceptance criteria

- [ ] Introduce a package-level coordinator runner that can execute with fake Hub, fake dispatch, fake clock, and fake W&B services.
- [ ] Keep Hub JSON/text download and upload semantics behavior-compatible, including retry behavior and swallowed missing-state handling where currently used.
- [ ] Add integration seam tests where fake worker statuses drive one complete coordinator pass without network calls.
- [ ] Prove no-ready-workers inside the timeout window exits without aggregation or state mutation.
- [ ] Prove timed-out partial readiness aggregates available workers and demotes missing workers exactly as before.
- [ ] Prove no state found exits cleanly with the current message/behavior shape.
- [ ] Preserve W&B metric names and step behavior when W&B is enabled, while allowing tests to run with a fake/no-op logger.

---

## Phase 6: Root adapter thinning and guardrails

**User stories**: 1, 5, 6

### What to build

Thin `diloco_coordinator.py` into a compatibility adapter only after the package seams are covered. The adapter should retain the public executable path and delegate to package code. Add guardrail tests so the root coordinator does not silently regrow into a monolith.

### Acceptance criteria

- [ ] Move CLI construction/delegation into package code or keep only minimal adapter-owned startup concerns in the root file.
- [ ] Keep `python diloco_coordinator.py --help` compatible with existing options.
- [ ] Keep GitHub Actions coordinator invocation compatible.
- [ ] Add an adapter contract test that fails if the root file regrows substantial business logic.
- [ ] Run the full local test gate.
- [ ] Leave training adapter, Kaggle notebook launch magic, worker shard math, and signal artifact tracking untouched unless tests prove no drift.

---

## Verification checklist

- [ ] Every phase is independently verifiable.
- [ ] No phase is merely a horizontal module move; every slice preserves a runnable coordinator path.
- [ ] Durable decisions are separated from volatile implementation details.
- [ ] Remote services are fakeable before orchestration moves.
- [ ] Root coordinator remains runnable until the final adapter-thinning slice.
- [ ] The first commit checkpoint is significant: PRD + plan + aggregation tests + aggregation extraction.
