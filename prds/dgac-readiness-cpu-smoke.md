# PRD: DGAC Readiness CPU-Smoke Gate

## Problem Statement

DGAC launch readiness is currently blocked by workflow confidence, not only model quality. The latest coordinator/Kaggle path can be unit-tested locally, but the real GitHub Actions → Kaggle API → Kaggle notebook path was not a closed-loop end-to-end gate. A Kaggle push can succeed while notebook runtime plumbing, repo/ref checkout, worker env injection, or early CPU validation reporting still fails. Waiting until the next GPU quota window to discover those problems risks losing scarce GPU time and delaying Stage 10 → DGAC handoff.

## Solution

Add a read-only CPU-smoke workflow validation gate to the coordinator workflow. When `workflow_validate=cpu-smoke` is selected from `workflow_dispatch`, the coordinator should default to Worker A unless explicit worker IDs are provided, push a CPU-only Kaggle notebook, have the notebook exit before `torchrun`, publish a validation status/report to Hugging Face under a workflow-validation namespace, and have the GitHub coordinator job verify that remote artifact before passing.

## User Stories

1. As the project operator, I want a one-click CPU smoke workflow, so that I can validate the live GitHub Actions → Kaggle → Hub path without consuming GPU quota.
2. As the project operator, I want CPU smoke to default to Worker A, so that validation is deterministic and does not accidentally dispatch all accounts.
3. As the project operator, I want CPU smoke to publish a remote validation artifact, so that the coordinator can verify Kaggle notebook execution instead of trusting a push-only result.
4. As the project operator, I want CPU smoke to be read-only with respect to DiLoCo round state, so that validation cannot corrupt Stage 10 training progress.
5. As the project operator, I want explicit tests around the validation branch, so that future refactors cannot reintroduce GPU requests, `torchrun`, or hidden Hub-state mutation.

## Implementation Decisions

- Add a `workflow_validate` workflow-dispatch input with supported value `cpu-smoke`.
- Add coordinator CLI arguments for validation mode, validation run ID, timeout, and poll interval.
- Treat workflow validation as a separate coordinator mode that runs before reading `round_state.json`.
- Default CPU smoke to Worker A when `force_worker_ids` is empty; respect explicit `force_worker_ids` for multi-worker validation.
- Inject `OUROBOROS_WORKFLOW_VALIDATE=cpu-smoke`, a stable validation run ID, a validation state repo, and `OUROBOROS_WORKFLOW_VALIDATION_PUBLISH=1` into staged Kaggle notebooks.
- Continue disabling Kaggle GPU metadata and omitting `--accelerator NvidiaTeslaT4` when CPU smoke is active.
- Publish status/report JSON under `diloco_state/workflow_validation/<run_id>/worker_<id>_{status,report}.json`.
- Poll those validation status artifacts from GitHub Actions and fail the coordinator job if they do not appear or do not match the requested run/worker/mode.
- Do not mutate `diloco_state/round_state.json`, anchors, worker checkpoint paths, or stage counters during workflow validation.

## Testing Decisions

- Extend CPU-smoke tests to cover remote status/report path construction and publish behavior with a fake uploader.
- Extend dispatch tests to verify the staged notebook receives validation publish env vars and still omits accelerator for CPU smoke.
- Extend coordinator orchestration tests to verify validation mode defaults to Worker A, respects explicit worker IDs, polls remote validation status, and does not read or write live `round_state.json`.
- Extend source-of-truth workflow tests so the `workflow_dispatch` CPU-smoke input and run-id plumbing cannot drift out of the YAML.

## Out of Scope

- DGAC algorithm changes.
- Model quality thresholds and halt-step distribution analysis.
- CPU/Mamba portability beyond the smoke path.
- Replacing the signal mechanism.
- Real GPU training launch behavior changes.

## Further Notes

DGAC readiness remains a two-part gate after this work: workflow readiness via CPU smoke, then model readiness via Stage 10 metrics and DGAC-specific launch criteria.
