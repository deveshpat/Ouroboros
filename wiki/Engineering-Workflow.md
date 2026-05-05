# Engineering Workflow
> Load this page before starting a new implementation phase or converting a PRD into code.

---

## Canonical Loop

All major engineering phases follow this flow:

```text
latest PRD
  -> implementation plan
  -> issues / tracer bullets
  -> TDD implementation loop
  -> documentation promotion
  -> retire obsolete planning artifacts
```

The implementation loop is continuous and phase-scoped:

```text
for each phase:
    while phase is not complete:
        choose one tracer bullet
        write or adjust the characterization/contract test first
        confirm RED when the behavior is new or the guardrail is missing
        implement the smallest behavior-preserving slice
        run targeted tests
        run the relevant regression subset
        prune only obvious local redundancy from that slice
        mark the tracer bullet complete
    move to the next phase
exit only after every phase is complete
```

This avoids both failure modes:

- no bulk blind refactor;
- no stopping after each tiny red/green step.

---

## Commit Policy

Do not commit minor changes or planning-only deltas.

A commit should represent a significant repo improvement. Good commit boundaries include:

- a completed meaningful implementation slice plus its tests and docs;
- a fully completed phase;
- a cleanup that removes obsolete artifacts after their durable decisions have been documented.

Avoid commits that only add a PRD, only add a plan, only tweak wording, or only fix one incidental test assertion unless they are part of a larger checkpoint.

---

## PRD and Plan Lifecycle

PRDs and plans are working artifacts, not permanent architecture docs.

1. Write the PRD when the problem and acceptance criteria need alignment.
2. Convert it to an implementation plan with phases and tracer bullets.
3. Execute the plan using TDD.
4. Promote durable decisions, final ownership, validation gates, and deferred work into `wiki/`.
5. Delete obsolete files from `prds/` and `plans/` once the wiki has the canonical record; in other words, delete obsolete planning artifacts after documentation promotion.

The wiki is the durable project memory. The `prds/` and `plans/` directories should not accumulate stale completed artifacts that compete with the current source of truth.

---

## TDD Rules for This Repo

- Characterization tests come before behavior movement.
- Public entrypoints remain runnable while internals move.
- Remote services must be faked before orchestration is moved.
- Kaggle GPU runs validate; they are not the local development loop.
- Tests must not depend on ambient developer secrets or shell environment tokens.
- Runtime/generated artifacts must not be tracked.
- Runtime signal files are disposable doorbells, but the signal mechanism and `.github/workflows/diloco_coordinator.yml` path trigger stay until a replacement trigger exists.
- Optimization waits until correctness seams are tested.

---

## Next PRD Candidate

The next logical PRD after training and coordinator adapter thinning is:

```text
Kaggle CPU/API Workflow Validation
```

Purpose:

- exercise the end-to-end Kaggle workflow through API/local fake seams;
- validate notebook staging, runtime env injection, repository ref selection, worker identity, and coordinator trigger/reconciliation without burning GPU hours;
- create a safe path for later benchmarking, quantization, CPU/edge portability, and wrapper cleanup.

Out of scope for that PRD until the validation seam is green:

- model quality benchmarking;
- quantization;
- CPU/Mamba portability experiments;
- edge-device inference optimization;
- large wrapper cleanup.
