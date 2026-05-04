# PRD: DiLoCo Coordinator Zero-Drift Extraction and Adapter Thinning

## Problem Statement

The training monolith has been extracted and thinned into a compatibility adapter, but the DiLoCo coordinator still concentrates CLI parsing, Hub state I/O, round-state transition logic, CPU aggregation math, Kaggle dispatch staging, dispatch reconciliation, W&B logging, and orchestration in one root script.

That script is operationally critical: it drives round advancement, anchor promotion, worker dispatch, timeout demotion, attendance promotion, and recovery from unconfirmed Kaggle launches. Because it currently owns the full coordination path, direct cleanup risks breaking live training even if the changes look mechanically simple.

The next engineering phase is to extract the coordinator into tested package modules with zero runtime drift, then thin the root coordinator into a compatibility adapter only after characterization tests prove the extracted behavior matches the current monolith.

## Solution

Extract coordinator behavior into the DiLoCo package behind stable, testable seams while keeping the root coordinator runnable throughout the migration.

The migration must preserve all non-negotiable public contracts:

1. CLI compatibility: existing coordinator invocations continue to work.
2. Hub state compatibility: round-state schema, field names, defaults, and semantics remain compatible with existing Hub state.
3. Kaggle dispatch compatibility: staged worker notebook dispatch remains behaviorally identical, including accelerator metadata and push-only behavior.
4. Aggregation compatibility: solo promotion and DiLoCo weighted-delta aggregation produce the same tensors as the current coordinator.
5. Recovery compatibility: waiting mode, timeout demotion, attendance promotion, `triggered_at=0` re-dispatch, and post-dispatch reconciliation preserve current behavior exactly.

The first extraction seam should be aggregation math because it is deterministic, CPU-only, isolated from remote services, and easiest to prove against the current coordinator before touching orchestration or dispatch.

## User Stories

1. As the repository maintainer, I want the coordinator root script to remain runnable during extraction so live DiLoCo training is not blocked by refactor churn.
2. As the repository maintainer, I want aggregation behavior covered by characterization tests so anchor updates do not drift during module extraction.
3. As the repository maintainer, I want round-state transitions tested with fixtures so timeout, waiting, attendance, and stage-advance behavior are preserved.
4. As the repository maintainer, I want Kaggle dispatch staging tested without calling Kaggle so notebook metadata and runtime environment injection can be changed safely later.
5. As the repository maintainer, I want the root coordinator to become a thin adapter only after package modules are proven so future CPU/API workflow validation has clean seams to target.
6. As a future implementer, I want clear module boundaries so optimization work can happen after correctness is locked, not during extraction.

## Implementation Decisions

- Keep the existing coordinator root as the authoritative public entrypoint until extracted modules are proven by tests.
- Use characterization tests before each movement of behavior. The current coordinator behavior is the oracle during extraction.
- Preserve current behavior even when it looks awkward. Behavioral cleanup belongs to a later optimization phase.
- Extract aggregation before state orchestration or dispatch.
- Prefer deep modules with narrow responsibilities:
  - aggregation math and anchor promotion;
  - round-state parsing, serialization, and transition helpers;
  - Kaggle dispatch metadata, runtime environment injection, staging, and result reconciliation;
  - orchestration that wires state, aggregation, dispatch, and logging together.
- Keep shared DiLoCo primitives stdlib-only where they are imported by both training and coordinator paths.
- Do not change worker-side round-state expectations during this phase.
- Do not change accelerator selection, Kaggle push semantics, or notebook launch semantics during this phase.
- Treat existing Hub state as production data. New typed helpers must round-trip unknown fields and tolerate existing state defaults.
- Keep W&B logging semantics stable. Logging may move behind an adapter seam, but metric names and step behavior must not change.
- The final adapter should own only process startup concerns: argument parsing, environment/bootstrap setup if needed, and delegation into packaged coordinator code.

## Testing Decisions

The local test gate remains the primary confidence signal. Kaggle runs are validation, not the development loop.

Required test coverage:

1. Aggregation characterization
   - weighted delta aggregation matches the current coordinator for small CPU tensor fixtures;
   - sample weighting is preserved;
   - solo mode promotes worker weights directly;
   - zero or invalid sample cases preserve current behavior or raise the same failure shape.

2. Round-state compatibility
   - existing round-state dictionaries round-trip without losing unknown fields;
   - worker ordering and deduplication are stable;
   - `triggered_at=0` means unconfirmed dispatch and causes immediate re-dispatch;
   - `triggered_at>0` inside the timeout window waits;
   - timeout demotes missing active workers into attendance;
   - attendance workers that respond are promoted for the next round;
   - all workers absent enters waiting mode without advancing the round;
   - stage completion advances stage and resets round count.

3. Dispatch staging and reconciliation
   - generated Kaggle metadata preserves the expected accelerator value;
   - Kaggle push arguments preserve the expected accelerator flag;
   - runtime environment payload includes worker identity and required tokens without leaking secrets in assertions;
   - failed dispatches move failed workers out of triggered workers and into attendance;
   - `triggered_at` is corrected to zero when no worker was actually dispatched.

4. CLI and adapter contract
   - the root coordinator remains executable;
   - help and argument parsing remain compatible;
   - root script becomes thin only after module-level behavior is covered;
   - tests prevent the coordinator root from regrowing after thinning.

5. Integration seam tests without remote services
   - Hub I/O uses fakes or fixtures;
   - Kaggle dispatch uses fake subprocess results;
   - W&B logging is optional and fakeable;
   - no test requires Hugging Face, Kaggle, W&B, GPU, or network credentials.

## Acceptance Criteria

- The coordinator root script remains runnable with the same CLI throughout the phase.
- Aggregation behavior is extracted first and covered by characterization tests.
- Round-state transition behavior is covered before orchestration movement.
- Kaggle dispatch staging and reconciliation are covered before dispatch movement.
- The root coordinator becomes a thin adapter only after all extracted package behavior is covered.
- Existing training adapter and Kaggle notebook contracts remain untouched except where tests prove no drift.
- The final local gate passes with the existing pytest command used for the training extraction phase.

## Out of Scope

- Changing DiLoCo protocol semantics.
- Changing worker shard math.
- Changing worker-side training behavior.
- Changing Kaggle notebook structure beyond preserving current dispatch behavior.
- Changing GitHub Actions scheduling, trigger semantics, or concurrency behavior.
- Changing Hugging Face repository layout.
- Optimizing CPU aggregation performance.
- Benchmarking model quality or throughput.
- Quantization.
- CPU/edge portability experiments.
- Cleaning up old wrappers beyond the coordinator adapter thinning explicitly covered here.

## Further Notes

This phase intentionally mirrors the successful training monolith extraction pattern: characterize first, extract behavior behind narrow seams, keep the root adapter runnable, then thin only after tests prove no drift.

The coordinator is more operationally sensitive than the training adapter because it mutates shared Hub state and triggers external workers. That makes remote side effects the main risk. The extraction should therefore prioritize pure functions and fakeable seams before any orchestration rewrite.

After this PRD is accepted, the next artifact should be an implementation plan with vertical slices. The first implementation slice should be aggregation characterization and extraction. Subsequent slices should move state transitions, dispatch staging, reconciliation, and finally root adapter thinning.
