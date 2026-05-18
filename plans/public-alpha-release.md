# Plan: Public Alpha Release Spine

> Supersedes the deleted public-alpha PRD.
>
> Scope: make Ouroboros release-ready by repairing public CLIs, wrapping the existing Coconut/Kaggle-tested validation runtime, producing reproducible artifacts, then updating docs/demo/optimization only after evidence exists.

## Durable architectural decisions

- **Canonical in-domain holdout**: `WeirdRunner/Ouroboros`, config `coconut-v1`, split `validation`, revision `6a52cd0c47be1e7b85d9018225387950aefc4631`.
- **Claim boundary**: Coconut validation is an ID-backed in-domain holdout, not an external benchmark. It may support base-vs-candidate sanity comparisons, but not broad superiority/SOTA claims by itself.
- **Runtime source of truth**: existing Coconut runtime owns latent passes, DGAC HaltGate, dataset loading, and Kaggle-safe eval behavior. `ouroboros.eval` must wrap/reuse it, not reimplement it.
- **lm-eval position**: later bridge/adapter layer over the faithful Ouroboros runtime. Do not start by rewriting around lm-eval.
- **Candidate model path**: base Jamba + `<|lat|>` token + Ouroboros PEFT adapter + DGAC HaltGate + Coconut latent runtime.
- **Baseline model path**: `ai21labs/AI21-Jamba-Reasoning-3B` evaluated under the same dataset/prompt/template/scoring constraints where applicable.
- **Artifacts before claims**: docs/model-card tables are filled only from generated `run_config.json`, `summary.json`, and `results.jsonl`.
- **Heavy work opt-in**: GPU/model-loading evals stay out of default tests; default tests cover public CLI/help and dry-run artifact behavior.

---

## Phase 1: Public CLI smoke repair

**User stories covered**: developer needs documented commands to match package behavior; release workflow needs stable entrypoints.

### What to build

Repair missing lightweight public command surfaces without loading model weights.

Expected command surface:

```bash
python -m ouroboros.coconut --help
python -m ouroboros.coordinator --help
python -m ouroboros.inference --help
python -m ouroboros.eval --help
```

Implementation notes:

```text
add ouroboros/inference/__main__.py -> delegate to ouroboros.inference.generation.main
add minimal ouroboros/eval package -> help-only CLI first
add tests/test_public_cli.py -> subprocess help checks
```

### Acceptance criteria

- [ ] `python -m ouroboros.inference --help` exits 0 and does not load model weights.
- [ ] `python -m ouroboros.eval --help` exits 0 and describes the Coconut validation commands.
- [ ] Existing Coconut/coordinator help commands still pass.
- [ ] `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` passes for the new smoke tests.

---

## Phase 2: Coconut validation artifact wrapper

**User stories covered**: researcher needs reproducible validation evidence; benchmark reviewer needs audit metadata; developer needs a Kaggle-compatible path.

### What to build

Create `ouroboros.eval` as a thin orchestration/artifact layer over existing Coconut eval. First deliver dry-run and metadata inspection before real model loading.

Suggested package shape:

```text
ouroboros/eval/
  __init__.py
  __main__.py
  cli.py
  artifacts.py
  coconut_val.py
```

Suggested commands:

```bash
python -m ouroboros.eval inspect-coconut-val \
  --data_dir data/coconut_v1

python -m ouroboros.eval dry-run-coconut-val \
  --data_dir data/coconut_v1 \
  --dataset_repo WeirdRunner/Ouroboros \
  --dataset_config coconut-v1 \
  --dataset_split validation \
  --dataset_revision 6a52cd0c47be1e7b85d9018225387950aefc4631 \
  --output_dir runs/eval/coconut_val_dryrun
```

Artifact contract:

```text
run_config.json -> model/dataset/runtime/scoring config
summary.json    -> aggregate status, counts, metrics if available
results.jsonl   -> per-ID rows when real eval runs; dry-run may be empty or schema-only
```

Minimum dataset metadata to record:

```json
{
  "dataset_repo": "WeirdRunner/Ouroboros",
  "dataset_config": "coconut-v1",
  "dataset_split": "validation",
  "dataset_revision": "6a52cd0c47be1e7b85d9018225387950aefc4631",
  "id_field": "id",
  "source_field": "source",
  "claim_boundary": "ID-backed in-domain holdout; not external benchmark"
}
```

### Acceptance criteria

- [ ] Dry-run creates `run_config.json` and `summary.json` without loading model weights.
- [ ] If local `val.jsonl` exists, inspect command reports row count, source counts, and ID coverage.
- [ ] Missing IDs or duplicate IDs are reported as validation failures.
- [ ] Dataset revision and claim boundary are written to artifacts.
- [ ] No fresh custom manifest code is introduced.

---

## Phase 3: Faithful Coconut validation comparison

**User stories covered**: researcher wants base Jamba vs Ouroboros comparison; reviewer wants same data/settings; user wants honest evidence.

### What to build

Add a real comparison command that evaluates baseline and candidate using the existing Coconut/Kaggle-safe runtime path wherever possible.

Suggested command:

```bash
python -m ouroboros.eval compare-coconut-val \
  --data_dir data/coconut_v1 \
  --dataset_repo WeirdRunner/Ouroboros \
  --dataset_config coconut-v1 \
  --dataset_split validation \
  --dataset_revision 6a52cd0c47be1e7b85d9018225387950aefc4631 \
  --baseline_model_id ai21labs/AI21-Jamba-Reasoning-3B \
  --candidate_repo_id WeirdRunner/Ouroboros \
  --candidate_subdir diloco_state/anchor \
  --stage_k 10 \
  --use_halt_gate \
  --val_batch_size 2 \
  --output_dir runs/eval/coconut_val_anchor
```

Implementation notes:

```text
reuse ouroboros.coconut.data.load_canonical_dataset
reuse ouroboros.coconut.evaluation.evaluate_stage / run_eval_only
reuse ouroboros.models loading seams
preserve existing Kaggle env/timeout/memory assumptions
```

### Acceptance criteria

- [ ] Candidate eval uses real adapter + `<|lat|>` + HaltGate + latent runtime.
- [ ] Baseline and candidate artifacts are written under the same run directory.
- [ ] Artifacts include stage, seed, dtype, max sequence length, batch size, device/hardware notes, dataset revision, and exact model IDs.
- [ ] Results are keyed by validation IDs when per-example rows are emitted.
- [ ] Summary clearly labels the result as in-domain holdout validation, not benchmark superiority.

---

## Phase 4: README and HF model card generated from artifacts

**User stories covered**: user needs clear status; researcher needs non-overclaiming public docs.

### What to build

Update public docs only after Phase 3 emits artifacts. Manual text is allowed for status/limitations, but metric tables should be copied from `summary.json` or generated by a tiny helper once the artifact schema stabilizes.

Docs to update:

```text
README.md
wiki/STATUS.md
docs/release/HF_MODEL_CARD_DRAFT.md
terminal_log.md, only when preserving run evidence
```

### Acceptance criteria

- [ ] No table claims a win unless linked to a generated artifact.
- [ ] Validation result is described as ID-backed in-domain holdout.
- [ ] External benchmarks remain TBD until lm-eval bridge runs.
- [ ] The deleted PRD is not referenced from docs.

---

## Phase 5: Faithful cloud demo

**User stories covered**: demo user wants to try real runtime; researcher wants demo to reflect actual model behavior.

### What to build

Build a minimal hosted demo only after inference CLI and validation artifacts are stable.

Preferred shape:

```text
Hugging Face Space or equivalent free/low-cost demo
Transformers + PEFT runtime
loads base model + adapter + halt_gate.pt
calls ouroboros.inference rather than copying generation logic
small max_new_tokens default
queue enabled
alpha warning visible
```

### Acceptance criteria

- [ ] Demo starts from a clean environment.
- [ ] Demo exposes model/version/runtime metadata.
- [ ] Demo fails clearly when token/GPU/model access is unavailable.
- [ ] Demo does not silently fall back to base model only.

---

## Phase 6: lm-eval bridge

**User stories covered**: benchmark reviewer wants standard benchmarks; researcher wants comparable external metrics.

### What to build

Wrap the faithful Ouroboros runtime for lm-eval after the Coconut validation comparison works.

Order:

```text
one small lm-eval task smoke
-> anchor suite: arc_easy, hellaswag, winogrande
-> reasoning suite: arc_challenge, openbookqa, piqa, gsm8k, truthfulqa_mc2
```

### Acceptance criteria

- [ ] lm-eval candidate path still uses latent/HaltGate runtime.
- [ ] Baseline and candidate use equivalent prompt/template/scoring settings where possible.
- [ ] Results are emitted as artifacts before README/HF card updates.
- [ ] Unsupported benchmark/runtime combinations fail loudly.

---

## Phase 7: Optimization and edge compatibility

**User stories covered**: edge/local user wants speed only if behavior is preserved.

### What to build

Optimize after correctness is measurable.

Order:

```text
faithful FP16/BF16 runtime
-> memory/latency report
-> 8-bit or 4-bit load experiment
-> merged adapter experiment
-> serving/vLLM compatibility check
-> GGUF/Ollama/export notes only if latent/HaltGate behavior can be preserved
```

### Acceptance criteria

- [ ] Optimized path is compared against faithful path.
- [ ] Behavior drift is measured and documented.
- [ ] Unsupported edge/export paths are clearly marked unsupported.

---

## Implementation loop for the next thread

Use this exact order:

```text
1. Phase 1 with TDD: help CLI smoke tests -> inference __main__ -> eval help shell.
2. Phase 2 with TDD: dry-run artifact command -> inspect local val IDs/source counts.
3. Phase 3: real Coconut validation comparison on Kaggle/runtime hardware.
4. Phase 4: docs/model-card update from produced artifacts.
5. Stop before demo/lm-eval/optimization unless Phase 3 evidence exists.
```

Validation command set for every commit-worthy slice:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m compileall -q ouroboros
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m ouroboros.coconut --help
python -m ouroboros.coordinator --help
python -m ouroboros.inference --help
python -m ouroboros.eval --help
```
