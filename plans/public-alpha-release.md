# Plan: Public Alpha Release Spine — Finalized

> Source context: `plans/public-alpha-release.md`, repo inspection, and planning discussion on faithful validation, HaltGate use, generated-answer evaluation, and lm-eval boundaries.
>
> Status: replaces the previous public alpha plan. The old plan was directionally useful but unsafe because it allowed committed test bloat, teacher-forced metrics to stand in for real progress, non-HaltGate stage validation, and an underspecified lm-eval bridge.

## Durable architectural decisions

- **Canonical in-domain holdout**: `WeirdRunner/Ouroboros`, config `coconut-v1`, split `validation`, revision `6a52cd0c47be1e7b85d9018225387950aefc4631`.
- **Claim boundary**: Coconut validation is an ID-backed in-domain holdout. It can support base-vs-candidate sanity/progress claims, but it is not an external benchmark or SOTA claim.
- **Runtime source of truth**: existing Ouroboros Coconut/inference runtime owns latent passes, DGAC HaltGate, tokenization, generation, and Kaggle/runtime assumptions. `ouroboros.eval` wraps/reuses this runtime instead of reimplementing it.
- **Training-health vs real-progress metrics**:
  - `evaluate_stage()` remains available as a teacher-forced training-health eval.
  - `evaluate_stage()` must become HaltGate-aware by default whenever a HaltGate object exists.
  - Generated-answer exact match is the release gate and real progress metric.
- **Candidate path**: base Jamba + `<|lat|>` token + Ouroboros PEFT adapter + DGAC HaltGate + Coconut latent runtime.
- **Baseline path**: true `ai21labs/AI21-Jamba-Reasoning-3B` base model. No LoRA, no `<|lat|>`, no resized tokenizer, no HaltGate, no latent runtime.
- **HaltGate release rule**: if a candidate is declared DGAC/HaltGate-backed, release comparison must fail loudly when `halt_gate.pt` is missing. No silent fixed-depth fallback for release artifacts.
- **Artifacts before claims**: README/HF/model-card metrics must come from generated `run_config.json`, `summary.json`, and `results.jsonl` artifacts.
- **No repo-bloating tests**: permanent test files are out of scope. Implementers validate with temporary scripts/snippets and standard smoke commands.
- **lm-eval position**: later optional bridge for external benchmark compatibility/generalization. Do not add lm-eval to the default install path and do not run MC benchmark suites until latent-aware loglikelihood is implemented.

---

## Phase 0: Plan amendment and guardrails

**User stories covered**: maintainer needs implementation-safe scope; future implementer needs clear metric semantics; reviewer needs no misleading claims.

### What to build

Update the public alpha plan before code work starts.

Replace unsafe assumptions:

```text
committed tests -> implementer-only validation snippets
teacher-forced eval as progress -> teacher-forced health metric only
fixed-depth DGAC eval -> HaltGate-aware eval when gate exists
lm-eval monolith -> smoke, generative bridge, loglikelihood bridge, suites
```

### Acceptance criteria

- [ ] Plan no longer asks to add permanent `tests/` files.
- [ ] Plan names generated-answer exact match as the release progress metric.
- [ ] Plan states teacher-forced CE/token accuracy is a health side metric only.
- [ ] Plan requires `evaluate_stage()` to use HaltGate when one is present.
- [ ] Plan requires release comparison to hard-fail if required HaltGate artifact is missing.
- [ ] Plan separates Coconut holdout evidence from lm-eval external benchmark evidence.

---

## Phase 1: Public CLI smoke repair

**User stories covered**: developer needs documented commands to match package behavior; release workflow needs stable entrypoints; implementer needs a fast validation loop.

### What to build

Repair lightweight public command surfaces without loading model weights.

Expected command surface:

```bash
python -m ouroboros.coconut --help
python -m ouroboros.coordinator --help
python -m ouroboros.inference --help
python -m ouroboros.eval --help
```

Suggested package changes:

```text
ouroboros/inference/__main__.py
ouroboros/eval/__init__.py
ouroboros/eval/__main__.py
ouroboros/eval/cli.py
```

Implementation notes:

- `python -m ouroboros.inference --help` must not require model weights.
- If delegating to `ouroboros.inference.generation.main`, ensure help path remains bootstrap-safe and does not accidentally load models before argparse exits.
- `ouroboros.eval` starts as a help-only CLI shell with subcommands planned for later phases.

### Acceptance criteria

- [ ] `python -m ouroboros.inference --help` exits 0.
- [ ] `python -m ouroboros.eval --help` exits 0.
- [ ] Existing Coconut/coordinator help commands still exit 0.
- [ ] Help commands do not load model weights or require GPU/HF credentials.
- [ ] `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m compileall -q ouroboros` passes.

### Implementer-only validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m compileall -q ouroboros
python -m ouroboros.coconut --help
python -m ouroboros.coordinator --help
python -m ouroboros.inference --help
python -m ouroboros.eval --help
```

Do not commit validation-only test files unless a later explicit decision changes this policy.

---

## Phase 2: Coconut validation artifact shell

**User stories covered**: researcher needs reproducible validation evidence; benchmark reviewer needs audit metadata; developer needs dry-run verification without model loading.

### What to build

Create `ouroboros.eval` as a thin orchestration/artifact layer over existing Coconut runtime. First deliver inspection and dry-run artifacts only.

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
results.jsonl   -> per-ID rows for real eval; dry-run may be empty/schema-only
```

Minimum metadata:

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
- [ ] Dry-run does not auto-download datasets unless an explicit `--download-if-missing` style flag is provided.
- [ ] If local `val.jsonl` exists, inspect reports row count, source counts, duplicate IDs, and missing IDs.
- [ ] Dataset revision and claim boundary are written to artifacts.
- [ ] No fresh custom manifest system is introduced.

### Implementer-only validation

```bash
tmp="$(mktemp -d)"
python -m ouroboros.eval dry-run-coconut-val \
  --data_dir data/coconut_v1 \
  --dataset_repo WeirdRunner/Ouroboros \
  --dataset_config coconut-v1 \
  --dataset_split validation \
  --dataset_revision 6a52cd0c47be1e7b85d9018225387950aefc4631 \
  --output_dir "$tmp"

python - <<'PY'
import json, pathlib, os
root = pathlib.Path(os.environ.get("TMP_EVAL_DIR", ""))
# Implementer may replace with actual temp path.
PY
```

Use a temporary assertion snippet to inspect JSON shape. Do not commit it.

---

## Phase 3: Runtime correctness seams

**User stories covered**: researcher needs a true base baseline; trainer needs HaltGate-aware stage validation; release evaluator needs no silent runtime drift.

### What to build

Add minimal runtime seams instead of rewriting model loading or evaluation.

#### 3A. Base-only loading seam

Create a small function/arg path that returns a true base model:

```python
def load_base_model_and_tokenizer(args, device, *, add_lat_token: bool = False):
    ...
```

Or equivalent private helper:

```python
def _load_base_model_and_tokenizer(args, device, *, add_lat_token: bool):
    ...

def load_base_model_and_tokenizer(args, device):
    return _load_base_model_and_tokenizer(args, device, add_lat_token=False)

def load_model_and_tokenizer(args, device):
    model, tokenizer, d_model, lat_token_id = _load_base_model_and_tokenizer(
        args, device, add_lat_token=True
    )
    # existing adapter/HaltGate candidate path continues here
```

Rules:

```text
baseline: add_lat_token=False, no adapter, no HaltGate
candidate: add_lat_token=True, adapter, HaltGate when required
```

#### 3B. HaltGate-aware stage eval

Keep `evaluate_stage()`, but pass HaltGate into the latent forward path whenever present.

Target behavior:

```text
halt_gate object exists -> evaluate_stage uses HaltGate
halt_gate is None       -> evaluate_stage uses fixed-depth latent eval
```

Expected seam:

```python
def forward_latent_batch(..., halt_gate=None, ...):
    latent_ctx, _, actual_k = run_latent_passes(
        runtime=runtime,
        ctx=q_ctx,
        ctx_mask=q_ctx_mask,
        n_latent=actual_target_latents,
        halt_gate=halt_gate,
        args=args,
    )
```

Then:

```python
result = forward_latent_batch(
    runtime=runtime,
    batch=batch,
    args=args,
    halt_gate=halt_gate,
    include_hidden_sequences=halt_gate is not None,
    include_token_accuracy=True,
)
```

Metric labels must be clear:

```text
health_metrics.teacher_forced.ce
health_metrics.teacher_forced.token_acc
health_metrics.teacher_forced.halt_gate_used
health_metrics.teacher_forced.actual_latents_mean
health_metrics.teacher_forced.actual_latents_min
health_metrics.teacher_forced.actual_latents_max
```

### Acceptance criteria

- [ ] Baseline loader returns true base Jamba: no adapter, no `<|lat|>`, no resized embeddings, no HaltGate.
- [ ] Candidate loader continues to reuse existing adapter and latent runtime paths.
- [ ] `evaluate_stage()` remains available and teacher-forced, but is HaltGate-aware by default when a HaltGate exists.
- [ ] Pre-DGAC/no-HaltGate stages still work with fixed-depth validation.
- [ ] Stage eval logs/returns whether HaltGate was used and aggregate actual latent usage.
- [ ] Release comparison has a hard-fail path for missing required HaltGate artifacts.

### Implementer-only validation

Use temporary snippets/mocks to verify loader branching without loading huge models when possible. For real runtime checks, use sampled/limited GPU validation only after compile/help passes.

---

## Phase 4: Faithful Coconut generated-answer comparison

**User stories covered**: researcher wants base Jamba vs Ouroboros evidence; reviewer wants same data/settings; user wants honest progress metrics.

### What to build

Add a generated-answer comparison command. This is the release gate.

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
  --candidate_requires_halt_gate \
  --gen_max_tokens 128 \
  --limit_samples 10 \
  --output_dir runs/eval/coconut_val_anchor_sample
```

Primary metric:

```text
generated_answer_exact_match
```

Baseline flow:

```text
question only
-> true base Jamba
-> greedy decode
-> normalize_pred(output)
-> compare against answer_norm
```

Candidate flow:

```text
question only
-> base Jamba + <|lat|> + adapter + required HaltGate
-> Coconut latent runtime
-> same greedy decode budget
-> normalize_pred(output)
-> compare against answer_norm
```

Allowed prompt field:

```text
question
```

Allowed scoring field:

```text
answer_norm
```

Forbidden prompt leakage:

```text
steps
answer_full
stage labels
latent supervision
```

Per-row `results.jsonl` shape:

```json
{
  "id": "...",
  "source": "...",
  "answer_norm": "...",
  "baseline_text": "...",
  "baseline_pred_norm": "...",
  "baseline_correct": false,
  "candidate_text": "...",
  "candidate_pred_norm": "...",
  "candidate_correct": true,
  "candidate_actual_latents": 7
}
```

Summary shape:

```json
{
  "primary_metric": "generated_answer_exact_match",
  "claim_boundary": "ID-backed in-domain holdout; not external benchmark",
  "baseline": {
    "model_id": "ai21labs/AI21-Jamba-Reasoning-3B",
    "generated_answer_exact_match": 0.0
  },
  "candidate": {
    "model_id": "WeirdRunner/Ouroboros",
    "subdir": "diloco_state/anchor",
    "halt_gate_required": true,
    "halt_gate_used": true,
    "generated_answer_exact_match": 0.0,
    "actual_latents_mean": 0.0
  },
  "health_metrics": {
    "teacher_forced": "optional side metric only; not used for claims"
  }
}
```

### Acceptance criteria

- [ ] Candidate eval uses real adapter + `<|lat|>` + required HaltGate + latent runtime.
- [ ] Missing required `halt_gate.pt` fails loudly.
- [ ] Baseline is true base Jamba with no latent token or adapter modifications.
- [ ] Baseline and candidate use the same validation IDs, prompt policy, decode budget, normalization, and scoring function.
- [ ] Prompt uses question only; no answer/step leakage.
- [ ] Results are keyed by validation IDs.
- [ ] Summary clearly labels the result as in-domain holdout validation, not external benchmark superiority.
- [ ] Teacher-forced metrics, if emitted, are nested under health/side metrics only.

### Implementer-only validation

Run sampled first:

```bash
python -m ouroboros.eval compare-coconut-val ... --limit_samples 10
```

Manually inspect:

```text
run_config.json
summary.json
results.jsonl
```

Only then run the full validation split.

---

## Phase 5: README, wiki, and HF model card from artifacts

**User stories covered**: user needs clear status; researcher needs non-overclaiming public docs; maintainer needs docs that do not drift.

### What to build

Update public docs only after Phase 4 artifacts exist. Manual text is allowed for scope/status/limitations, but metric tables must be copied from artifacts or generated by a tiny helper once schema stabilizes.

Docs to update:

```text
README.md
wiki/STATUS.md
docs/release/HF_MODEL_CARD_DRAFT.md
terminal_log.md only if preserving run evidence
```

Required language:

```text
Coconut validation = ID-backed in-domain holdout
not external benchmark
teacher-forced CE/token accuracy = training-health side metric
generated exact match = real progress metric
lm-eval = pending external benchmark bridge until artifacts exist
```

### Acceptance criteria

- [ ] No table claims a win unless linked to/generated from an artifact.
- [ ] Validation result is described as ID-backed in-domain holdout.
- [ ] External benchmarks remain TBD until lm-eval artifacts exist.
- [ ] Deleted/obsolete PRDs are not referenced from public docs.
- [ ] README does not overexplain internal planning details to end users.

---

## Phase 6: Faithful cloud demo

**User stories covered**: demo user wants to try real runtime; researcher wants demo to reflect actual model behavior; maintainer wants no fake demo fallbacks.

### What to build

Build a minimal hosted demo only after inference CLI and generated-answer artifacts are stable.

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
- [ ] Demo uses the same candidate runtime path as Phase 4 as much as practical.

---

## Phase 7: lm-eval bridge

**User stories covered**: benchmark reviewer wants standard benchmark artifacts; researcher wants comparable external metrics; maintainer wants no invalid benchmark claims.

### What to build

Add an optional lm-eval bridge over the faithful Ouroboros runtime. This phase is not required for public alpha existence, but is required before external benchmark tables/claims.

Do not add lm-eval to the default installation path. Keep it optional/release-only and record exact lm-eval version/extras in artifacts.

### Phase 7A: generation-only smoke

Build:

```text
ouroboros/eval/lm_eval_bridge.py
```

Wrapper behavior:

```text
generate_until -> uses ouroboros.inference / Coconut latent runtime
loglikelihood -> explicit unsupported error for now
loglikelihood_rolling -> explicit unsupported error for now
```

Acceptance:

- [ ] lm-eval can import/register/call the Ouroboros wrapper.
- [ ] Candidate generation path uses adapter + `<|lat|>` + required HaltGate.
- [ ] Missing required HaltGate fails loudly.
- [ ] Smoke artifact records lm-eval version, task, limit, model IDs, adapter path, prompt settings, and runtime mode.
- [ ] Multiple-choice/loglikelihood tasks are not claimed supported yet.

### Phase 7B: latent-aware loglikelihood

Implement loglikelihood after generation smoke works.

Candidate scoring flow:

```text
context tokens
-> run Ouroboros latent passes / HaltGate on context
-> append continuation embeddings/tokens according to existing runtime constraints
-> compute continuation token logprobs
```

Baseline scoring flow:

```text
same wrapper mode=baseline
no adapter
no <|lat|>
no latent passes
```

Important rule:

```text
For first valid MC comparisons, run baseline and candidate through the same Ouroboros lm-eval wrapper in different modes. Do not compare lm-eval's stock HF backend baseline against a custom Ouroboros candidate until wrapper parity is understood.
```

Acceptance:

- [ ] `loglikelihood` returns valid scores and `is_greedy` flags.
- [ ] Batch size can be `1` initially.
- [ ] Unsupported rolling tasks fail clearly until implemented.
- [ ] A tiny-limit MC task run succeeds and emits per-sample logs.

### Phase 7C: anchor suite

Only after 7B passes.

Initial tiny-limit tasks:

```text
arc_easy
hellaswag
winogrande
```

Acceptance:

- [ ] Baseline and candidate both run through the same wrapper modes.
- [ ] Same task versions, fewshot counts, prompt policy, batch size, and scoring path are recorded.
- [ ] Per-sample logs are emitted.
- [ ] Full run is attempted only after tiny-limit artifact inspection.

### Phase 7D: reasoning suite

After anchor suite stabilizes:

```text
arc_challenge
openbookqa
piqa
gsm8k
truthfulqa_mc2
```

Separate reporting buckets:

```json
{
  "anchor_suite": {},
  "reasoning_mc_suite": {},
  "reasoning_generative_suite": {},
  "unsupported_or_skipped": []
}
```

Acceptance:

- [ ] MC and generative tasks are not collapsed into a single ambiguous score.
- [ ] GSM8K/extraction-heavy tasks have their own interpretation boundary.
- [ ] README/HF external benchmark tables remain absent until artifacts exist.

---

## Phase 8: Optimization and edge compatibility

**User stories covered**: edge/local user wants speed only if behavior is preserved; maintainer wants optimization after measurable correctness.

### What to build

Optimize after faithful behavior is measurable.

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

- [ ] Optimized path is compared against faithful path using Phase 4 generated eval.
- [ ] Behavior drift is measured and documented.
- [ ] Unsupported edge/export paths are clearly marked unsupported.
- [ ] No optimization path becomes the default unless it preserves candidate behavior.

---

## Implementation loop

Use this order:

```text
0. Replace old plan with this finalized plan.
1. Phase 1: public CLI smoke repair.
2. Phase 2: dry-run/inspect artifact shell.
3. Phase 3: loader seam + HaltGate-aware stage eval.
4. Phase 4: sampled generated-answer compare-coconut-val.
5. Phase 4: full Coconut validation after sample artifacts pass inspection.
6. Phase 5: docs/model-card update from produced artifacts.
7. Phase 6: faithful demo only after artifacts exist.
8. Phase 7: lm-eval generation smoke -> loglikelihood -> suites.
9. Phase 8: optimization only after faithful metrics exist.
```

Stop conditions:

```text
- Stop before docs claims if artifacts are missing.
- Stop before demo if faithful runtime cannot load.
- Stop before lm-eval MC suites if loglikelihood is unsupported.
- Stop before optimization if Phase 4 generated eval is not stable.
```

Standard validation per commit-worthy slice:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m compileall -q ouroboros
python -m ouroboros.coconut --help
python -m ouroboros.coordinator --help
python -m ouroboros.inference --help
python -m ouroboros.eval --help
```

No permanent test files are required. Use temporary snippets under `/tmp` or inline `python - <<'PY'` checks for artifact/schema validation.

---

## Verification checklist

- [ ] Every phase is independently verifiable.
- [ ] No phase is merely “backend”, “CLI”, “docs”, or another horizontal slice without behavior acceptance.
- [ ] Durable decisions are separated from volatile implementation details.
- [ ] Teacher-forced metrics cannot be confused with generated-answer progress.
- [ ] HaltGate is default whenever a trained HaltGate exists.
- [ ] Baseline is a true unmodified base model.
- [ ] lm-eval external claims are blocked until valid bridge artifacts exist.
