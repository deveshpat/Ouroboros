# PRD: Ouroboros Public Alpha Release

## Problem Statement

Ouroboros has a promising healthy checkpoint signal, but it is not yet ready for public claims or broad deployment. The current anchor can be restored and evaluated through the Coconut runtime, but the repo still needs an unbiased comparison evaluation, public-facing research documentation, Hugging Face presentation materials, and a faithful deployment path.

The risk is publishing too early and confusing three different states:

```text
healthy checkpoint != benchmark win != deployable public model
```

The release process must preserve that distinction.

## Solution

Create a public-alpha release pipeline that moves from evidence to presentation to deployment:

```text
health signal
-> unbiased comparison eval
-> research-style README + model card
-> faithful cloud demo
-> optimization/edge experiments
```

The first public alpha should be honest, reproducible, and useful. It should let people see what Ouroboros is, run or inspect the model path, and understand what has and has not been proven.

## User Stories

1. As the researcher, I want to compare base Jamba and Ouroboros without dataset leakage, so that I can make claims only when the evidence supports them.
2. As a user, I want a clear README and model card, so that I understand the model, method, status, limitations, and how to try it.
3. As a developer, I want public CLI smoke tests, so that documented commands do not drift from the implementation.
4. As a benchmark reviewer, I want immutable eval manifests and reproducible settings, so that results can be audited.
5. As a demo user, I want a hosted path that uses the real latent/HaltGate runtime, so that the demo reflects the actual model rather than a simplified approximation.
6. As an edge/local user, I want quantization/export experiments only after correctness checks, so that speed does not silently change behavior.

## Implementation Decisions

### Phase 0 — Documentation alignment

Goal: make docs reflect current truth before implementation begins.

Decisions:

```text
root README -> alpha research overview, current status, non-claims, release blockers
wiki/STATUS.md -> update latest eval-only health result and next gates
terminal_log.md -> replace old H100-only status with latest T4 eval-only anchor result
BLUEPRINT.md -> distinguish implemented surfaces from planned release blockers
HF model card draft -> prepare but mark benchmark table pending
PRD -> define release work before code changes
```

Acceptance:

```text
reader can tell what works now
reader can tell what is planned
no doc claims that `ouroboros.eval` exists today
no doc claims that Ouroboros beats Jamba before comparison eval
```

### Phase 1 — Public CLI and smoke-test repair

Goal: ensure documented commands match package behavior.

Required behavior:

```text
python -m ouroboros.coconut --help        -> works
python -m ouroboros.coordinator --help    -> works
python -m ouroboros.inference --help      -> works
python -m ouroboros.eval --help           -> works after eval package lands
```

Implementation shape:

```text
add ouroboros/inference/__main__.py -> delegates to existing inference main
add ouroboros/eval package -> owns benchmark/eval public surface
add tests that execute help commands without loading heavy models
```

Outcomes:

```text
public docs stop drifting from runtime
release scripts can call stable package surfaces
```

### Phase 2 — Unbiased comparison evaluation

Goal: compare base Jamba against Ouroboros while minimizing bias.

Models:

```text
baseline  -> ai21labs/AI21-Jamba-Reasoning-3B
candidate -> base Jamba + Ouroboros adapter + <|lat|> token + DGAC HaltGate + latent runtime
```

Bias controls:

```text
same prompt text
same chat template policy
same max sequence length
same max new tokens
same decoding settings
same hardware class where possible
same batch policy
same scoring scripts
same random seed for stochastic paths
same output schema
same eval commit
```

Eval sets:

```text
A. in-domain sanity
   data/coconut_v1/val only if it was not used for training/tuning/checkpoint selection

B. standard benchmark suite
   arc_easy
   hellaswag
   winogrande
   arc_challenge
   openbookqa
   piqa
   gsm8k
   truthfulqa_mc2

C. fresh custom reasoning set
   new prompts not used in training/debugging
   immutable eval_manifest.jsonl
   versioned in repo or release artifact
```

Required outputs:

```text
results.jsonl -> per-example results
summary.json -> aggregate metrics and metadata
run_config.json -> model ids, adapter path, prompt template, decode params, hardware notes
README table -> only after results are generated
```

Acceptance:

```text
baseline and candidate both run from clean commands
candidate uses real latent/HaltGate runtime
dataset contamination status is explicit
results are reproducible from checked-in config
```

### Phase 3 — Research-style GitHub README and Hugging Face model card

Goal: make the project presentable without overstating results.

GitHub README must include:

```text
abstract-style summary
method overview
current status
runtime map
reproducible commands
comparison eval table once available
limitations and non-claims
release blockers
license
```

Hugging Face model card must include:

```text
model summary
base model
adapter path
training/eval data disclosure
intended use
out-of-scope use
how to load
how to run inference
benchmark table once available
limitations
citation/attribution
```

Acceptance:

```text
a first-time reader understands alpha status in under one minute
no benchmark claim appears without a linked/committed eval artifact
HF card can be copied into the Hub model page with minimal edits
```

### Phase 4 — Faithful cloud demo

Goal: deploy a small public demo that uses the real Ouroboros runtime.

Preferred first deployment:

```text
Hugging Face Space or equivalent free/low-cost cloud demo
Transformers + PEFT path
loads base model + adapter + halt_gate.pt
uses Ouroboros inference package
small max_new_tokens default
queue enabled
clear alpha warning
```

Non-goal for first demo:

```text
not high-throughput production serving
not aggressive quantization first
not a simplified base-model-only demo
```

Acceptance:

```text
demo starts from clean environment
demo can answer a small prompt
demo exposes model/version metadata
demo does not require private local files
```

### Phase 5 — Optimization and edge compatibility

Goal: optimize only after correctness is measurable.

Order:

```text
faithful Transformers demo
-> memory/latency measurement
-> 8-bit or 4-bit weight loading experiment
-> merged adapter experiment
-> vLLM/serving compatibility experiment if runtime allows
-> quantized export experiment
-> edge-device compatibility notes
```

Edge warning:

```text
base-model GGUF compatibility does not automatically mean Ouroboros compatibility
```

The adapter, `<|lat|>` vocabulary change, latent pass logic, and HaltGate behavior must be preserved or explicitly marked as unsupported.

Acceptance:

```text
optimized path is compared against faithful path
behavior drift is measured
unsupported edge/runtime features are documented honestly
```

## Testing Decisions

Public behavior tests:

```text
compileall passes
help commands pass without loading model weights
eval manifest parser accepts valid manifest and rejects malformed manifest
comparison eval dry-run creates expected metadata files
inference CLI validates prompt/prompt_file inputs
HF model card examples remain syntactically valid
```

Recommended commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m compileall -q ouroboros
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m ouroboros.coconut --help
python -m ouroboros.coordinator --help
python -m ouroboros.inference --help
python -m ouroboros.eval --help
```

Heavy tests must be opt-in because model loading requires GPU/VRAM and external model access.

## Out of Scope

```text
claiming SOTA or Jamba-beating results before comparison eval
production autoscaling
full safety alignment work
custom llama.cpp/Ollama runtime for latent/HaltGate behavior
training a new base model
changing the training method itself during release-readiness work
```

## Further Notes

Current evidence:

```text
stage=10 eval-only
val_ce=0.4114
val_token_acc=0.8693
adapter restored=yes
halt_gate restored=yes
mamba fast path=active
status=healthy checkpoint signal
```

Current known repo gaps:

```text
no top-level README existed before this PRD/docs pass
`python -m ouroboros.inference --help` is documented but currently fails without __main__.py
`python -m ouroboros.eval --help` is documented but no eval package exists yet
no tests directory is present in the uploaded snapshot
```

Go/no-go:

```text
Go -> docs + PRD + release-readiness implementation plan
No-go -> public superiority claims before comparison eval
```
