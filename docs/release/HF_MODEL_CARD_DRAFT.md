---
language:
- en
license: apache-2.0
base_model: ai21labs/AI21-Jamba-Reasoning-3B
tags:
- reasoning
- mamba
- jamba
- peft
- lora
- research
- alpha
library_name: transformers
pipeline_tag: text-generation
---

# Ouroboros — Alpha Research Checkpoint

## Model Summary

Ouroboros is an alpha research checkpoint built as a PEFT adapter and DGAC HaltGate runtime on top of `ai21labs/AI21-Jamba-Reasoning-3B`.

The experiment tests a lightweight latent-reasoning path:

```text
base Jamba Reasoning 3B
+ <|lat|> token
+ PEFT adapter
+ Coconut latent passes
+ DGAC HaltGate
```

This model card is a draft for the public alpha release. Comparison benchmarks are pending and must be added before making any superiority claims.

## Status

```text
release stage -> alpha research checkpoint
latest health signal -> stage 10 teacher-forced eval-only pass
teacher-forced CE -> 0.4114
teacher-forced token accuracy -> 0.8693
anchor path -> WeirdRunner/Ouroboros/diloco_state/anchor
claim status -> healthy checkpoint, not generated-answer progress or benchmark superiority
```

## Intended Use

Suitable for:

```text
research inspection
reasoning-runtime experiments
adapter/HaltGate evaluation
comparison against the base model
small public demo after release gates pass
```

Not yet suitable for:

```text
production deployment
safety-critical decisions
medical/legal/financial advice
claims of SOTA performance
edge deployment without behavior-preservation checks
```

## How to Use

Planned faithful runtime path:

```bash
python -m ouroboros.inference \
  --prompt "Solve: ..." \
  --adapter_repo WeirdRunner/Ouroboros \
  --adapter_subfolder diloco_state/anchor \
  --use_halt_gate
```

The package CLI now has a lightweight help path. Real inference still requires access to the base model, adapter, and `halt_gate.pt` when HaltGate-backed runtime is required.

## Evaluation

Pending real generated-answer comparison artifacts:

| Suite | Dataset/Split | Jamba baseline | Ouroboros | Notes |
|---|---|---:|---:|---|
| In-domain holdout | `WeirdRunner/Ouroboros`, config `coconut-v1`, split `validation`, revision `6a52cd0c47be1e7b85d9018225387950aefc4631` | TBD | TBD | ID-backed Coconut validation; not an external benchmark |
| Anchor suite | `arc_easy`, `hellaswag`, `winogrande` | TBD | TBD | Later optional lm-eval bridge after latent-aware loglikelihood |
| Reasoning suite | `arc_challenge`, `openbookqa`, `piqa`, `gsm8k`, `truthfulqa_mc2` | TBD | TBD | Later optional lm-eval bridge; no external claims until artifacts exist |

## Training and Data

Known current data signal:

```text
train samples -> 36,906
validation samples -> 1,940
stage stats -> median=10, mean=10.42, max=16
```

Before public release, the final card must disclose:

```text
training data source and construction process
validation split policy and exact revision
whether validation influenced checkpoint selection
source/ID fields used for auditability
known contamination and claim-boundary risks
```

## Limitations

```text
comparison against base Jamba is not complete
Coconut validation result is an ID-backed in-domain holdout signal, not a public external benchmark claim
latent/HaltGate runtime may not export cleanly to GGUF/Ollama yet
quantized paths must be compared against faithful runtime before release
model inherits limitations and risks from the base model
```

## Release Checklist

```text
[x] public inference CLI help works
[x] eval package help/dry-run shell exists
[ ] ID-backed Coconut generated-answer Jamba-vs-Ouroboros eval completed with real artifacts
[ ] benchmark artifacts uploaded or committed
[ ] README table filled from artifacts
[ ] demo uses faithful runtime
[ ] limitations/non-claims preserved
```

## Citation / Attribution

Base model: `ai21labs/AI21-Jamba-Reasoning-3B`.

Ouroboros adapter/runtime: `WeirdRunner/Ouroboros`.
