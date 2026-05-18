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
latest health signal -> stage 10 eval-only pass
validation CE -> 0.4114
validation token accuracy -> 0.8693
anchor path -> WeirdRunner/Ouroboros/diloco_state/anchor
claim status -> healthy checkpoint, not yet benchmark superiority
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

Note: the package CLI is a release blocker until `ouroboros/inference/__main__.py` is added and tested. Until then, use the repo's internal inference API or Coconut eval-only path.

## Evaluation

Pending comparison eval:

| Suite | Dataset/Split | Jamba baseline | Ouroboros | Notes |
|---|---:|---:|---:|---|
| In-domain holdout | `WeirdRunner/Ouroboros`, config `coconut-v1`, split `validation`, revision `6a52cd0c47be1e7b85d9018225387950aefc4631` | TBD | TBD | ID-backed Coconut validation; not an external benchmark |
| Anchor suite | `arc_easy`, `hellaswag`, `winogrande` | TBD | TBD | Later lm-eval bridge; same prompt/template/decoding required |
| Reasoning suite | `arc_challenge`, `openbookqa`, `piqa`, `gsm8k`, `truthfulqa_mc2` | TBD | TBD | Later lm-eval bridge; candidate must use latent/HaltGate runtime |

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
[ ] public inference CLI works
[ ] eval package exists
[ ] ID-backed Coconut validation Jamba-vs-Ouroboros eval completed
[ ] benchmark artifacts uploaded or committed
[ ] README table filled from artifacts
[ ] demo uses faithful runtime
[ ] limitations/non-claims preserved
```

## Citation / Attribution

Base model: `ai21labs/AI21-Jamba-Reasoning-3B`.

Ouroboros adapter/runtime: `WeirdRunner/Ouroboros`.
