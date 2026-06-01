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

Ouroboros is an alpha research checkpoint built as a PEFT adapter plus latent-token/Coconut runtime on top of `ai21labs/AI21-Jamba-Reasoning-3B`.

The experiment tests a lightweight latent-reasoning path:

```text
base Jamba Reasoning 3B
+ <|lat|> token
+ PEFT adapter
+ Coconut latent passes
+ optional/experimental DGAC HaltGate
```

This card is a draft. It must not be used to claim model superiority until release-valid artifacts exist.

## Status

```text
release stage -> alpha research checkpoint, pre-claim
latest health signal -> stage 10 teacher-forced eval-only pass
teacher-forced CE -> 0.4114
teacher-forced token accuracy -> 0.8693
anchor path -> WeirdRunner/Ouroboros/diloco_state/anchor
claim status -> no public benchmark claim; generation/eval harness sanity remains the next gate
```

Current evidence summary:

```text
HaltGate-enabled sample -> degenerate/over-stopped generated tokens; not release-ready
fixed-depth sample diagnostic -> one earlier small sampled slice looked promising, but HaltGate was bypassed
fixed-depth longest-25 stress -> completed after memory fixes, but candidate regressed: baseline 0.12 vs candidate 0.08
PEFT/runtime warning -> adapter load reported ignored PEFT config keys; runtime fidelity must be verified before claims
```

## Intended Use

Suitable for:

```text
research inspection
reasoning-runtime experiments
adapter/HaltGate evaluation
generation-harness debugging
comparison against the base model after raw-output sanity checks
```

Not yet suitable for:

```text
production deployment
safety-critical decisions
medical/legal/financial advice
claims of SOTA performance
claims of benchmark superiority
claims that dynamic stopping works
edge deployment without behavior-preservation checks
```

## How to Use

Faithful runtime path is still under evaluation. Fixed-depth inference is diagnostic-only when HaltGate is disabled:

```bash
python -m ouroboros.inference   --prompt "Solve: ..."   --adapter_repo WeirdRunner/Ouroboros   --adapter_subfolder diloco_state/anchor   --no_halt_gate
```

HaltGate-backed inference must be treated as experimental until generated-answer artifacts show sane stopping and non-degenerate outputs.

## Evaluation

Current generated-answer artifacts are diagnostic, not release-valid model-card metrics:

| Suite | Dataset/Split | Jamba baseline | Ouroboros | Notes |
|---|---|---:|---:|---|
| In-domain holdout sample-25, HaltGate enabled | `WeirdRunner/Ouroboros`, config `coconut-v1`, split `validation`, revision `6a52cd0c47be1e7b85d9018225387950aefc4631` | 0.08 | 0.04 | Failed release gate; candidate over-stopped at one latent on 19/25 rows |
| In-domain holdout sample-25, fixed-depth diagnostic | same sample as above | 0.08 | 0.12 | Diagnostic-only pass with `--disable_candidate_halt_gate`; not valid for DGAC/HaltGate release claims |
| In-domain holdout longest-25, fixed-depth OOM/memory check | longest 25 scorable validation rows, max_seq_len 8192 | 0.12 | 0.08 | Completed after memory fixes but failed generated-answer gate; not a quality/promotion signal |
| lm-eval HF/PEFT smoke | TBD | TBD | TBD | CLI bridge exists; this is not yet latent-aware benchmark evidence |
| External benchmark suites | TBD | TBD | TBD | Blocked until faithful candidate/baseline wrappers and artifacts exist |

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
runtime library versions used to load the adapter and base model
```

## Limitations

```text
HaltGate-enabled generation is not release-ready and may stop too early
fixed-depth ablations bypass learned stopping and are diagnostic only
latest fixed-depth longest-25 stress run regressed against base Jamba
Coconut validation result is an ID-backed in-domain holdout signal, not a public external benchmark claim
PEFT config compatibility warnings must be resolved before public claims
latent/HaltGate runtime may not export cleanly to GGUF/Ollama yet
quantized paths must be compared against faithful runtime before release
model inherits limitations and risks from the base model
```

## Release Checklist

```text
[x] public inference CLI help works
[x] eval package help/dry-run shell exists
[x] sampled ID-backed Coconut generated-answer eval produced real artifacts
[x] fixed-depth longest-25 OOM/memory stress run completes after memory fixes
[ ] PEFT/runtime version fidelity verified
[ ] raw generated outputs inspected for degenerate tokens and answer-extraction failures
[ ] lm-eval smoke artifact produced and inspected
[ ] benchmark artifacts uploaded or committed
[ ] README table filled from release-valid artifacts
[ ] demo uses faithful runtime
[ ] limitations/non-claims preserved
```

## Citation / Attribution

Base model: `ai21labs/AI21-Jamba-Reasoning-3B`.

Ouroboros adapter/runtime: `WeirdRunner/Ouroboros`.
