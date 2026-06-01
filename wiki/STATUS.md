# Status

Current truth -> modular research runtime, simplified active path, no release-valid model-quality claim.

## Active Runtime

```text
Bootstrap -> runtime guardrails and cached-wheel setup
Coconut -> training, latent passes, DGAC/HaltGate, checkpoints
Models -> HF/PEFT loading, dtype, quant, runtime probes
Inference -> faithful adapter + latent generation
Eval -> Coconut artifacts plus lm-eval stock HF/PEFT smoke plus artifact-only readiness gate
Utils -> small env/W&B helpers
```

## Current Goal

```text
Make architecture experiments easy to run on Kaggle
Keep modular boundaries
Remove dead orchestration and unused helper layers
Preserve runtime bug fixes
Add normal and edge-oriented launch commands
```

## Kaggle Modes

```text
infer        -> faithful adapter/DGAC inference smoke
infer_edge   -> 4-bit edge-oriented inference smoke
train_canary -> short QLoRA canary
eval_lm      -> lm-eval stock HF/PEFT smoke
eval_compare -> faithful generated-answer comparison artifacts for the readiness gate
eval_gate    -> read existing eval artifacts and report whether architecture work is unblocked
```

## Preserved Guardrails

```text
Mamba fast-path runtime probe
T4 fp16 instead of bf16 emulation
cc < 7.5 fast-fail
generated-answer truncation audits
explicit HaltGate missing/disabled state
visible notebook launch command
artifact-only gate before JEPA/curriculum/HaltGate redesign
```

## Still Not Claimed

```text
SOTA quality
public benchmark win
working dynamic stopping
behavior-preserving quantized export
latent-aware lm-eval loglikelihood
```
