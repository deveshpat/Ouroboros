# Status

Current truth -> modular research runtime, simplified active path, no release-valid model-quality claim.

## Active Runtime

```text
Bootstrap -> runtime guardrails and cached-wheel setup
Coconut -> training, latent passes, DGAC/HaltGate, checkpoints
Models -> HF/PEFT loading, dtype, quant, runtime probes
Inference -> faithful adapter + latent generation
Eval -> Coconut artifacts plus lm-eval stock HF/PEFT smoke
Utils -> small provider helpers
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
```

## Preserved Guardrails

```text
Mamba fast-path runtime probe
T4 fp16 instead of bf16 emulation
cc < 7.5 fast-fail
generated-answer truncation audits
explicit HaltGate missing/disabled state
visible notebook launch command
```

## Still Not Claimed

```text
SOTA quality
public benchmark win
working dynamic stopping
behavior-preserving quantized export
latent-aware lm-eval loglikelihood
```
