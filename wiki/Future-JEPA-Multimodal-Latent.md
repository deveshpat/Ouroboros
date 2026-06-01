# Future -> JEPA / Multimodal Latent

Status -> research parking lot only.
Current runtime -> do not widen for JEPA yet.
Current project gate -> `python -m ouroboros.eval gate-experiment-readiness` before JEPA/curriculum work.

## Direction

```text
text Coconut -> current latent reasoning core
HaltGate -> experimental compute controller; current objective may not align with generated-answer utility
JEPA-style objective -> future representation/curriculum branch, not runtime dependency
future multimodal -> separate owner only after concrete PRD
```

## Why JEPA does not replace the current eval gate

JEPA-style training changes what latent representations learn. It does not remove the need for:

```text
faithful baseline/candidate loading
sane generation settings
raw output inspection
answer extraction audits
benchmark artifacts
bounded compute/controller decisions
```

The current HaltGate and fixed-depth results are not enough to justify jumping straight into JEPA. First separate these failure modes:

```text
PEFT/runtime mismatch
undertraining
bad decoding/answer extraction
HaltGate target/objective mismatch
latent curriculum weakness
```

## Guardrail

Do not add JEPA abstractions now.

Allowed next step only after the readiness gate says ready:

```text
benchmark need -> PRD -> tracer slice -> package owner -> isolated branch
```

A JEPA branch must not change the default Coconut/inference runtime until it produces release-valid artifacts.
