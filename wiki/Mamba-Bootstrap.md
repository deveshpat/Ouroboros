# Mamba Bootstrap

Load when Jamba/Mamba fast path fails.

## Rule

Bootstrap owns fast-path readiness.
Models hides family quirks.
Coconut should not branch on Mamba internals.

## Known-good

`mamba-ssm==1.2.2` -> supported path.
2.x API break -> known failure -> guardrail.
missing fast path -> fallback policy from Bootstrap.

## Debug order

check install -> check import -> check CUDA arch -> check model kwargs -> run package smoke.

## Runtime readiness contract

Bootstrap can verify wheel/import readiness, but Models owns loaded-model readiness.
Do not print `mamba CUDA kernels: fast path ACTIVE` until the shared post-load
Jamba probe passes for the actual model object. If generation raises `Fast Mamba
kernels are not available`, classify it as a runtime-readiness-contract failure
before burning a full validation run.
