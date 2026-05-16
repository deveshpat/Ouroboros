# Coconut Training Session

Coconut -> package, not script.

## Owners

CLI/help -> `ouroboros.coconut.cli`.
session orchestration -> `session`.
stage loop -> `stage_runner`.
data shaping -> `data`.
DGAC/HaltGate -> `dgac`.
latent passes/decode -> `latent`.
checkpoint/resume -> `checkpointing`.
val/gen hooks -> `evaluation`.
plan classifier -> `training_plan`.

## Flow

```text
python -m ouroboros.coconut
-> run_cli
-> run_training_session
-> stage_runner + evaluation + checkpointing
-> data/dgac/latent
-> models/utils/coordinator at explicit seams only
```

## Public rule

operator code -> package root.
package tests -> may inspect submodule seams.
Coordinator import inside stage loop -> no.

## Preserved contracts

help import -> bootstrap-safe.
DGAC anchor start -> `--use_halt_gate --resume_from_diloco_anchor`.
root training wrappers -> retired.
hard lessons -> tests/guardrails, not extraction docs.
