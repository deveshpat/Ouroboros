# Coconut Training Session Architecture

Durable record for the current Coconut-owned training runtime.

## Current ownership

Coconut is a package, not a monolithic script. Its package root exposes the supported public training interface while internal modules keep responsibilities separated.

| Concern | Owner |
|---|---|
| Training command | `python -m ouroboros.coconut` |
| CLI parsing/help | `ouroboros.coconut.cli` |
| Training session orchestration | `ouroboros.coconut.session` |
| Stage loop execution | `ouroboros.coconut.stage_runner` |
| Coconut dataset shaping | `ouroboros.coconut.data` |
| DGAC/HaltGate policy and training loss | `ouroboros.coconut.dgac` |
| Latent execution and latent-context decode | `ouroboros.coconut.latent` |
| Checkpoint save/load/prune and resume sync | `ouroboros.coconut.checkpointing` |
| Validation/generation hooks used during training | `ouroboros.coconut.evaluation` |
| Training plan resolution | `ouroboros.coconut.training_plan` |

## Dependency direction

```text
python -m ouroboros.coconut
  -> ouroboros.coconut.run_cli
     -> ouroboros.coconut.session.run_training_session
        -> ouroboros.coconut.stage_runner
        -> ouroboros.coconut.evaluation
        -> ouroboros.coconut.checkpointing
        -> ouroboros.coconut.training_plan
        -> ouroboros.coconut.{data,dgac,latent}
        -> ouroboros.models / ouroboros.utils / ouroboros.coordinator only at explicit seams
```

`ouroboros.coconut.stage_runner` must not import the coordinator. DiLoCo-specific behavior enters through coordinator/worker seams, not through a root training script.

## Public interface rule

New code should prefer `import ouroboros.coconut as coconut` for public capabilities. Direct submodule imports are acceptable only for tested package-internal seams or package-specific contract tests.

## Preserved behavior

- `python -m ouroboros.coconut --help` remains bootstrap-safe.
- Stage, DGAC, latent, checkpoint, and eval behavior remain test-covered after moving under Coconut.
- Root training wrappers are retired; package entrypoints are the operator surface.
- Hard lessons from the old training monolith are preserved as guardrails and behavior tests instead of passive extraction notes.

## Guardrails

| Test file | Contract protected |
|---|---|
| `tests/test_minimal_runtime_public_architecture.py` | seven-package surface and retired root scripts |
| `tests/test_training_deep_module_contracts.py` | Coconut ownership and dependency direction |
| `tests/test_checkpoint_smoke_training.py` | checkpoint/resume and fake training smoke |
| `tests/test_validation_no_drift.py` | eval/generation no-grad and current behavior seams |
| `tests/test_latent_execution_contract.py` | fixed/gated latent passes and latent-context decode |
| `tests/test_dgac_anchor_cli_contract.py` | DGAC CLI/eval-only contracts |
