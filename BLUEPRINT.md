# Ouroboros Blueprint

Goal -> make architecture experiments easy to run while preserving runtime fixes.

## Public Map

| Package | Owns | Surface |
|---|---|---|
| Bootstrap | Kaggle/runtime setup, Mamba/Jamba compatibility, hard-lesson triage | imported before heavy runtime |
| Coconut | curriculum, latent passes, DGAC/HaltGate, train/checkpoint/resume | `python -m ouroboros.coconut ...` |
| Models | Transformers/PEFT loading, dtype, quant, Accelerate device maps, runtime probes | `ouroboros.models` |
| Inference | faithful adapter + latent decode | `python -m ouroboros.inference ...` |
| Eval | Coconut artifacts, generated-answer comparison, lm-eval stock HF smoke | `python -m ouroboros.eval ...` |
| Utils | small env, Hub, W&B helpers | helper layer only |

## Active Commands

```bash
python -m ouroboros.coconut --help
python -m ouroboros.inference --help
python -m ouroboros.eval --help
python -m ouroboros.eval lm-eval-hf --help
```

## Removed From Active Path

```text
multi-worker control loop
stateful Kaggle launch bookkeeping
hidden launch command builders
unused Kaggle repo-sync helper abstraction
```

The lessons from those paths remain in `wiki/Lessons-Learned.md` and
`ouroboros/bootstrap/guardrails.py` so the bug fixes are not forgotten.

## Current Experiment Path

```text
Kaggle notebook smoke
-> normal inference launch
-> 4-bit edge inference launch
-> short training canary
-> lm-eval stock HF/PEFT smoke
-> decide next architecture experiment
-> only then JEPA/curriculum/HaltGate redesign
```

## Lightweight Validation

```bash
python -m compileall -q ouroboros
python -m ouroboros.coconut --help
python -m ouroboros.inference --help
python -m ouroboros.eval lm-eval-hf --help
```
