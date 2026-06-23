# Ouroboros Blueprint

Goal -> make architecture experiments easy to run while preserving runtime fixes.

## Design principle: inherit and extend

Every module extends a HF/torch base class rather than reimplementing what it
already gives you. `model.Ouroboros` *is* a `JambaForCausalLM` (latent passes
run inside `forward()`); `trainer.CurriculumTrainer` *is* a `transformers.Trainer`
(only `compute_loss`/`_save`/`evaluate` + the optimizer/scheduler factories are
overridden); `data.CoconutDataset` *is* a `torch.utils.data.Dataset`;
`train_args.OuroborosTrainingArguments` *is* a `TrainingArguments`. DDP, grad
accumulation, checkpointing machinery, autocast, device maps — all inherited,
not rebuilt. The one piece of non-inheritable math is the DGAC loss, which lives
as private methods on the trainer.

## Public Map

| File | Owns | Surface |
|---|---|---|
| `bootstrap.py` | runtime setup (CUDA profile, Mamba/Transformers patches, dep install, `OuroborosBootstrap` singleton) | imported before heavy runtime |
| `model.py` | `Ouroboros`(JambaForCausalLM), `OuroborosConfig`, `HaltGate`, `for_training`/`save_adapter`/`from_pretrained` | `from model import Ouroboros` |
| `data.py` | `CoconutDataset`(Dataset), `DGACConfig`, canonical-dataset loader, `get_max_stage` | `from data import CoconutDataset, DGACConfig` |
| `train_args.py` | `OuroborosTrainingArguments`(TrainingArguments) + Kaggle session-budget fields | imported by `train.py` |
| `callbacks.py` | `SessionTimeoutCallback`, `ValBudgetGuardCallback`, `CheckpointSidecarCallback` | imported by `train.py` |
| `checkpointing.py` | cross-session resume + Hub sync/prune | imported by `train.py` |
| `trainer.py` | `CurriculumTrainer`(Trainer): DGAC loss math, adapter-only `_save`, `evaluate` override | imported by `train.py` |
| `train.py` | session driver, stage loop, CLI | `python train.py --help` |
| `inference.py` | inference CLI collapsing onto `Ouroboros.from_pretrained` + inherited `.generate()` | `python inference.py --help` |
| `smoke.py` | CPU synthetic smoke (tiny all-attention Ouroboros + arithmetic data, 6 checks) | `python smoke.py` |
| `eval.py` | lm-eval / readiness-gate / answer-compare | **deferred — future surface** |

No `utils.py` (HF-token + wandb are one-liners at their call sites; rank helpers
dissolve into Accelerate via `Trainer.is_world_process_zero()`). No package —
flat, repo-root, single file per concern, because inheritance keeps each small.

## Active Commands

```bash
python train.py --help            # curriculum + DGAC training (torch-free --help)
python inference.py --help        # Ouroboros latent-reasoning inference
python smoke.py                   # CPU synthetic smoke (6 checks) — run in a torch env
```

All three defer heavy imports past argparse, so `--help` works without
torch/transformers/peft installed.

## Removed From Active Path

```text
the entire ouroboros/ package  (coconut/, models/, inference/, utils/)
  — superseded by the flat files above; deleted in the rewrite
manual DDP (init_process_group / all_reduce_gradients / broadcast_parameters)
  — Accelerate owns DDP now; launch via torchrun
PeftModel-wrapped base + external latent runtime
  (prepare_latent_runtime / run_latent_passes / decode_from_latent_context)
  — a real Ouroboros subclass runs latent passes inside forward(); .generate() is inherited
duplicated HaltGate, B9-redundant collate fields, hand-rolled 644-line training loop
multi-worker control loop, stateful Kaggle launch bookkeeping,
hidden launch command builders, unused Kaggle repo-sync helper abstraction
```

The lessons from those paths remain in `wiki/Lessons-Learned.md` so the bug
fixes are not forgotten.

## Current Experiment Path

```text
CPU synthetic smoke (smoke.py)  — fast feedback loop, no Kaggle session
-> normal inference launch (inference.py)
-> 4-bit edge inference launch (inference.py --use_4bit --baseline)
-> short training canary (train.py, limited steps)
-> generated-answer comparison artifacts  (eval.py, future)
-> lm-eval stock HF/PEFT smoke             (eval.py, future)
-> artifact-only experiment-readiness gate  (eval.py, future)
-> decide next architecture experiment
```

## Lightweight Validation

```bash
python -m compileall -q bootstrap.py model.py data.py train_args.py \
  callbacks.py checkpointing.py trainer.py train.py inference.py smoke.py
python train.py --help
python inference.py --help
# smoke.py executes the 6 checks + ablations — run in a torch environment:
python smoke.py
```
