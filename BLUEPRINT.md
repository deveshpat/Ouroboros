# Ouroboros -> Minimal Runtime

Read first -> inspect owning package -> avoid root-wrapper thinking.

## Public map

| Package | Owns | Surface | State |
|---|---|---|---|
| Bootstrap | runtime -> device/dtype -> guardrails -> known-failure triage | imported before heavy runtime | implemented |
| Coconut | curriculum -> latent passes -> DGAC/HaltGate -> train/checkpoint/resume | `python -m ouroboros.coconut ...` | implemented |
| Models | HF CausalLM -> tokenizer -> adapter -> LoRA/PEFT -> quant/memory policy | `ouroboros.models` | implemented |
| Inference | prompt -> latent decode -> generated output | package API; module CLI pending `__main__.py` | partial |
| Eval | anchor eval -> gen checks -> named lm-eval suites/benchmark -> smoke | planned `python -m ouroboros.eval ...` | release blocker |
| Coordinator | DiLoCo/solo/DDP -> dispatch -> aggregate -> promote/repair | `python -m ouroboros.coordinator ...` | implemented |
| Utils | env/provider -> Hub/W&B/Kaggle/Azure/Mac helpers | `ouroboros.utils` | implemented |

## Ownership rule

runtime? -> Bootstrap
stage/latent/DGAC/train/checkpoint? -> Coconut
model/tokenizer/adapter/quant/memory? -> Models
prompt/generate/decode? -> Inference
eval/gen/lm-eval/benchmark/suite? -> Eval
worker/DDP/dispatch/aggregate/promote/repair? -> Coordinator
provider/env/Hub/W&B helper only? -> Utils

## Commands

Implemented now:

```bash
python -m ouroboros.coconut --help
python -m ouroboros.coconut --use_halt_gate --resume_from_diloco_anchor --eval_only
python -m ouroboros.coordinator --help
```

Release blockers before public demo/claims:

```bash
python -m ouroboros.inference --help   # needs ouroboros/inference/__main__.py
python -m ouroboros.eval --help        # needs planned eval package
```

Root scripts -> retired.
Package roots -> public surface.
Submodules -> internal unless doc says seam.

## Validation

Current lightweight validation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m compileall -q ouroboros
```

Future validation after tests land:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m compileall -q ouroboros tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Tests -> behavior + hard lessons.
No tests -> old extraction shape.

## State

Canonical anchor -> `WeirdRunner/Ouroboros/diloco_state/anchor`.
Latest eval-only health signal -> stage 10, val CE 0.4114, token acc 0.8693.
Coordinator -> dispatch/aggregate/promote.
Eval -> release-blocking comparison gates before public claims.
Hard lesson -> executable guardrail/test/classifier, not prose only.

## Release path

```text
healthy anchor
-> unbiased Jamba-vs-Ouroboros comparison eval
-> research README + HF model card
-> faithful hosted demo
-> optimization/edge experiments
```
