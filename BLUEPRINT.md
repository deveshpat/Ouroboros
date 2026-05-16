# Ouroboros -> Minimal Runtime

Read first -> inspect owning package -> avoid root-wrapper thinking.

## Public map

| Package | Owns | Surface |
|---|---|---|
| Bootstrap | runtime -> device/dtype -> guardrails -> known-failure triage | imported before heavy runtime |
| Coconut | curriculum -> latent passes -> DGAC/HaltGate -> train/checkpoint/resume | `python -m ouroboros.coconut ...` |
| Models | HF CausalLM -> tokenizer -> adapter -> LoRA/PEFT -> quant/memory policy | `ouroboros.models` |
| Inference | prompt -> latent decode -> generated output | `python -m ouroboros.inference ...` |
| Eval | anchor eval -> gen checks -> diagnostics -> lm-eval/benchmark -> smoke | `python -m ouroboros.eval ...` |
| Coordinator | DiLoCo/solo/DDP -> dispatch -> aggregate -> promote/repair | `python -m ouroboros.coordinator ...` |
| Utils | env/provider -> Hub/W&B/Kaggle/Azure/Mac helpers | `ouroboros.utils` |

## Ownership rule

runtime? -> Bootstrap
stage/latent/DGAC/train/checkpoint? -> Coconut
model/tokenizer/adapter/quant/memory? -> Models
prompt/generate/decode? -> Inference
eval/gen/diagnostics/lm-eval/benchmark? -> Eval
worker/DDP/dispatch/aggregate/promote/repair? -> Coordinator
provider/env/Hub/W&B helper only? -> Utils

## Commands

```bash
python -m ouroboros.coconut --help
python -m ouroboros.coconut --use_halt_gate --resume_from_diloco_anchor --eval_only
python -m ouroboros.coordinator --help
python -m ouroboros.eval --help
python -m ouroboros.inference --help
```

Root scripts -> retired.
Package roots -> public surface.
Submodules -> internal unless doc says seam.

## Validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m compileall -q ouroboros tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Tests -> behavior + hard lessons.
No tests -> old extraction shape.

## State

Canonical anchor -> `WeirdRunner/Ouroboros/diloco_state/anchor`.
Coordinator -> dispatch/aggregate/promote.
Eval -> quality gates before claims.
Hard lesson -> executable guardrail/test/classifier, not prose only.
