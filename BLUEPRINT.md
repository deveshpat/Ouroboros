# Ouroboros -> Minimal Runtime

Read first -> inspect owning package -> avoid root-wrapper thinking.

## Public map

| Package | Owns | Surface | State |
|---|---|---|---|
| Bootstrap | runtime -> device/dtype -> guardrails -> known-failure triage | imported before heavy runtime | implemented |
| Coconut | curriculum -> latent passes -> DGAC/HaltGate -> train/checkpoint/resume | `python -m ouroboros.coconut ...` | implemented |
| Models | HF CausalLM -> tokenizer -> adapter -> LoRA/PEFT -> quant/memory policy | `ouroboros.models` | implemented; PEFT fidelity check required before claims |
| Inference | prompt -> latent decode -> generated output | package API + `python -m ouroboros.inference ...` | implemented; raw generation sanity required |
| Eval | Coconut validation comparison -> artifacts -> lm-eval/benchmark bridge -> smoke | `python -m ouroboros.eval ...` | implemented; lm-eval generation sanity is next |
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
python -m ouroboros.inference --help
python -m ouroboros.eval --help
python -m ouroboros.eval inspect-coconut-val ...
python -m ouroboros.eval compare-coconut-val ...
```

## Current gate before public demo/claims

```text
Do not release or benchmark-claim the anchor yet.
HaltGate-enabled generated-answer eval -> degenerate/over-stopped.
Earlier fixed-depth sample-25 -> diagnostic-only candidate win, not release-valid.
Latest fixed-depth longest-25 -> completed after OOM fixes but regressed: candidate 0.08 vs baseline 0.12.
Next gate -> PEFT/runtime fidelity + raw generation sanity + lm-eval-compatible generation harness.
```

Root scripts -> retired.
Package roots -> public surface.
Submodules -> internal unless doc says seam.

## Validation

Current lightweight validation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m compileall -q ouroboros
```

Behavior-changing implementation should use temporary, uncommitted validation snippets plus the relevant smoke commands. Do not add permanent validation scripts/tests unless explicitly requested.

## State

Canonical anchor -> `WeirdRunner/Ouroboros/diloco_state/anchor`.
Latest eval-only health signal -> stage 10, val CE 0.4114, token acc 0.8693.
Latest generated-answer diagnosis -> HaltGate failed/degenerate; fixed-depth longest-25 completed after memory fixes but failed candidate gate.
Coordinator -> dispatch/aggregate/promote.
Eval -> generated-answer comparison implemented; benchmark/lm-eval generation sanity is now the priority before more training.
Hard lesson -> executable guardrail/test/classifier, not prose only.

## Release path

```text
healthy anchor
-> PEFT/runtime fidelity check
-> raw generation and answer-extraction sanity
-> lm-eval-compatible generation smoke
-> small benchmark sanity with faithful baseline/candidate wrappers
-> full in-domain validation only after sample-level generation is sane
-> decide: benchmark phase vs HaltGate objective repair vs JEPA/curriculum branch
-> research README + HF model card from release-valid artifacts
-> faithful hosted demo
-> optimization/edge experiments
```
