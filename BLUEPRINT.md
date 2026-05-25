# Ouroboros -> Minimal Runtime

Read first -> inspect owning package -> avoid root-wrapper thinking.

## Public map

| Package | Owns | Surface | State |
|---|---|---|---|
| Bootstrap | runtime -> device/dtype -> guardrails -> known-failure triage | imported before heavy runtime | implemented |
| Coconut | curriculum -> latent passes -> DGAC/HaltGate -> train/checkpoint/resume | `python -m ouroboros.coconut ...` | implemented |
| Models | HF CausalLM -> tokenizer -> adapter -> LoRA/PEFT -> quant/memory policy | `ouroboros.models` | implemented |
| Inference | prompt -> latent decode -> generated output | package API + `python -m ouroboros.inference ...` | implemented |
| Eval | Coconut validation comparison -> artifacts -> lm-eval bridge later -> smoke | `python -m ouroboros.eval ...` | implemented; HaltGate quality is release blocker |
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

Release blocker before public demo/claims:

```text
HaltGate-enabled sampled generated-answer eval must beat or match baseline without over-stopping.
Current gate-enabled sample-25 -> candidate 0.04 vs baseline 0.08, 19/25 rows at one latent.
Fixed-depth diagnostic sample-25 -> candidate 0.12 vs baseline 0.08, but not release-valid because HaltGate decisions were bypassed.
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
Latest generated-answer diagnosis -> HaltGate enabled failed sample-25; fixed-depth diagnostic passed sample-25.
Coordinator -> dispatch/aggregate/promote.
Eval -> generated-answer comparison implemented; HaltGate training/calibration blocks release claims.
Hard lesson -> executable guardrail/test/classifier, not prose only.

## Release path

```text
healthy anchor
-> sampled ID-backed Coconut val Jamba-vs-Ouroboros comparison
-> fixed-depth ablation only when HaltGate behavior is suspect
-> train/calibrate HaltGate until sampled HaltGate-enabled comparison passes
-> full validation + research README + HF model card from release-valid artifacts
-> faithful hosted demo
-> optimization/edge experiments
```
