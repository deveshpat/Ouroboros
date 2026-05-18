# Status

Current truth -> alpha research runtime with healthy DGAC anchor signal, pre-claim release state.

## Anchor

canonical anchor -> `WeirdRunner/Ouroboros/diloco_state/anchor`.
source checkpoint -> `runs/azure_h100_dgac/stage_10/checkpoint-0001154`.
adapter + config + `halt_gate.pt` -> promoted.

## Latest eval-only health signal

```text
dataset -> 36,906 train / 1,940 val
stage -> 10
gpu -> Tesla T4 16GB fp16
mamba fast path -> active
anchor restore -> adapter + halt_gate.pt restored from canonical anchor
mode -> eval-only
val_ce -> 0.4114
val_token_acc -> 0.8693
result -> healthy checkpoint signal
```

## Caveat

Healthy anchor != benchmark win.

The latest eval-only run proves the canonical anchor can be restored and evaluated. It does not prove Ouroboros beats `ai21labs/AI21-Jamba-Reasoning-3B`, and it does not prove the validation split is independent enough for public claims.

Next gate -> unbiased comparison eval:

```text
baseline -> ai21labs/AI21-Jamba-Reasoning-3B
candidate -> same base + Ouroboros adapter + <|lat|> + DGAC HaltGate + latent runtime
```

## Release-readiness workflow

```text
docs alignment
-> public CLI smoke repair
-> unbiased comparison eval
-> research README + HF model card
-> faithful cloud demo
-> optimization/edge experiments
```

## Current docs/release artifacts

```text
README.md -> research-style alpha overview
docs/prds/public-alpha-release.md -> PRD for comparison eval, docs, deployment, optimization
docs/release/HF_MODEL_CARD_DRAFT.md -> Hugging Face model card draft
```

## Runtime/package truth

Implemented package roots:

```text
Bootstrap -> runtime guardrails
Coconut -> training/DGAC/eval-only
Models -> HF CausalLM compatibility
Inference -> package API exists; module CLI needs __main__.py
Coordinator -> dispatch/aggregate/promote
Utils -> provider IO
```

Planned/release-blocking package root:

```text
Eval -> comparison eval + benchmark manifests + lm-eval wrapper
```

## Dispatch controls

manual inputs -> `force_worker_ids`, `skip_trigger`, `dry_run`, `kaggle_run_mode`, `benchmark_suite`, optional `benchmark_tasks`, `dgac_anchor_eval_resume_mode`.

## Active risks

```text
validation contamination -> do not claim benchmark win from reused val split
missing eval package -> docs must not imply implemented comparison harness
missing inference __main__ -> public CLI repair required before demo
quota exhaustion -> attendance/timeout path
Kaggle push false-success -> strict output parser
wrong GPU -> accelerator + runtime fast-fail
DGAC eval OOM -> inference-mode guard
optimization drift -> quantized/edge path must be compared against faithful runtime
```
