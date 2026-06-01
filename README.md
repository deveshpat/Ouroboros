# Ouroboros

Ouroboros is a modular research runtime for experimenting with Coconut/DGAC-style latent reasoning on top of `ai21labs/AI21-Jamba-Reasoning-3B`.

The current direction is intentionally simple without becoming flat: keep the package boundaries that make architecture work easy, remove dead orchestration, and lean on proven libraries such as Hugging Face Hub, Transformers, PEFT, Accelerate-backed device maps, bitsandbytes, and lm-evaluation-harness.

## Current Shape

```text
bootstrap   -> Kaggle/runtime setup and hard-won Jamba/Mamba guardrails
coconut     -> dataset, latent passes, DGAC HaltGate, training, checkpoints
models      -> Transformers/PEFT loading, dtype, quant, runtime probes
inference   -> faithful adapter + latent generation smoke path
eval        -> Coconut artifacts plus generated-answer and lm-eval smoke paths
utils       -> small env and W&B helpers
```

There is no active multi-worker control loop. The older orchestration lessons remain documented as failure patterns, but Kaggle runs now go through the visible `kaggle-utils.ipynb` command cell.

## Install

```bash
python -m pip install -r requirements.txt
```

On Kaggle, attach the cached `ouroboros-cache` dataset so the Mamba wheels and HF cache are reused instead of rebuilt each session.

## Kaggle

Open `kaggle-utils.ipynb`, set `OUROBOROS_KAGGLE_RUN_MODE`, and run all cells.

```text
infer        -> normal faithful adapter/DGAC inference smoke
infer_edge   -> 4-bit edge-oriented inference smoke
train_canary -> short QLoRA training canary
eval_lm      -> lm-eval stock HF/PEFT smoke; not latent-aware scoring
eval_compare -> faithful generated-answer comparison artifacts
eval_gate    -> artifact-only readiness check before architecture experiments
```

The notebook prints the exact command before launching it.

## Inference

Normal runtime:

```bash
python -m ouroboros.inference \
  --adapter_repo WeirdRunner/Ouroboros \
  --adapter_subfolder diloco_state/anchor \
  --prompt "Explain the idea in one paragraph."
```

Edge-oriented 4-bit runtime:

```bash
python -m ouroboros.inference \
  --adapter_repo WeirdRunner/Ouroboros \
  --adapter_subfolder diloco_state/anchor \
  --prompt "Explain the idea in one paragraph." \
  --use_4bit \
  --model_device_map auto \
  --dtype float16
```

## Training

Sequential canary:

```bash
python -m ouroboros.coconut \
  --data_dir data/coconut_v1 \
  --max_stage 1 \
  --max_samples 128 \
  --max_train_steps 10 \
  --use_4bit \
  --output_dir runs/stage3_canary
```

DGAC continuation from a Hub anchor:

```bash
python -m ouroboros.coconut \
  --use_halt_gate \
  --resume_from_anchor \
  --resume_anchor_repo_id WeirdRunner/Ouroboros \
  --resume_anchor_subdir diloco_state/anchor \
  --max_train_steps 10 \
  --output_dir runs/stage3_dgac_canary
```

## Evaluation

Coconut generated-answer comparison remains the faithful latent runtime path:

```bash
python -m ouroboros.eval compare-coconut-val \
  --data_dir data/coconut_v1 \
  --dataset_repo WeirdRunner/Ouroboros \
  --dataset_config coconut-v1 \
  --dataset_split validation \
  --dataset_revision 6a52cd0c47be1e7b85d9018225387950aefc4631 \
  --baseline_model_id ai21labs/AI21-Jamba-Reasoning-3B \
  --candidate_repo_id WeirdRunner/Ouroboros \
  --candidate_subdir diloco_state/anchor \
  --output_dir runs/coconut_val_compare
```

lm-eval integration is available for standard HF/PEFT smoke tests:

```bash
python -m ouroboros.eval lm-eval-hf \
  --model_id ai21labs/AI21-Jamba-Reasoning-3B \
  --adapter WeirdRunner/Ouroboros \
  --adapter_subfolder diloco_state/anchor \
  --tasks arc_easy \
  --limit 10 \
  --output_path runs/lm_eval_smoke
```

Boundary: `lm-eval-hf` uses lm-evaluation-harness' stock HF backend. It is useful for harness setup and PEFT smoke testing, but it does not execute Coconut latent passes yet.

Before widening into JEPA/curriculum/HaltGate redesign, read the generated artifacts with the readiness gate:

```bash
python -m ouroboros.eval gate-experiment-readiness \
  --comparison_dir runs/coconut_val_compare \
  --lm_eval_dir runs/lm_eval_smoke \
  --output_path runs/experiment_readiness.json
```

This gate loads no model weights. It checks whether the generated-answer comparison was clean, prompt budgets were not truncated, candidate scoring did not regress, HaltGate behavior was not suspicious, and optional lm-eval smoke artifacts exist.

## Guardrails Kept

The simplification keeps the fixes that were expensive to learn:

```text
Mamba/Jamba fast-path probe before generation/eval loops
T4 uses fp16 instead of bf16 emulation
cc < 7.5 fast-fails before cached-wheel training runs
prompt truncation audits for generated-answer eval
HaltGate missing/disabled states are explicit
Kaggle launch command stays visible in the notebook
```

## Non-Claims

Ouroboros does not currently claim SOTA quality, a public benchmark win, working dynamic stopping, or behavior-preserving quantized export. The goal is to make architecture experiments fast, faithful, and easy to run.

## License

See `LICENSE`.
