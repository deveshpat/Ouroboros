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

lm-eval integration drives the stock HF/PEFT backend. Single-GPU smoke:

```bash
python -m ouroboros.eval lm-eval-hf \
  --model_id ai21labs/AI21-Jamba-Reasoning-3B \
  --adapter WeirdRunner/Ouroboros \
  --adapter_subfolder diloco_state/anchor \
  --tasks arc_easy \
  --limit 10 \
  --output_path runs/lm_eval_smoke
```

Multi-GPU data parallelism (one full model copy per GPU, data split across them;
the launcher shells out to `accelerate launch -m lm_eval` per the harness docs).
Do not pass `--device` here — accelerate places the replicas:

```bash
python -m ouroboros.eval lm-eval-hf \
  --adapter WeirdRunner/Ouroboros --adapter_subfolder diloco_state/anchor \
  --suite reasoning_core \
  --data_parallel 2 --main_process_port 29501 \
  --batch_size auto \
  --output_path runs/lm_eval_reasoning
```

Model parallelism (shard one copy across GPUs for a model too big for one card;
runs outside accelerate via `parallelize=True`): `--model_parallel`.

Curated `--suite` presets fix the tasks plus conventional few-shot / chat-template
/ generation defaults so a score is meaningful and comparable:

```text
smoke           arc_easy                                            0-shot
reasoning_core  arc_challenge,hellaswag,winogrande,piqa,openbookqa  0-shot
knowledge       mmlu                                                5-shot
math            gsm8k                                               5-shot, chat, gen
instruction     ifeval                                              0-shot, chat, gen
truthful        truthfulqa_mc2                                      0-shot
leaderboard     arc_c,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k  approximate*
```

Any explicit flag (`--tasks`, `--num_fewshot`, `--apply_chat_template`/
`--no_apply_chat_template`, `--gen_kwargs`, ...) overrides the preset. For the
instruct/reasoning model, chat-template usually gives the most representative
numbers; multiple-choice loglikelihood tasks are often reported without it, so
choose deliberately. `--log_samples` (on by default with `--output_path`) keeps
per-sample outputs for the raw-output inspection the Lessons-Learned require.

\* `leaderboard` uses a single global `--num_fewshot`; the official Open LLM
Leaderboard v1 uses mixed per-task shots (arc 25 / hella 10 / mmlu 5 / wino 5 /
gsm8k 5), which needs a per-task group config rather than this preset.

Boundary 1: `lm-eval-hf` uses lm-evaluation-harness' stock HF backend. It scores
the plain HF/PEFT model and does not execute Coconut latent passes yet.

Boundary 2: the harness loads the model in a fresh subprocess, so Ouroboros'
import-time Jamba/Mamba fast-path patches do not run inside it. The cached Mamba
wheel + triton source patch from a prior bootstrap must already be in the
environment. On Kaggle, run the notebook bootstrap cell first, or pass
`--bootstrap` (full runtime setup) / `--require_fast_path` (hard-fail instead of
warn) before a long multi-GPU run.

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
