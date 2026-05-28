# Ouroboros

Ouroboros is a compact latent-reasoning research repo built directly on
HuggingFace, PEFT, Accelerate, and lm-eval. It no longer carries coordinators,
workers, runtime bootstrapping, or custom Hub sync systems.

The whole active path is intentionally flat:

```text
train.py                  -> python train.py
eval.py                   -> python eval.py
ouroboros/config.py       -> defaults and tiny dataclasses
ouroboros/data.py         -> JSON/JSONL loading and latent stage samples
ouroboros/latent.py       -> Coconut wrapper plus DGAC HaltGate
ouroboros/generation.py   -> latent-aware greedy generation
ouroboros/train.py        -> staged Accelerate training loop
ouroboros/eval.py         -> teacher-forced eval and lm-eval bridge
ouroboros/callbacks.py    -> save bundle, W&B, push to Hub
ouroboros/utils.py        -> small shared helpers
```

If an old infrastructure path is needed later, Git history has it. The active
repo should stay boring and experiment-shaped.

## Install

```bash
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for the machine first if
the default pip resolver does not pick the right CUDA wheel.

## Train

```bash
python train.py \
  --train data/coconut_v1/train.jsonl \
  --validation data/coconut_v1/val.jsonl \
  --base-model ai21labs/AI21-Jamba-Reasoning-3B \
  --stages 0-10 \
  --epochs-per-stage 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --output-dir runs/ouroboros
```

This saves a release bundle to:

```text
runs/ouroboros/final
```

To train the optional DGAC HaltGate regularizer:

```bash
python train.py --use-halt-gate --stages 0-10
```

## Publish

```bash
python -m ouroboros publish \
  --bundle-dir runs/ouroboros/final \
  --hub-model-id WeirdRunner/Ouroboros
```

Publishing uses HuggingFace Hub APIs directly. There is no internal checkpoint
sync service.

## Evaluate

```bash
python eval.py \
  --adapter WeirdRunner/Ouroboros \
  --data data/coconut_v1/val.jsonl \
  --max-samples 128
```

This reports teacher-forced latent loss. It is a training sanity check, not a
public benchmark claim.

For standard benchmarks, call lm-eval through the repo wrapper:

```bash
python eval.py \
  --adapter WeirdRunner/Ouroboros \
  --tasks hellaswag,arc_easy \
  --lm-eval
```

That path evaluates the HF model plus PEFT adapter through lm-eval. The local
teacher-forced path is the latent-aware sanity check.

## Infer

```bash
python -m ouroboros infer \
  --adapter WeirdRunner/Ouroboros \
  --prompt "Explain the idea in one paragraph."
```

## Data Format

JSONL or JSON rows should contain:

```json
{
  "question": "question text",
  "steps": ["reasoning step 1", "reasoning step 2"],
  "answer": "final answer"
}
```

The loader also accepts the existing local fields `answer_full` and
`answer_norm`.

## What This Does Not Do

This simplified repo does not run distributed orchestration, worker lifecycle
management, benchmark automation, or environment self-repair. The current goal
is to make the core model experiment easy to run, save, publish, and inspect.

## License

See `LICENSE`.
