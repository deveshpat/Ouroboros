# Ouroboros

Ouroboros is now a small Coconut-style latent reasoning adapter workflow.

The codebase has one runtime file and one command-line file:

```text
ouroboros/coconut.py -> model wrapper, dataset shaping, generation, save/publish helpers
ouroboros/cli.py     -> train, eval, infer, publish commands
```

If a removed historical path is needed later, Git history has it. The working
path here should stay boring.

## Install

```bash
bash requirements.sh
```

For GPU training, install the CUDA build of PyTorch appropriate for the machine
before running the requirements file, or edit `requirements.sh` for that runtime.

## Train And Publish

```bash
python -m ouroboros train \
  --train data/coconut_v1/train.jsonl \
  --validation data/coconut_v1/val.jsonl \
  --base-model ai21labs/AI21-Jamba-Reasoning-3B \
  --stage 10 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8 \
  --output-dir runs/ouroboros \
  --push-to-hub \
  --hub-model-id WeirdRunner/Ouroboros
```

That command saves a complete release bundle to:

```text
runs/ouroboros/final
```

When `--push-to-hub` is set, the same bundle is uploaded with the Hugging Face
Hub API. No custom Hub sync layer is involved.

## Publish An Existing Bundle

```bash
python -m ouroboros publish \
  --bundle-dir runs/ouroboros/final \
  --hub-model-id WeirdRunner/Ouroboros
```

## Evaluate

```bash
python -m ouroboros eval \
  --adapter WeirdRunner/Ouroboros \
  --data data/coconut_v1/val.jsonl \
  --max-samples 128
```

This reports teacher-forced loss. It is a training sanity check, not a public
benchmark claim.

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

This simplified repo does not run distributed orchestration, benchmark
automation, or environment self-repair. The current goal is to make the core
model experiment easy to run, save, publish, and inspect.

## License

See `LICENSE`.
