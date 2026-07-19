<p align="center">
  <img src="assets/ouroboros.png" alt="Ouroboros" width="420">
  <br><em>Teaching a small LLM to think in latent space before it speaks.</em>
</p>

# Ouroboros

Ouroboros fine-tunes [AI21's Jamba-Reasoning-3B](https://huggingface.co/ai21labs/AI21-Jamba-Reasoning-3B)
— a hybrid **attention + Mamba + MoE** model — to perform **Coconut-style latent
reasoning**: instead of emitting every reasoning step as text, the model does part of
its "thinking" as continuous passes through its own hidden state, then decodes only the
answer. It is built to run end-to-end on **free Kaggle GPUs**, with cross-session
checkpointing so a multi-day curriculum survives Kaggle's session limits.

> **Design principle — inherit, don't reimplement.** Every module extends a HF/torch base
> class. `Ouroboros` *is* a `JambaForCausalLM`; `CurriculumTrainer` *is* a `transformers.Trainer`;
> `CoconutDataset` *is* a `torch.utils.data.Dataset`. DDP, grad accumulation, autocast,
> device maps, and `.generate()` are all inherited, not rebuilt.

---

## What makes it interesting

- **Latent reasoning inside `forward()`.** A real `JambaForCausalLM` subclass runs latent
  passes *unconditionally* whenever `<|lat|>` tokens are present in `input_ids` — a
  data-driven trigger, so latent reasoning is mandatory rather than a flag that defaults
  to off. A learned **HaltGate** decides when the model has thought enough (adaptive compute).
- **Custom DGAC loss + curriculum.** `CurriculumTrainer` overrides only `compute_loss`,
  `_save`, and `evaluate`; the non-inheritable reasoning-supervision math (DGAC) lives as
  private methods on the trainer. Training walks a curriculum of increasing latent-reasoning
  difficulty.
- **LoRA on a hybrid stack.** Adapters target Jamba's attention projections
  (`q/k/v/o_proj`), Mamba SSM projections (`in/x/dt/out_proj`), and the MoE expert output —
  so only a few million parameters train, and checkpoints are adapter-only.
- **Cross-session training on free hardware.** Checkpoints sync to the HuggingFace Hub and
  resume across separate Kaggle sessions, with callbacks that respect the session's time
  and validation budget.
- **Edge inference.** 4-bit quantized inference path for running the tuned model on
  constrained hardware.
- **Fast feedback without a GPU.** `smoke.py` builds a tiny all-attention Ouroboros on
  synthetic arithmetic data and runs a 6-check CPU smoke suite, so the whole training loop
  is validated before spending a GPU session.

## Architecture

```
              <|lat|> tokens in input_ids
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  Ouroboros(JambaForCausalLM)                   │
   │    forward():                                  │
   │      run N latent passes over hidden state ────┼──► HaltGate ──► stop when confident
   │      then decode answer tokens                 │
   └──────────────────────────────────────────────┘
                        ▲
         LoRA adapters on attn + Mamba SSM + MoE
                        ▲
   CurriculumTrainer(Trainer): DGAC loss · adapter-only _save · curriculum stages
                        ▲
   CoconutDataset(Dataset)  ·  Accelerate/torchrun DDP  ·  HF Hub checkpoint resume
```

## Repository map

| File | Owns |
|------|------|
| `bootstrap.py` | Runtime setup (CUDA profile, Mamba/Transformers patches, dep install) |
| `model.py` | `Ouroboros`, `OuroborosConfig`, `HaltGate`, adapter save/load |
| `data.py` | `CoconutDataset`, `DGACConfig`, canonical-dataset loader |
| `train_args.py` | `OuroborosTrainingArguments` (+ Kaggle session-budget fields) |
| `trainer.py` | `CurriculumTrainer`: DGAC loss math, adapter-only save, evaluate |
| `checkpointing.py` | Cross-session resume + Hub sync/prune |
| `train.py` | Session driver, curriculum stage loop, CLI |
| `inference.py` | Latent-reasoning inference CLI |
| `smoke.py` | CPU synthetic smoke suite (6 checks) |
| `BLUEPRINT.md` | Full design rationale |

## Quick start

```bash
# 1. Validate the whole pipeline on CPU — no GPU, no Kaggle session needed
python smoke.py

# 2. See the training / inference surfaces (argparse works without torch installed)
python train.py --help
python inference.py --help

# 3. Run inference against the tuned adapter
python inference.py                     # standard
python inference.py --use_4bit --baseline   # 4-bit edge inference
```

## Status

Active research project. The current experiment path runs CPU smoke → normal inference
→ 4-bit edge inference → a short training canary → generated-answer comparison, then
decides the next architecture experiment. See [`BLUEPRINT.md`](BLUEPRINT.md) for the full
design and `wiki/Lessons-Learned.md` for bugs already fixed and why.

## License

See [LICENSE](LICENSE).
