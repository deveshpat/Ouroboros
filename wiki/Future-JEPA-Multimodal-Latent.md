# Future Track — JEPA-Style Latent Prediction and Multimodal Ouroboros
> Deferred architecture note. This is not an active implementation plan and does not change the current DGAC launch path.

---

## Status

| Item | State |
|---|---|
| Current training priority | Finish and evaluate DGAC from the Stage 10 terminal DiLoCo anchor |
| JEPA-style latent prediction | Deferred research track |
| Multimodal input/output architecture | Deferred research track after text-only latent prediction is validated |
| Code changes implied by this page | None |

This page records the agreed direction so it does not get lost while the project focuses on nearer gates: DGAC correctness, post-DGAC evaluation, benchmarking, quantization, and CPU/edge portability.

---

## Core Thesis

Ouroboros should evolve from:

```text
text question -> continuous latent thoughts -> text answer
```

into:

```text
any modality input -> shared latent reasoning trajectory -> target latent representation -> modality-specific decoder/output head
```

The central model should eventually be a **latent reasoning / world-model core** that predicts useful representations. Text, image, audio, video, code, and actions should be adapters around that core rather than hard-coded assumptions inside it.

The model should not be framed as only "a language model that hides chain-of-thought." The long-term target is:

```text
Ouroboros core = modality-agnostic latent predictor / planner / reasoner
Encoders        = modality -> shared latent space
Decoders/heads  = shared latent space -> modality output
```

---

## Important Boundary

Predicting embeddings makes the core more modality-agnostic, but embeddings are **not** final user-visible outputs.

A predicted image embedding still needs an image decoder or generator. A predicted audio embedding still needs an audio decoder or vocoder. A predicted video embedding still needs a video decoder or renderer. A predicted text representation still needs a token head when the required output is text.

Therefore the intended architecture is:

```text
Ouroboros predicts target representations.
Modality heads/decoders realize those representations.
```

Not:

```text
Ouroboros alone directly emits every final modality.
```

---

## Why JEPA Fits Ouroboros

JEPA-style training predicts latent representations of missing, future, or target content instead of reconstructing raw tokens/pixels/audio samples directly.

Ouroboros already has the right primitive: Coconut-style continuous latent states are inserted into `<|lat|>` slots and then used by the normal model forward pass. DGAC adds adaptive halting and diversity pressure over those latent states.

The missing piece is a direct objective that teaches each latent thought to become semantically predictive, instead of relying only on delayed final-token cross entropy.

The useful near-term objective is:

```text
latent thought h_i -> predictor -> representation of the reasoning step h_i replaced
```

This should be an auxiliary loss, not a replacement for the existing CE/DGAC losses.

---

## Current Repo Seams

The current text-only latent execution seam is `ouroboros/latent.py`. `ouroboros/dgac.py` remains the DGAC/HaltGate policy seam and public training-loss contract.

Relevant functions:

| Function | Current role | Future JEPA relevance |
|---|---|---|
| `prepare_latent_runtime(...)` | Resolves the model runtime handles used by latent execution. | Lets future objectives reuse latent execution without importing model-private helpers. |
| `run_latent_passes(...)` | Builds the continuous latent trajectory from question context, including optional HaltGate shortening. | Source of latent states to train with representation prediction. |
| `forward_latent_batch(...)` | Injects latent states into `<|lat|>` positions, runs full forward, computes CE, and returns latent artifacts. | Natural place to expose target/prediction tensors for `jepa_loss` once target representations exist. |
| `collect_latent_hidden_sequences(...)` | Collects latent hidden states from the latent context. | Can be reused or extended for JEPA target alignment. |
| `decode_from_latent_context(...)` | Greedy-decodes from a precomputed latent context. | Keeps generation/evaluation from duplicating latent decode internals. |
| `dgac.coconut_forward(...)` | Combines CE and DGAC losses into the training loss. | Future location for `total_loss += jepa_lambda * jepa_loss` while JEPA mechanics stay behind `ouroboros.coconut`. |

Future JEPA work should target public `ouroboros.coconut` functions rather than private DGAC compatibility wrappers such as `_run_latent_passes` or `_forward_batched_latent`.

The current data seam is `ouroboros/data.py`.

Relevant behavior:

| Function | Current role | Future JEPA relevance |
|---|---|---|
| `build_sample_at_stage(...)` | Replaces the first `stage_k` reasoning steps with `<|lat|>` and supervises remaining steps + answer. | Needs optional preservation of replaced step text/metadata so target representations can be generated. |
| `collate_stage_k(...)` | Batches `input_ids`, `labels`, `q_lens`, `n_latents`, and `pad_id`. | Future batches may carry target-representation handles or cached target tensors. |

---

## Near-Term Design: Reasoning-JEPA v1

The first implementation should stay text-only.

Goal:

```text
Teach each latent thought to predict the representation of the reasoning content it replaced.
```

Recommended objective:

```text
loss = ce
     + dgac_lambda_ponder * ponder
     + dgac_lambda_diversity * diversity
     + jepa_lambda * jepa_loss
```

Recommended v1 target:

```text
replaced reasoning step text -> frozen/cached teacher representation -> target_h
latent_h_i -> small predictor -> pred_h
jepa_loss = cosine/distance loss(pred_h, stop_grad(target_h))
```

Recommended first predictor:

```text
LayerNorm(d_model)
Linear(d_model -> 2*d_model)
GELU
Linear(2*d_model -> d_model)
```

Do not start with a second full EMA teacher model on Kaggle T4. That is a later ablation if v1 proves useful. The first version should use frozen or cached target representations to protect quota and reduce moving parts.

---

## Future Design: OuroLatent Bus

After text-only Reasoning-JEPA is validated, introduce a canonical latent container rather than coupling all future modalities directly to Jamba internals.

Sketch:

```text
OuroLatent
  z:        Tensor[B, T, D]
  mask:     Tensor[B, T]
  modality: text | image | audio | video | code | action | mixed
  metadata: dict
```

Architecture:

```text
text/image/audio/video/code/action encoder
        -> modality projector
        -> shared Ouroboros latent space
        -> JEPA/DGAC latent prediction and planning
        -> modality projector/output head
        -> text/image/audio/video/code/action decoder
```

The shared latent dimension should be explicit. Do not permanently assume every modality must live in the raw Jamba hidden space. Projectors make it possible to change the base model, plug in frozen encoders, or attach modality decoders without rewriting the core.

---

## Future Design: Multimodal Outputs

Output multimodality should be staged after the shared latent bus is stable.

Potential output heads:

| Head | Output target | Decoder/realizer needed |
|---|---|---|
| `TextHead` | tokens or text hidden states | LM head / tokenizer |
| `ImageHead` | image-generation conditioning latent | image decoder/generator |
| `AudioHead` | audio embedding or codec tokens | vocoder/audio decoder |
| `VideoHead` | video latent sequence | video decoder/renderer |
| `CodeHead` | code tokens, AST-like embedding, or tool-call representation | tokenizer/compiler/tool runtime |
| `ActionHead` | action/state embedding | environment actuator/simulator |

The core rule remains:

```text
The Ouroboros core predicts representations.
Output heads convert representations into modality-specific artifacts.
```

---

## Execution Order

Do not jump straight to multimodal training.

Recommended order:

1. Finish current DGAC run from the Stage 10 terminal anchor.
2. Evaluate DGAC against the Stage 10 anchor baseline.
3. Stabilize any coordinator/training bugs found during DGAC.
4. Run benchmarking and quality checks needed before new research objectives.
5. Implement **Reasoning-JEPA v1** as a text-only auxiliary loss.
6. Validate whether JEPA improves latent quality, halt behavior, CE, generation quality, or sample efficiency.
7. Only then design `OuroLatent` as a shared multimodal latent bus.
8. Add frozen modality encoders first.
9. Add output heads/decoders only after shared latent alignment is testable.

---

## Validation Gates Before Implementation

Before a PRD for Reasoning-JEPA v1 is accepted, the project should have:

- post-DGAC eval results from the Stage 10 terminal anchor path;
- a clear baseline for `val_ce`, `val_acc`, generation samples, Mean UWR, halt-step distribution, and diversity;
- a CPU-safe smoke or fake-model test path for the new objective;
- a quota-safe target-generation strategy;
- tests proving default behavior is unchanged when JEPA is disabled.

---

## Reasoning-JEPA v1 Acceptance Criteria

A future PRD should require at least these tests:

- `--use_jepa_aux=false` preserves current loss/metrics behavior.
- The data pipeline can preserve replaced reasoning-step targets without changing current `input_ids`, `labels`, `q_lens`, or `n_latents` semantics.
- The JEPA predictor is trainable and checkpointable without changing existing LoRA/HaltGate checkpoint behavior.
- `jepa_loss` is logged separately from CE, ponder, and diversity.
- `total_loss` includes JEPA only when explicitly enabled.
- The implementation runs on fake CPU fixtures without requiring GPU, Hub, W&B, or Kaggle secrets.
- A small `max_samples` run verifies the objective decreases or at least produces finite stable metrics.

---

## Out of Scope Until Later

These are intentionally deferred:

- replacing CE with representation prediction;
- full multimodal pretraining;
- direct image/audio/video generation inside the Ouroboros core;
- a large EMA teacher model in the first JEPA pass;
- action-conditioned latent planning;
- decoder training;
- large-scale cross-modal alignment;
- CPU/edge decoder optimization.

---

## PRD Names to Use Later

When this track resumes, split it into separate PRDs:

1. `Reasoning-JEPA v1: Text-Only Latent Prediction for Ouroboros`
2. `OuroLatent: Shared Multimodal Latent Bus`
3. `Frozen Multimodal Encoder Alignment`
4. `Multimodal Output Heads and Decoders`
5. `Latent Planning over OuroLatent Trajectories`

This keeps the implementation path testable and avoids turning the current DGAC project into an unbounded multimodal rewrite.
