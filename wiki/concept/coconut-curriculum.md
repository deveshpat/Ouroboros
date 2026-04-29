---
title: Coconut Curriculum
type: concept
sources:
  - BLUEPRINT.md
  - ouroboros/coconut/curriculum.py
  - ouroboros/coconut/latent.py
  - ouroboros/coconut/dgac.py
  - ouroboros/coconut/dataset_runtime.py
updated: 2026-04-30
---

# Coconut Curriculum

## The Core Idea

Each training stage progressively replaces the leftmost chain-of-thought reasoning
steps with *latent passes* — iterations of the Mamba recurrent state that carry
compressed reasoning without emitting tokens. By stage K=10, the model produces an
answer using only latent computation; no intermediate tokens are visible.

This is Meta's Coconut (arXiv:2412.06769) extended with DGAC (see below).

---

## Stage Structure

```
Stage 0:  [Q] [S1 S2 S3 ... Sn] [A]         — standard CoT; labels on all steps + A
Stage k:  [Q] [●●...●k] [S_{k+1} ... Sn] [A] — first k steps replaced; labels shift right
Stage K:  [Q] [●●...●K] [A]                  — all steps replaced; labels on A only
```

`●` is the latent pass token. Each `●` corresponds to one forward pass through the
model where the hidden state evolves but no token is emitted.

**Why label shifting matters:** at stage k, loss is only computed over visible steps
(index k+1 onward) and the answer. The model must have compressed S1..Sk into its
recurrent state to predict S_{k+1} correctly. This is enforced by the data
construction in `build_stage_sample()`, not by a special loss mask — the labels array
simply doesn't include the latent steps.

---

## Data Representation

A training sample at stage k is built from raw dataset rows with fields:
`question`, `steps` (list), `answer_full`, `answer_norm`.

`build_stage_sample(example, stage_k)` splits `steps[:k]` → `latent_steps` and
`steps[k:]` → `visible_steps`. The training sequence is then assembled from
`visible_steps` + `answer_full`.

`normalize_steps()` handles the dataset's steps field being stored as a JSON-encoded
string in some records (common in the Hub version). Always call it before slicing.

---

## Stage Parameters (locked decisions)

| Parameter | Value | Rationale |
|---|---|---|
| K (max stages) | 10 | Matches n_steps_median of the coconut-v1 dataset |
| `--max_seq_len` | 1024 | Longer sequences OOM on T4; shorter filters too many samples |
| `--epochs_per_stage` | 1 | DiLoCo shards already divide data across workers; more epochs would re-train on the same shard |
| `--batch_size` | 4 (2 per GPU on Dual T4) | VRAM limit |
| `--max_grad_norm` | 0.3 for k≥2 | Exploding gradients observed at stage 2 without clipping |
| `--val_batch_size` | 2 | OOM at 4 during val |
| `val accuracy samples` | 50 | Speed vs signal tradeoff |

---

## Performance Model

Each latent pass adds one full model forward. Step time scales roughly linearly:

| Stage | Step time |
|---|---|
| 1 | ~41s |
| 2 | ~48–53s |
| 3 | ~69s |
| 5 | ~92s |
| 10 | ~149s (projected) |

Kaggle T4 sessions have a 12h hard wall. `--session_timeout_hours 12.0` with a
20-minute graceful exit buffer. Stages 4–6 each completed in ~1 day with 3 workers.

---

## Pre-Val Accuracy Trend

Monotonically increasing across stages — the primary signal that curriculum transfer
is working:

| Stage entry | Pre-val accuracy (Worker A round 0) |
|---|---|
| Stage 3 | 0.0% (expected: first stage with full latent replacement) |
| Stage 4 | ~2% |
| Stage 5 | ~3–4% |
| Stage 6 | ~4–6% |

CE remains in a healthy range (0.4–0.8) throughout. No intervention has been needed
on the training signal.

---

## DGAC (Phase 3.4 — stages 7–10 remaining)

DGAC (Diversity-Gated Adaptive Coconut) adds a halt gate to the latent pass loop:

```
L_total = L_ce + λ₁(t)·L_ponder + λ₂·L_diversity
HaltGate: Linear(2·d_model → 1), zero-init
```

`λ₁` increases progressively with stage depth (`compute_dgac_lambda1()`).
The gate is added via `--use_halt_gate`. Not yet active — planned after stage 10
baseline completes.

The primary research question is whether DGAC's halt step distribution at K≥2
provides a meaningful signal about latent compute allocation. This is tracked as
an open question in BLUEPRINT.md.

---

## Model Architecture Context

Jamba Reasoning 3B has 28 layers (26 Mamba + 2 Attention, 13:1 ratio). The Mamba
SSM recurrent state is the natural scratch-pad for latent passes — it evolves across
the `●` tokens and carries compressed reasoning into the visible sequence prefix.

LoRA targets: `q/k/v/o_proj, in_proj, x_proj, dt_proj, out_proj` — `conv1d` excluded
because it caused training instability. `r=32`, 4-bit NF4 QLoRA.

Only 26,851,328 params (0.88%) are trained.
