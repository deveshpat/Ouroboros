# Project Ouroboros — Blueprint Index
> **Single source of truth entry point.** Load this first, then pull wiki pages relevant to the day's task.
> Source-of-truth rule: if this file and `.py`/`.ipynb` files disagree, the Python/notebook file wins.

---

## What This Project Is

Coconut-Ouroboros: latent reasoning injection into Jamba Reasoning 3B (Transformer-Mamba hybrid).
Mamba SSM recurrent state acts as compressed scratch-pad across K latent thought passes.
Based on Meta's Coconut (arXiv:2412.06769) + DGAC (Diversity-Gated Adaptive Coconut).

---

## Current Status → [wiki/STATUS.md](wiki/STATUS.md)

| Stage | Status |
|---|---|
| Stages 0–9 | ✅ COMPLETE |
| Stage 10 | 🔄 IN PROGRESS — waiting on Kaggle GPU quota |
| DGAC | ⬜ BLOCKED until Stage 10 quality gate; CPU-smoke workflow gate already passed live |

**Compute:** DiLoCo dynamic workers with attendance/waiting fallback. Current stage 10 Hub state shows Worker A contributed 10,912 samples; A/B/C dispatch attempts are blocked by weekly GPU quota.

---

## DGAC Launch Command (Phase 3.4 — after stage 10)

```bash
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --use_halt_gate --resume_from_diloco_anchor \
  --diloco_state_repo WeirdRunner/Ouroboros --hf_token "$HF_TOKEN" \
  --data_dir data/coconut_v1 --use_4bit \
  --epochs_per_stage 3 --max_stage 10 --max_grad_norm 0.3 \
  --batch_size 4 --grad_accum 8 --val_batch_size 2 \
  --val_skip_buffer_minutes 60 \
  --session_timeout_hours 12.0 --graceful_exit_buffer_minutes 20 \
  --push_to_hub --output_dir runs/stage3_dgac \
  --wandb_project "ouroboros-stage3-jamba"
```

---

## Wiki — Load Pages Relevant to Today's Task

| Page | Load when… |
|---|---|
| [wiki/STATUS.md](wiki/STATUS.md) | Every session — full status, decisions, open questions |
| [wiki/DiLoCo-Protocol.md](wiki/DiLoCo-Protocol.md) | Debugging round state, attendance/waiting, shard assignment |
| [wiki/Coordinator-State-Machine.md](wiki/Coordinator-State-Machine.md) | Debugging coordinator runs, dispatch failures, aggregation |
| [wiki/Mamba-Bootstrap.md](wiki/Mamba-Bootstrap.md) | Wheel install, CUDA kernel, fast-path issues |
| [wiki/GPU-Guardrails.md](wiki/GPU-Guardrails.md) | P100 assignment, GPU fast-fail, T4 requirement |
| [wiki/Checkpoint-Hub-Sync.md](wiki/Checkpoint-Hub-Sync.md) | Checkpoint resume, Hub uploads, DiLoCo anchor |
| [wiki/Lessons-Learned.md](wiki/Lessons-Learned.md) | Recurring failure debugging |
| [wiki/SessionLog.md](wiki/SessionLog.md) | Historical coordinator run record |
| [wiki/Architecture-Extraction.md](wiki/Architecture-Extraction.md) | Completed extraction history, adapter ownership, retired PRD/plan decisions |
| [wiki/Engineering-Workflow.md](wiki/Engineering-Workflow.md) | PRD → plan → tracer-bullet → TDD loop, commit policy, artifact retirement |
| [wiki/Kaggle-CPU-API-Workflow-Validation.md](wiki/Kaggle-CPU-API-Workflow-Validation.md) | CPU-safe Kaggle workflow validation, manual API smoke, dispatch guardrails |

---

## File Registry

| File | Role |
|---|---|
| `jamba_coconut_finetune.py` | Thin worker-training compatibility adapter |
| `diloco_coordinator.py` | Thin coordinator compatibility adapter |
| `ouroboros/` | Packaged training, worker, coordinator, dispatch, state, and aggregation behavior |
| `.github/workflows/diloco_coordinator.yml` | CI trigger + dependencies |
| `kaggle-utils.ipynb` | Kaggle notebook runtime adapter; preserves `!torchrun` shell magic |
| `signals/.gitkeep` | Keeps the runtime signal directory present; generated `signals/*.json` files are ignored; signal JSON pushes still trigger the coordinator |
| `wiki/` | Durable operational and architectural knowledge base |
| `terminal_log.md` | Last-session verbatim output (rolling, ≤80 lines) |

---

## Engineering Architecture

Training and coordinator monolith extraction is complete. The root scripts are thin compatibility adapters, while reusable behavior lives in `ouroboros/`. Kaggle CPU/API workflow validation is implemented for quota-safe runtime smoke checks, including remote Hub artifact verification through `workflow_validate=cpu-smoke`. See [wiki/Architecture-Extraction.md](wiki/Architecture-Extraction.md) and [wiki/Kaggle-CPU-API-Workflow-Validation.md](wiki/Kaggle-CPU-API-Workflow-Validation.md).

## Architecture Snapshot

```
Model:    Jamba Reasoning 3B  |  d_model=2560  |  28 layers (26 Mamba + 2 Attn)
Trainable: 26.8M params (0.88% — LoRA r=32 adapters only)
Dataset:  36 906 samples  |  batch=4  |  grad_accum=8  |  max_seq_len=1024
DiLoCo:   3 workers (A=weirdrunner, B=weirdrunner007, C=weirdrunner008)
          Outer LR=0.7  |  outer update = weighted pseudo-gradient average
```
