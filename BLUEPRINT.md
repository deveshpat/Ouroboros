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
| Stage 10 | ✅ COMPLETE — terminal DiLoCo anchor uploaded (2026-05-09) |
| DGAC | 🟢 CLEARED FOR LAUNCH — Stage 10 terminal anchor eval passed; launch explicitly via workflow |

**Compute:** DiLoCo dynamic workers with attendance/waiting fallback. The 2026-05-09 coordinator run aggregated Workers A/B/C for stage 10 round 2, reached 36,906/36,906 stage samples, uploaded the terminal DiLoCo anchor, and stopped at the DGAC manual gate with no stage-11 dispatch. The follow-up anchor eval-only run loaded `diloco_state/anchor` and reported `val_ce=0.4863`, `val_acc=0.0889`, coherent generation samples, and `Mean UWR=0.733`, so Phase 3.4 DGAC is cleared for explicit launch.

---

## Stage 10 Anchor Quality Review Command

Use this before launching DGAC because W&B only has Stage 10 `pre_val` / accuracy from the start of the stage, not a final validation pass for the terminal anchor. This loads `diloco_state/anchor`, evaluates at Stage 10, optionally runs the existing generation callback, and exits without optimizer steps or checkpoint writes.

```bash
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --use_halt_gate --resume_from_diloco_anchor --eval_only \
  --diloco_state_repo WeirdRunner/Ouroboros --hf_token "$HF_TOKEN" \
  --data_dir data/coconut_v1 --use_4bit \
  --max_stage 10 --max_grad_norm 0.3 \
  --batch_size 4 --grad_accum 8 --val_batch_size 2 \
  --val_skip_buffer_minutes 60 \
  --session_timeout_hours 12.0 --graceful_exit_buffer_minutes 20 \
  --output_dir runs/stage10_anchor_eval \
  --wandb_project "ouroboros-stage3-jamba"
```

Workflow path: GitHub Actions → `coordinate` → **Run workflow** with `kaggle_run_mode=dgac-anchor-eval`, `force_worker_ids=A`, `skip_trigger=false`, `dry_run=false`, and empty `workflow_validate`. This pushes one GPU Kaggle notebook in eval-only mode; it does not mutate `round_state` and still loads the latest anchor from `diloco_state/anchor` on Hub.

**Latest eval result:** PASS — `val_ce=0.4863`, `val_acc=0.0889`, coherent generation samples, `Mean UWR=0.733`. `k_actual=10` for all samples is expected before DGAC because HaltGate starts at zero-init.

## DGAC Launch Command (Phase 3.4 — Stage 10 quality gate passed)

**Recommended path: parallel DGAC DiLoCo.** Cancel/stop any older single-worker `dgac-train` Kaggle run before using this, otherwise it will burn quota while the parallel workers train from the same anchor. This path initializes `round_state` for DGAC, resets the Stage-10 DGAC sample counter to 0 while preserving the pre-DGAC totals under `pre_dgac_total_samples_seen`, dispatches all selected workers, runs one local DGAC epoch per worker shard without redundant worker pre-val/gen, and aggregates both LoRA adapter weights and `halt_gate.pt` into `diloco_state/anchor`.

Workflow path: GitHub Actions → `coordinate` → **Run workflow** with `kaggle_run_mode=dgac-diloco`, `force_worker_ids=A,B,C` (or empty to use every credentialed worker), `skip_trigger=false`, `dry_run=false`, and empty `workflow_validate`. Worker signals resume the normal coordinator loop until DGAC reaches 36,906/36,906 Stage-10 samples and writes `mode=dgac-complete`. Launch another `dgac-diloco` pass from the aggregated anchor only if the post-DGAC eval says HaltGate needs more training.

Equivalent worker command shape:

```bash
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit \
  --use_halt_gate --resume_from_diloco_anchor \
  --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 --max_grad_norm 0.3 \
  --batch_size 4 --grad_accum 8 --val_batch_size 2 \
  --val_skip_buffer_minutes 60 \
  --session_timeout_hours 12.0 --graceful_exit_buffer_minutes 20 \
  --diloco_mode --diloco_worker_id "$DILOCO_WORKER_ID" \
  --diloco_outer_lr "$OUROBOROS_DILOCO_OUTER_LR" \
  --diloco_state_repo WeirdRunner/Ouroboros \
  --diloco_signal_repo deveshpat/Ouroboros \
  --push_to_hub --output_dir runs/dgac_diloco
```

**Fallback path: sequential single-worker DGAC.** Use only when you intentionally want one worker to train DGAC without mutating DiLoCo `round_state`.

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

Workflow fallback: GitHub Actions → `coordinate` → **Run workflow** with `kaggle_run_mode=dgac-train`, `force_worker_ids=A`, `skip_trigger=false`, `dry_run=false`, and empty `workflow_validate`. This pushes one GPU Kaggle notebook, loads the terminal DiLoCo anchor, writes local checkpoints under `runs/stage3_dgac`, pushes Hub checkpoints under `runs/stage3_dgac`, and does not mutate DiLoCo `round_state`.

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
| [wiki/Future-JEPA-Multimodal-Latent.md](wiki/Future-JEPA-Multimodal-Latent.md) | Deferred JEPA-style latent prediction, shared multimodal latent bus, output-head roadmap |

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
