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
| DGAC | ✅ COMPLETE / 🟡 QUALITY PENDING — corrected Azure H100 epoch-0 checkpoint uploaded; quality eval still pending |

**Compute:** DiLoCo dynamic workers with attendance/waiting fallback. The 2026-05-09 coordinator run aggregated Workers A/B/C for stage 10 round 2, reached 36,906/36,906 stage samples, uploaded the terminal DiLoCo anchor, and stopped at the DGAC manual gate with no stage-11 dispatch. The pre-DGAC anchor eval-only run loaded `diloco_state/anchor` and reported `val_ce=0.4863`, `val_acc=0.0889`, coherent generation samples, and `Mean UWR=0.733`. The 2026-05-10 DGAC DiLoCo coordinator run then completed 36,906/36,906 samples and uploaded a new aggregated anchor. The 2026-05-15 Azure H100 corrected DGAC run restored the anchor `halt_gate.pt`, completed epoch 0, and uploaded `runs/azure_h100_dgac/stage_10/checkpoint-0001154`; val/gen did not run because the 720 min validation buffer tripped with 299 min remaining. Evaluate that checkpoint, or resume from it explicitly, before deciding whether another DGAC pass is needed.

---

## Current Anchor Quality Review Command

Use this for both pre-DGAC terminal-anchor review and post-DGAC aggregated-anchor review. It loads `diloco_state/anchor`, restores `halt_gate.pt` when the anchor contains it, evaluates at Stage 10, optionally runs the existing generation callback, and exits without optimizer steps or checkpoint writes.

For local runs, copy `.env.example` to `.env.local`, fill the secret values,
then load it before running coordinator, canary, or Mac fallback commands:

```bash
set -a; source .env.local; set +a
```

When `WANDB_API_KEY`/`WANDB_KEY` is set and `wandb` is installed, local fallback
workers run with `--wandb_mode online` so DGAC/DiLoCo train metrics appear live
under `OUROBOROS_WANDB_PROJECT`.

Accelerator-neutral speedups should stay enabled on CUDA/Kaggle paths: the
Kaggle launch builders and notebook command templates include `--latent_cache`,
and CE now projects only supervised next-token positions through the LM head.
CUDA bootstrap also caches arch-specific `causal_conv1d`, `mamba_ssm`, and
`flash_attn` wheels on Hugging Face; `flash_attn` is enabled only on sm80+
(A100/H100-class GPUs) and verified with a real CUDA forward/backward probe
before model load.
Mac-only speedups (`--mac_mps_mamba_kernels`, reduced `--max_seq_len`, and
`--dgac_halt_probe_steps stage_k`) stay on the strict local fallback path because
CUDA already uses the native Mamba kernels and the full DGAC probe schedule
preserves the GPU training target selection.

```bash
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --use_halt_gate --resume_from_diloco_anchor --eval_only \
  --diloco_state_repo WeirdRunner/Ouroboros --hf_token "$HF_TOKEN" \
  --data_dir data/coconut_v1 --use_4bit --latent_cache \
  --max_stage 10 --max_grad_norm 0.3 \
  --batch_size 4 --grad_accum 8 --val_batch_size 2 \
  --val_skip_buffer_minutes 60 \
  --session_timeout_hours 12.0 --graceful_exit_buffer_minutes 20 \
  --output_dir runs/dgac_anchor_eval \
  --wandb_project "ouroboros-stage3-jamba"
```

Workflow path: GitHub Actions → `coordinate` → **Run workflow** with `kaggle_run_mode=dgac-anchor-eval`, `force_worker_ids=A`, `skip_trigger=false`, `dry_run=false`, and empty `workflow_validate`. This pushes one GPU Kaggle notebook in eval-only mode; it does not mutate `round_state` and loads the latest adapter + `halt_gate.pt` from `diloco_state/anchor` on Hub when present.

**Latest completed eval result:** Stage 10 terminal anchor PASS — `val_ce=0.4863`, `val_acc=0.0889`, coherent generation samples, `Mean UWR=0.733`. `k_actual=10` for all samples was expected before DGAC because HaltGate started at zero-init. After DGAC aggregation, rerun this same command and require `diloco` logs to show `Loaded halt gate from diloco_state/anchor/halt_gate.pt` before trusting halt-step distribution.

## Azure H100 Corrected DGAC Checkpoint Review (latest active gate)

Latest evidence: `Azure H100 SCUS DGAC full budgeted` loaded `diloco_state/anchor`, restored `diloco_state/anchor/halt_gate.pt`, verified H100 BF16 + flash-attn + Mamba CUDA fast path, ran stage 10 epoch 0, then saved and uploaded `runs/azure_h100_dgac/stage_10/checkpoint-0001154` with `training_state.pt`, adapter weights, and `halt_gate.pt`. Validation/generation were skipped because the run had 299 min remaining and `--val_skip_buffer_minutes 720`.

Use the normal checkpoint resume path for this artifact. Do **not** pass `--resume_from_diloco_anchor` when evaluating or resuming `checkpoint-0001154`, because that flag intentionally reloads `diloco_state/anchor` and starts a fresh optimizer from the anchor.

```bash
torchrun --standalone --nproc_per_node=1 jamba_coconut_finetune.py \
  --use_halt_gate --eval_only \
  --hf_token "$HF_TOKEN" --hf_stage_subdir runs/azure_h100_dgac \
  --data_dir data/coconut_v1 --use_4bit --latent_cache \
  --max_stage 10 --max_seq_len 2048 --max_grad_norm 0.3 \
  --batch_size 4 --grad_accum 8 --val_batch_size 2 \
  --output_dir runs/azure_h100_dgac_eval \
  --wandb_project "ouroboros-stage3-jamba"
```

If the H100 run should continue training instead of only evaluating, resume the same checkpoint stream with `--use_halt_gate`, `--hf_stage_subdir runs/azure_h100_dgac`, `--output_dir runs/azure_h100_dgac`, and no `--resume_from_diloco_anchor`. Lower the validation buffer from 720 min if an epoch-end val/gen checkpoint is required before timeout.

## DGAC Launch Command (Phase 3.4 — available if another pass is needed)

**Recommended path: DGAC dedicated rounds.** Cancel/stop any older single-worker `dgac-train` Kaggle run before using this, otherwise it will burn quota while the parallel workers train from the same anchor. This path initializes `round_state` for DGAC, resets the Stage-10 DGAC sample counter to 0 while preserving the pre-DGAC totals under `pre_dgac_total_samples_seen`, dispatches all selected workers, runs one local DGAC epoch per worker shard without redundant worker pre-val/gen, and aggregates both LoRA adapter weights and `halt_gate.pt` into `diloco_state/anchor`.

Workflow path: GitHub Actions → `coordinate` → **Run workflow** with `kaggle_run_mode=dgac-diloco`, `force_worker_ids=A,B,C` (or empty to use every credentialed worker), `skip_trigger=false`, `dry_run=false`, and empty `workflow_validate`. Worker signals resume the normal coordinator loop until DGAC reaches 36,906/36,906 Stage-10 samples and writes `mode=dgac-complete`. Do not launch another `dgac-diloco` pass unless the post-DGAC eval says HaltGate needs more training.

DGAC training now uses correctness-aware HaltGate supervision by default. The target halt depth is the smallest probed latent depth whose CE is within tolerance of the full stage-`k` path, with ponder/diversity retained as regularizers. Defaults: `--dgac_halt_supervision_weight 0.1`, `--dgac_halt_ce_tolerance 0.02`, `--dgac_halt_probe_steps 1,2,4,stage_k`, `--dgac_halt_target_mode ce_within_tolerance`.

Equivalent worker command shape:

```bash
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit --latent_cache \
  --use_halt_gate --resume_from_diloco_anchor \
  --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 --max_grad_norm 0.3 \
  --dgac_halt_supervision_weight 0.1 --dgac_halt_ce_tolerance 0.02 \
  --dgac_halt_probe_steps 1,2,4,stage_k \
  --batch_size 4 --grad_accum 8 --val_batch_size 2 \
  --val_skip_buffer_minutes 60 \
  --session_timeout_hours 12.0 --graceful_exit_buffer_minutes 20 \
  --diloco_mode --diloco_worker_id "$DILOCO_WORKER_ID" \
  --diloco_outer_lr "$OUROBOROS_DILOCO_OUTER_LR" \
  --diloco_state_repo WeirdRunner/Ouroboros \
  --diloco_signal_repo deveshpat/Ouroboros \
  --push_to_hub --output_dir runs/dgac_dedicated
```

**Fallback path: sequential single-worker DGAC.** Use only when you intentionally want one worker to train DGAC without mutating DiLoCo `round_state`.

```bash
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --use_halt_gate --resume_from_diloco_anchor \
  --diloco_state_repo WeirdRunner/Ouroboros --hf_token "$HF_TOKEN" \
  --data_dir data/coconut_v1 --use_4bit --latent_cache \
  --epochs_per_stage 3 --max_stage 10 --max_grad_norm 0.3 \
  --dgac_halt_supervision_weight 0.1 --dgac_halt_ce_tolerance 0.02 \
  --dgac_halt_probe_steps 1,2,4,stage_k \
  --batch_size 4 --grad_accum 8 --val_batch_size 2 \
  --val_skip_buffer_minutes 60 \
  --session_timeout_hours 12.0 --graceful_exit_buffer_minutes 20 \
  --push_to_hub --output_dir runs/stage3_dgac \
  --wandb_project "ouroboros-stage3-jamba"
```

Workflow fallback: GitHub Actions → `coordinate` → **Run workflow** with `kaggle_run_mode=dgac-train`, `force_worker_ids=A`, `skip_trigger=false`, `dry_run=false`, and empty `workflow_validate`. This pushes one GPU Kaggle notebook, loads the terminal DiLoCo anchor, writes local checkpoints under `runs/stage3_dgac`, pushes Hub checkpoints under `runs/stage3_dgac`, and does not mutate DiLoCo `round_state`.

**Strict local Mac fallback (only after preflight).** Use this only when Kaggle
quota/dispatch is intentionally paused and the live Hub state still matches the
expected DGAC waiting round. The command checks MPS, confirms CUDA is absent,
probes `mamba-ssm-macos`, proves a Jamba FP16/MPS forward/backward pass,
confirms `diloco_state/anchor` plus `halt_gate.pt`, refuses `--use_4bit`, writes a
short-lived Hub claim, runs one local Mac worker, then aggregates with
`--skip_trigger` under the matching claim.

```bash
python -m ouroboros.mac_dgac_fallback \
  --repo_id WeirdRunner/Ouroboros \
  --workers A \
  --local_grad_accum 8 \
  --expected_stage_k 10 \
  --expected_round_n 3 \
  --expected_mode waiting \
  --expected_total_samples_seen 23481 \
  --output_root runs/mac_dgac_fallback
```

While `diloco_state/locks/mac_dgac_fallback.json` has an active foreign claim,
the GitHub Actions coordinator exits before dispatching or aggregating. Manual
`workflow_dispatch` and signal pushes remain present, but the scheduled watchdog
cron is disabled so GitHub cannot race the local fallback on a timer.

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
| `ouroboros/mac_dgac_fallback.py` | Strict local Apple Silicon DGAC fallback preflight, claim lock, sequential worker commands, and local aggregation command |
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
