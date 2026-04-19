# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in every new session.**
> **Source of truth:** If this doc and `.py`/`.ipynb` files ever disagree, the Python/notebook file wins.
> **DRY rule:** Session details and verbatim logs live in `terminal_log.md` only. This file holds decisions, status, and next actions.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros: latent reasoning injection into Jamba Reasoning 3B (Transformer-Mamba hybrid). The Mamba SSM recurrent state acts as compressed scratch-pad across K latent thought passes, replacing token generation during reasoning. Based on Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — a novel anti-collapse halt gate.

### Current Status (2026-04-19)

| Curriculum Stage | Status | Best val |
|---|---|---|
| Stage 0 — CoT warmup | ✅ COMPLETE | ce=0.4041, acc=0.0222 |
| Stage 1 — 1 latent pass | ✅ COMPLETE | ce=0.4912, acc=0.0444 |
| Stage 2 — 2 latent passes | 🟡 59% (679/1154 steps) — anchor on Hub at checkpoint-0002987 | — |
| Stages 3–10 | ⬜ NOT STARTED | — |
| Phase 3.4 — DGAC | ⬜ after Stage 10 | — |
| Phase 4 — GRPO | ⬜ after DGAC | — |

**Compute mode: DiLoCo 3-way parallel (3× speedup over sequential relay)**

---

## Part 0.1 — Immediate Next Steps (strict order, no ambiguity)

### Step 1 — Verify Cell 5 in `kaggle-utils.ipynb` on Kaggle ⚠️ CONFIRM FIRST

The notebook file in this repo still shows the old Session 15 sequential Cell 5. The new Cell 5 (subprocess auto-detect via `DILOCO_WORKER_ID` secret) is a **manual Kaggle edit** — the coding agent cannot apply it. Confirm on each account's `kaggle-utils` notebook that Cell 5 reads:

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient
_secrets = UserSecretsClient()
WORKER_ID = _secrets.get_secret("DILOCO_WORKER_ID").strip().upper()
assert WORKER_ID in ("A", "B", "C"), ...
cmd = ("torchrun ... --diloco_mode "
       f"--diloco_worker_id {WORKER_ID} ...")
subprocess.run(cmd, shell=True, check=True)
```

If not yet updated: open each account's `kaggle-utils` → Edit → replace Cell 5 → Save version.

### Step 2 — Run `bootstrap_diloco.py` (once, any machine with HF access)

Seeds `diloco_state/anchor/` from `runs/stage3/checkpoint-0002987` and writes initial `round_state.json`:

```bash
python bootstrap_diloco.py \
  --hf_token "$HF_TOKEN" \
  --repo_id "WeirdRunner/Ouroboros" \
  --source_checkpoint "runs/stage3/checkpoint-0002987" \
  --stage_k 2 \
  --round_n 0 \
  --samples_seen 21728 \
  --completed_stages 0 1
```

Expected Hub output:
```
diloco_state/anchor/adapter_model.safetensors
diloco_state/anchor/adapter_config.json
diloco_state/round_state.json  ← {"stage_k": 2, "round_n": 0, ...}
```

### Step 3 — Start all three workers simultaneously

Trigger `kaggle-utils` manually on all three accounts at the same time. Each reads `DILOCO_WORKER_ID` and launches with `--diloco_worker_id {A,B,C}`. This is the Stage 2 DiLoCo proof-of-concept.

```bash
# Worker A (weirdrunner) — Account A
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit \
  --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 \
  --batch_size 4 --grad_accum 8 --val_batch_size 2 \
  --val_skip_buffer_minutes 60 \
  --session_timeout_hours 12.0 --graceful_exit_buffer_minutes 20 \
  --diloco_mode --diloco_worker_id A \
  --diloco_outer_lr 0.7 \
  --diloco_state_repo WeirdRunner/Ouroboros \
  --diloco_signal_repo deveshpat/Ouroboros \
  --push_to_hub \
  --output_dir runs/diloco
```
(Workers B and C identical except `--diloco_worker_id B/C`)

### Step 4 — Watch coordinator

GitHub Actions fires when workers push `signals/worker_*.json`. Monitor at `github.com/deveshpat/Ouroboros/actions`. After each round: coordinator aggregates, uploads new anchor, triggers next sessions automatically.

### Step 5 — Validate and continue

After Stage 2 completes via DiLoCo, Worker A of Stage 3 runs pre-val (`--diloco_run_val`). If `val_acc` is within 10% of sequential Stage 2 baseline (acc=0.0444), DiLoCo is confirmed. All remaining stages proceed with the same setup.

---

## Part 0.2 — Hub State: What's There, What Matters

```
WeirdRunner/Ouroboros/
  runs/stage3/
    best/                     ← misplaced (stage_1 artifact) — IGNORE
    checkpoint-0002308/       ← misplaced (stage_1 artifact) — IGNORE
    checkpoint-0002987/       ← Stage 2 anchor — SOURCE for bootstrap_diloco.py
    stage_0/best/             ← correct ✓
    stage_1/                  ← correct ✓
  diloco_state/               ← created by bootstrap_diloco.py (Step 2 above)
    anchor/
    round_state.json
    workers/{A,B,C}/          ← written by workers during training
```

The misplaced files are a legacy artifact from Session 15 (path bug fixed in `save_checkpoint()`). No cleanup needed — DiLoCo uses `diloco_state/` prefix entirely.

---

## Part 0.3 — Resolved Decisions

| Decision | Value |
|---|---|
| Model | Jamba Reasoning 3B (`ai21labs/AI21-Jamba-Reasoning-3B`) |
| Fine-tuning | QLoRA (4-bit NF4) + LoRA r=32 |
| LoRA targets | q/k/v/o_proj, in_proj, x_proj, dt_proj, out_proj — conv1d excluded |
| Curriculum K | 10 stages |
| `--max_seq_len` | 1024 |
| `--max_grad_norm` | 0.3 (k≥2 stages) |
| `--session_timeout_hours` | 12.0 (headless wall-clock) |
| `--val_batch_size` | 2 |
| val accuracy samples | 50 |
| `--val_skip_buffer_minutes` | 60 |
| NCCL timeout | `timedelta(hours=4)` |
| `--epochs_per_stage` | 1 |
| `--batch_size` | 4 (2 per GPU on Dual T4) |
| amp_dtype T4 (sm75) | FP16 |
| amp_dtype A100+ (sm80+) | BF16 |
| Gradient checkpointing | Auto-disabled at VRAM≥40GB |
| Multi-account strategy | DiLoCo 3-way parallel |
| Notebook strategy | One `kaggle-utils` per account (no separate `ouroboros-worker-*` notebooks) |
| Worker auto-detection | `DILOCO_WORKER_ID` Kaggle secret per account (`A`/`B`/`C`) |
| Kaggle trigger auth | Per-worker credentials; Kaggle API is owner-authenticated (403 on cross-account) |
| W&B worker run ID | `diloco-{worker_lower}-s{stage_k}` — persists and resumes across rounds within a stage |
| W&B coordinator run ID | `diloco-coordinator-s{stage_k}` |
| W&B step axis | `round_n × shard_step_estimate + local_step` (monotonic across rounds) |
| `shard_step_estimate` | `ceil(36906 / 3 / (batch_size × grad_accum))` = 385 at defaults |
| DiLoCo wandb init timing | Deferred to `run_diloco_worker()` where stage_k/round_n are known |
| Stage advancement (DiLoCo) | When `sum(all workers' samples_seen_this_stage) >= len(train_set)` |
| Val in DiLoCo mode | Worker A only, once per stage (round_n == 0) |
| DiLoCo outer LR | 0.7 (DiLoCo paper default) |
| DiLoCo min_workers | 2 of 3 |
| TRC quota | TPU only — incompatible. Email requesting GPU conversion pending. |

---

## Part 0.4 — DRY Refactors (pending — lower priority than DiLoCo first run)

Apply after DiLoCo Stage 2 is confirmed working. Feed as a separate agent prompt.

| Refactor | Description |
|---|---|
| R1 — Merge token resolution | `_bootstrap_resolve_token()` and `_resolve_hf_token_common()` are identical. Keep one. |
| R2 — Extract latent pass loop | `evaluate_stage()`, `run_generation_callback()`, `_forward_batched_latent()` share the same latent loop. Extract to `_run_latent_passes()`. ✅ Already done in current code. |
| R3 — Cache backbone/embed/lm_head | `_get_backbone()` etc. already use `_cache_model_lookup()`. ✅ Done. |
| R4 — Collapse `_forward_batched_stage0` | Already unified into `_forward_batched_latent`. ✅ Done. |
| R5 — `_ddp_sum()` helper | ✅ Already implemented in current code. |

R2–R5 already done. R1 is the only remaining refactor.

---

## Part 0.5 — Pre-flight Blockers

All resolved.

| Blocker | Resolution |
|---|---|
| `attn_implementation` crash | try/except fallback ✅ |
| `use_mamba_kernels` old TF | `_safe_from_pretrained` retry ✅ |
| `last_hidden_state` None | assert in all forward paths ✅ |
| Graceful session timeout | `make_timeout_checker()` ✅ |
| `conv1d` in LoRA | Excluded ✅ |
| OOM at val | `empty_cache()` + `val_batch_size=2` ✅ |
| Stage 1+ samples filtered by short seq_len | `--max_seq_len 1024` ✅ |
| Exploding gradients k≥2 | `--max_grad_norm 0.3` ✅ |
| mamba-ssm 2.x API break | Pinned to 1.2.2 via git URL ✅ |
| Val at 200 samples too slow | Capped at 50 ✅ |
| NCCL watchdog kills DDP val | `timedelta(hours=4)` + env var ✅ |
| BF16 emulation on T4 | `_amp_dtype` checks `cc >= (8,0)` ✅ |
| GC wastes compute on A100 | Auto-disable at VRAM≥40GB ✅ |
| `_amp_dtype` called in hot loop | `@lru_cache` ✅ |
| Hub upload missing `stage_{k}/` subdir | Fixed in `save_checkpoint()` ✅ |
| Prior-stage `best/` accumulating | `startup_hub_sync_and_prune` prunes them ✅ |
| Sequential relay bottleneck | DiLoCo 3-way parallel ✅ |
| Kaggle `/run` API is owner-authenticated | Per-worker credential pairs as GitHub secrets ✅ |
| Separate worker notebooks create sync overhead | One `kaggle-utils` per account + `DILOCO_WORKER_ID` secret ✅ |
| `global_step_offset` inside wandb guard → `NameError` | Computed unconditionally before wandb block ✅ |
| Pre-val `step=round_n` (tiny int) breaks W&B timeline | `step=global_step_offset` ✅ |
| Pre-val `wandb_run` always None in DiLoCo mode | Uses `diloco_wandb_run` ✅ |
| Two conflicting agent prompts (multiworker + wandb) | Merged into `AGENT_PROMPT_diloco_v2.md` ✅ |

---

## Part 1 — Architecture

### Jamba Reasoning 3B
```
HuggingFace : ai21labs/AI21-Jamba-Reasoning-3B   License: Apache 2.0
Layers      : 28 (26 Mamba + 2 Attention) — 13:1 ratio
Attention   : MQA (20 Q heads, 1 KV head)
Vocab / Context : 64K / 256K tokens
d_model     : 2560
Trainable   : 26,851,328 params (0.88% — LoRA adapters only)
```

### Coconut Curriculum
```
Stage 0:  [Q][S1..Sn][A]              standard CoT; labels on all steps + A
Stage k:  [Q][●*k][S_{k+1}..Sn][A]   first k steps → latent; labels shift right
Stage K:  [Q][●*K][A]                 all steps replaced; labels on A only
K = 10
```

### DGAC (Phase 3.4 only)
```
L_total = L_ce + λ₁(t)·L_ponder + λ₂·L_diversity
L_diversity = mean( Σ_k relu(cos_sim(h_k, h_{k-1}) − τ) ),  τ=0.9
λ₁: 0 for steps 0-200, ramp 0→0.01 over steps 200-500
HaltGate: Linear(2·d_model → 1), zero-init → outputs 0.5 at start
```

---

## Part 2 — Performance Model

```
t_fp16(k) ≈ 34 + 11.5·k  seconds/step  (empirical, Dual T4)
Stage 2: ~52-57s  Stage 3: ~69s  Stage 5: ~92s  Stage 10: ~149s
```

| Mode | Stages 3–10 |
|---|---|
| Sequential relay | ~278h (~3.1 weeks) |
| DiLoCo 3-way parallel | ~93h (~1 week) |
| DiLoCo + A100 (if TRC) | ~19h (~2 days) |

**Per-worker shard:** ~12,302 samples → ~385 steps/worker/stage. Stages 1-9 fit in one 12h session. Stage 10 (~149s/step × 385 = ~16h) needs 2 rounds; coordinator handles transparently.

---

## Part 3 — File Registry

| File | Status |
|---|---|
| `jamba_coconut_finetune.py` | ✅ Complete — all DiLoCo + W&B changes verified |
| `diloco_coordinator.py` | ✅ Complete — per-worker creds, W&B coordinator run, all changes verified |
| `bootstrap_diloco.py` | ✅ Code complete — not yet run (Step 2 above) |
| `.github/workflows/diloco_coordinator.yml` | ✅ Complete — WANDB_KEY + 6 Kaggle secrets verified |
| `kaggle-utils.ipynb` Cell 5 | ⚠️ Manual Kaggle edit — cannot verify from repo context; confirm on each account |
| `prepare_coconut_dataset.py` | ✅ Done |
| `build_wheels_kaggle.py` | ✅ Done |
| `AGENT_PROMPT_diloco_v2.md` | ✅ Delivered — supersedes both predecessor prompts |
| `AGENT_PROMPT_multiworker.md` | ⛔ Superseded — do not use |
| `AGENT_PROMPT_wandb_and_notebook.md` | ⛔ Superseded — do not use |

---

## Part 4 — Open Questions

| Question | Status |
|---|---|
| Stage 2 gn spikes (~1.9 pre-clip): transient? | 🟡 Monitor — CE not diverging at step 2987 |
| Kaggle API `/kernels/{slug}/run` — verify endpoint works | 🟡 Test before first DiLoCo run (Step 3) |
| TRC GPU quota conversion | 🟡 Draft email ready in `terminal_log.md` |
| DGAC halt_step distribution at K≥2 | 🔴 Open — primary research question |
| Prefix re-computation optimization | 🟡 Pre-A100, before Stage 5 if TRC succeeds |
| Cell 5 replaced on all 3 accounts? | 🟡 Verify manually — cannot confirm from repo context |

---

## Part 5 — Hard Lessons

| Lesson | Fix |
|---|---|
| val_batch_size=16 → OOM | `--val_batch_size 2` |
| NCCL watchdog at 60min | `timedelta(hours=4)` + env var |
| `max_seq_len=512` filtered stage 1+ | `--max_seq_len 1024` |
| gn=36.9 at k=2 | `--max_grad_norm 0.3` |
| mamba-ssm 2.x broke fast path | Pinned 1.2.2 via git URL |
| mamba-ssm PyPI sdist is a stub | Must use `git+https://...@v1.2.2` |
| Per-sample loop → 113s/step | Batched forward |
| Val 200 samples → 5.5h | Cap at 50 |
| `is_bf16_supported()` true on T4 (emulation) | `cc >= (8,0)` check |
| GC wastes 20-40% on A100 | Auto-disable at VRAM≥40GB |
| `--push_to_hub` omitted → silent no-op | Always add explicitly |
| Checkpoint pruning per-stage only | `startup_hub_sync_and_prune` at session start |
| Step-time model `34 + 6k` too optimistic | Empirical: `34 + 11.5k` |
| `save_checkpoint` missing `stage_{k}/` in remote path | Fixed |
| "Parallel sessions impossible on Kaggle" | DiLoCo makes it viable |
| TRC assumed GPU access | TPU only — incompatible |
| P100 assumed better than T4 | No Tensor Cores, single GPU — worse |
| Headless sessions capped by quota display | Wall-clock = 12h; `--session_timeout_hours 12.0` |
| Kaggle `/run` API is owner-authenticated | Per-worker credential pairs as GitHub secrets |
| Separate worker notebooks create sync overhead | One `kaggle-utils` per account + `DILOCO_WORKER_ID` Kaggle secret |
| `global_step_offset` inside wandb guard → `NameError` | Computed unconditionally before wandb block |
| Pre-val `step=round_n` (tiny int) breaks W&B timeline | `step=global_step_offset` |
| Pre-val logging used `wandb_run` (always None in DiLoCo) | Uses `diloco_wandb_run` |
| Two conflicting agent prompts defining same constants differently | Merge into one canonical prompt before handing to agent |
