# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in every new session.**
> **Source of truth:** If this doc and `.py`/`.ipynb` files ever disagree, the Python/notebook file wins.
> **DRY rule:** Session details and verbatim logs live in `terminal_log.md` only. This file holds decisions, status, and next actions.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros: latent reasoning injection into Jamba Reasoning 3B (Transformer-Mamba hybrid). The Mamba SSM recurrent state acts as compressed scratch-pad across K latent thought passes, replacing token generation during reasoning. Based on Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — a novel anti-collapse halt gate.

### Current Status (2026-04-20)

| Curriculum Stage | Status | Best val |
|---|---|---|
| Stage 0 — CoT warmup | ✅ COMPLETE | ce=0.4041, acc=0.0222 |
| Stage 1 — 1 latent pass | ✅ COMPLETE | ce=0.4912, acc=0.0444 |
| Stage 2 — 2 latent passes | 🟡 DiLoCo Round 0 running — Workers A+B at ~100/159 steps | — |
| Stages 3–10 | ⬜ NOT STARTED | — |
| Phase 3.4 — DGAC | ⬜ after Stage 10 | — |
| Phase 4 — GRPO | ⬜ after DGAC | — |

**Compute mode: DiLoCo 3-way parallel (Worker C quota exhausted; A+B sufficient at min_workers=2)**

---

## Part 0.1 — Immediate Next Steps (strict order)

### Step 1 — Wait for Stage 2 Round 0 to finish ⏳
Workers A and B are at ~100/159 steps. Both finish within the same ~20min window. Monitor `github.com/deveshpat/Ouroboros/actions` — coordinator fires when both signal files arrive.

### Step 2 — Verify coordinator fires correctly
Look for in the Actions log:
```
[coordinator] Worker A: NNNN samples ready
[coordinator] Worker B: NNNN samples ready
[coordinator] Aggregating on CPU...
[coordinator] New anchor uploaded: ...
[coordinator] round_state.json updated: stage=3 round=0
[coordinator] Triggered Worker A: weirdrunner/kaggle-utils
[coordinator] Triggered Worker B: weirdrunner007/kaggle-utils
```

### Step 3 — If auto-trigger fails, start workers manually
The aggregation step is independent of the trigger. If `Triggered Worker` lines show errors, just open `kaggle-utils` on Account A and B and run manually. `round_state.json` will already have `stage=3 round=0`.

### Step 4 — Stages 3–10 run unattended after that

---

## Part 0.2 — Hub State: What's There, What Matters

```
WeirdRunner/Ouroboros/
  runs/stage3_curriculum/
    stage_0/best/             ← correct ✓
    stage_1/                  ← correct ✓
    stage_2/checkpoint-0002987/  ← Stage 2 sequential anchor (source for bootstrap_diloco.py)
  diloco_state/
    anchor/                   ← seeded from checkpoint-0002987 ✓
    round_state.json          ← {"stage_k": 2, "round_n": 0, ...}
    workers/{A,B}/            ← written by workers during training
  runs/stage3/
    best/ checkpoint-0002308/ checkpoint-0002987/  ← legacy path artifacts — IGNORE
```

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
| Notebook launch cells | `!torchrun` magic commands only — never `subprocess.run`. Magic commands stream output to notebook in real time; subprocess suppresses it, hides crashes, and prevented session-kill on fatal errors. |
| Worker auto-detection | `DILOCO_WORKER_ID` Kaggle secret per account (`A`/`B`/`C`) |
| Kaggle trigger auth | Per-worker credentials; Kaggle API is owner-authenticated (403 on cross-account) |
| Kaggle trigger mechanism | SDK pull → re-push (`kernels_pull` + `kernels_push`). No standalone `/run` endpoint exists. |
| W&B worker run ID | `diloco-{worker_lower}-s{stage_k}` — persists and resumes across rounds within a stage |
| W&B coordinator run ID | `diloco-coordinator-s{stage_k}` |
| W&B entity | `default=None` — auto-resolved from API key. Explicit entity arg removed. |
| W&B account | New account (old trial expired). Entity auto-resolved; no hardcoding needed. |
| W&B step axis | `round_n × shard_step_estimate + local_step` (monotonic across rounds) |
| `shard_step_estimate` | `ceil(36906 / 3 / (batch_size × grad_accum))` = 385 at defaults |
| DiLoCo wandb init timing | Deferred to `run_diloco_worker()` where stage_k/round_n are known |
| Stage advancement (DiLoCo) | When `sum(all workers' samples_seen_this_stage) >= len(train_set)` |
| Val in DiLoCo mode | Worker A only, once per stage (round_n == 0, is_new_stage == True) |
| DiLoCo outer LR | 0.7 (DiLoCo paper default) |
| DiLoCo min_workers | 2 of 3 |
| Timeout clock anchor | `_SCRIPT_START = time.perf_counter()` at **module import**, before `_bootstrap()`. |
| DiLoCo shard computation | Subtracts `total_samples_seen[stage_k]` before A/B/C split. |
| Hub auto-resume (DDP) | Rank 0 resolves; path broadcast via marker file; `.hub_resume/` cleanup deferred. |
| Pre-val guard | Only when `is_new_stage == True` (i.e. `stage_samples_seen == 0`). |
| TRC quota | TPU only — incompatible. Email requesting GPU conversion sent. |

---

## Part 0.4 — DRY Refactors (all complete)

| Refactor | Status |
|---|---|
| R1 — Merge token resolution | ✅ `_resolve_hf_token_common` |
| R2 — Extract latent pass loop | ✅ `_run_latent_passes()` |
| R3 — Cache backbone/embed/lm_head | ✅ `_cache_model_lookup()` |
| R4 — Collapse `_forward_batched_stage0` | ✅ Unified into `_forward_batched_latent` |
| R5 — `_ddp_sum()` helper | ✅ Implemented |

---

## Part 0.5 — Pre-flight Blockers (all resolved)

| Blocker | Resolution |
|---|---|
| `attn_implementation` crash | try/except fallback ✅ |
| `use_mamba_kernels` old TF | `_safe_from_pretrained` retry ✅ |
| `last_hidden_state` None | assert in all forward paths ✅ |
| Graceful session timeout | `make_timeout_checker()` using `_SCRIPT_START` ✅ |
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
| Prior-stage `best/` accumulating | `startup_hub_sync_and_prune` ✅ |
| Sequential relay bottleneck | DiLoCo 3-way parallel ✅ |
| Kaggle `/run` endpoint doesn't exist | SDK pull → re-push via `kernels_pull` + `kernels_push` ✅ |
| Kaggle trigger is owner-authenticated | Per-worker credential pairs as GitHub secrets ✅ |
| `--wandb_entity` pointing at dead account | Removed; `entity=None` auto-resolves from API key ✅ |
| `global_step_offset` inside wandb guard → `NameError` | Computed unconditionally before wandb block ✅ |
| Pre-val `step=round_n` breaks W&B timeline | `step=global_step_offset` ✅ |
| Pre-val `wandb_run` always None in DiLoCo | Uses `diloco_wandb_run` ✅ |
| **Timeout clock starts too late** | `_SCRIPT_START` at module import ✅ |
| **DiLoCo sharding ignores stage remainder** | Subtracts `total_samples_seen[stage_k]` ✅ |
| **Pre-val fires on resumed partial stage** | `is_new_stage = (stage_samples_seen == 0)` ✅ |
| **Hub auto-resume not DDP-safe** | Rank 0 resolves; path broadcast; deferred cleanup ✅ |
| **subprocess in notebook suppressed output** | `!torchrun` magic command ✅ |

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
t_fp16(k) ≈ 34 + 11.5·k  seconds/step  (theoretical, Dual T4)
Empirical Session 16: Worker A 48.3s/step, Worker B 53.4s/step at k=2 (~10% faster than model)
Stage 1: ~41s  Stage 2: ~48–54s  Stage 3: ~69s  Stage 5: ~92s  Stage 10: ~149s
```

| Mode | Stages 3–10 |
|---|---|
| Sequential relay | ~278h (~3.1 weeks) |
| DiLoCo 3-way parallel | ~93h (~1 week) |
| DiLoCo 2-way (A+B only) | ~139h (~1.5 weeks) — current fallback |
| DiLoCo + A100 (if TRC) | ~19h (~2 days) |

**Per-worker shard (full stage):** ~12,302 samples → ~385 steps/worker/stage.
**Stage 2 remainder shard:** ~5,060 samples → ~159 steps/worker. ✅ Confirmed empirically.

---

## Part 3 — File Registry

| File | Status |
|---|---|
| `jamba_coconut_finetune.py` | ✅ Complete |
| `diloco_coordinator.py` | ✅ Complete — Kaggle trigger fixed (pull→push); `wandb_entity` removed |
| `bootstrap_diloco.py` | ✅ Run and confirmed |
| `.github/workflows/diloco_coordinator.yml` | ✅ Complete — `wandb_entity` removed; `kaggle` added to pip install |
| `kaggle-utils.ipynb` Cell 5 | ✅ `!torchrun` magic + `DILOCO_WORKER_ID` secret |
| `prepare_coconut_dataset.py` | ✅ Done |
| `build_wheels_kaggle.py` | ✅ Done |

---

## Part 4 — Open Questions

| Question | Status |
|---|---|
| Stage 2 DiLoCo Round 0: do A+B complete and coordinator fire? | 🟡 ~100/159 steps done |
| Stage 2 DiLoCo: does aggregated model match sequential baseline? | 🟡 Pre-val by Worker A at Stage 3 start |
| TRC GPU quota conversion | 🟡 Email sent — awaiting response |
| DGAC halt_step distribution at K≥2 | 🔴 Open — primary research question |
| Prefix re-computation optimization | 🟡 Before Stage 5 if TRC succeeds |
| Worker C quota: replenishment timeline? | 🟡 Nice-to-have; A+B sufficient |
| PyTorch 2.9 `use_reentrant` warning | 🟡 Minor cleanup — add `gradient_checkpointing_kwargs={"use_reentrant": True}` |

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
| Step-time model `34 + 6k` too optimistic | `34 + 11.5k`; empirical confirms workers run ~10% faster |
| `save_checkpoint` missing `stage_{k}/` in remote path | Fixed |
| "Parallel sessions impossible on Kaggle" | DiLoCo makes it viable |
| TRC assumed GPU access | TPU only — incompatible |
| P100 assumed better than T4 | No Tensor Cores, single GPU — worse |
| Headless sessions capped by quota display | Wall-clock = 12h; `--session_timeout_hours 12.0` |
| Kaggle `/run` API does not exist | Pull → re-push via official SDK |
| Kaggle trigger is owner-authenticated | Per-worker credential pairs |
| Wandb entity hardcoded to stale account | Remove `--wandb_entity`; let SDK auto-resolve |
| `global_step_offset` inside wandb guard → `NameError` | Computed unconditionally before wandb block |
| Pre-val `step=round_n` breaks W&B timeline | `step=global_step_offset` |
| Two conflicting agent prompts | Merge into one canonical prompt |
| **Timeout clock reset inside `main()`** | `_SCRIPT_START` at module import |
| **DiLoCo sharding used full dataset** | Subtract `total_samples_seen[stage_k]` |
| **Pre-val fired on resumed partial stage** | `is_new_stage = (stage_samples_seen == 0)` |
| **Hub resume path lost between DDP ranks** | Rank 0 resolves, broadcasts, cleanup deferred |
| **`subprocess.run` hid output** | `!torchrun` magic command |
