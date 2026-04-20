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
| Stage 2 — 2 latent passes | 🟡 DiLoCo Round 0 active — Workers A+B running (159 steps each) | — |
| Stages 3–10 | ⬜ NOT STARTED | — |
| Phase 3.4 — DGAC | ⬜ after Stage 10 | — |
| Phase 4 — GRPO | ⬜ after DGAC | — |

**Compute mode: DiLoCo 3-way parallel (Worker C quota exhausted; A+B sufficient at min_workers=2)**

---

## Part 0.1 — Immediate Next Steps (strict order)

### Step 1 — Monitor Stage 2 Round 0 completion ⏳

Workers A and B are running ~159 steps each. When both finish:
1. Each pushes `signals/worker_{id}_stage_2_round_0.json` to GitHub
2. GitHub Actions fires `diloco_coordinator.yml`
3. Coordinator finds 2/2 ready workers, aggregates, uploads new anchor, re-triggers A+B for Stage 3

Monitor at `github.com/deveshpat/Ouroboros/actions`.

### Step 2 — Update W&B entity in coordinator ⚠️ REQUIRED

The `--wandb_entity "devesh-patel0922-weirdrunner"` arg in `.github/workflows/diloco_coordinator.yml` may be stale after the W&B account change. Confirm the new entity slug and update:
- `.github/workflows/diloco_coordinator.yml` line: `--wandb_entity`
- Any hardcoded entity in `diloco_coordinator.py`

W&B logging failure is silent (doesn't break training), but fix before Stage 3 begins.

### Step 3 — Validate Stage 2 DiLoCo result

After coordinator completes Round 0, Worker A of Stage 3 runs pre-val automatically (`round_n == 0`, new stage). Target: `val_acc` within 10% of sequential Stage 2 baseline (acc=0.0444 from Stage 1 best). If confirmed, DiLoCo is validated and all remaining stages proceed identically.

### Step 4 — Stages 3–10 run unattended

Coordinator auto-triggers workers after each round. Human intervention only needed if:
- A GitHub Action fails (check Actions tab)
- Worker quota depleted on A or B (C already exhausted)
- Val accuracy collapse at a stage boundary

---

## Part 0.2 — Hub State: What's There, What Matters

```
WeirdRunner/Ouroboros/
  runs/stage3_curriculum/
    stage_0/best/             ← correct ✓
    stage_1/                  ← correct ✓
    stage_2/checkpoint-0002987/  ← Stage 2 sequential anchor (source for bootstrap_diloco.py)
  diloco_state/               ← created by bootstrap_diloco.py ✓
    anchor/                   ← seeded from checkpoint-0002987
    round_state.json          ← {"stage_k": 2, "round_n": 0, ...}
    workers/{A,B,C}/          ← written by workers during training
  runs/stage3/
    best/                     ← misplaced stage_1 artifact — IGNORE
    checkpoint-0002308/       ← misplaced stage_1 artifact — IGNORE
    checkpoint-0002987/       ← misplaced (legacy path bug, now fixed) — IGNORE
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
| W&B worker run ID | `diloco-{worker_lower}-s{stage_k}` — persists and resumes across rounds within a stage |
| W&B coordinator run ID | `diloco-coordinator-s{stage_k}` |
| W&B account | New account created (old account's free trial expired). Entity slug TBD — update `--wandb_entity` in coordinator workflow before Stage 3. |
| W&B step axis | `round_n × shard_step_estimate + local_step` (monotonic across rounds) |
| `shard_step_estimate` | `ceil(36906 / 3 / (batch_size × grad_accum))` = 385 at defaults |
| DiLoCo wandb init timing | Deferred to `run_diloco_worker()` where stage_k/round_n are known |
| Stage advancement (DiLoCo) | When `sum(all workers' samples_seen_this_stage) >= len(train_set)` |
| Val in DiLoCo mode | Worker A only, once per stage (round_n == 0, is_new_stage == True) |
| DiLoCo outer LR | 0.7 (DiLoCo paper default) |
| DiLoCo min_workers | 2 of 3 |
| Timeout clock anchor | `_SCRIPT_START = time.perf_counter()` at **module import**, before `_bootstrap()`. `make_timeout_checker()` uses this value — never resets the clock. |
| DiLoCo shard computation | Subtracts `total_samples_seen[stage_k]` from full dataset before partitioning A/B/C. Ensures partial-stage resumes only process the true remainder. |
| Hub auto-resume (DDP) | Rank 0 resolves and downloads resume checkpoint; path broadcast to all ranks via marker file. `.hub_resume/` cleanup deferred until all code paths are done. |
| Pre-val guard | Auto pre-val runs only when `is_new_stage == True` (i.e. `stage_samples_seen == 0`). Resumed partial stages skip it. |
| TRC quota | TPU only — incompatible. Email requesting GPU conversion sent. |

---

## Part 0.4 — DRY Refactors (all complete)

| Refactor | Status |
|---|---|
| R1 — Merge token resolution (`_bootstrap_resolve_token` / `_resolve_hf_token_common`) | ✅ One function — `_resolve_hf_token_common` |
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
| Prior-stage `best/` accumulating | `startup_hub_sync_and_prune` prunes them ✅ |
| Sequential relay bottleneck | DiLoCo 3-way parallel ✅ |
| Kaggle `/run` API is owner-authenticated | Per-worker credential pairs as GitHub secrets ✅ |
| Separate worker notebooks create sync overhead | One `kaggle-utils` per account + `DILOCO_WORKER_ID` secret ✅ |
| `global_step_offset` inside wandb guard → `NameError` | Computed unconditionally before wandb block ✅ |
| Pre-val `step=round_n` (tiny int) breaks W&B timeline | `step=global_step_offset` ✅ |
| Pre-val `wandb_run` always None in DiLoCo mode | Uses `diloco_wandb_run` ✅ |
| Two conflicting agent prompts (multiworker + wandb) | Merged into `AGENT_PROMPT_diloco_v2.md` ✅ |
| **Timeout clock starts too late** | `_SCRIPT_START` captured at module import before `_bootstrap()` ✅ |
| **DiLoCo sharding ignores stage remainder** | Subtracts `total_samples_seen[stage_k]` before A/B/C split ✅ |
| **Pre-val fires on resumed partial stage** | Guarded by `is_new_stage = (stage_samples_seen == 0)` ✅ |
| **Hub auto-resume not DDP-safe** | Rank 0 resolves/downloads once; path broadcast; cleanup deferred ✅ |
| **subprocess in notebook Cell 5 suppressed output** | Replaced with `!torchrun` magic command; subprocess hid crashes and blocked session kill ✅ |

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
Stage 2: ~50-56s  Stage 3: ~69s  Stage 5: ~92s  Stage 10: ~149s
```
*Session 16 Worker A observed ~49.9s/step at k=2, consistent with model.*

| Mode | Stages 3–10 |
|---|---|
| Sequential relay | ~278h (~3.1 weeks) |
| DiLoCo 3-way parallel | ~93h (~1 week) |
| DiLoCo 2-way (A+B only) | ~139h (~1.5 weeks) — current fallback if C stays exhausted |
| DiLoCo + A100 (if TRC) | ~19h (~2 days) |

**Per-worker shard (full stage):** ~12,302 samples → ~385 steps/worker/stage.
**Stage 2 remainder shard (this round):** ~5,060 samples → ~159 steps/worker. ✅ Confirmed.

---

## Part 3 — File Registry

| File | Status |
|---|---|
| `jamba_coconut_finetune.py` | ✅ Complete — all DiLoCo + W&B + timeout + sharding fixes verified |
| `diloco_coordinator.py` | ✅ Complete — per-worker creds, W&B coordinator run verified; ⚠️ `wandb_entity` may need update |
| `bootstrap_diloco.py` | ✅ Run and confirmed (seeded `diloco_state/anchor/`) |
| `.github/workflows/diloco_coordinator.yml` | ✅ Complete — ⚠️ `--wandb_entity` may need update |
| `kaggle-utils.ipynb` Cell 5 | ✅ Uses `!torchrun` magic command + `DILOCO_WORKER_ID` secret |
| `prepare_coconut_dataset.py` | ✅ Done |
| `build_wheels_kaggle.py` | ✅ Done |

---

## Part 4 — Open Questions

| Question | Status |
|---|---|
| Stage 2 DiLoCo Round 0: do A+B complete within quota? | 🟡 In progress |
| Stage 2 gn spikes (~1.9 pre-clip in Session 15): resolved by DiLoCo reset? | 🟡 Monitor Round 0 gn values |
| W&B entity slug for new account | 🔴 Confirm and update `--wandb_entity` in coordinator workflow |
| TRC GPU quota conversion | 🟡 Email sent — awaiting response |
| DGAC halt_step distribution at K≥2 | 🔴 Open — primary research question |
| Prefix re-computation optimization | 🟡 Pre-A100, before Stage 5 if TRC succeeds |
| Worker C quota: replenishment timeline? | 🟡 Nice-to-have; A+B sufficient for now |

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
| **Timeout clock reset inside `main()`** | `_SCRIPT_START` at module import; `make_timeout_checker()` takes it as param — never restarts |
| **DiLoCo sharding used full dataset, not stage remainder** | Subtract `total_samples_seen[stage_k]` from indices before A/B/C partition |
| **Pre-val fired on every round_n==0, even mid-stage resumes** | `is_new_stage = (stage_samples_seen == 0)` guard added |
| **Hub resume path lost between DDP ranks** | Rank 0 writes resolved path to marker file; all ranks read it; cleanup deferred past all uses |
| **`subprocess.run` in notebook Cell 5 hid all output** | `!torchrun` magic command used instead; subprocess suppressed stdout/stderr, masked crash messages, and held the session open after fatal errors |
