# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in every new session.**
> **Source of truth:** If this doc and `.py`/`.ipynb` files ever disagree, the Python/notebook file wins.
> **DRY rule:** Session details and verbatim logs live in `terminal_log.md` only. This file holds decisions, status, and next actions.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros: latent reasoning injection into Jamba Reasoning 3B (Transformer-Mamba hybrid). The Mamba SSM recurrent state acts as compressed scratch-pad across K latent thought passes, replacing token generation during reasoning. Based on Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — a novel anti-collapse halt gate.

### Current Status (2026-04-21)

| Curriculum Stage | Status | Best val |
|---|---|---|
| Stage 0 — CoT warmup | ✅ COMPLETE | ce=0.4041, acc=0.0222 |
| Stage 1 — 1 latent pass | ✅ COMPLETE | ce=0.4912, acc=0.0444 |
| Stage 2 — 2 latent passes | ⚠️ Round 1 complete (33,533/36,906 samples, ~91%); Round 2 pending — trigger workers manually (one last time) | — |
| Stages 3–10 | ⬜ NOT STARTED | — |
| Phase 3.4 — DGAC | ⬜ after Stage 10 | — |
| Phase 4 — GRPO | ⬜ after DGAC | — |

**Compute mode: DiLoCo dynamic (Worker C quota exhausted; A+B active)**

---

## Part 0.1 — Immediate Next Steps (strict order)

### Step 1 — Run workers manually for Stage 2 Round 2 ⚡ BLOCKING (last manual round)
Open kaggle-utils on Worker A and Worker B accounts and run Cell 5 manually, in parallel.
`round_state.json` is now `{stage_k: 2, round_n: 2}`. Each worker trains on ~562 samples (~35 steps).
After both signal files are pushed, coordinator auto-fires. **This is the round that tests the patched `kernels push` trigger.** Watch Actions logs for:
```
[coordinator] Triggered Worker A: weirdrunner/kaggle-utils  (...)
[coordinator] Triggered Worker B: weirdrunner007/kaggle-utils  (...)
```

### Step 2 — Deploy Dynamic DiLoCo patch ⚡ IMPORTANT
Feed the agent prompt (Part 6) to the coding agent. Changes: `diloco_coordinator.py`, `jamba_coconut_finetune.py`, `diloco_coordinator.yml`. Then push to `deveshpat/Ouroboros`.
This patch makes the system fully self-managing after round 2. Stage 2 will auto-close (~6 more rounds), then stage 3 begins automatically.

### Step 3 — Verify auto-trigger patch after Round 2 completes
After Round 2 coordinator runs, confirm in Actions that the new `kernels push` trigger worked. If a `WARNING: kernels push failed` appears, the likely cause is the `--accelerator` flag string. Fix: remove `--kaggle_accelerator` from the coordinator workflow YAML (GPU is already requested via `enable_gpu: true` in kernel metadata JSON).

---

## Part 0.2 — Hub State: What's There, What Matters

```
WeirdRunner/Ouroboros/
  runs/stage3_curriculum/
    stage_0/best/             ← correct ✓
    stage_1/                  ← correct ✓
    stage_2/checkpoint-0002987/  ← Stage 2 sequential anchor (pre-DiLoCo)
  diloco_state/
    anchor/                   ← Updated after Round 1 aggregation ✓
    round_state.json          ← {"stage_k": 2, "round_n": 2}  ← current ✓
    workers/A/                ← Stage 2 Round 1 weights ✓
    workers/B/                ← Stage 2 Round 1 weights ✓
  runs/stage3/
    best/ checkpoint-0002308/ checkpoint-0002987/  ← legacy path — IGNORE
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
| Multi-account strategy | DiLoCo dynamic parallel |
| Notebook launch cells | `!torchrun` magic commands only |
| Worker auto-detection | `DILOCO_WORKER_ID` Kaggle secret per account (`A`/`B`/`C`) |
| Kaggle trigger auth | Per-worker credentials; owner-authenticated |
| **Kaggle trigger mechanism** | Local `kernels push`: stage checked-in `kaggle-utils.ipynb` + generated `kernel-metadata.json` → `kaggle kernels push -p <tmpdir>`. No pull needed. ✅ |
| **Kaggle SDK version** | Pinned `kaggle==1.6.17`. ✅ |
| W&B worker run ID | `diloco-{worker_lower}-s{stage_k}` — persists across rounds within a stage |
| W&B coordinator run ID | `diloco-coordinator-s{stage_k}` |
| W&B step axis | Monotonic: `round_n × (shard_step_estimate + 1) + local_step` ✅ |
| `shard_step_estimate` | `ceil(36906 / 3 / (batch_size × grad_accum))` = 385 at defaults |
| DiLoCo outer LR | 0.7 (diloco mode) / 1.0 effective (solo mode = direct promotion) |
| **`min_workers`** | **REMOVED** — superseded by `min_shard_samples` dynamic logic |
| **`min_shard_samples`** | **32** (1 optimizer step = batch_size × grad_accum). Workers projected below this are not triggered. |
| **Solo mode** | 1 active worker → direct weight promotion (skip outer update). Cleaner than blended update with stale anchor. |
| **Stage close** | remaining < `min_shard_samples` → declare stage complete, advance. Max error: 32/36906 ≈ 0.09%. |
| **Coordinator planning** | Coordinator computes projected shards before triggering. Only active workers (shard ≥ min_shard_samples) get triggered. `round_state.json` stores `triggered_workers` + `mode`. |
| **`insufficient_worker_rounds` streak** | REMOVED — obsolete with dynamic worker selection. |
| **`workflow_dispatch` inputs** | `force_worker_ids` (e.g. "A,B"), `skip_trigger` (aggregate only), `dry_run` (plan only). Full control from Actions UI. |
| DiLoCo wandb init timing | Deferred to `run_diloco_worker()` where stage_k/round_n are known |
| Stage advancement (DiLoCo) | When `sum(all workers' samples_seen_this_stage) >= len(train_set)` OR `remaining < min_shard_samples` |
| Val in DiLoCo mode | Worker A only, once per stage (round_n == 0, is_new_stage == True) |
| Timeout clock anchor | `_SCRIPT_START = time.perf_counter()` at **module import** |
| DiLoCo shard computation | Subtracts `total_samples_seen[stage_k]` before split |
| **GITHUB_TOKEN scope** | Classic PAT from `deveshpat` account with `repo` scope. Updated on all worker accounts. ✅ |
| **`numpy` in coordinator** | Installed. ✅ |
| **Coordinator torch** | CPU-only wheel. ✅ |
| **Coordinator timeout** | 30 minutes. ✅ |

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

## Part 0.5 — Pre-flight Blockers

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
| mamba-ssm 2.x API break | Pinned to 1.2.2 ✅ |
| NCCL watchdog kills DDP val | `timedelta(hours=4)` + env var ✅ |
| BF16 emulation on T4 | `_amp_dtype` checks `cc >= (8,0)` ✅ |
| **W&B step collision between rounds** | `round_step_span = shard_step_estimate + 1` ✅ |
| **Kaggle `kernels pull` → 403** | Replaced with local `kernels push` ✅ |
| **`kaggle==2.0.x` gRPC → 403** | Pinned `kaggle==1.6.17` ✅ |
| **Coordinator crashes on single worker** | Dynamic worker selection + solo mode ✅ (in progress) |
| **Stage never closes with tiny remainder** | `min_shard_samples` close-out logic ✅ (in progress) |
| **Workers triggered unnecessarily** | Coordinator pre-computes projected shards ✅ (in progress) |
| **No manual trigger granularity** | `workflow_dispatch` inputs: `force_worker_ids`, `skip_trigger`, `dry_run` ✅ (in progress) |

---

## Part 1 — Architecture

### Jamba Reasoning 3B
```
HuggingFace : ai21labs/AI21-Jamba-Reasoning-3B   License: Apache 2.0
Layers      : 28 (26 Mamba + 2 Attention) — 13:1 ratio
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

### Dynamic DiLoCo Protocol (new)
```
Each round:
  Coordinator
    1. remaining = total_train - total_samples_seen[stage_k]
    2. if remaining < min_shard_samples → advance stage
    3. projected_shard[w] = _partition_contiguous_range(remaining, 3, w_idx)
    4. active = [w for w if projected_shard[w] >= min_shard_samples AND w has creds]
    5. mode = "complete"|"solo"|"diloco"
    6. Aggregate previous round (solo → direct promotion, diloco → weighted avg)
    7. Trigger only active workers
    8. Write round_state.json with triggered_workers, mode, projected_shards

  Workers (triggered subset only)
    - Train on shard, upload weights + status.json, push GitHub signal
    - On empty shard (safety net): push signal with samples_seen=0, upload passthrough status
```

### DGAC (Phase 3.4 only)
```
L_total = L_ce + λ₁(t)·L_ponder + λ₂·L_diversity
HaltGate: Linear(2·d_model → 1), zero-init
```

---

## Part 2 — Performance Model

```
Stage 1: ~41s/step  Stage 2: ~48–53s/step  Stage 3: ~69s/step
Stage 5: ~92s/step  Stage 10: ~149s/step
```

**Stage 2 remaining rounds (from round 2):**

| Round | Remaining | A shard | B shard | New total seen | Done? |
|---|---|---|---|---|---|
| 2 (manual) | 3,373 | 1,125 | 1,124 | 35,471 | No |
| 3 | 1,435 | 479 | 478 | 36,428 | No |
| 4 | 478 | 160 | 159 | 36,747 | No |
| 5 | 159 | 53 | 53 | 36,853 | No |
| 6 | 53 | 18 | 18 | 36,889 | No |
| 7 | 17 | 6 | 6 | 36,901 | No |
| 8 | 5 | ≤32 total | — | — | **Close-out** ✅ |

Round 8: remaining (5) < min_shard_samples (32) → coordinator declares stage complete, advances to stage 3.

---

## Part 3 — File Registry

| File | Status |
|---|---|
| `jamba_coconut_finetune.py` | ✅ Complete + W&B step fix + empty-shard passthrough signal |
| `diloco_coordinator.py` | ⚠️ Dynamic DiLoCo patch in progress |
| `bootstrap_diloco.py` | ✅ Run and confirmed |
| `.github/workflows/diloco_coordinator.yml` | ⚠️ `workflow_dispatch` inputs patch in progress |
| `kaggle-utils.ipynb` Cell 5 | ✅ `!torchrun` magic + `DILOCO_WORKER_ID` secret |
| `prepare_coconut_dataset.py` | ✅ Done |
| `build_wheels_kaggle.py` | ✅ Done |

---

## Part 4 — Open Questions

| Question | Status |
|---|---|
| Stage 2 Round 2 auto-trigger: does patched `kernels push` work? | 🟡 Pending — testing after manual round 2 run |
| Stage 2 DiLoCo: does aggregated model match sequential baseline? | 🟡 Pre-val by Worker A at Stage 3 start |
| TRC GPU quota conversion | 🟡 Email sent — awaiting response |
| DGAC halt_step distribution at K≥2 | 🔴 Open — primary research question |
| Worker C quota: replenishment timeline? | 🟡 Nice-to-have; A+B sufficient |
| PyTorch 2.9 `use_reentrant` warning | 🟡 Minor — add `gradient_checkpointing_kwargs={"use_reentrant": True}` |

---

## Part 5 — Hard Lessons

*(Appending new lessons; see git history for full list)*

| Lesson | Fix |
|---|---|
| `kaggle kernels pull` → 403 in CI | Use local `kernels push` instead; pull is download-only |
| W&B step collision between rounds | `round_step_span = shard_step_estimate + 1` |
| Fixed `min_workers` causes deadlock when B has empty shard | Dynamic `min_shard_samples` pre-computation |
| Stage never closes with geometric remainder | `remaining < min_shard_samples` → declare stage complete |
| Coordinator triggers all workers even when some have nothing to do | Pre-compute projected shards, trigger only active workers |
| Solo mode with outer_lr=0.7 blends stale anchor into new weights | Direct weight promotion in solo mode (outer_lr=1.0 equivalent) |

---

## Part 6 — Agent Prompt: Dynamic DiLoCo

See separate file: `agent_prompt_dynamic_diloco.md`
