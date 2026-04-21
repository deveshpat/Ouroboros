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
| Stage 2 — 2 latent passes | ⚠️ Round 3 complete (~36,531/36,906 samples); Round 4 stalled (Worker C deadlock — apply attendance patch first) | — |
| Stages 3–10 | ⬜ NOT STARTED | — |
| Phase 3.4 — DGAC | ⬜ after Stage 10 | — |
| Phase 4 — GRPO | ⬜ after DGAC | — |

**Compute mode: DiLoCo dynamic (Worker C quota exhausted; A+B active. C managed via attendance mechanism.)**

---

## Part 0.1 — Immediate Next Steps (strict order)

### Step 1 — Apply Worker Attendance & Timeout patch ⚡ BLOCKING
Feed `agent_prompt_attendance_mechanism.md` (Part 7) to the coding agent.
Modifies: `diloco_coordinator.py`, `jamba_coconut_finetune.py`, `.github/workflows/diloco_coordinator.yml`.
Push to `deveshpat/Ouroboros`.

### Step 2 — One-time `round_state.json` fix ⚡ BLOCKING
Manually update `diloco_state/round_state.json` on `WeirdRunner/Ouroboros`:
- Move `"C"` from `triggered_workers` → `attendance_workers`
- Set `triggered_at: 0` (so coordinator doesn't immediately timeout on first run)

```json
{
  "stage_k": 2, "round_n": 4,
  "triggered_workers": ["A", "B"],
  "attendance_workers": ["C"],
  "triggered_at": 0
}
```

### Step 3 — Trigger `workflow_dispatch` with `skip_trigger=false`
Let the patched coordinator aggregate round 3 weights and kick off round 4 automatically.
A+B train normally; C is pinged. Watch Actions for:
```
[coordinator] Triggering training: ['A', 'B']  attendance: ['C']
```

### Step 4 — Stage 2 self-completes (~2 more rounds after round 4)
Rounds 4–6 run fully automatically. Coordinator closes Stage 2 when remaining < min_shard_samples per worker.

---

## Part 0.2 — Hub State: What's There, What Matters

```
WeirdRunner/Ouroboros/
  runs/stage3_curriculum/
    stage_0/best/             ← correct ✓
    stage_1/                  ← correct ✓
    stage_2/checkpoint-0002987/  ← Stage 2 sequential anchor (pre-DiLoCo)
  diloco_state/
    anchor/                   ← Updated after Round 3 aggregation ✓
    round_state.json          ← {"stage_k": 2, "round_n": 4, triggered_workers: ["A","B","C"]} ← BROKEN, fix per Step 2
    workers/A/                ← Stage 2 Round 3 weights ✓
    workers/B/                ← Stage 2 Round 3 weights ✓
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
| **Solo mode** | 1 active worker → direct weight promotion (skip outer update). |
| **Stage close** | remaining < `min_shard_samples` per active worker → declare stage complete, advance. |
| **Coordinator planning** | Coordinator computes projected shards before triggering. Only active workers (shard ≥ min_shard_samples) get triggered. `round_state.json` stores `triggered_workers` + `mode`. |
| **`workflow_dispatch` inputs** | `force_worker_ids`, `skip_trigger`, `dry_run`. Full control from Actions UI. |
| DiLoCo wandb init timing | Deferred to `run_diloco_worker()` where stage_k/round_n are known |
| Stage advancement (DiLoCo) | When `sum(all workers' samples_seen_this_stage) >= len(train_set)` OR `remaining < min_shard_samples` |
| Val in DiLoCo mode | Worker A only, once per stage (round_n == 0, is_new_stage == True) |
| **Worker timeout threshold** | **13h** (Kaggle 12h hard wall + 1h grace). Set via `--worker_timeout_hours`. |
| **`triggered_at` field** | Written to `round_state.json` on every worker dispatch. Used to compute timeout deadline. Reset on re-dispatch in waiting mode. |
| **Attendance round** | Worker in `attendance_workers` → skips training, downloads anchor, uploads `status.json` with `samples_seen=0`, pushes signal. Proves quota active. |
| **Attendance promotion** | Worker responds to attendance → added to `eligible_for_training` on NEXT round. `round_n` advances normally. |
| **Waiting mode** | All credentialed workers in `attendance_workers`, none training. `round_n` frozen. Exited by any worker signal push OR manual `workflow_dispatch`. |
| **Worker C permanent exclusion** | NOT needed. Attendance mechanism handles it: C stays in `attendance_workers` until quota renews, then auto-promotes. No config changes required. |
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
| W&B step collision between rounds | `round_step_span = shard_step_estimate + 1` ✅ |
| `kaggle kernels pull` → 403 | Replaced with local `kernels push` ✅ |
| `kaggle==2.0.x` gRPC → 403 | Pinned `kaggle==1.6.17` ✅ |
| `kaggle kernels push --accelerator` → unrecognized arg | Removed flag; GPU requested via `enable_gpu: true` in metadata JSON ✅ |
| Coordinator stalls on quota-dead worker | `triggered_at` + 13h timeout → attendance mechanism ✅ (patch in progress) |
| Worker C deadlock (round 4) | One-time `round_state.json` fix (Step 2) + attendance patch ✅ (in progress) |

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

### Dynamic DiLoCo Protocol
```
Each round:
  Coordinator
    1. remaining = total_train - total_samples_seen[stage_k]
    2. if remaining < min_shard_samples (per active worker) → advance stage
    3. projected_shard[w] = _partition_contiguous_range(remaining, 3, w_idx)
    4. active = [w for w if projected_shard[w] >= min_shard_samples AND w has creds AND w not in attendance_workers]
    5. mode = "complete"|"solo"|"diloco"|"waiting"
    6. Aggregate previous round (solo → direct promotion, diloco → weighted avg)
    7. Trigger active workers (training) + attendance_workers (ping-only)
    8. Write round_state.json with triggered_workers, attendance_workers, mode, triggered_at

  Workers (triggered subset)
    - If in attendance_workers → download anchor, upload status(samples=0), push signal, return
    - Else → train on shard, upload weights + status, push signal

  Timeout (coordinator, next run after 13h)
    - Non-responsive triggered workers → demote to attendance_workers
    - All in attendance → mode = "waiting", round_n frozen
    - Any attendance worker signals → promote to training next round
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

**Stage 2 remaining rounds (from round 4, A+B only, C in attendance):**

| Round | Remaining | A shard | B shard | New total seen | Done? |
|---|---|---|---|---|---|
| 4 (current) | 375 | 125 | 125 | 36,781 | No |
| 5 | 125 | 42 | 42 | 36,865 | No |
| 6 | 41 | all < 32 | — | — | **Close-out** ✅ |

Round 6: remaining (41) → projected per worker ~14 < min_shard_samples (32) → coordinator declares stage complete, advances to Stage 3. Max samples missed: 41/36906 ≈ 0.11%.

---

## Part 3 — File Registry

| File | Status |
|---|---|
| `jamba_coconut_finetune.py` | ⚠️ Attendance check patch pending (Part 7) |
| `diloco_coordinator.py` | ⚠️ Attendance + timeout patch pending (Part 7) |
| `bootstrap_diloco.py` | ✅ Run and confirmed |
| `.github/workflows/diloco_coordinator.yml` | ⚠️ `--worker_timeout_hours` arg pending (Part 7) |
| `kaggle-utils.ipynb` Cell 5 | ✅ `!torchrun` magic + `DILOCO_WORKER_ID` secret |
| `prepare_coconut_dataset.py` | ✅ Done |
| `build_wheels_kaggle.py` | ✅ Done |

---

## Part 4 — Open Questions

| Question | Status |
|---|---|
| Stage 2 DiLoCo: does aggregated model match sequential baseline? | 🟡 Pre-val by Worker A at Stage 3 start |
| TRC GPU quota conversion | 🟡 Email sent — awaiting response |
| DGAC halt_step distribution at K≥2 | 🔴 Open — primary research question |
| Worker C quota: replenishment timeline? | 🟡 Attendance mechanism handles automatically when renewed |
| PyTorch 2.9 `use_reentrant` warning | 🟡 Minor — add `gradient_checkpointing_kwargs={"use_reentrant": True}` |

---

## Part 5 — Hard Lessons

| Lesson | Fix |
|---|---|
| `kaggle kernels pull` → 403 in CI | Use local `kernels push` instead |
| W&B step collision between rounds | `round_step_span = shard_step_estimate + 1` |
| Fixed `min_workers` causes deadlock when B has empty shard | Dynamic `min_shard_samples` pre-computation |
| Stage never closes with geometric remainder | `remaining < min_shard_samples` → declare stage complete |
| Coordinator triggers all workers even when some have nothing to do | Pre-compute projected shards, trigger only active workers |
| Solo mode with outer_lr=0.7 blends stale anchor into new weights | Direct weight promotion in solo mode |
| `kaggle kernels push --accelerator NvidiaTeslaT4` → unrecognized argument | Remove `--accelerator` flag; GPU is already set via `enable_gpu: true` in kernel metadata JSON |
| Worker C in `triggered_workers` with exhausted quota → coordinator stalls forever | `triggered_at` + 13h timeout + attendance mechanism |
| Removing dead worker's credentials is a temporary fix only | Attendance mechanism: permanent, self-healing, zero config on quota renewal |

---

## Part 6 — Agent Prompt: Dynamic DiLoCo

See separate file: `agent_prompt_dynamic_diloco.md`

---

## Part 7 — Agent Prompt: Worker Attendance & Timeout Mechanism

See separate file: `agent_prompt_attendance_mechanism.md`

Changes: `diloco_coordinator.py`, `jamba_coconut_finetune.py`, `.github/workflows/diloco_coordinator.yml`

Key additions:
- `triggered_at` timestamp written on every worker dispatch
- `attendance_workers` list in `round_state.json`
- 13h timeout → demote non-responsive triggered workers to attendance
- Attendance round: worker proves quota active with 0-sample ping, no training
- Waiting mode: `round_n` frozen, coordinator re-pings on manual dispatch or incoming signal
- Fully self-healing: Worker C auto-promotes when quota renews, no config changes needed
