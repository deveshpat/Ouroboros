# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in every new session.**
> **Source of truth:** If this doc and `.py`/`.ipynb` files ever disagree, the Python/notebook file wins.
> **DRY rule:** Session details and verbatim logs live in `terminal_log.md` only. This file holds decisions, status, and next actions.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros: latent reasoning injection into Jamba Reasoning 3B (Transformer-Mamba hybrid). The Mamba SSM recurrent state acts as compressed scratch-pad across K latent thought passes, replacing token generation during reasoning. Based on Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — a novel anti-collapse halt gate.

### Current Status (2026-04-22)

| Curriculum Stage | Status | Best val |
|---|---|---|
| Stage 0 — CoT warmup | ✅ COMPLETE | ce=0.4041, acc=0.0222 |
| Stage 1 — 1 latent pass | ✅ COMPLETE | ce=0.4912, acc=0.0444 |
| Stage 2 — 2 latent passes | ✅ COMPLETE (6 rounds, ~36,865 samples) | — |
| Stage 3 — 3 latent passes | ⚠️ Round 0 ✅ (24,604 samples, 66.7%); Round 1 in progress — B cancelled (P100 GPU, needs fix below); A status unknown | ce=0.6087 (pre-val), acc=0.0000 |
| Stages 4–10 | ⬜ NOT STARTED | — |

**Compute mode: DiLoCo dynamic (Worker C quota exhausted; A+B active. C in attendance.)**

---

## Part 0.1 — Immediate Next Steps (strict order)

### Step 1 — Push GPU + W&B fixes ⚡ BLOCKING
Two surgical patches — use `agent_prompt_gpu_wandb_fixes.md`.

**Fix A — P100 → T4 (`diloco_coordinator.py`):**
In `_build_kaggle_kernel_metadata()`, add one line:
```python
"accelerator": "nvidiaTeslaT4",   # pins T4 explicitly in kernel-metadata.json
```
This is a `kernel-metadata.json` field — not the old `--accelerator` CLI flag.

**Fix B — W&B per-round grouping (`jamba_coconut_finetune.py`):**
In `run_diloco_worker()`, change the W&B init:
```python
# id:   diloco-a-s3-r1  (was: diloco-a-s3)
# group: diloco-a-s3    (new field)
# name: Worker A | Stage 3 | Round 1  (was: Worker A | Stage 3)
# remove: resume="allow"
```

### Step 2 — Wait for next coordinator run (≤30 min)
The coordinator will detect Worker B as not ready for round 1 and re-dispatch.
With the P100 fix deployed, B should now get T4.

### Step 3 — Stage 3 self-completes
Round 1: A+B train ~4101 samples each. Round 2: ~0 or trivial close-out.

---

## Part 0.2 — Hub State

```
WeirdRunner/Ouroboros/
  diloco_state/
    anchor/                   ← Stage 3 Round 0 aggregate ✓
    round_state.json          ← stage_k=3, round_n=1, triggered_workers=["A","B"], triggered_at=<live>
    workers/A/round_0000_stage_3/  ✓
    workers/B/round_0000_stage_3/  ✓
    workers/A/round_0001_stage_3/  ⚠️ status unknown (may have completed on T4)
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
| **`accelerator` in kernel-metadata.json** | **`"nvidiaTeslaT4"` — JSON field, pins T4. Distinct from the old `--accelerator` CLI flag. ✅** |
| W&B worker run ID | `diloco-{worker_lower}-s{stage_k}-r{round_n}` — unique per round |
| W&B worker group | `diloco-{worker_lower}-s{stage_k}` — groups all rounds for a stage |
| W&B coordinator run ID | `diloco-coordinator-s{stage_k}` |
| W&B step axis | Monotonic: `round_n × (shard_step_estimate + 1) + local_step` ✅ |
| `shard_step_estimate` | `ceil(36906 / 3 / (batch_size × grad_accum))` = 385 at defaults |
| DiLoCo outer LR | 0.7 (diloco mode) / 1.0 effective (solo mode = direct promotion) |
| **`min_shard_samples`** | **32** (1 optimizer step = batch_size × grad_accum) |
| **Solo mode** | 1 active worker → direct weight promotion (skip outer update). |
| **Stage close** | remaining < `min_shard_samples` per active worker → declare stage complete, advance. |
| **`workflow_dispatch` inputs** | `force_worker_ids`, `skip_trigger`, `dry_run`. Full control from Actions UI. |
| **Worker timeout threshold** | **13h** (Kaggle 12h hard wall + 1h grace). Set via `--worker_timeout_hours`. |
| **`triggered_at` field** | Written to `round_state.json` on every worker dispatch. Reset to 0 as manual "unconfirmed dispatch" signal. |
| **`triggered_at=0` semantics** | Canonical signal for "dispatch unconfirmed". Coordinator immediately re-dispatches on next run. ✅ VERIFIED |
| **Attendance round** | Worker in `attendance_workers` → skips training, uploads status(samples=0), pushes signal. |
| **Waiting mode** | All credentialed workers in `attendance_workers`. `round_n` frozen. |

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
| `kaggle kernels push --accelerator` → unrecognized arg | Removed CLI flag; GPU requested via JSON metadata ✅ |
| Coordinator stalls on quota-dead worker | `triggered_at` + 13h timeout → attendance mechanism ✅ |
| Round 1 deadlock (triggered_at≠0, workers never ran) | `triggered_at=0` manual reset → immediate re-dispatch ✅ VERIFIED (Session 19) |
| **Kaggle assigns P100 instead of T4** | **`"accelerator": "nvidiaTeslaT4"` in `_build_kaggle_kernel_metadata()` ⚠️ PENDING** |
| **W&B subsequent rounds not logged (wandb 0.25.0 resume bug)** | **Per-round run IDs + `group=` parameter ⚠️ PENDING** |

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

---

## Part 3 — File Registry

| File | Status |
|---|---|
| `jamba_coconut_finetune.py` | ⚠️ W&B grouping patch pending |
| `diloco_coordinator.py` | ⚠️ T4 accelerator patch pending |
| `bootstrap_diloco.py` | ✅ |
| `.github/workflows/diloco_coordinator.yml` | ✅ |
| `kaggle-utils.ipynb` Cell 5 | ✅ `!torchrun` magic + `DILOCO_WORKER_ID` secret |
| `prepare_coconut_dataset.py` | ✅ |
| `build_wheels_kaggle.py` | ✅ |

---

## Part 4 — Open Questions

| Question | Status |
|---|---|
| Stage 2 DiLoCo: does aggregated model match sequential baseline? | 🟡 Pre-val by Worker A at Stage 3 start (ce=0.6087, acc=0.0000 — expected at stage entry) |
| TRC GPU quota conversion | 🟡 Email sent — awaiting response |
| DGAC halt_step distribution at K≥2 | 🔴 Open — primary research question |
| Worker C quota: replenishment timeline? | 🟡 Attendance mechanism handles automatically when renewed |
| Worker A Stage 3 Round 1: succeeded on T4 or also P100? | 🟡 Needs confirmation from next coordinator run |

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
| `kaggle kernels push --accelerator` → unrecognized argument | Remove `--accelerator` CLI flag; GPU requested via `enable_gpu: true` in kernel metadata JSON |
| Worker C quota exhausted → coordinator stalls forever | `triggered_at` + 13h timeout + attendance mechanism |
| Coordinator writes `triggered_workers` but push fails silently | `triggered_at=0` manual reset → immediate re-dispatch on next run |
| `enable_gpu: true` in kernel-metadata.json → Kaggle assigns P100 | Add `"accelerator": "nvidiaTeslaT4"` JSON field explicitly |
| `wandb==0.25.0` `resume="allow"` on finished run creates ephemeral run | Per-round run IDs + `group=` for stage-level grouping |
