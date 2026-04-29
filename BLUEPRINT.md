# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in every new session.**
> **Source of truth:** If this doc and `.py`/`.ipynb` files ever disagree, the Python/notebook file wins.
> **DRY rule:** Session details and verbatim logs live in `terminal_log.md` only. This file holds decisions, status, and next actions.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros: latent reasoning injection into Jamba Reasoning 3B (Transformer-Mamba hybrid). The Mamba SSM recurrent state acts as compressed scratch-pad across K latent thought passes, replacing token generation during reasoning. Based on Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — a novel anti-collapse halt gate.

### Current Status (2026-04-30)

| Curriculum Stage | Status | Notes |
|---|---|---|
| Stage 0 — CoT warmup | ✅ COMPLETE | ce=0.4041, acc=0.0222 |
| Stage 1 — 1 latent pass | ✅ COMPLETE | ce=0.4912, acc=0.0444 |
| Stage 2 — 2 latent passes | ✅ COMPLETE | 6 rounds, A+B, ~36,906 samples |
| Stage 3 — 3 latent passes | ✅ COMPLETE | 6 rounds; A solo r2–r3 (B only signals); C rejoined r5 |
| Stage 4 — 4 latent passes | ✅ COMPLETE | 1 round, A+B+C all active |
| Stage 5 — 5 latent passes | ✅ COMPLETE | 1 round, A+B+C all active |
| Stage 6 — 6 latent passes | ✅ COMPLETE | 1 round, A+B+C all active |
| Stage 7 — 7 latent passes | 🔄 IN PROGRESS | Round 0 done (A+B+C signals); awaiting coordinator next run |
| Stages 8–10 | ⬜ NOT STARTED | — |

**Compute mode: DiLoCo 3-worker (A+B+C all active.)**

**Training velocity:** ~1 stage/day for stages 4–7. System is running autonomously.

---

## Part 0.1 — Immediate Next Steps

### Step 1 — Monitor Stage 7 completion (no action required)
Stage 7 round 0 completed (all three worker signals present — verified from signal files).
The coordinator will aggregate and trigger round 1 automatically within 30 minutes.

No code changes needed. All fixes from previous sessions are verified deployed and working.

### Step 2 — Watch W&B for pre-val accuracy trend
Pre-val accuracy at stage entry (Worker A round 0):
- Stage 3: 0.00% (expected — first stage with full latent replacement in effect)
- Stage 4: ~2%
- Stage 5: ~3–4%
- Stage 6: ~4–6%
- Stage 7: ~6–8% (expected — monotonic trend continuing)

This is a positive monotonic trend. CE is also in a healthy range (0.4–0.8). No intervention needed.

### Step 3 — DGAC prep (stages 8–10 remaining before Phase 3.4)
When stage 10 completes, run Phase 3.4:
```bash
python jamba_coconut_finetune.py \
  --use_halt_gate \
  --resume_from runs/stage3_curriculum/stage_10/best \
  --output_dir runs/stage3_dgac [...]
```

### Step 4 — Stage 3 rounds 2–3: A signal gap (low priority)
Worker A has no signal files for stage 3 rounds 2 and 3. B completed both. This likely means A was in solo/attendance mode those rounds (coordinator handled it correctly — B's weights were promoted directly in solo mode). No correctness issue; flag for review in W&B if A's stage 3 CE looks discontinuous.

---

## Part 0.2 — Hub State

```
WeirdRunner/Ouroboros/
  diloco_state/
    anchor/                                  ← Stage 7 Round 0 aggregate (latest coordinator run)
    round_state.json                         ← stage_k=7, round_n=1 (or higher), triggered_workers=[A,B,C]
    workers/A/round_0000_stage_3/  ✓
    workers/B/round_0000_stage_3/  ✓
    workers/A/round_0001_stage_3/  ✓ (confirmed — signal present)
    workers/B/round_0001_stage_3/  ✓ (confirmed — signal present)
    workers/{A,B,C}/round_0000_stage_4/  ✓
    workers/{A,B,C}/round_0000_stage_5/  ✓
    workers/{A,B,C}/round_0000_stage_6/  ✓
    workers/{A,B,C}/round_0000_stage_7/  ✓
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
| **Kaggle SDK version** | **`kaggle>=1.8.4`** — first version with `--accelerator` for `kernels push` (PR #907). ✅ VERIFIED WORKING |
| **`"accelerator"` field in kernel-metadata.json** | **`"NvidiaTeslaT4"` (capital N).** Belt-and-suspenders alongside `--accelerator` CLI flag. ✅ |
| **`"accelerator"` CLI flag in push command** | **`--accelerator NvidiaTeslaT4`** added to `push_args` in `_trigger_single_worker()`. ✅ VERIFIED WORKING |
| **GPU fast-fail safety net** | `cc < (7,5)` → `_diloco_reset_triggered_at()` + signal + `sys.exit(0)`. ✅ DEPLOYED |
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
| **`triggered_at` field** | Written to `round_state.json` on every worker dispatch. Reset to 0 as "unconfirmed dispatch" signal. |
| **`triggered_at=0` semantics** | Canonical signal for "dispatch unconfirmed". Coordinator immediately re-dispatches on next run. ✅ VERIFIED |
| **Attendance round** | Worker in `attendance_workers` → skips training, uploads status(samples=0), pushes signal. |
| **Waiting mode** | All credentialed workers in `attendance_workers`. `round_n` frozen. |
| **Wiki** | `WIKI.md` + `wiki/` directory at repo root. LLM-maintained synthesized understanding layer. Init: 2026-04-30. |

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
| `kaggle==2.0.x` gRPC → 403 | Pinned `kaggle==1.6.17` ✅ (now superseded by `>=1.8.4`) |
| `kaggle kernels push --accelerator` → unrecognized arg | Removed CLI flag; GPU requested via JSON metadata ✅ (belt-and-suspenders: also added CLI flag after upgrade to >=1.8.4) |
| Coordinator stalls on quota-dead worker | `triggered_at` + 13h timeout → attendance mechanism ✅ |
| Round 1 deadlock (triggered_at≠0, workers never ran) | `triggered_at=0` manual reset → immediate re-dispatch ✅ VERIFIED (Session 19) |
| **`kaggle==1.6.17` predates `--accelerator` feature → P100 assigned** | **Fixed: `kaggle>=1.8.4` + `--accelerator NvidiaTeslaT4` in `push_args` + `"NvidiaTeslaT4"` JSON cap + runtime fast-fail. ✅ VERIFIED WORKING (no P100 since deploy)** |
| **W&B subsequent rounds not logged (wandb 0.25.0 resume bug)** | **Per-round run IDs + `group=` parameter. ✅ VERIFIED WORKING (dashboard shows separate runs per round)** |
| Worker C quota exhausted (stages 2–3 early rounds) | Quota renewed; C rejoined at stage 3 round 5. All three workers active from stage 4 onward. ✅ |

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
    - GPU guard: if sm60 (P100) → reset triggered_at=0 + push signal + exit
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
Observed velocity (stages 4–7): ~1 stage/day with 3 workers on 12h Kaggle sessions
```

---

## Part 3 — File Registry

| File | Status |
|---|---|
| `jamba_coconut_finetune.py` | ✅ All patches deployed and verified |
| `diloco_coordinator.py` | ✅ All patches deployed and verified |
| `.github/workflows/diloco_coordinator.yml` | ✅ `kaggle>=1.8.4` deployed |
| `kaggle-utils.ipynb` Cell 5 | ✅ No changes needed |
| `prepare_coconut_dataset.py` | ✅ |
| `build_wheels_kaggle.py` | ✅ |
| `WIKI.md` | ✅ Wiki schema — init 2026-04-30 |
| `wiki/index.md` | ✅ Content catalog |
| `wiki/log.md` | ✅ Append-only operation log |
| `wiki/concept/coconut-curriculum.md` | ✅ Stage-k mechanics, perf model, DGAC status |
| `wiki/concept/diloco-protocol.md` | ✅ Round state machine, triggered_at, modes |
| `wiki/debug/kaggle-gpu-p100-fallback.md` | ✅ P100 root cause + full fix chain |
| `wiki/debug/wandb-resume-ephemeral-runs.md` | ✅ wandb 0.25.0 resume bug + fix |
| `wiki/pattern/coordinator-retry-flow.md` | ✅ Decision tree + invariants |

---

## Part 4 — Open Questions

| Question | Status |
|---|---|
| Stage 2 DiLoCo: does aggregated model match sequential baseline? | 🟡 Pre-val acc rising monotonically (0%→2%→4%→6%→~8%) — promising signal |
| Stage 3 rounds 2–3: where are Worker A signals? | 🟡 Likely solo/attendance — B signals present, coordinator handled correctly |
| TRC GPU quota conversion | 🟡 Email sent — awaiting response |
| DGAC halt_step distribution at K≥2 | 🔴 Open — primary research question (Phase 3.4, stages 8–10 remaining) |
| Worker C quota: stable for remainder of curriculum? | 🟢 Active since stage 3 round 5 — stable through stage 7 |
| Pre-val accuracy at stage 10: target threshold for DGAC? | 🔴 Open — define success criteria before Phase 3.4 |

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
| `kaggle kernels push --accelerator` → unrecognized argument (old CLI) | Upgrade to `kaggle>=1.8.4`; add `--accelerator NvidiaTeslaT4` to `push_args` |
| Worker C quota exhausted → coordinator stalls forever | `triggered_at` + 13h timeout + attendance mechanism |
| Coordinator writes `triggered_workers` but push fails silently | `triggered_at=0` manual reset → immediate re-dispatch on next run |
| `kaggle==1.6.17` + `enable_gpu:true` + `"accelerator": "nvidiaTeslaT4"` → still P100 | Root cause 1: `--accelerator` CLI flag added in v1.8.4 (predates our pin). Root cause 2: wrong cap. Fix: `kaggle>=1.8.4` + `--accelerator NvidiaTeslaT4` + fix JSON cap + runtime fast-fail. All verified. |
| `wandb==0.25.0` `resume="allow"` on finished run creates ephemeral run | Per-round run IDs + `group=` for stage-level grouping |
| W&B dashboard becomes unreadable with many overlapping runs | Unique `id` per round + `group` by stage keeps it navigable |
| BLUEPRINT.md status table drifts behind signal files between sessions | Signal files are ground truth — always reconcile against `signals/` on session start |
