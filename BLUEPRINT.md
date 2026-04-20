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
| Stage 2 — 2 latent passes | ⚠️ Workers done; coordinator crashed (numpy missing) — fix deployed | — |
| Stages 3–10 | ⬜ NOT STARTED | — |
| Phase 3.4 — DGAC | ⬜ after Stage 10 | — |
| Phase 4 — GRPO | ⬜ after DGAC | — |

**Compute mode: DiLoCo 3-way parallel (Worker C quota exhausted; A+B sufficient at min_workers=2)**

---

## Part 0.1 — Immediate Next Steps (strict order)

### Step 1 — Deploy the workflow fix ⚡ BLOCKING
Push `.github/workflows/diloco_coordinator.yml` to `deveshpat/Ouroboros` with these changes:
- `pip install numpy` added (crash root cause — `safetensors.torch.save_file` calls numpy internally)
- `torch` switched to CPU-only wheel (`--index-url https://download.pytorch.org/whl/cpu`)
- `timeout-minutes: 30` (up from 15 — anchor upload is ~1.5 GB)
- `workflow_dispatch:` trigger added (allows manual re-run from Actions UI without a signal commit)

### Step 2 — Re-trigger the coordinator
After pushing the fix, either:
- **Option A (simplest):** Actions tab → DiLoCo Coordinator → "Run workflow" (now available via `workflow_dispatch`)
- **Option B:** Edit either signal file (e.g. bump the timestamp) and commit

`round_state.json` is still clean at `{stage_k: 2, round_n: 0}`. Worker weights are on HF Hub. The coordinator will pick up exactly where it left off.

### Step 3 — Verify coordinator completes
Expected log:
```
[coordinator] Worker A: 5060 samples ready
[coordinator] Worker B: 5059 samples ready
[coordinator] Aggregating on CPU...
[coordinator] New anchor uploaded: DiLoCo anchor: stage 2 round 0 ...
[coordinator] round_state.json updated: stage=3 round=0
[coordinator] Triggered Worker A: weirdrunner/kaggle-utils
[coordinator] Triggered Worker B: weirdrunner007/kaggle-utils
[coordinator] Done.
```

### Step 4 — Fix GITHUB_TOKEN on both Kaggle accounts (for future rounds)
Generate a classic PAT from the **`deveshpat` GitHub account** with `repo` scope.
Update `GITHUB_TOKEN` Kaggle secret on **weirdrunner** and **weirdrunner007** accounts.
Worker B's current token is expired (401) — urgent for future auto-signalling.

### Step 5 — If Kaggle auto-trigger fails, start workers manually
Aggregation is independent of the trigger. If `Triggered Worker` lines show errors, open kaggle-utils on Account A and B and run manually. `round_state.json` will already have `stage=3 round=0`.

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
    round_state.json          ← {"stage_k": 2, "round_n": 0}  ← STALE — coordinator crashed before update
    workers/A/                ← Stage 2 Round 0 weights uploaded ✓ (5060 samples)
    workers/B/                ← Stage 2 Round 0 weights uploaded ✓ (5059 samples)
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
| Notebook launch cells | `!torchrun` magic commands only |
| Worker auto-detection | `DILOCO_WORKER_ID` Kaggle secret per account (`A`/`B`/`C`) |
| Kaggle trigger auth | Per-worker credentials; Kaggle API is owner-authenticated (403 on cross-account) |
| Kaggle trigger mechanism | SDK pull → re-push (`kernels_pull` + `kernels_push`). No standalone `/run` endpoint exists. |
| W&B worker run ID | `diloco-{worker_lower}-s{stage_k}` — persists and resumes across rounds within a stage |
| W&B coordinator run ID | `diloco-coordinator-s{stage_k}` |
| W&B entity | `default=None` — auto-resolved from API key |
| W&B step axis | `round_n × shard_step_estimate + local_step` (monotonic across rounds) |
| `shard_step_estimate` | `ceil(36906 / 3 / (batch_size × grad_accum))` = 385 at defaults |
| DiLoCo wandb init timing | Deferred to `run_diloco_worker()` where stage_k/round_n are known |
| Stage advancement (DiLoCo) | When `sum(all workers' samples_seen_this_stage) >= len(train_set)` |
| Val in DiLoCo mode | Worker A only, once per stage (round_n == 0, is_new_stage == True) |
| DiLoCo outer LR | 0.7 (DiLoCo paper default) |
| DiLoCo min_workers | 2 of 3 |
| Timeout clock anchor | `_SCRIPT_START = time.perf_counter()` at **module import** |
| DiLoCo shard computation | Subtracts `total_samples_seen[stage_k]` before A/B/C split |
| Hub auto-resume (DDP) | Rank 0 resolves; path broadcast via marker file; `.hub_resume/` cleanup deferred |
| Pre-val guard | Only when `is_new_stage == True` (i.e. `stage_samples_seen == 0`) |
| TRC quota | TPU only — incompatible. Email requesting GPU conversion sent. |
| **Coordinator torch** | CPU-only wheel (`--index-url https://download.pytorch.org/whl/cpu`). Coordinator is GPU-free. |
| **Coordinator numpy** | Must be explicitly installed. `safetensors.torch.save_file` → `_tobytes` → `import numpy`. |
| **Coordinator timeout** | 30 minutes. Anchor upload is ~1.5 GB; 15 min is too tight. |
| **workflow_dispatch** | Added to coordinator yml — allows manual re-trigger from Actions UI. |
| **GITHUB_TOKEN scope** | Must be classic PAT from `deveshpat` account with `repo` scope. Read-only works for `git clone` but fails the Contents API `PUT` (write). |

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

## Part 0.5 — Pre-flight Blockers (all resolved except one pending)

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
| Kaggle `/run` endpoint doesn't exist | SDK pull → re-push ✅ |
| Kaggle trigger is owner-authenticated | Per-worker credential pairs ✅ |
| `--wandb_entity` pointing at dead account | Removed; `entity=None` ✅ |
| `global_step_offset` inside wandb guard → `NameError` | Computed unconditionally ✅ |
| Timeout clock starts too late | `_SCRIPT_START` at module import ✅ |
| DiLoCo sharding ignores stage remainder | Subtracts `total_samples_seen[stage_k]` ✅ |
| Pre-val fires on resumed partial stage | `is_new_stage = (stage_samples_seen == 0)` ✅ |
| Hub auto-resume not DDP-safe | Rank 0 resolves; path broadcast ✅ |
| subprocess in notebook suppressed output | `!torchrun` magic command ✅ |
| **`numpy` missing in coordinator workflow** | Added to pip install in `diloco_coordinator.yml` ✅ |
| **Coordinator CUDA torch bloat** | Switched to CPU-only wheel ✅ |
| **Coordinator timeout too short** | Bumped to 30 min ✅ |
| **No manual re-trigger path** | `workflow_dispatch:` added to yml ✅ |
| **GITHUB_TOKEN read-only → 403/401 on signal push** | Use `deveshpat` classic PAT with `repo` scope on all worker Kaggle accounts ⚠️ PENDING |

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
Empirical Session 16: Worker A 48.5s/step, Worker B 53.0s/step at k=2 (~10% faster than model)
Stage 1: ~41s  Stage 2: ~48–53s  Stage 3: ~69s  Stage 5: ~92s  Stage 10: ~149s
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
| `diloco_coordinator.py` | ✅ Complete |
| `bootstrap_diloco.py` | ✅ Run and confirmed |
| `.github/workflows/diloco_coordinator.yml` | ✅ Fixed — numpy, CPU torch, 30min timeout, workflow_dispatch |
| `kaggle-utils.ipynb` Cell 5 | ✅ `!torchrun` magic + `DILOCO_WORKER_ID` secret |
| `prepare_coconut_dataset.py` | ✅ Done |
| `build_wheels_kaggle.py` | ✅ Done |

---

## Part 4 — Open Questions

| Question | Status |
|---|---|
| Coordinator re-run with numpy fix: succeeds? | 🟡 Pending — deploy fix and re-trigger |
| Stage 2 DiLoCo: does aggregated model match sequential baseline? | 🟡 Pre-val by Worker A at Stage 3 start |
| TRC GPU quota conversion | 🟡 Email sent — awaiting response |
| DGAC halt_step distribution at K≥2 | 🔴 Open — primary research question |
| Worker C quota: replenishment timeline? | 🟡 Nice-to-have; A+B sufficient |
| PyTorch 2.9 `use_reentrant` warning | 🟡 Minor — add `gradient_checkpointing_kwargs={"use_reentrant": True}` |

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
| GC wastes 20-40% on A100 | Auto-disable at VRAM≥40GB ✅ |
| Checkpoint pruning per-stage only | `startup_hub_sync_and_prune` at session start |
| "Parallel sessions impossible on Kaggle" | DiLoCo makes it viable |
| TRC assumed GPU access | TPU only — incompatible |
| Headless sessions capped by quota | `--session_timeout_hours 12.0` |
| Kaggle `/run` API does not exist | Pull → re-push via official SDK |
| Kaggle trigger is owner-authenticated | Per-worker credential pairs |
| Wandb entity hardcoded to stale account | Remove `--wandb_entity`; let SDK auto-resolve |
| Timeout clock reset inside `main()` | `_SCRIPT_START` at module import |
| DiLoCo sharding used full dataset | Subtract `total_samples_seen[stage_k]` |
| Pre-val fired on resumed partial stage | `is_new_stage = (stage_samples_seen == 0)` |
| Hub resume path lost between DDP ranks | Rank 0 resolves, broadcasts, cleanup deferred |
| `subprocess.run` hid output | `!torchrun` magic command |
| `git clone` works but GitHub Contents API PUT fails | `git clone` = read; API push needs `repo` scope. Use `deveshpat` classic PAT. 401=expired, 403=wrong scope. |
| **`safetensors.torch.save_file` needs numpy** | `safetensors` has Rust backend but PyTorch tensor-to-bytes path calls `import numpy`. Add numpy to coordinator pip install. |
| **Coordinator job timeout too short** | Anchor is ~1.5 GB; 15 min not enough. Use 30 min. |
| **No manual coordinator re-trigger** | Add `workflow_dispatch:` to yml — Actions UI gets "Run workflow" button. |
