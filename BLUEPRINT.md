# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in any new session.**
> **DRY Guidelines:** Session details live in `terminal_log.md`. Blueprint holds decisions, architecture, status, and next actions only.
> **Source of truth:** If docs and `.py`/`.ipynb` files ever disagree, the Python/notebook file takes precedence.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros latent reasoning injection into a Transformer-Mamba hybrid (Jamba Reasoning 3B). Mamba SSM recurrent state acts as compressed scratch-memory across K latent thought passes, bypassing token generation during reasoning. Core mechanism from Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — our novel anti-collapse halt gate.

### Strategic Status (Updated 2026-04-19)

| Stage | Name | Status |
|---|---|---|
| 0 | Architecture & Viability (nano) | ✅ COMPLETE |
| 1 | Pre-training (nano) | ✅ Pipeline test only; retired |
| 2 | SFT (nano) | 🔴 RETIRED |
| 3 | Coconut-Ouroboros + DGAC on Jamba Reasoning 3B | 🟡 ACTIVE |

**Stage 3 sub-stages:**

| Sub-stage | Status | Best val |
|---|---|---|
| Stage 0 (CoT warmup) | ✅ COMPLETE | ce=0.4041, acc=0.0222 |
| Stage 1 (1 latent pass) | ✅ COMPLETE | ce=0.4912, acc=0.0444 |
| Stage 2 (2 latent passes) | 🟡 IN PROGRESS — 679/1154 steps (59%), checkpoint-0002987 on Hub | — |
| Stages 3–10 | ⬜ NOT STARTED | — |

| Stage | Name | Status |
|---|---|---|
| 4 | GRPO on Jamba Reasoning 3B | ⬜ NOT STARTED |
| 5 | Quantization / Edge Deploy | ⬜ NOT STARTED |

### TRC Status
✅ Accepted (email 2026-04-07). **TRC grants Cloud TPU quota only — fundamentally incompatible with our CUDA/Mamba stack** (no XLA kernels for mamba_ssm/causal_conv1d; bitsandbytes is CUDA-only). TRC cannot be used for this workload as granted.

**Action pending:** Email trc-support@google.com requesting conversion to A100 GPU-hours or GCP credits applicable to GPU VMs. Draft in `terminal_log.md`.

### Compute Strategy — Three-Account Relay Training
Three Kaggle accounts (A/B/C) do **sequential relay handoff** via Hub checkpoints. This is NOT parallel DDP — true multi-node DDP is impossible on Kaggle (no public IPs, no inter-container networking). The Hub is the sole communication mechanism.

**Relay rule: never run two accounts simultaneously. Sequential only.**

| Account | Quota Now | Projected Work |
|---|---|---|
| A (current) | 8.5h | Stage 2 finish (~6.8h remaining) |
| B | 30h | Stage 3 (~21.9h) + partial Stage 4 |
| C | 30h | Remainder of Stage 4 + Stage 5 |
| Weekly refresh | 90h/week | ~3 weeks total for stages 3–10 |

### Dataset
- **Train:** 36,906 samples  **Val:** 1,940 samples
- **median_steps=10  mean=10.42  max=16**
- **`--max_stage=10` for all production runs**

### GPU Arch / Hub Wheel Status
| Arch | causal_conv1d | mamba_ssm |
|---|---|---|
| sm75 (T4) | ✅ on Hub | ✅ on Hub |
| sm100 (B100) | ✅ on Hub | ❌ not yet built |
| sm60 (P100) | ❌ not built | ❌ not built |

**P100 is not viable:** no Tensor Cores (would regress to ~120-150s/step), single GPU (no DDP), sm60 wheels not on Hub. Stick to Dual T4.

---

### Part 0.1 — Immediate Next Actions (ordered)

1–10. **[ALL DONE]** See previous sessions.

11. **[ACTION — BEFORE NEXT RUN]** Apply relay path fix to `jamba_coconut_finetune.py`:
    - See `AGENT_PROMPT_relay_path_fix.md`
    - One-line change in `save_checkpoint()`: `remote_prefix=subdir` → `remote_prefix=f"{subdir.strip('/')}/stage_{stage_k}"`
    - **This is relay-blocking.** Account B/C cannot find Hub checkpoints without this fix.
    - Commit fix to GitHub repo so `kaggle-utils.ipynb` cell 3 (git sync) pulls it automatically.

12. **[Account A — 8.5h remaining]** Finish Stage 2:
    ```bash
    torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
      --data_dir data/coconut_v1 --use_4bit \
      --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 \
      --batch_size 4 --grad_accum 8 \
      --val_batch_size 2 \
      --val_skip_buffer_minutes 60 \
      --session_timeout_hours 8.5 --graceful_exit_buffer_minutes 20 \
      --push_to_hub \
      --output_dir runs/stage3_curriculum
    ```
    - `--session_timeout_hours 8.5` — match actual remaining quota
    - Auto-resumes from `stage_2/checkpoint-0002987` on Hub
    - Stage 2 remaining: ~475 steps × ~52s ≈ 6.8h — should complete within this session

13. **[Account B — 30h]** Same command with `--session_timeout_hours 11.0`:
    - No local data, no `--resume_from` needed — Hub scan is automatic
    - Expected startup: `[resume] No local checkpoints found. Scanning Hub...`

14. **[Account C — 30h]** Same as B after B exhausts quota.

15. **[Weekly rotation]** When all accounts exhaust quota, wait for weekly reset, relay continues.

16. **[Email TRC]** Send draft from `terminal_log.md` — 5 minutes, potentially unlocks A100.

---

### Part 0.1.1 — Relay Handoff Checklist (per account)

Before starting a new account's session:
- [ ] Previous account's session fully ended and Hub upload confirmed complete
- [ ] Path fix applied and committed to GitHub (`AGENT_PROMPT_relay_path_fix.md`)
- [ ] Cell 3 of `kaggle-utils.ipynb` will git-pull the fix automatically
- [ ] Command includes `--push_to_hub`
- [ ] `--session_timeout_hours` set to actual remaining quota for that account

Expected startup log on Account B/C (no local checkpoints):
```
  [resume] No local checkpoints found. Scanning Hub...
  [hub] downloading stage_2/checkpoint-XXXXXXX ...
  [resume] using stage_2/checkpoint-XXXXXXX as resume checkpoint
```

---

### Part 0.2 — Pre-flight Blockers

| Blocker | Resolution |
|---|---|
| `attn_implementation` hardcoded crash | try/except fallback to `eager` ✅ |
| `use_mamba_kernels` kwarg on old TF | `_safe_from_pretrained` retries without kwarg ✅ |
| `last_hidden_state` silent None | assert in Stage 0 and Phase B ✅ |
| No graceful timeout → lost Kaggle work | `make_timeout_checker()` integrated ✅ |
| `conv1d` in LoRA targets | Explicitly excluded ✅ |
| OOM at first val step | `empty_cache()` + `val_batch_size=2` ✅ |
| `--max_seq_len 512` filtered stage 1+ | Default changed to 1024 ✅ |
| Exploding gradients at k≥2 | `--max_grad_norm 0.3` ✅ |
| mamba-ssm 2.x API | Pinned to 1.2.2 ✅ |
| Val decode time (200 samples) | Cap at 50 samples ✅ |
| NCCL watchdog (60min) kills DDP val | `timedelta(hours=4)` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` ✅ |
| Val skip threshold used wrong buffer | `--val_skip_buffer_minutes 60` separate arg ✅ |
| BF16 on T4 uses FP32 compute paths | `_amp_dtype` checks `cc >= (8, 0)`; T4 uses FP16 ✅ |
| GC wastes 20-40% on A100 | Auto-disabled at VRAM≥40GB ✅ |
| `_amp_dtype` called in hot loop | `@functools.lru_cache(maxsize=None)` ✅ |
| `--push_to_hub` never passed | ✅ FIXED — confirmed Session 15 |
| `prune_epoch_checkpoints` per-stage only | ✅ FIXED — `startup_hub_sync_and_prune` Session 15 |
| `save_checkpoint` hub upload missing stage subdir | 🔴 RELAY-BLOCKING — `AGENT_PROMPT_relay_path_fix.md` |
| Prefix re-computation at Stage k | 🟡 FUTURE — before Stage 5 on A100 |

---

## Part 1 — Architecture

### Jamba Reasoning 3B
```
HuggingFace : ai21labs/AI21-Jamba-Reasoning-3B   License: Apache 2.0
Layers      : 28 total (26 Mamba, 2 Attention) → 13:1 Mamba:Attention ratio
Attention   : MQA (20 Q heads, 1 KV head)
Vocab       : 64K / Context: 256K tokens
d_model     : 2560
```

---

## Part 2 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Primary research model | Jamba Reasoning 3B |
| Fine-tuning approach | QLoRA (4-bit NF4) + LoRA |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, in_proj, x_proj, dt_proj, out_proj; conv1d excluded |
| Coconut curriculum | Progressive step replacement per Meta paper |
| max_stage K | **10** |
| DGAC halt gate | Phase 3.4 only; λ₁ annealed from 0 |
| `--max_seq_len` | 1024 |
| `--max_grad_norm` | 0.3 for k≥2 stages |
| `--session_timeout_hours` | Set to actual remaining quota per account (not fixed at 11.0) |
| val_batch_size | **2** |
| val accuracy samples | **50** |
| val skip threshold | **`--val_skip_buffer_minutes 60`** |
| NCCL timeout | **`timedelta(hours=4)`** |
| epochs_per_stage | **1** |
| batch_size | **4** |
| DDP val strategy | All ranks participate (interleaved shard) |
| amp_dtype on T4 (sm75) | ✅ **FP16** — ~41s/step (k=1), ~52-57s/step (k=2) |
| amp_dtype on A100+ (sm80+) | ✅ **BF16** |
| Gradient checkpointing | Auto-disabled at VRAM≥40GB |
| Multi-account strategy | **Sequential relay via Hub** — NOT parallel DDP (impossible on Kaggle) |
| TRC quota | **TPU only — incompatible with stack** — email requesting GPU conversion |
| P100 viability | **No** — no Tensor Cores, single GPU. Dual T4 strictly better. |

---

## Part 3 — Open Questions

| Question | Status |
|---|---|
| Stage 2 elevated gn (up to ~1.9 pre-clip): transient? | 🟡 MONITOR — CE not diverging |
| DGAC Phase 3.4: halt_step distribution across K≥2? | 🔴 OPEN — primary research |
| Prefix re-computation optimization | 🟡 OPEN — before Stage 5 on A100 |
| TRC GPU conversion via email? | 🟡 PENDING — draft ready, not sent |

---

## Part 4 — Performance Analysis

### Step-time model (empirical)

```
t_fp16(k) ≈ 34 + ~11.5k  seconds/step

Stage 0: ~34s  Stage 1: ~41s ✓  Stage 2: ~52-57s ✓
Stage 3: ~69s  Stage 5: ~92s   Stage 10: ~149s  (projected)
```

### Stage hours on Dual T4 + 3-account relay

| Stage | Est. Hours | Account |
|---|---|---|
| 2 (remaining ~475 steps) | ~6.8h | A (8.5h quota) |
| 3 | ~21.9h | B (30h quota) |
| 4 | ~25.7h | B remainder + C |
| 5 | ~29.3h | C remainder + reset |
| 6–10 | ~201h | Weekly resets |
| **Stages 3–10 total** | **~278h** | **~3.1 weeks at 90h/week** |

A100 (if TRC converts): ~5× faster → ~55h → ~2 days.

---

## Part 5 — File Registry

| File | Status | Notes |
|---|---|---|
| `jamba_coconut_finetune.py` | 🔴 NEEDS PATH FIX | Relay-blocking bug in `save_checkpoint` |
| `kaggle-utils.ipynb` | ✅ CURRENT | Cell 5 has `--push_to_hub`; git-pull in cell 3 gets fix |
| `AGENT_PROMPT_relay_path_fix.md` | ✅ READY | One-line fix, apply before Account B |
| `AGENT_PROMPT_hub_prune_fix.md` | ✅ APPLIED | Session 15 confirmed |
| `AGENT_PROMPT_perf_fix.md` | ✅ APPLIED | FP16 + lru_cache + auto-GC |
| `AGENT_PROMPT_nccl_val_fix.md` | ✅ APPLIED | Session 13 confirmed |
| `prepare_coconut_dataset.py` | ✅ DONE | coconut-v1 on Hub |
| `build_wheels_kaggle.py` | ✅ DONE | sm75 wheels on Hub |

---

## Part 6 — Coconut Curriculum Design

```
Stage 0:  [Q][S1][S2]...[Sn][A]    ← standard CoT; labels on S1..Sn + A
Stage k:  [Q][●*k][S_{k+1}..Sn][A] ← first k steps replaced; labels shift
Stage K:  [Q][●*K][A]              ← all steps replaced; labels on A only
K = 10
```

---

## Part 7 — DGAC

```
L_total = L_ce  +  λ₁(t) · L_ponder  +  λ₂ · L_diversity

L_diversity = mean_batch( Σ_k relu(cos_sim(h_k, h_{k-1}) − τ) ),  τ = 0.9
λ₁ schedule: 0 for steps 0-200, ramp 0→0.01 over steps 200-500, flat after
```

**HaltGate:** Linear(2*d_model → 1), zero-initialized → outputs 0.5 at Phase 3.4 start.

---

## Part 8 — Checkpoint Format

```
output_dir/
  stage_0/best/             ← acc=0.0222 ce=0.4041
  stage_1/best/             ← acc=0.0444 ce=0.4912
  stage_2/
    checkpoint-0002987/     ← in progress (679/1154 steps)
    best/
  stage_k/best/
    adapter_model/
    training_state.pt
    halt_gate.pt            ← Phase 3.4 only

Hub (after path fix applied):
  WeirdRunner/Ouroboros/runs/stage3/stage_{k}/checkpoint-XXXXXXX/
  WeirdRunner/Ouroboros/runs/stage3/stage_{k}/best/
```

---

## Part 9 — Compute Plan

| Phase | Platform | Estimate |
|---|---|---|
| Stage 2 finish | Account A (8.5h) | ~6.8h ✅ fits |
| Stage 3 | Account B (30h) | ~21.9h ✅ fits |
| Stages 4–5 | Account B+C | ~55h |
| Stages 6–10 | Weekly relay resets | ~223h |
| Phase 3.4 (DGAC) | After Stage 10 | ~6–8h |
| Phase 4 (GRPO) | After DGAC | ~8–12h |

---

## Part 10 — Hard Lessons

| Lesson | Codified As |
|---|---|
| val_batch_size=16 → OOM | `--val_batch_size 2` |
| NCCL watchdog kills DDP | `timedelta(hours=4)` + env var |
| max_seq_len=512 filtered stage 1+ | `--max_seq_len 1024` |
| gn=36.9 at k=2 | `--max_grad_norm 0.3` |
| mamba-ssm 2.x broke fast path | Pinned to 1.2.2 |
| mamba-ssm PyPI sdist is 35kB stub | `git+https://...@v1.2.2` |
| Per-sample loop → 113s/step | Batched forward path |
| Val at 200 samples → 5.5h | Cap at 50 |
| `is_bf16_supported()` True on sm75 (emulation) | `_amp_dtype` checks `cc >= (8, 0)` |
| GC wastes 20-40% on A100 | Auto-disable at VRAM≥40GB |
| `--push_to_hub` never added → silent no-op | Add flag explicitly |
| `prune_epoch_checkpoints` per-stage only | `startup_hub_sync_and_prune` at session start |
| Step-time model `34 + 6k` too optimistic | Empirical k=2 → ~11.5s/latent pass |
| `save_checkpoint` remote_prefix missing stage subdir | 🔴 RELAY-BLOCKING — fix before Account B |
| "Parallel sessions" misread as multi-node DDP | Impossible on Kaggle — Hub relay only |
| TRC assumed to give GPU access | TPU-only — incompatible with CUDA/Mamba stack |
| P100 assumed better than T4 | No Tensor Cores, single GPU — strictly worse |
| `--session_timeout_hours` fixed at 11.0 | Set to actual remaining quota per account |
