# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in any new session.**
> **DRY Guidelines:** Session details live in `terminal_log.md`. Blueprint holds decisions, architecture, status, and next actions only.
> **Source of truth:** If docs and `.py`/`.ipynb` files ever disagree, the Python/notebook file takes precedence. Update docs to reflect reality.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros latent reasoning injection into a Transformer-Mamba hybrid (Jamba Reasoning 3B). The Mamba SSM recurrent state acts as compressed scratch-memory across K latent thought passes, bypassing token generation during reasoning. Core mechanism from Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — our novel anti-collapse halt gate.

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
| Stage 2 (2 latent passes) | 🟡 IN PROGRESS — step ~2860/3462 (≈step 564/1154 in epoch) | — |
| Stages 3–10 | ⬜ NOT STARTED | — |

| Stage | Name | Status |
|---|---|---|
| 4 | GRPO on Jamba Reasoning 3B | ⬜ NOT STARTED |
| 5 | Quantization / Edge Deploy | ⬜ NOT STARTED |

### TRC Status
✅ Accepted (email 2026-04-07).
⚠️ **REVISED CLAIM TIMING:** Claim TRC immediately after Stage 1 completes → **Stage 1 is now COMPLETE. Claim TRC now.**

### Dataset
- **Train:** 36,906 samples  **Val:** 1,940 samples
- **median_steps=10  mean=10.42  max=16**
- **`--max_stage=10` for all production runs**

### GPU Arch / Hub Wheel Status
| Arch | causal_conv1d | mamba_ssm |
|---|---|---|
| sm75 (T4) | ✅ on Hub | ✅ on Hub |
| sm100 (B100) | ✅ on Hub | ❌ not yet built |

---

### Part 0.1 — Immediate Next Actions (ordered)

1. **[DONE]** Phase 2.5/2.6 ordering swap ✅
2. **[DONE]** Batched forward for Stage 0 ✅
3. **[DONE]** Profile Dual T4 throughput ✅
4. **[DONE]** Batched latent injection for stages k>0 ✅
5. **[DONE]** Stage 0 training to completion ✅ (checkpoint-0001154)
6. **[DONE]** NCCL + val timeout fixes ✅
7. **[DONE]** Stage 0 val ✅ (ce=0.4041, acc=0.0222)
8. **[DONE]** Stage 1 training + val ✅ (ce=0.4912, acc=0.0444)
9. **[DONE]** FP16 patch confirmed ✅ (~41s/step at k=1)
10. **[DONE]** Hub+prune fix applied and confirmed ✅ (Session 15)

11. **[ACTION NOW] Claim TRC** — Stage 1 is complete. Submit TRC claim.

12. **[NEXT SESSION] Resume Stage 2** (already in progress at step ~2860):
    ```bash
    torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
      --data_dir data/coconut_v1 --use_4bit \
      --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 \
      --batch_size 4 --grad_accum 8 \
      --val_batch_size 2 \
      --val_skip_buffer_minutes 60 \
      --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \
      --push_to_hub \
      --output_dir runs/stage3_curriculum
    ```
    - Startup log **must** show: `[GPU] Tesla T4  cc=sm75  VRAM=16GB  amp_dtype=float16`
    - Auto-resumes from latest stage_2 checkpoint
    - Stage 2 target step time: **~57s/step** (empirical, Session 15)
    - Stage 2 remaining: ~590 steps × 57s ≈ 9.3h — expect 1 session + possible timeout

13. **On A100 (post-TRC):** Move stages 3–10. Confirm: `[perf] 80GB VRAM detected: disabling gradient checkpointing`

14. **After K=4–5 gate passes on A100:** Integrate DGAC (Phase 3.4).

15. **[MINOR BUG — low priority]** `save_checkpoint` hub upload uses wrong remote path — missing stage subdir (e.g. uploads to `runs/stage3/checkpoint-0002308` instead of `runs/stage3/stage_1/checkpoint-0002308`). Startup sync corrects this next session. Fix in `save_checkpoint`: pass `f"{subdir}/stage_{stage_k}"` as `remote_prefix`.

---

### Part 0.1.1 — Profiler snippet (use if step time regresses)

```python
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    loss, metrics = coconut_forward(...)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# Confirm: volta_h884gemm (FP16 tensor cores) NOT volta_sgemm (FP32)
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
| `--max_seq_len 512` filtered stage 1+ samples | Default changed to 1024 ✅ |
| Exploding gradients at k≥2 | `--max_grad_norm 0.3` ✅ |
| mamba-ssm 2.x API | Pinned to 1.2.2 ✅ |
| Val decode time (200 samples → 5+hrs) | Cap at 50 samples ✅ |
| NCCL watchdog (60min) kills DDP val | `timedelta(hours=4)` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` ✅ |
| Val skip threshold used wrong buffer | `--val_skip_buffer_minutes 60` separate arg ✅ |
| BF16 on T4 uses FP32 compute paths | `_amp_dtype` checks `cc >= (8, 0)`; T4 uses FP16 ✅ |
| GC wastes 20–40% on A100 | Auto-disabled at VRAM≥40GB ✅ |
| `_amp_dtype` called in hot loop | `@functools.lru_cache(maxsize=None)` ✅ |
| `--push_to_hub` never passed | ✅ FIXED — added to command; hub+prune confirmed Session 15 |
| `prune_epoch_checkpoints` only after val; per-stage only | ✅ FIXED — `startup_hub_sync_and_prune` confirmed Session 15 |
| `save_checkpoint` hub upload missing stage subdir | 🟡 LOW PRIORITY — startup sync corrects it; fix when convenient |
| Prefix re-computation at Stage k | 🟡 FUTURE — Cache Mamba state at q_len before Stage 5 on A100 |

---

## Part 1 — Architecture

### Jamba Reasoning 3B (primary research model)
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
| Session timeout | `--session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20` |
| val_batch_size | **2** |
| val accuracy samples | **50** (capped from 200) |
| val skip threshold | **`--val_skip_buffer_minutes 60`** |
| NCCL init_process_group timeout | **`timedelta(hours=4)`** |
| Stage 0 forward pass | ✅ Batched — single fused backbone call per micro-batch |
| Stage k≥1 forward pass | ✅ `_forward_batched_latent()` |
| epochs_per_stage | **1 for all stages** |
| batch_size | **4** |
| Stage 0 skip | **NO** — domain adaptation. 1 epoch sufficient. |
| DDP val strategy | All ranks participate (interleaved shard) |
| TRC claim timing | **After Stage 1 completes** → Stage 1 now COMPLETE |
| amp_dtype on T4 (sm75) | ✅ **FP16** — confirmed ~41s/step (k=1), ~57s/step (k=2) |
| amp_dtype on A100+ (sm80+) | ✅ **BF16** |
| Gradient checkpointing | ✅ Auto-disabled at VRAM≥40GB |
| Hub checkpoint sync | ✅ `startup_hub_sync_and_prune` confirmed working Session 15 |

---

## Part 3 — Open Questions

| Question | Status |
|---|---|
| Does `inputs_embeds` → `last_hidden_state` work at k≥2? | 🟢 CONFIRMED — Stage 2 training, ce healthy ~0.45–0.65 |
| Step time at Stage 2 post-FP16? | 🟢 **~57s/step** (empirical, Session 15). Model predicted ~46s; actual slightly higher. |
| DGAC Phase 3.4: halt_step distribution across K≥2? | 🔴 OPEN — primary research validation |
| Prefix re-computation optimization | 🟡 OPEN — implement before Stage 5 on A100 |
| Is 1 epoch per stage sufficient for convergence? | 🟡 MONITOR — Stage 2 gn up to 1.9 (pre-clip); CE trending ~0.5 vs Stage 1 ~0.44 |
| Stage 2 elevated gn (up to ~1.9 pre-clip): instability or transient? | 🟡 MONITOR — max_grad_norm=0.3 clipping correctly; CE not diverging |

---

## Part 4 — Performance Analysis

### Step-time model (empirical)

```
Post-patch FP16 on T4 (CONFIRMED):
  t_fp16(k) ≈ 34 + ~11.5k  seconds/step  [revised from empirical k=1 and k=2]
  Stage 0: ~34s  Stage 1: ~41s ✓  Stage 2: ~57s ✓  Stage 5: ~92s  Stage 10: ~149s

  Note: original estimate was 34 + 6k; actual k=2 data (57s) suggests ~11.5s per latent pass.
  Revising upward; A100 still required for stages 3–10.
```

| Platform | Condition | Stages 1–10 total | Feasible? |
|---|---|---|---|
| Dual T4 | FP16 patch | ~550h (revised) | ❌ No |
| A100 80GB | BF16 native + no GC (~4–6× vs T4 FP16) | ~90–140h | ✅ Yes |

**Conclusion:** A100 required for stages 3–10. Stage 2 is finishable on T4 (~1–2 sessions).

### Disk usage (Kaggle)
Hub+prune confirmed working. Each session: all local checkpoints pushed to Hub, all except resume pruned. Disk stays bounded.

---

## Part 5 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `jamba_coconut_finetune.py` | 3 | ✅ Hub+prune CONFIRMED | Minor bug: `save_checkpoint` remote path missing stage subdir |
| `kaggle-utils.ipynb` | 3 | ✅ CURRENT | Cell 5 has `--push_to_hub`; confirmed working |
| `AGENT_PROMPT_hub_prune_fix.md` | 3 | ✅ APPLIED & CONFIRMED | Session 15 |
| `AGENT_PROMPT_perf_fix.md` | 3 | ✅ APPLIED | FP16 + lru_cache + auto-GC disable |
| `AGENT_PROMPT_nccl_val_fix.md` | 3 | ✅ APPLIED | Session 13 confirmed |
| `prepare_coconut_dataset.py` | 3 | ✅ DONE | coconut-v1 on Hub |
| `build_wheels_kaggle.py` | 3 | ✅ DONE | sm75 wheels on Hub |

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
λ₁ schedule: 0 for steps 0-200, ramp 0→0.01 over steps 200-500, flat 0.01 after
```

**HaltGate:** Linear(2*d_model → 1), zero-initialized → outputs 0.5 at Phase 3.4 start.

---

## Part 8 — Checkpoint Format

```
output_dir/
  stage_0/
    checkpoint-0001154/
    best/                  ← acc=0.0222 ce=0.4041
  stage_1/
    checkpoint-0002308/
    best/                  ← acc=0.0444 ce=0.4912
  stage_2/
    checkpoint-XXXXXXX/    ← in progress
    best/
  stage_k/best/
    adapter_model/
    training_state.pt      ← {stage_k, step, epoch, step_in_epoch, val_ce, val_acc, optimizer, scheduler}
    halt_gate.pt           ← Phase 3.4 only
```

---

## Part 9 — Compute Plan

| Phase | Platform | Estimate |
|---|---|---|
| Stage 2 (in progress, ~590 steps remaining) | Kaggle Dual T4 | ~590 × 57s ≈ 9.3h (1–2 sessions) |
| Stages 3–10 | TRC A100 80GB | ~90–140h total (revised) |
| Phase 3.4 (DGAC) | TRC A100 80GB | ~6–8h |
| Phase 4 (GRPO) | TRC A100 80GB | ~8–12h |

---

## Part 10 — Hard Lessons

| Lesson | Codified As |
|---|---|
| val_batch_size=16 → OOM | `--val_batch_size 2` default |
| NCCL watchdog kills DDP | `timedelta(hours=4)` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` |
| max_seq_len=512 filtered stage 1+ | `--max_seq_len 1024` |
| gn=36.9 at k=2 | `--max_grad_norm 0.3` |
| mamba-ssm 2.x broke fast path | Pinned to 1.2.2 |
| mamba-ssm 1.2.2 PyPI sdist is a 35kB stub | `git+https://github.com/state-spaces/mamba.git@v1.2.2` |
| Single-name generation shim insufficient | Comprehensive 10-alias shim |
| Per-sample loop at batch=1 for stage 0 → 113s/step | Batched forward path |
| Profile run (200 samples) shows 30s/step; full run shows 137s/step | Always profile at full scale |
| Blueprint marked stage k≥1 batching as PENDING but code already had it | Audit .py before marking blockers |
| Stage 0 skip tempting for capable base model | Stage 0 is domain adaptation |
| Val with no KV cache at 200 samples → 5.5h | Cap at 50 |
| Timeout checker only fires inside training step loop | Pre-val checkpoint save + timeout check before val |
| `timedelta(minutes=60)` kills rank 1 during val | `timedelta(hours=4)` + env var |
| `graceful_exit_buffer=20min` used as val skip threshold | `--val_skip_buffer_minutes 60` separate arg |
| `torch.cuda.is_bf16_supported()` True on sm75 (soft emulation) | `_amp_dtype` checks `cc >= (8, 0)` |
| GC mandatory on T4 but wastes 20–40% on A100 | Auto-disable at VRAM≥40GB |
| `_amp_dtype` called in hot loop | `@functools.lru_cache(maxsize=None)` |
| Conservative 1.5–2.5× FP16 speedup; actual ~4× | Always verify empirically |
| `--push_to_hub` never added → hub upload silently never fires | Add flag explicitly |
| `prune_epoch_checkpoints` per-stage only; skipped on timeout | `startup_hub_sync_and_prune` at session start |
| Blueprint showed `--val_skip_buffer_minutes 120` but code used 60 | Python/notebook files take precedence |
| Step-time model t_fp16(k) ≈ 34 + 6k was too optimistic | Empirical k=2 = 57s → ~11.5s per latent pass |
| `save_checkpoint` hub upload missing stage subdir in remote path | Startup sync corrects it; fix `remote_prefix` in `save_checkpoint` |
