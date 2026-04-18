# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in any new session.**
> **DRY Guidelines:** Session details live in `terminal_log.md`. Blueprint holds decisions, architecture, status, and next actions only.
> **Source of truth:** If docs and `.py`/`.ipynb` files ever disagree, the Python/notebook file takes precedence. Update docs (next steps, resolved decisions, hard lessons) to reflect reality.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros latent reasoning injection into a Transformer-Mamba hybrid (Jamba Reasoning 3B). The Mamba SSM recurrent state acts as compressed scratch-memory across K latent thought passes, bypassing token generation during reasoning. Core mechanism from Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — our novel anti-collapse halt gate.

### Strategic Status (Updated 2026-04-18)

| Stage | Name | Status |
|---|---|---|
| 0 | Architecture & Viability (nano) | ✅ COMPLETE |
| 1 | Pre-training (nano) | ✅ Pipeline test only; retired |
| 2 | SFT (nano) | 🔴 RETIRED |
| 3 | Coconut-Ouroboros + DGAC on Jamba Reasoning 3B | 🟡 ACTIVE — Stage 0 val COMPLETE (ce=0.4041, acc=0.0222). Stage 1 IN PROGRESS: step=1338 saved, ~184/1154 steps done. **FP16 patch confirmed: ~41s/step (~4× speedup over pre-patch 162s/step).** |
| 4 | GRPO on Jamba Reasoning 3B | ⬜ NOT STARTED |
| 5 | Quantization / Edge Deploy | ⬜ NOT STARTED |

### TRC Status
✅ Accepted (email 2026-04-07).  
⚠️ **REVISED CLAIM TIMING:** Claim TRC immediately after Stage 1 completes. Run stages 2–10 on A100.

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
6. **[DONE]** NCCL + val timeout fixes ✅ (Session 13 confirmed val succeeded)
7. **[DONE]** Stage 0 val ✅ (ce=0.4041, acc=0.0222, DDP val ~2h21m)
8. **[DONE]** Stage 1 training started ✅
9. **[DONE]** FP16 patch confirmed ✅ (~41s/step empirical, ~4× speedup)

10. **[NEXT SESSION] Apply hub+prune fix, then resume Stage 1:**

    See `AGENT_PROMPT_hub_prune_fix.md` for the code changes. Once applied:
    ```bash
    torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
      --data_dir data/coconut_v1 --use_4bit \
      --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 \
      --batch_size 4 --grad_accum 8 \
      --val_batch_size 2 \
      --val_skip_buffer_minutes 60 \
      --no-gen_every_stage \
      --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \
      --push_to_hub \
      --output_dir runs/stage3_curriculum
    ```
    - Startup log **must** show: `[GPU] Tesla T4  cc=sm75  VRAM=16GB  amp_dtype=float16`
    - On session start: all local checkpoints pushed to Hub, then all pruned except resume checkpoint
    - Profile step time at steps 1–5. Target: ~41s/step (FP16 confirmed)
    - Auto-resumes from `stage_1/checkpoint-0001338`

11. **After Stage 1 completes:** Claim TRC. Move stages 2–10 to A100.

12. **On A100:** Confirm startup log: `[perf] 80GB VRAM detected: disabling gradient checkpointing`

13. **After K=4–5 gate passes on A100:** Integrate DGAC (Phase 3.4).

---

### Part 0.1.1 — Profiler snippet (use if step time regresses)

```python
# Temporary diagnostic — add inside the micro-step loop, run for 1 step, then remove
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
| Val skip threshold used wrong buffer | `--val_skip_buffer_minutes 60` (actual default in use) ✅ |
| **BF16 on T4 uses FP32 compute paths** | ✅ FIXED — `_amp_dtype` gates BF16 on cc≥(8,0); T4 uses FP16 |
| **GC wastes 20–40% on A100** | ✅ FIXED — auto-disabled at VRAM≥40GB |
| **`_amp_dtype` called in hot loop repeatedly** | ✅ FIXED — `@functools.lru_cache(maxsize=None)` |
| **`--push_to_hub` never passed → hub upload never fires** | 🔴 ADD to command — see Part 0.1 #10 |
| **`prune_epoch_checkpoints` only fires after successful val; scoped per-stage only** | 🔴 Fix in `AGENT_PROMPT_hub_prune_fix.md` |
| **Prefix re-computation at Stage k** | 🟡 FUTURE — Cache Mamba state at q_len. Implement before Stage 5 on A100. |

---

## Part 1 — Architecture

### Jamba Reasoning 3B (primary research model)
```
HuggingFace : ai21labs/AI21-Jamba-Reasoning-3B   License: Apache 2.0
Layers      : 28 total (26 Mamba, 2 Attention) → 13:1 Mamba:Attention ratio
Attention   : MQA (20 Q heads, 1 KV head)
Vocab       : 64K / Context: 256K tokens
d_model     : 2560   (confirmed Session 13)
```

---

## Part 2 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Primary research model | Jamba Reasoning 3B |
| Fine-tuning approach | QLoRA (4-bit NF4) + LoRA |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, in_proj, x_proj, dt_proj, out_proj; conv1d excluded |
| Coconut curriculum | Progressive step replacement per Meta paper |
| max_stage K | **10** (confirmed from dataset median_steps) |
| DGAC halt gate | Phase 3.4 only; λ₁ annealed from 0 |
| `--max_seq_len` | 1024 |
| `--max_grad_norm` | 0.3 for k≥2 stages |
| Session timeout | `--session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20` |
| val_batch_size | **2** |
| val accuracy samples | **50** (capped from 200) |
| val skip threshold | **`--val_skip_buffer_minutes 60`** (empirical; 120 was never in the actual command) |
| NCCL init_process_group timeout | **`timedelta(hours=4)`** |
| Stage 0 forward pass | ✅ Batched — single fused backbone call per micro-batch |
| Stage k≥1 forward pass | ✅ `_forward_batched_latent()` |
| epochs_per_stage | **1 for all stages** |
| batch_size | **4** |
| Stage 0 skip | **NO** — domain adaptation. 1 epoch sufficient. |
| gen_every_stage | **`--no-gen_every_stage`** in production (was omitted in sessions 13/14, gen ran at stage 0 end) |
| DDP val strategy | All ranks participate (interleaved shard) |
| TRC claim timing | **After Stage 1 completes** |
| amp_dtype on T4 (sm75) | ✅ **FP16** — confirmed empirical: `amp_dtype=float16`, ~41s/step |
| amp_dtype on A100+ (sm80+) | ✅ **BF16** — native BF16 tensor cores available |
| `_amp_dtype` hot-loop overhead | ✅ `@functools.lru_cache(maxsize=None)` |
| Gradient checkpointing | ✅ **Auto-disabled at VRAM≥40GB** (A100). Mandatory on T4. |

---

## Part 3 — Open Questions

| Question | Status |
|---|---|
| Does `inputs_embeds` → `last_hidden_state` work with Jamba + fast Mamba kernels at k≥1? | 🟢 CONFIRMED at Stage 1 (ce=0.41, training progressing) |
| What is step time at Stage 1 post-FP16 patch? | 🟢 **~41s/step** (empirical, Session 14). ~4× speedup over pre-patch. |
| DGAC Phase 3.4: does halt_step distribute across K≥2 after training? | 🔴 OPEN — primary research validation |
| Prefix re-computation optimization (Mamba state caching) | 🟡 OPEN — implement before Stage 5 on A100. ~3× additional speedup at Stage 10. |
| Is 1 epoch per stage sufficient for convergence? | 🟡 MONITOR — Stage 1 ce trajectory healthy so far |
| Stage 0 val_acc=0.0222: normalize_pred too strict or genuine? | 🟡 MONITOR — expected at Stage 0; CE=0.40 is the signal |

---

## Part 4 — Performance Analysis

### Step-time model (empirical)

```
Pre-patch BF16/FP32 on T4:
  t_bf16(k) ≈ 137 + 25k  seconds/step
  Stage 0: 137s  Stage 1: 162s  Stage 5: 262s  Stage 10: 387s

Post-patch FP16 on T4 (CONFIRMED Session 14):
  t_fp16(k) ≈ 34 + 6k  seconds/step   [estimated from Stage 1 empirical = 41s]
  Stage 0: ~34s  Stage 1: ~41s ✓  Stage 5: ~64s  Stage 10: ~94s
  Speedup: ~4× end-to-end (beats conservative 1.5–2.5× prediction)
```

| Platform | Condition | Stages 1–10 total | Feasible? |
|---|---|---|---|
| Dual T4 | BF16/FP32 paths (pre-patch) | ~880h | ❌ No |
| Dual T4 | FP16 patch (confirmed ~4× gain) | ~220h | ❌ Marginal |
| A100 80GB | BF16 native + no GC (~4–6× vs T4 FP16) | ~40–55h | ✅ Yes |

**Conclusion:** FP16 fix meaningfully extends Stage 1 on T4. A100 still required for stages 2–10.

### Disk usage concern (Kaggle)
Kaggle `/kaggle/working/` ≈ 20GB. Each LoRA checkpoint ≈ 1.5–2GB. With timeout saves every ~10h and no prune-on-timeout, disk fills after ~2–3 stages. Must push-then-prune on session start (see `AGENT_PROMPT_hub_prune_fix.md`).

### Known future optimization (Mamba state caching at Stage k)
Re-processing positions `0..q_len-1` in every prefix pass at Stage k wastes compute.
Caching Mamba recurrent state at `q_len` gives ~3.1× at Stage 10.
Requires `use_cache=True` → incompatible with GC → safe on A100. Implement before Stage 5.

---

## Part 5 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `jamba_coconut_finetune.py` | 3 | ✅ FP16 PATCHED | Hub+prune fix pending (see agent prompt) |
| `kaggle-utils.ipynb` | 3 | 🟡 NEEDS UPDATE | Add `--push_to_hub` and `--no-gen_every_stage` to Cell 5 command |
| `AGENT_PROMPT_hub_prune_fix.md` | 3 | 🔴 NEW — apply next session | Hub push + global prune fix |
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
K = 10 (confirmed from dataset)
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
    checkpoint-0001338/    ← emergency timeout save (no val_ce/val_acc)
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
| Stage 1 (in progress) | Kaggle Dual T4 | ~970 steps remaining × 41s ≈ 11h (1–2 sessions) |
| Stages 2–10 | TRC A100 80GB | ~40–55h total |
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
| Profile run (200 samples) shows 30s/step; full run (36906) shows 137s/step | Always profile at full scale |
| Blueprint marked stage k≥1 batching as PENDING but code already had it | Audit .py before marking blockers |
| Stage 0 skip tempting for capable base model | Stage 0 is domain adaptation, not capability training |
| Val with no KV cache at 200 samples → 5.5h decode loop | Cap at 50 |
| Timeout checker only fires inside training step loop | Pre-val checkpoint save + timeout check before val entry |
| `timedelta(minutes=60)` kills rank 1 during val | `timedelta(hours=4)` + env var |
| `graceful_exit_buffer=20min` used as val skip threshold | `--val_skip_buffer_minutes 60` separate arg |
| `torch.cuda.is_bf16_supported()` returns True on sm75 (CUDA 12 soft emulation) | `_amp_dtype` checks `cc >= (8, 0)` |
| GC mandatory on T4 but wastes 20–40% on A100 | Auto-disable GC at VRAM≥40GB |
| `_amp_dtype` called in hot loop repeatedly | `@functools.lru_cache(maxsize=None)` |
| Conservative 1.5–2.5× end-to-end FP16 speedup estimate; actual was ~4× | Always verify empirically; FP16 on T4 is more impactful than expected |
| `--push_to_hub` never added to command → hub upload silently never fires | Add flag explicitly; absence of errors ≠ uploads happening |
| `prune_epoch_checkpoints` scoped per-stage; only called after successful val → disk fills on timeout sessions | Global prune + hub push must happen on session startup |
| Blueprint showed `--val_skip_buffer_minutes 120` but actual command used 60; doc lagged behind `.py` | Python/notebook files take precedence — update docs to match code, not vice versa |
| `--no-gen_every_stage` omitted from session 13/14 commands; gen ran unexpectedly | Keep flag explicit in production command |
