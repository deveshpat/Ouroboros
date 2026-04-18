# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in any new session.**
> **DRY Guidelines:** Session details live in `terminal_log.md`. Blueprint holds decisions, architecture, status, and next actions only.

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
| 3 | Coconut-Ouroboros + DGAC on Jamba Reasoning 3B | 🟡 ACTIVE — Stage 0 val COMPLETE (ce=0.4041, acc=0.0222). Stage 1 IN PROGRESS: ~143/1154 steps, ce=0.41, step_time=~162s/step. |
| 4 | GRPO on Jamba Reasoning 3B | ⬜ NOT STARTED |
| 5 | Quantization / Edge Deploy | ⬜ NOT STARTED |

### TRC Status
✅ Accepted (email 2026-04-07).  
⚠️ **REVISED CLAIM TIMING:** Stages 1–10 on Dual T4 = ~880 GPU-hours (infeasible). **Claim TRC immediately after Stage 1 completes.** Run stages 2–10 on A100.

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
8. **[DONE]** Stage 1 training started ✅ (~162s/step confirmed empirically)

9. **[NOW] Apply performance patches before next session:**
   - See `AGENT_PROMPT_perf_fix.md` for exact changes
   - Fix 1: `_amp_dtype` — use FP16 for sm75 and below (BF16 on T4 uses FP32 paths, not tensor cores)
   - Fix 2: Auto-disable gradient checkpointing on A100/H100 (saves 20–40%)
   - Fix 3: Log GPU compute capability + effective dtype prominently at startup

10. **[NEXT SESSION] Resume Stage 1 with perf patches applied:**
    ```bash
    torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
      --data_dir data/coconut_v1 --use_4bit \
      --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 \
      --batch_size 4 --grad_accum 8 \
      --val_batch_size 2 \
      --val_skip_buffer_minutes 120 \
      --no-gen_every_stage \
      --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \
      --output_dir runs/stage3_curriculum
    ```
    - Auto-resumes from latest Stage 1 checkpoint
    - Monitor step time: target <100s with FP16 fix (was 162s)
    - **Profile immediately at step 1:** `tqdm` rate within first 5 steps

11. **After Stage 1 completes:** Claim TRC. Move stages 2–10 to A100.

12. **On A100:** Run Stage 2+ with `--no-grad-checkpoint` (auto-detected) for additional 20–40% speedup.

13. **After K=4–5 gate passes on A100:** Integrate DGAC (Phase 3.4).

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
| NCCL watchdog (60min) kills DDP val | `timedelta(hours=4)` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` ✅ (Session 13 confirmed) |
| Val skip threshold used wrong buffer | `--val_skip_buffer_minutes 120` ✅ |
| **BF16 on T4 uses FP32 compute paths** | **🔴 OPEN — Fix: cc-aware `_amp_dtype`. See `AGENT_PROMPT_perf_fix.md`.** |
| **GC wastes 20–40% on A100 (not needed at 80GB)** | **🔴 OPEN — Fix: auto-disable GC at VRAM≥40GB. See `AGENT_PROMPT_perf_fix.md`.** |
| **Prefix re-computation at Stage k** | 🟡 FUTURE — Cache Mamba state at q_len within each forward pass. Implement after Stage 2 validated on A100. |

---

## Part 1 — Architecture

### Jamba Reasoning 3B (primary research model)
```
HuggingFace : ai21labs/AI21-Jamba-Reasoning-3B   License: Apache 2.0
Layers      : 28 total (26 Mamba, 2 Attention) → 13:1 Mamba:Attention ratio
Attention   : MQA (20 Q heads, 1 KV head)
Vocab       : 64K / Context: 256K tokens
d_model     : 2560   (confirmed in Session 13)
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
| val skip threshold | **`--val_skip_buffer_minutes 120`** |
| NCCL init_process_group timeout | **`timedelta(hours=4)`** |
| Stage 0 forward pass | ✅ Batched — single fused backbone call per micro-batch |
| Stage k≥1 forward pass | ✅ `_forward_batched_latent()` processes all active samples in one call per latent step |
| epochs_per_stage | **1 for all stages** |
| batch_size | **4** for all stages |
| Stage 0 skip | **NO** — necessary for LoRA domain adaptation. 1 epoch sufficient. |
| gen_every_stage | **`--no-gen_every_stage`** in production |
| DDP val strategy | **All ranks participate** (interleaved shard); eliminates idle-rank NCCL hangs |
| TRC claim timing | **After Stage 1 completes** (revised from K=4; T4 cannot complete stages 2–10 in reasonable time) |
| amp_dtype on T4 | **FP16** (pending patch; BF16 on sm75 uses FP32 paths, not tensor cores) |
| amp_dtype on A100+ | **BF16** (cc≥8.0 has native BF16 tensor core support) |
| Gradient checkpointing | **Auto-disabled at VRAM≥40GB** (A100 does not need it; saves 20–40%) |

---

## Part 3 — Open Questions

| Question | Status |
|---|---|
| Does `inputs_embeds` → `last_hidden_state` work with Jamba + fast Mamba kernels at k≥1? | 🟢 CONFIRMED at Stage 1 (ce=0.41, training progressing normally) |
| What is step time at Stage 1 on Dual T4? | 🟢 **~162s/step** (empirically confirmed over 7 × 20-step intervals) |
| Does BF16→FP16 fix materially reduce step time on T4? | 🟡 VERIFY in next session. Theoretical: 3–5× improvement. |
| DGAC Phase 3.4: does halt_step distribute across K≥2 after training? | 🔴 OPEN — primary research validation |
| Prefix re-computation optimization (Mamba state caching within forward) | 🟡 OPEN — implement post-Stage 2 |
| Is 1 epoch per stage sufficient for convergence? | 🟡 MONITOR — Stage 1 ce trajectory healthy so far |
| Stage 0 val_acc=0.0222: normalize_pred too strict or genuine model weakness? | 🟡 MONITOR — expected at Stage 0; CE=0.40 is the signal |

---

## Part 4 — Performance Analysis

### Step-time model (empirical)

```
t(k) ≈ 137 + 25k  seconds/step   (k = stage number)

Stage 0 : 137s  (1 full forward pass)
Stage 1 : 162s  (1 prefix pass of ~q_len + 1 full pass)  ← empirically confirmed
Stage 5 : 262s
Stage 10: 387s
```

Each additional latent pass adds ~25s because prefix passes process sequences of length `q_len+j ≈ 150–160` tokens, which is ~30% of a full sequence forward pass (~83s = 0.30 × ~137s × 2 for prefix+full).

### Total compute (stages 1–10, 1154 steps each)

| Platform | Total stage 1–10 | Sessions needed | Feasible? |
|----------|-----------------|-----------------|-----------|
| Dual T4 (BF16/FP32 paths) | ~880h | ~80 Kaggle | ❌ No |
| Dual T4 (FP16 fix, 3× gain) | ~293h | ~27 Kaggle | ❌ No |
| A100 80GB (BF16 native, 4–6× vs T4 FP16) | ~150–220h | ~5–9 TRC sessions | ✅ Feasible |

**Conclusion:** BF16→FP16 fix makes T4 viable for Stage 1 and possibly Stage 2 (buys time while TRC is claimed). A100 is required for stages 2–10.

### Known algorithmic inefficiency (future optimization)

At Stage k, `_forward_batched_latent` re-processes positions `0..q_len-1` in every prefix pass. For Stage 10:
- Current: 2045 token-passes per micro-step
- Optimal (Mamba state caching): 660 token-passes
- Potential: ~3.1× additional speedup

Implementation requires exposing Mamba recurrent state (`past_key_values`) within the training forward pass while preserving gradient flow. Feasible on A100 (no GC constraint). Implement after Stage 2 is validated.

---

## Part 5 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `jamba_coconut_finetune.py` | 3 | 🔴 NEEDS PATCH | BF16→FP16 + auto-GC disable. See `AGENT_PROMPT_perf_fix.md`. |
| `kaggle-utils.ipynb` | 3 | ✅ UP TO DATE | Cell 5 command current |
| `AGENT_PROMPT_perf_fix.md` | 3 | ✅ READY | Apply before next session |
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
    checkpoint-0001154/    ← Stage 0 pre-val (acc=0.0222 ce=0.4041 — now updated)
    best/                  ← Stage 0 best (acc=0.0222 ce=0.4041)
  stage_1/
    checkpoint-XXXXXXX/    ← latest Stage 1 checkpoint (in progress)
    best/                  ← written after Stage 1 val completes
  stage_k/best/
    adapter_model/
    training_state.pt      ← {stage_k, step, epoch, step_in_epoch, val_ce, val_acc, optimizer, scheduler}
    halt_gate.pt           ← Phase 3.4 only
```

---

## Part 9 — Compute Plan

| Phase | Platform | Estimate |
|---|---|---|
| Stage 1 (in progress) | Kaggle Dual T4 | ~50h remaining at 162s/step; ~15h if FP16 fix works |
| Stages 2–10 | TRC A100 80GB | ~150–220h total (~5–9 sessions) |
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
| `graceful_exit_buffer=20min` used as val skip threshold but val needs 120min | `--val_skip_buffer_minutes 120` separate arg |
| **`torch.cuda.is_bf16_supported()` returns True on sm75 (CUDA 12 soft emulation)** | **`_amp_dtype` must check cc≥8.0 for native BF16; use FP16 on sm75** |
| **GC mandatory on T4 but wastes 20–40% on A100** | **Auto-disable GC at VRAM≥40GB** |
| **Stage k step time scales as ~137+25k s — stages 2–10 take ~880h on T4** | **Claim TRC after Stage 1; T4 cannot complete the curriculum** |
| Rank 0 running val solo → idle rank 1 hits NCCL barrier → watchdog fires | DDP val: all ranks participate with interleaved sharding |
