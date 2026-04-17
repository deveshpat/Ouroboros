# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in any new session.**
> **DRY Guidelines:** Session details live in `terminal_log.md`. Blueprint holds decisions, architecture, status, and next actions only.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros latent reasoning injection into a Transformer-Mamba hybrid (Jamba Reasoning 3B). The Mamba SSM recurrent state acts as compressed scratch-memory across K latent thought passes, bypassing token generation during reasoning. Core mechanism from Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — our novel anti-collapse halt gate.

### Strategic Status (Updated 2026-04-17)

| Stage | Name | Status |
|---|---|---|
| 0 | Architecture & Viability (nano) | ✅ COMPLETE |
| 1 | Pre-training (nano) | ✅ Pipeline test only; retired |
| 2 | SFT (nano) | 🔴 RETIRED |
| 3 | Coconut-Ouroboros + DGAC on Jamba Reasoning 3B | 🟡 ACTIVE — Stage 0 training COMPLETE at step 1154/1154. Val killed by NCCL watchdog (rank 1 timeout after 60min while rank 0 ran val). **checkpoint-0001154 saved and valid.** Next session: apply NCCL fix → resume → val runs → Stage 1 begins. |
| 4 | GRPO on Jamba Reasoning 3B | ⬜ NOT STARTED |
| 5 | Quantization / Edge Deploy | ⬜ NOT STARTED |

### TRC Status
✅ Accepted (email 2026-04-07). **Do not claim quota yet.** Claim after K=4 gate passes on Kaggle Dual T4.

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

1. **[DONE] Phase 2.5/2.6 ordering swap** ✅
2. **[DONE] Batched forward for Stage 0** ✅  ~30s/step on Dual T4 profile run
3. **[DONE] Profile Dual T4 throughput** ✅  val_acc=0.4000 after 12 steps
4. **[DONE] Batched latent injection for stages k>0** ✅  Audited 2026-04-15
5. **[DONE] Stage 0 training to completion** ✅  260 steps at 137s/step; checkpoint-0001154 saved
6. **[NOW — BEFORE NEXT SESSION] Apply NCCL + val timeout fixes:**
   - See `AGENT_PROMPT_nccl_val_fix.md` for exact changes
   - 5 targeted changes to `jamba_coconut_finetune.py`
   - 1 command update in `kaggle-utils.ipynb` Cell 5
   - **Must be applied before the next Kaggle session or val will fail again**

7. **[NEXT SESSION] Resume from checkpoint-0001154 with fixes applied:**
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
   - Auto-resumes from checkpoint-0001154 (step 1154/1154 — 0 training steps remaining)
   - Goes directly to val: ~98min with val_batch_size=2, 50 accuracy samples
   - If val succeeds: saves post-val checkpoint, loads best, advances to Stage 1
   - Estimated Stage 1 start: ~2h into session

8. **After K=4 gate passes**: claim TRC, run K=10 + DGAC Phase 3.4 on A100.

---

### Part 0.2 — Pre-flight Blockers

| Blocker | Resolution |
|---|---|
| `attn_implementation` hardcoded crash | try/except fallback to `eager` ✅ |
| `use_mamba_kernels` kwarg on old TF | `_safe_from_pretrained` retries without kwarg ✅ |
| `last_hidden_state` silent None | assert in Stage 0 and Phase B ✅ |
| No graceful timeout → lost Kaggle work | `make_timeout_checker()` integrated ✅ |
| `conv1d` in LoRA targets | Explicitly excluded ✅ |
| NCCL watchdog (inter-rank) | `timeout=timedelta(minutes=60)` + env var ✅ (was 60min) |
| OOM at first val step | `empty_cache()` + `val_batch_size=1` ✅ |
| `--max_seq_len 512` filtered stage 1+ samples | Default changed to 1024 ✅ |
| Exploding gradients at k≥2 | `--max_grad_norm 0.3` ✅ |
| mamba-ssm 2.x API | Pinned to 1.2.2 ✅ |
| Val with no KV cache (200 decode samples → 5+ hrs) | Cap at 50; pre-val checkpoint save ✅ |
| Timeout checker only fires inside training step loop → val kills lose epoch | Timeout check before val/gen entry; pre-val checkpoint save ✅ |
| **NCCL watchdog fires after 60min while rank 0 runs val (rank 1 at barrier())** | **🔴 OPEN — Fix: `timedelta(hours=4)` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` + `val_skip_buffer_minutes=120`. See AGENT_PROMPT_nccl_val_fix.md.** |
| **`check_timeout()` uses 20min buffer for skipping val, but val needs 120min** | **🔴 OPEN — Fix: `--val_skip_buffer_minutes 120` arg. See AGENT_PROMPT_nccl_val_fix.md.** |

**One item still requires empirical verification:**
- `inputs_embeds` → `last_hidden_state` path for Jamba Reasoning 3B with fast Mamba kernels at stage k≥1

---

## Part 1 — Architecture

### Jamba Reasoning 3B (primary research model)
```
HuggingFace : ai21labs/AI21-Jamba-Reasoning-3B   License: Apache 2.0
Layers      : 28 total (26 Mamba, 2 Attention) → 13:1 Mamba:Attention ratio
Attention   : MQA (20 Q heads, 1 KV head)
Vocab       : 64K / Context: 256K tokens
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
| val_batch_size | **2** (increased from 1; halves CE loop time) |
| val accuracy samples | **50** (capped from 200; saves ~5h) |
| val skip threshold | **`--val_skip_buffer_minutes 120`** (new; skip val if <2h remaining) |
| NCCL init_process_group timeout | **`timedelta(hours=4)`** (increased from minutes=60) |
| Stage 0 forward pass | ✅ Batched — single fused backbone call per micro-batch when `n_latent=0` |
| Stage k≥1 forward pass | ✅ `_forward_batched_latent()` processes all active samples in one call per latent step |
| epochs_per_stage | **1 for all stages** (`--stage_0_epochs 1 --epochs_per_stage 1`) |
| batch_size | **4** for all stages |
| Stage 0 skip | **NO** — necessary for LoRA domain adaptation + curriculum anchor. 1 epoch sufficient. |
| gen_every_stage | **`--no-gen_every_stage`** in production (saves 30-60min per stage) |

---

## Part 3 — Open Questions

| Question | Status |
|---|---|
| Does `inputs_embeds` → `last_hidden_state` work with Jamba + fast Mamba kernels at k≥1? | 🟡 VERIFY at stage k=1 |
| DGAC Phase 3.4: does halt_step distribute across K≥2 after training? | 🔴 OPEN — primary research validation |
| What is step time at stage k=1 on Dual T4? | 🟡 OPEN — estimate higher than stage 0 (batched latent injection adds prefill passes) |
| Is 1 epoch per stage sufficient for convergence? | 🟡 MONITOR — watch val_acc trajectory across stages |

---

## Part 4 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `jamba_coconut_finetune.py` | 3 | 🔴 NEEDS FIX | NCCL timeout + val_skip_buffer patches pending. See AGENT_PROMPT_nccl_val_fix.md. |
| `kaggle-utils.ipynb` | 3 | 🔴 NEEDS UPDATE | Cell 5 command needs --val_batch_size 2, --val_skip_buffer_minutes 120, --no-gen_every_stage |
| `AGENT_PROMPT_nccl_val_fix.md` | 3 | ✅ READY | Apply before next session |
| `prepare_coconut_dataset.py` | 3 | ✅ DONE | coconut-v1 on Hub |
| `build_wheels_kaggle.py` | 3 | ✅ DONE | sm75 wheels on Hub |

---

## Part 5 — Coconut Curriculum Design

```
Stage 0:  [Q][S1][S2]...[Sn][A]    ← standard CoT; labels on S1..Sn + A
Stage k:  [Q][●*k][S_{k+1}..Sn][A] ← first k steps replaced; labels shift
Stage K:  [Q][●*K][A]              ← all steps replaced; labels on A only
K = 10 (confirmed from dataset)
```

---

## Part 6 — DGAC

```
L_total = L_ce  +  λ₁(t) · L_ponder  +  λ₂ · L_diversity

L_diversity = mean_batch( Σ_k relu(cos_sim(h_k, h_{k-1}) − τ) ),  τ = 0.9
λ₁ schedule: 0 for steps 0-200, ramp 0→0.01 over steps 200-500, flat 0.01 after
```

**HaltGate:** Linear(2*d_model → 1), zero-initialized → outputs 0.5 at Phase 3.4 start.

---

## Part 7 — Checkpoint Format

```
output_dir/
  stage_0/
    checkpoint-0001154/    ← current live checkpoint (pre-val, acc=None ce=None)
    best/                  ← written after val completes next session
  stage_k/best/            ← loaded before advancing to stage k+1
    adapter_model/
    training_state.pt      ← {stage_k, step, epoch, step_in_epoch, val_ce, val_acc, optimizer, scheduler}
    halt_gate.pt           ← Phase 3.4 only
```

---

## Part 8 — Compute Plan

| Phase | Platform | Estimate |
|---|---|---|
| Stage 0 val (next session start) | Kaggle Dual T4 | ~98min |
| Stages 1–10 | Kaggle Dual T4 | TBD — profile step time at k=1 immediately |
| Phase 3.4 (DGAC) | TRC A100 80GB | ~6–8h |
| Phase 4 (GRPO) | TRC A100 80GB | ~8–12h |

---

## Part 9 — Hard Lessons

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
| Stage 0 skip tempting for capable base model | Stage 0 is domain adaptation, not capability training. Cannot skip. 1 epoch sufficient. |
| Val with no KV cache at 200 samples → 5.5h decode loop | Cap at 50; this is already in code |
| Timeout checker only fires inside training step loop | Pre-val checkpoint save + timeout check before val entry |
| **`timedelta(minutes=60)` in `init_process_group` kills rank 1 during val** | **`timedelta(hours=4)` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` env var** |
| **`graceful_exit_buffer_minutes=20` used as val skip threshold but val needs 120min** | **`--val_skip_buffer_minutes 120` separate arg** |
