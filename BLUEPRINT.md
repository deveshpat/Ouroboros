# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in any new session.**
> **DRY Guidelines:** Session details live in `terminal_log.md`. Blueprint holds decisions, architecture, status, and next actions only.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros latent reasoning injection into a Transformer-Mamba hybrid (Jamba Reasoning 3B). The Mamba SSM recurrent state acts as compressed scratch-memory across K latent thought passes, bypassing token generation during reasoning. Core mechanism from Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — our novel anti-collapse halt gate.

### Strategic Status (Updated 2026-04-13)

| Stage | Name | Status |
|---|---|---|
| 0 | Architecture & Viability (nano) | ✅ COMPLETE |
| 1 | Pre-training (nano) | ✅ Pipeline test only; retired |
| 2 | SFT (nano) | 🔴 RETIRED |
| 3 | Coconut-Ouroboros + DGAC on Jamba Reasoning 3B | 🔴 BLOCKED — 2 bootstrap fixes required before smoke test can pass |
| 4 | GRPO on Jamba Reasoning 3B | ⬜ NOT STARTED |
| 5 | Quantization / Edge Deploy | ⬜ NOT STARTED |

### TRC Status
✅ Accepted (email 2026-04-07). **Do not claim quota yet.** Claim after K=4 gate passes on Kaggle Dual T4.

### Dataset Confirmed (2026-04-13)
- **Train:** 36,906 samples  **Val:** 1,940 samples
- **median_steps=10  mean=10.42  max=16**
- **`--max_stage=10` for all production runs**

---

### Part 0.1 — Immediate Next Actions (ordered)

1. **Fix `_bootstrap()` in `jamba_coconut_finetune.py`** — TWO required changes:

   **Fix A (easy, confirmed):** Pin bitsandbytes version floor:
   ```python
   # BEFORE
   "bitsandbytes",
   # AFTER
   "bitsandbytes>=0.46.1",
   ```

   **Fix B (critical, mamba_ssm still not installing):** The source build runs (~30 min) but the module is absent post-install. Root cause unclear — build stderr was in truncated log. Two changes needed:
   - Inject `TORCH_CUDA_ARCH_LIST` into the environment before the pip subprocess call (same logic as `build_wheels_kaggle.py`'s `_build_env_vars()`)
   - Capture build stderr to a file so the next session can diagnose the exact failure

   Proposed `_bootstrap()` approach:
   ```python
   import torch, subprocess, sys, os
   def _bootstrap():
       env = os.environ.copy()
       env["MAX_JOBS"] = "4"
       if torch.cuda.is_available():
           major, minor = torch.cuda.get_device_capability(0)
           env["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}+PTX"
           print(f"[bootstrap] TORCH_CUDA_ARCH_LIST={env['TORCH_CUDA_ARCH_LIST']}")
       packages = [
           "transformers>=4.54.0", "peft", "datasets", "tqdm", "wandb",
           "bitsandbytes>=0.46.1", "accelerate", "huggingface_hub",
           "mamba-ssm==1.2.2", "causal-conv1d>=1.4.0",
       ]
       result = subprocess.run(
           [sys.executable, "-m", "pip", "install", "-q",
            "--no-build-isolation", *packages],
           env=env, check=False,
       )
       ...
   ```
   Note: `torch.cuda` must be importable before `_bootstrap()` runs. Since torch is pre-installed in the Kaggle container, this is safe.

2. **Smoke test** (after fixes above):
   ```
   python jamba_coconut_finetune.py \
     --data_dir data/coconut_v1 --use_4bit \
     --epochs_per_stage 1 --max_stage 2 --max_samples 200 \
     --max_seq_len 1024 --max_grad_norm 0.3 \
     --session_timeout_hours 1.5 --wandb_mode disabled --output_dir runs/smoke
   ```
   Must see `mamba CUDA kernels: OK` and `val_acc > 0` to proceed.

3. **K=0→K_max curriculum** on Kaggle Dual T4 with `torchrun --nproc_per_node=2`

4. **If K=4 gate passes**: claim TRC, run K=10 + DGAC Phase 3.4 on A100

### Part 0.2 — Pre-flight Blockers

| Blocker | Resolution |
|---|---|
| `attn_implementation` hardcoded crash | try/except fallback to `eager` ✅ |
| `use_mamba_kernels` kwarg on old TF | `_safe_from_pretrained` retries without kwarg ✅ |
| `use_mamba_kernels=False` hardcoded | Replaced with kernel probe ✅ |
| `last_hidden_state` silent None | assert in Stage 0 and Phase B ✅ |
| No graceful timeout → lost Kaggle work | `make_timeout_checker()` integrated ✅ |
| `conv1d` in LoRA targets | Explicitly excluded ✅ |
| NCCL watchdog (S5–S7) | `timeout=timedelta(minutes=60)` + `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` ✅ |
| OOM at first val step (S8) | `torch.cuda.empty_cache()` + `val_batch_size=1` ✅ |
| `--max_seq_len 512` filtered all stage 1+ samples | Default changed to 1024 ✅ |
| Exploding gradients at k≥2 | `--max_grad_norm 0.3` ✅ |
| Wheel/ABI mismatch | Retired wheel workflow; `_bootstrap()` self-installs ✅ |
| mamba-ssm 2.x API | Pinned to `mamba-ssm==1.2.2` ✅ |
| **bitsandbytes < 0.46.1 in container** | **`"bitsandbytes>=0.46.1"` in `_bootstrap()` 🔴 NOT YET APPLIED** |
| **mamba_ssm source build silently fails** | **Inject `TORCH_CUDA_ARCH_LIST` + capture stderr in `_bootstrap()` 🔴 NOT YET APPLIED** |

One item still requires empirical verification during smoke test:
- `inputs_embeds` → `last_hidden_state` path for Jamba Reasoning 3B **with fast Mamba kernels active**

---

## Part 1 — Architecture

### Jamba Reasoning 3B (primary research model)
```
HuggingFace : ai21labs/AI21-Jamba-Reasoning-3B   License: Apache 2.0
Layers      : 28 total (26 Mamba, 2 Attention) → 13:1 Mamba:Attention ratio
Attention   : MQA (20 Q heads, 1 KV head)
Vocab       : 64K
Context     : 256K tokens
```

**Why Reasoning 3B over Jamba2-3B:** Reasoning 3B already has explicit CoT traces that Coconut progressively replaces with latent passes.

---

## Part 2 — Resolved Decisions

| Decision | Resolution |
|---|---|
| Primary research model | Jamba Reasoning 3B |
| Fine-tuning approach | QLoRA (4-bit NF4) + LoRA via `--use_4bit` |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, in_proj, x_proj, dt_proj, out_proj; conv1d excluded |
| Coconut curriculum | Progressive step replacement per Meta paper |
| Stage advancement | Epoch-based + best-accuracy checkpoint selection |
| max_stage K | **10** (confirmed from dataset median_steps) |
| DGAC halt gate | Phase 3.4 only; λ₁ annealed from 0 |
| Dataset Hub config | `coconut-v1` under `WeirdRunner/Ouroboros` |
| `attn_implementation` | Runtime detection: flash_attention_2 if available, else eager |
| `use_mamba_kernels` | Runtime probe; only False on ImportError |
| Mamba install strategy | `_bootstrap()` self-installs `mamba-ssm==1.2.2` + `causal-conv1d>=1.4.0` via `--no-build-isolation`. `build_wheels_kaggle.py` retained for TRC A100 only. |
| `--max_seq_len` | 1024 (512 filtered ~100% of traces at stage ≥ 1) |
| `--max_grad_norm` | 0.3 for k≥2 stages |
| Session timeout | `--session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20` |
| val_batch_size | 1 (Stage 2 S8 OOM'd with 16 at seq=2048) |

---

## Part 3 — Open Questions

| Question | Status |
|---|---|
| Does `inputs_embeds` → `last_hidden_state` work with Jamba + fast Mamba kernels active? | 🟡 VERIFY next smoke test |
| Why does mamba_ssm source build run (~30 min) but module not install? | 🔴 OPEN — needs verbose stderr capture |
| DGAC Phase 3.4: does halt_step distribute across K≥2 after training? | 🔴 OPEN — primary research validation |

---

## Part 4 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `baseline_trm_mamba.py` | 0 | ✅ COMPLETE | Retired; architecture reference only |
| `viability_gate.py` | 0 | ✅ COMPLETE | |
| `training_utils.py` | All nano | ✅ COMPLETE | Not used in Jamba scripts |
| `pretrain.py` | 1 | ✅ COMPLETE | Hub: ckpt-0021000 |
| `prepare_sft_dataset.py` | 2 | ✅ DONE | sft-mix-v1 cached; not reused for Coconut |
| `train_sft.py` | 2 | ✅ PATCHED | Retired |
| `prepare_coconut_dataset.py` | 3 | ✅ DONE | coconut-v1 on Hub confirmed (36906/1940 samples) |
| `jamba_coconut_finetune.py` | 3 | 🔴 NEEDS PATCH | `_bootstrap()` missing `bitsandbytes>=0.46.1` + ARCH_LIST injection |
| `build_wheels_kaggle.py` | 3 | 🟡 FALLBACK ONLY | For TRC A100 sessions only |

---

## Part 5 — Coconut Curriculum Design

```
Stage 0:  [Q][S1][S2][S3][A]     ← standard CoT; labels on S1..Sn + A
Stage k:  [Q][●*k][S_{k+1}..Sn][A]   ← first k steps replaced; labels shift
Stage K:  [Q][●*K][A]            ← all steps replaced; labels on A only
K = 10 (confirmed from dataset)
```

---

## Part 6 — DGAC: Diversity-Gated Adaptive Coconut

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
  stage_0/best/
    adapter_model/         ← PEFT LoRA weights
    training_state.pt      ← {stage_k, step, epoch, val_ce, val_acc, optimizer, scheduler}
  ...
  stage_10/best/           ← resume_from for Phase 3.4 DGAC run
    halt_gate.pt           ← HaltGate state dict (Phase 3.4 only)
```

---

## Part 8 — Compute Plan

| Phase | Platform | Estimate |
|---|---|---|
| Smoke test | Kaggle T4 | ~30-60 min bootstrap (first session) + ~10 min training |
| Stage 0→10 | Kaggle Dual T4 (2×16GB), QLoRA + DDP | ~4-8h training per session (multiple sessions) |
| Phase 3.4 (DGAC) | TRC A100 80GB | ~6-8h |
| Phase 4 (GRPO) | TRC A100 80GB | ~8-12h |

---

## Part 9 — Hard Lessons (Do Not Repeat)

| Lesson | Codified As |
|---|---|
| val_batch_size=16 → OOM | `--val_batch_size 1` default + `empty_cache()` |
| NCCL watchdog kills DDP | `timedelta(minutes=60)` + graceful exit |
| max_seq_len=512 filtered all stage 1+ | `--max_seq_len 1024` |
| gn=36.9 at k=2 | `--max_grad_norm 0.3` |
| `use_mamba_kernels=False` hardcoded → 100× slow | Runtime probe |
| mamba-ssm 2.x broke fast path | Pinned to 1.2.2 |
| Kaggle GPU arch unpredictable | `TORCH_CUDA_ARCH_LIST` injection at build time |
| bitsandbytes not upgraded → crash on 4-bit load | `bitsandbytes>=0.46.1` in bootstrap (🔴 not yet applied) |
| mamba_ssm build runs but module absent — stderr invisible | Verbose build log + stderr capture (🔴 not yet applied) |
