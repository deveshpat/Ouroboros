# Project Ouroboros — Master Blueprint

> **Thread-resume header. Read Part 0 first in any new session.**
> **DRY Guidelines:** Session details live in `terminal_log.md`. Blueprint holds decisions, architecture, status, and next actions only.

---

## Part 0 — Quick-Resume Context

### What this project is
Coconut-Ouroboros latent reasoning injection into a Transformer-Mamba hybrid (Jamba Reasoning 3B). The Mamba SSM recurrent state acts as compressed scratch-memory across K latent thought passes, bypassing token generation during reasoning. Core mechanism from Meta's Coconut (arXiv:2412.06769), extended with DGAC (Diversity-Gated Adaptive Coconut) — our novel anti-collapse halt gate.

### Strategic Status (Updated 2026-04-14)

| Stage | Name | Status |
|---|---|---|
| 0 | Architecture & Viability (nano) | ✅ COMPLETE |
| 1 | Pre-training (nano) | ✅ Pipeline test only; retired |
| 2 | SFT (nano) | 🔴 RETIRED |
| 3 | Coconut-Ouroboros + DGAC on Jamba Reasoning 3B | 🟡 BLOCKED — awaiting one clean build_wheels_kaggle.py run with git+https fix |
| 4 | GRPO on Jamba Reasoning 3B | ⬜ NOT STARTED |
| 5 | Quantization / Edge Deploy | ⬜ NOT STARTED |

### TRC Status
✅ Accepted (email 2026-04-07). **Do not claim quota yet.** Claim after K=4 gate passes on Kaggle Dual T4.

### Dataset Confirmed (2026-04-13)
- **Train:** 36,906 samples  **Val:** 1,940 samples
- **median_steps=10  mean=10.42  max=16**
- **`--max_stage=10` for all production runs**

### GPU Arch Confirmed
- Kaggle T4: **sm_75** (most sessions)
- Kaggle Blackwell: **sm_100** (B100, allocated occasionally)
- `TORCH_CUDA_ARCH_LIST` injection auto-detects correctly in both scripts

---

### Part 0.1 — Immediate Next Actions (ordered)

1. **Apply the two-line fix to both scripts (see Part 0.2 — Resolved Blockers)**

2. **Build + upload mamba_ssm-1.2.2 wheel on Kaggle (whichever GPU you get):**
   ```bash
   python build_wheels_kaggle.py --hf_token YOUR_TOKEN 2>&1 | tee build.log
   ```
   Expected outcome: full GitHub source cloned, compiled for current arch, wheel uploaded to Hub.
   After success: future bootstrap sessions skip compilation entirely.

3. **Smoke test** (after mamba_ssm wheel is on Hub):
   ```bash
   python jamba_coconut_finetune.py \
     --data_dir data/coconut_v1 --use_4bit \
     --epochs_per_stage 1 --max_stage 2 --max_samples 200 \
     --max_seq_len 1024 --max_grad_norm 0.3 \
     --session_timeout_hours 1.5 --wandb_mode disabled --output_dir runs/smoke
   ```
   Must see `Mamba fast path: ACTIVE ✓` in output.

4. **K=0→K_max curriculum** on Kaggle Dual T4 with `torchrun --nproc_per_node=2`

5. **If K=4 gate passes**: claim TRC, run K=10 + DGAC Phase 3.4 on A100

### Part 0.2 — Pre-flight Blockers

| Blocker | Resolution |
|---|---|
| `attn_implementation` hardcoded crash | try/except fallback to `eager` ✅ |
| `use_mamba_kernels` kwarg on old TF | `_safe_from_pretrained` retries without kwarg ✅ |
| `use_mamba_kernels=False` hardcoded | Replaced with runtime probe ✅ |
| `last_hidden_state` silent None | assert in Stage 0 and Phase B ✅ |
| No graceful timeout → lost Kaggle work | `make_timeout_checker()` integrated ✅ |
| `conv1d` in LoRA targets | Explicitly excluded ✅ |
| NCCL watchdog (S5–S7) | `timeout=timedelta(minutes=60)` + `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` ✅ |
| OOM at first val step (S8) | `torch.cuda.empty_cache()` + `val_batch_size=1` ✅ |
| `--max_seq_len 512` filtered all stage 1+ samples | Default changed to 1024 ✅ |
| Exploding gradients at k≥2 | `--max_grad_norm 0.3` ✅ |
| Wheel/ABI mismatch | Retired wheel workflow; `_bootstrap()` self-installs ✅ |
| mamba-ssm 2.x API | Pinned to `mamba-ssm==1.2.2` ✅ |
| bitsandbytes version floor missing | `bitsandbytes>=0.46.1` in `_bootstrap()` ✅ |
| causal_conv1d Hub wheel (sm_100, sm_75) | Built + uploaded for both arches ✅ |
| **mamba_ssm 1.2.2 PyPI sdist is a 35kB stub — CUDA source files absent** | **Fix: change source-build pip spec from `"mamba-ssm==1.2.2"` to `"git+https://github.com/state-spaces/mamba.git@v1.2.2"` in `_bootstrap()` and `build_wheels_kaggle.py`. PyPI has only Python metadata; GitHub has full source. ✅ PATCHED 2026-04-14** |

**Exact code changes (both files):**

`jamba_coconut_finetune.py` — in `_bootstrap()`, source-build fallback block:
```python
# Find these lines:
_parts    = _base.split("-")
_pkg_name = _parts[0]
_pkg_ver  = _parts[1]
_pip_spec = f"{_pkg_name.replace('_', '-')}=={_pkg_ver}"

# Replace with:
_parts    = _base.split("-")
_pkg_name = _parts[0]
_pkg_ver  = _parts[1]
# mamba_ssm PyPI sdist (35kB) omits CUDA source files. Use git+https for full source.
if _pkg_name == "mamba_ssm":
    _pip_spec = f"git+https://github.com/state-spaces/mamba.git@v{_pkg_ver}"
else:
    _pip_spec = f"{_pkg_name.replace('_', '-')}=={_pkg_ver}"
```

`build_wheels_kaggle.py` — line 3:
```python
# OLD:
MAMBA_SSM_VERSION  = "mamba-ssm==1.2.2"
# NEW:
MAMBA_SSM_VERSION  = "git+https://github.com/state-spaces/mamba.git@v1.2.2"
```

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
| mamba-ssm version | **1.2.2 from GitHub source** (`git+https://github.com/state-spaces/mamba.git@v1.2.2`) |
| mamba install strategy | `_bootstrap()` downloads pre-built arch wheels from Hub; falls back to git+https source build; uploads result to Hub for future sessions |
| `--max_seq_len` | 1024 |
| `--max_grad_norm` | 0.3 for k≥2 stages |
| Session timeout | `--session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20` |
| val_batch_size | 1 |

---

## Part 3 — Open Questions

| Question | Status |
|---|---|
| Does `inputs_embeds` → `last_hidden_state` work with Jamba + fast Mamba kernels active? | 🟡 VERIFY next smoke test |
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
| `jamba_coconut_finetune.py` | 3 | 🟡 PATCHED | git+https fix applied; needs one clean build run |
| `build_wheels_kaggle.py` | 3 | 🟡 PATCHED | git+https fix applied; run once to cache sm75/sm100 wheels |

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
| Wheel build (one-time) | Kaggle any GPU | ~20-30 min; then cached on Hub forever |
| Smoke test | Kaggle T4 | ~10 min once wheel is on Hub |
| Stage 0→10 | Kaggle Dual T4, QLoRA + DDP | ~4-8h per session |
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
| Kaggle GPU arch unpredictable | `TORCH_CUDA_ARCH_LIST` auto-injected from `torch.cuda.get_device_capability()` |
| bitsandbytes not upgraded | `bitsandbytes>=0.46.1` in bootstrap |
| **mamba-ssm 1.2.2 PyPI sdist is a 35kB stub** | **Use `git+https://github.com/state-spaces/mamba.git@v1.2.2` for source build — never `pip install mamba-ssm==1.2.2` from PyPI directly. ~20h of GPU quota lost to this.** |
