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
| 3 | Coconut-Ouroboros + DGAC on Jamba Reasoning 3B | 🟡 BLOCKED — comprehensive shim works; fast-path verifier import path was wrong; replace script with patched version and re-run smoke test |
| 4 | GRPO on Jamba Reasoning 3B | ⬜ NOT STARTED |
| 5 | Quantization / Edge Deploy | ⬜ NOT STARTED |

### TRC Status
✅ Accepted (email 2026-04-07). **Do not claim quota yet.** Claim after K=4 gate passes on Kaggle Dual T4.

### Dataset Confirmed (2026-04-13)
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

1. **Replace `jamba_coconut_finetune.py` with the patched version**.
   Keep the comprehensive Phase 1.5 10-alias transformers shim, but also fix Phase 3 verification so `selective_state_update` is imported from `mamba_ssm.ops.triton.selective_state_update` rather than `mamba_ssm.ops.selective_scan_interface`.

2. **Re-run the smoke test immediately — do not rebuild the sm75 wheel yet.**
   The latest Kaggle notebook run shows the 10-name shim worked and both sm75 wheels installed successfully; the observed failure is now the verifier's wrong import path, so the wheel is not yet proven bad.
   ```bash
   python jamba_coconut_finetune.py \
     --data_dir data/coconut_v1 --use_4bit \
     --epochs_per_stage 1 --max_stage 2 --max_samples 200 \
     --max_seq_len 1024 --max_grad_norm 0.3 \
     --session_timeout_hours 1.5 --wandb_mode disabled --output_dir runs/smoke
   ```
   Expected bootstrap evidence now:
   - `[bootstrap] Shim: patched 10 removed transformers.generation names ✓`
   - wheels install from Hub on sm75
   - `Mamba fast path: ACTIVE ✓`

3. **If the corrected smoke test passes**: run the K=0→K_max curriculum on Kaggle Dual T4.
   ```bash
   torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
     --data_dir data/coconut_v1 --use_4bit \
     --epochs_per_stage 3 --max_stage 10 --batch_size 2 --grad_accum 8 \
     --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \
     --output_dir runs/stage3_curriculum
   ```

4. **If the corrected smoke test still fails inside a real CUDA/Triton op**: treat that as the first actual wheel-health failure and add Phase 2.5 auto-heal logic (purge + source rebuild + re-upload) before retrying.

5. **If K=4 gate passes**: claim TRC, then run K=10 + DGAC Phase 3.4 on A100.

6. **If an sm100 session is allocated**: run `build_wheels_kaggle.py` to cache the `mamba_ssm` sm100 wheel.

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
| causal_conv1d Hub wheel (sm100, sm75) | Built + uploaded for both arches ✅ |
| mamba_ssm 1.2.2 PyPI sdist is a 35kB stub | pip spec changed to `git+https://github.com/state-spaces/mamba.git@v1.2.2` ✅ |
| `GreedySearchDecoderOnlyOutput` removed in `transformers>=4.44` | Partial shim applied in prior session ✅ |
| **`SampleDecoderOnlyOutput` (and 9 other generation output classes) removed in `transformers>=4.44`; mamba_ssm 1.2.2 imports them all** | **Fix: replace single-name shim with comprehensive 10-alias patch. See exact code below. ✅ PATCHED 2026-04-14** |
| **`_bootstrap_verify_fast_path()` imported `selective_state_update` from the wrong module** | **Fix: import it from `mamba_ssm.ops.triton.selective_state_update`, keep `selective_scan_fn` / `mamba_inner_fn` from `mamba_ssm.ops.selective_scan_interface`, and verify with real tiny CUDA/Triton ops. ✅ PATCHED LOCAL 2026-04-14** |
| **sm75 Hub wheel was suspected broken after notebook Cell 5** | **Status corrected: latest evidence only proves the verifier import path was wrong. Do not rebuild the wheel unless the corrected verifier fails a real kernel op. ✅ DIAGNOSIS UPDATED 2026-04-14** |

**Exact code change — replace the entire Phase 1.5 block in `_bootstrap()` in `jamba_coconut_finetune.py`:**

```python
    # ── Phase 1.5: transformers / mamba_ssm compatibility shim ───────────────
    # mamba_ssm 1.2.2 imports multiple generation output class names that were
    # removed in transformers>=4.44 (GreedySearch*, Sample*, BeamSearch*, etc.).
    # Backfill ALL removed names as aliases for their modern replacements in
    # one pass so we never debug one missing name per session.
    try:
        import importlib as _il2
        _il2.invalidate_caches()
        import transformers.generation as _tg_mod

        # Full mapping: removed name → replacement class in transformers>=4.44
        _GENERATION_COMPAT_ALIASES = {
            # Decoder-only
            "GreedySearchDecoderOnlyOutput":      "GenerateDecoderOnlyOutput",
            "SampleDecoderOnlyOutput":            "GenerateDecoderOnlyOutput",
            "ContrastiveSearchDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
            "BeamSearchDecoderOnlyOutput":        "GenerateBeamDecoderOnlyOutput",
            "BeamSampleDecoderOnlyOutput":        "GenerateBeamDecoderOnlyOutput",
            # Encoder-decoder (mamba_ssm may import these for seq2seq completeness)
            "GreedySearchEncoderDecoderOutput":      "GenerateEncoderDecoderOutput",
            "SampleEncoderDecoderOutput":            "GenerateEncoderDecoderOutput",
            "ContrastiveSearchEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
            "BeamSearchEncoderDecoderOutput":        "GenerateBeamEncoderDecoderOutput",
            "BeamSampleEncoderDecoderOutput":        "GenerateBeamEncoderDecoderOutput",
        }
        _patched = []
        for _old, _new in _GENERATION_COMPAT_ALIASES.items():
            if getattr(_tg_mod, _old, None) is None:
                _repl = getattr(_tg_mod, _new, None)
                if _repl is not None:
                    setattr(_tg_mod, _old, _repl)
                    _patched.append(_old)
        if _patched:
            print(f"[bootstrap] Shim: patched {len(_patched)} removed "
                  f"transformers.generation names ✓")
        else:
            print("[bootstrap] Shim: all generation names present (no patch needed)")
    except ImportError:
        pass  # transformers not yet importable; Phase 1 likely failed above
    except Exception as _shim_err:
        print(f"[bootstrap] WARNING: transformers shim failed: {_shim_err}")
        print("[bootstrap]          mamba_ssm import may fail at Phase 3 verification.")
```

One item still requires empirical verification during smoke test:
- `inputs_embeds` → `last_hidden_state` path for Jamba Reasoning 3B **with fast Mamba kernels active**
- **Important:** the current notebook failure no longer supports the conclusion that the sm75 `mamba_ssm` wheel is broken; rerun with the corrected verifier first.

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
| mamba install strategy | `_bootstrap()` downloads pre-built arch wheels from Hub; falls back to git+https source build; uploads result to Hub; shim backfills ALL removed transformers generation output class names in one pass |
| `--max_seq_len` | 1024 |
| `--max_grad_norm` | 0.3 for k≥2 stages |
| Session timeout | `--session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20` |
| val_batch_size | 1 |

---

## Part 3 — Open Questions

| Question | Status |
|---|---|
| Does `inputs_embeds` → `last_hidden_state` work with Jamba + fast Mamba kernels active? | 🟡 VERIFY next smoke test |
| Does the sm75 Hub `mamba_ssm` wheel pass the corrected fast-path verifier and tiny runtime ops? | 🟡 VERIFY next smoke test |
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
| `jamba_coconut_finetune.py` | 3 | 🟡 NEEDS REPLACEMENT | Current local version must be replaced with the patched file that keeps the 10-alias shim and fixes the Phase 3 verifier import path |
| `jamba_coconut_finetune_patched.py` | 3 | ✅ PATCHED LOCAL | Corrected `selective_state_update` import path, retained comprehensive 10-alias shim, and strengthened fast-path verification |
| `build_wheels_kaggle.py` | 3 | ✅ DONE | git+https fix applied; sm75 + sm100 causal_conv1d on Hub; sm75 mamba_ssm on Hub |

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
| Smoke test | Kaggle T4 | ~10 min (wheels cached) |
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
| mamba-ssm 1.2.2 PyPI sdist is a 35kB stub | Use `git+https://github.com/state-spaces/mamba.git@v1.2.2`. ~20h GPU quota lost. |
| **Single-name shim → one removed class fixed per session** | **Comprehensive 10-alias shim covering the entire removed generation output family. Never patch one name at a time.** |
