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
| 3 | Coconut-Ouroboros + DGAC on Jamba Reasoning 3B | 🟡 BLOCKED — hardened composite verifier + auto-heal rebuild patch ready; smoke test pending |
| 4 | GRPO on Jamba Reasoning 3B | ⬜ NOT STARTED |
| 5 | Quantization / Edge Deploy | ⬜ NOT STARTED |

### TRC Status
✅ Accepted (email 2026-04-07). **Do not claim quota yet.** Claim only after Kaggle T4 curriculum clears K=4 cleanly.

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

1. **Replace `jamba_coconut_finetune.py` with the hardened script**.
   New file adds:
   - corrected `selective_state_update` import path
   - corrected `causal_conv1d_fn` verification weight shape `(dim, width)`
   - real kernel checks for `causal_conv1d_fn`, `causal_conv1d_update`, `selective_scan_fn`, `selective_state_update`, and `mamba_inner_fn`
   - **Phase 2.5 auto-heal**: on verify failure after Hub-wheel install, purge modules, rebuild both wheels from source once, reinstall, and retry
   - broader alias propagation for removed transformers generation outputs
   - explicit Phase 1 install of `einops` and `safetensors`

2. **Run the smoke test again on Kaggle T4**:
   ```bash
   python jamba_coconut_finetune.py      --data_dir data/coconut_v1 --use_4bit      --epochs_per_stage 1 --max_stage 2 --max_samples 200      --max_seq_len 1024 --max_grad_norm 0.3      --session_timeout_hours 1.5 --wandb_mode disabled --output_dir runs/smoke
   ```
   Must see `Mamba fast path: ACTIVE ✓` from the hardened composite verifier.

3. **If smoke test passes**: run K=0→K_max curriculum on Kaggle Dual T4:
   ```bash
   torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py      --data_dir data/coconut_v1 --use_4bit      --epochs_per_stage 3 --max_stage 10 --batch_size 2 --grad_accum 8      --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20      --output_dir runs/stage3_curriculum
   ```

4. **Only if Phase 2.5 source rebuild is triggered and still fails**:
   inspect the rebuilt wheel logs; at that point the next blocker is a true package/kernel incompatibility, not a bootstrap false positive.

5. **If K=4 gate passes**: claim TRC, then move to K=10 + DGAC on A100.

---

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
| mamba-ssm 2.x API | Pinned to `mamba-ssm==1.2.2` ✅ |
| bitsandbytes version floor missing | `bitsandbytes>=0.46.1` in `_bootstrap()` ✅ |
| `einops` / `safetensors` not guaranteed | Added explicitly to Phase 1 install ✅ |
| mamba_ssm 1.2.2 PyPI sdist is a 35kB stub | Use `git+https://github.com/state-spaces/mamba.git@v1.2.2` ✅ |
| Removed generation output classes in `transformers>=4.44` | 10-alias shim retained and broadened ✅ |
| `selective_state_update` imported from wrong module in verifier | Import from `mamba_ssm.ops.triton.selective_state_update` ✅ |
| `causal_conv1d_fn` verifier used wrong weight shape | Fixed to `(dim, width)` ✅ |
| False green from import-only verification | Replaced with composite real-op verifier ✅ |
| Broken Hub wheel causing repeated sessions | **Phase 2.5 auto-heal rebuild** added ✅ |
| Stale broken modules after reinstall | `sys.modules` purge before re-verify ✅ |

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
| max_stage K | **10** |
| DGAC halt gate | Phase 3.4 only; λ₁ annealed from 0 |
| Dataset Hub config | `coconut-v1` under `WeirdRunner/Ouroboros` |
| `attn_implementation` | runtime detection: flash_attention_2 if available, else eager |
| `use_mamba_kernels` | runtime probe; only False on ImportError |
| mamba-ssm version | **1.2.2 from GitHub source** |
| mamba install strategy | arch-aware Hub wheel download → composite verify → one-shot source auto-heal rebuild if verify fails |
| `--max_seq_len` | 1024 |
| `--max_grad_norm` | 0.3 for k≥2 stages |
| session timeout | `--session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20` |
| val_batch_size | 1 |

---

## Part 3 — Open Questions

| Question | Status |
|---|---|
| Does `inputs_embeds` → `last_hidden_state` work with Jamba + fast Mamba kernels active? | 🟡 VERIFY next smoke test |
| Does the hardened verifier pass on Kaggle T4 without triggering Phase 2.5 rebuild? | 🟡 VERIFY next smoke test |
| DGAC Phase 3.4: does halt_step distribute across K≥2 after training? | 🔴 OPEN |

---

## Part 4 — File Registry

| File | Stage | Status | Notes |
|---|---|---|---|
| `prepare_coconut_dataset.py` | 3 | ✅ DONE | dataset confirmed |
| `jamba_coconut_finetune.py` | 3 | 🟡 REPLACE NOW | old verifier path is stale |
| `jamba_coconut_finetune_hardened.py` | 3 | ✅ READY | corrected verifier + auto-heal rebuild |
| `build_wheels_kaggle.py` | 3 | ✅ DONE | still used for manual wheel build/debug only |

---

## Part 5 — Coconut Curriculum Design

```
Stage 0:  [Q][S1][S2][S3][A]
Stage k:  [Q][●*k][S_{k+1}..Sn][A]
Stage K:  [Q][●*K][A]
K = 10
```

---

## Part 6 — DGAC: Diversity-Gated Adaptive Coconut

```
L_total = L_ce  +  λ₁(t) · L_ponder  +  λ₂ · L_diversity
```

---

## Part 7 — Checkpoint Format

Unchanged.

---

## Part 8 — Compute Plan

| Phase | Platform | Estimate |
|---|---|---|
| Smoke test | Kaggle T4 | ~10-20 min depending on whether Phase 2.5 rebuild triggers |
| Stage 0→10 | Kaggle Dual T4 | ~4-8h per session |
| DGAC | TRC A100 80GB | ~6-8h |

---

## Part 9 — Hard Lessons (Do Not Repeat)

- Never trust import-only kernel verification.
- Never assume `selective_state_update` lives beside `selective_scan_fn` in old Mamba.
- Never pass `(dim, 1, width)` directly to `causal_conv1d_fn`; its interface expects `(dim, width)`.
- If a Hub wheel installs but verify fails, rebuild automatically before spending another session debugging it.
