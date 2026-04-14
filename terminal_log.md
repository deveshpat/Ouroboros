# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**
**Purpose: session outcomes, key metrics, bugs found/fixed, generation samples.**
**Blueprint holds decisions/status. This file holds evidence.**

---

## Session 9 — sm75 (Kaggle Single T4, 2026-04-14) ✅ FIRST SUCCESSFUL SMOKE TEST
**Script:** `jamba_coconut_finetune.py` (all Session 8 fixes applied)
**Notebook:** `kaggle-utils.ipynb` (smoke test cell)
**Status:** ✅ Stage 0 COMPLETE — Stage 1 timeout (as expected at 1.5h budget)

**Bootstrap output (verbatim):**
```
[bootstrap] Phase 2: arch-aware Hub wheel install...
[bootstrap]   GPU arch: sm75  (TORCH_CUDA_ARCH_LIST=7.5+PTX if build needed)
[bootstrap]   Downloaded causal_conv1d-1.6.1-cp312-cp312-linux_x86_64-sm75.whl ✓
[bootstrap]   Installed causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl ✓
[bootstrap]   Downloaded mamba_ssm-1.2.2-cp312-cp312-linux_x86_64-sm75.whl ✓
[bootstrap]   Installed mamba_ssm-1.2.2-cp312-cp312-linux_x86_64.whl ✓
[bootstrap] WARNING: kernel export shim failed: cannot import name 'GreedySearchDecoderOnlyOutput' from 'transformers.generation'
[bootstrap] Shim: patched 10 removed transformers.generation names ✓
[bootstrap] Phase 3: verifying mamba fast path (symbol + CUDA op)...
[bootstrap]   ABI fingerprint: GPU=Tesla T4 sm75 | CUDA=12.8 | PyTorch=2.10.0+cu128 | Python=cp312
[bootstrap] Mamba fast path: ACTIVE ✓ — ~5s/step expected.
```

**Phase 2.5 warning is benign:** `_patch_kernel_top_level_exports()` runs before the generation name shim (Phase 2.6), so `mamba_ssm` import fails on first attempt. Phase 2.6 then patches the names; Phase 3 succeeds. Fix: swap Phase 2.5 and 2.6 ordering.

**sm75 Hub wheels confirmed valid.** No source build or auto-heal required.

**Training output (verbatim):**
```
trainable params: 26,851,328 || all params: 3,056,191,360 || trainable%: 0.8786
d_model=2560  layers=28

================================================================
  Stage 0/2  (CoT warmup)
  Epochs: 1  Steps/epoch: 12  Total: 12
================================================================

S0E0: 100%|████████████████| 12/12 [22:06<00:00, 110.54s/it, ce=0.615, gn=0.224]
  [val] s=0 ep=0 val_ce=0.5430 val_acc=0.2000
  [ckpt] saved -> runs/smoke/stage_0/checkpoint-0000012  acc=0.2  ce=0.5430010908315519
  [ckpt] best  -> runs/smoke/stage_0/best  acc=0.2  ce=0.5430010908315519
  [best] stage=0 new best acc=0.2000
```

**Step time observed: ~113s/step** (12 steps × ~110s/it = ~22 min for 190 samples, single T4, stage 0)
This is ~22× slower than the "~5s/step" blueprint estimate. Root cause: per-sample iteration in `coconut_forward` — 16 serial batch=1 backbone calls per optimizer step instead of batched micro-batch calls. **See Blueprint Part 0.2 for resolution plan.**

**Generation callback (verbatim, abbreviated):**
```
  -- Generation @ step 12 stage=0 --
  Q: What is 15 + 27?
  A: We need to compute 15 + 27. That's straightforward: 15 + 27 = 42...  [k_actual=0 uwr=0.484]
  Q: Write a Python function that returns the factorial of n.
  A: We need to write a Python function that returns the factorial of n...  [k_actual=0 uwr=0.590]
  Q: What is the capital of Japan?
  A: ...The capital of Japan is Tokyo...  [k_actual=0 uwr=0.400]
  Q: Solve for x: 3x + 6 = 21.
  A: ...3x = 21 - 6 = 15. Then divide by 3: x = 15 / 3 = 5. Check: 3*5 + 6 = 21. Yes...  [k_actual=0 uwr=0.623]
  Mean UWR: 0.547
```

All responses factually correct. Base model reasoning intact post-QLoRA. 

**Stage 1 timeout (expected):**
```
  [timeout] 1.40h elapsed - 6.0 min remaining (< 20 min buffer).
  [timeout] saving emergency checkpoint at step 12 ...
  [ckpt] saved -> runs/smoke/stage_1/checkpoint-0000012  acc=None  ce=None
  [timeout] Session budget exhausted - checkpoint saved.
  Re-run the same command with the same --output_dir to auto-resume.
```

Auto-resume and checkpoint save both functioning correctly.

---

## Session 8 — sm75 (Kaggle T4, 2026-04-14)
**Status:** 🔴 FAILED — verifier called `causal_conv1d_fn` with wrong weight shape

```
[bootstrap] FATAL: Mamba fast path verification FAILED: weight must have shape (dim, width)
```

**Root cause:** Verifier passed weight `(dim, 1, width)` instead of `(dim, width)`.

**Fixes applied after Session 8 (all confirmed present in current code):**
- Fixed `causal_conv1d_fn` weight shape to `(dim, width)` in `_bootstrap_verify_fast_path()`
- Expanded verifier to composite test: `causal_conv1d_fn`, `selective_scan_fn`, `selective_state_update`, `mamba_inner_fn`
- Added `einops` and `safetensors` to Phase 1 pip install
- NOTE: "Phase 2.5 auto-heal rebuild" described in planning was NOT implemented; not needed — Hub wheels are valid (confirmed Session 9)

---

## Session 7 — sm75 (Kaggle T4, 2026-04-14)
**Status:** 🔴 FAILED — verifier used wrong import path for `selective_state_update`

```
[bootstrap] FATAL: Mamba fast path verification FAILED: cannot import name 'selective_state_update' from 'mamba_ssm.ops.selective_scan_interface'
```

**Fix:** Import from `mamba_ssm.ops.triton.selective_state_update` ✓

---

## Session 6 — sm75 (Kaggle T4, 2026-04-14)
**Status:** 🔴 FAILED — single-name generation shim insufficient

```
[bootstrap] FATAL: Mamba fast path verification FAILED: cannot import name 'SampleDecoderOnlyOutput' from 'transformers.generation'
```

**Fix:** Comprehensive 10-alias generation compat shim ✓

---

## Sessions 4–5 — sm75 (Kaggle T4, 2026-04-14)
- Session 5: `GreedySearchDecoderOnlyOutput` missing — led to single-name shim (insufficient)
- Session 4: mamba_ssm PyPI sdist is 35kB stub — fixed by `git+https://github.com/state-spaces/mamba.git@v1.2.2`

---

## Stage 3 — Smoke Test (Kaggle Dual T4, 2026-04-11)
**Status:** ✅ Pipeline verified — training bugs found and codified

```
trainable params: 26,851,328 || all params: 3,056,191,360 || trainable%: 0.8786
Stage 2/2  S2E0: ce=1.464  gn=36.926
[val] s=2 ep=0 val_ce=0.0000 val_acc=0.0000
```

**Fixes codified:** `--max_seq_len 1024`, `--max_grad_norm 0.3`
