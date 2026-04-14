## Kaggle Utils Session 7 — sm75 (Kaggle T4, 2026-04-14)
**Script:** `jamba_coconut_finetune.py` launched from `kaggle-utils.ipynb` Cell 5
**Status:** 🟡 PATCH READY — latest blocker is a verifier bug, not yet a proven bad sm75 wheel

**Notebook output (verbatim key excerpt):**
```
[bootstrap] Shim: patched 10 removed transformers.generation names ✓
[bootstrap] Phase 2: arch-aware Hub wheel install...
[bootstrap]   GPU arch: sm75  (TORCH_CUDA_ARCH_LIST=7.5+PTX if build needed)
[bootstrap]   Downloaded causal_conv1d-1.6.1-cp312-cp312-linux_x86_64-sm75.whl ✓
[bootstrap]   Installed causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl ✓
[bootstrap]   Downloaded mamba_ssm-1.2.2-cp312-cp312-linux_x86_64-sm75.whl ✓
[bootstrap]   Installed mamba_ssm-1.2.2-cp312-cp312-linux_x86_64.whl ✓
[bootstrap] Phase 3: verifying mamba fast path (symbol + CUDA op)...
[bootstrap]   ABI fingerprint: GPU=Tesla T4 sm75 | CUDA=12.8 | PyTorch=2.10.0+cu128 | Python=cp312
[bootstrap] FATAL: Mamba fast path verification FAILED: cannot import name 'selective_state_update' from 'mamba_ssm.ops.selective_scan_interface'
[bootstrap]        Exiting now (no slow-path fallback — 500s/step is unusable).
```

**Corrected diagnosis:**
- The comprehensive transformers shim worked. The notebook shows `patched 10 removed transformers.generation names ✓`.
- Both sm75 wheels downloaded and installed from Hub successfully before Phase 3 failed.
- The current blocker is the verifier itself: it imported `selective_state_update` from `mamba_ssm.ops.selective_scan_interface`.
- For Jamba fast path, the correct split is:
  - `mamba_inner_fn`, `selective_scan_fn` from `mamba_ssm.ops.selective_scan_interface`
  - `selective_state_update` from `mamba_ssm.ops.triton.selective_state_update`
- Therefore the notebook run does **not** yet prove the sm75 `mamba_ssm` wheel is bad.

**Fix prepared locally:**
- Patched `jamba_coconut_finetune_patched.py`:
  - retains the 10-alias transformers compatibility shim
  - corrects the Phase 3 verifier import path for `selective_state_update`
  - strengthens fast-path verification with real tiny CUDA/Triton ops
  - adds explicit pure-Python dependency installs for `einops` and `safetensors`

**Next action:** Replace `jamba_coconut_finetune.py` with the patched version and rerun the smoke test before any wheel rebuild.

---

# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**
**Purpose: session outcomes, key metrics, bugs found/fixed, generation samples.**
**Blueprint holds decisions/status. This file holds evidence.**

---

## Wheel Build Session 6 — sm75 (Kaggle T4, 2026-04-14)
**Script:** `jamba_coconut_finetune.py` (comprehensive shim applied)
**Status:** 🟡 PATCH READY — smoke test pending

**New failure observed (verbatim):**
```
[bootstrap] Shim: GreedySearchDecoderOnlyOutput -> GenerateDecoderOnlyOutput ✓
[bootstrap] Phase 2: arch-aware Hub wheel install...
[bootstrap]   GPU arch: sm75  (TORCH_CUDA_ARCH_LIST=7.5+PTX if build needed)
[bootstrap]   Downloaded causal_conv1d-1.6.1-cp312-cp312-linux_x86_64-sm75.whl ✓
[bootstrap]   Installed causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl ✓
[bootstrap]   Downloaded mamba_ssm-1.2.2-cp312-cp312-linux_x86_64-sm75.whl ✓
[bootstrap]   Installed mamba_ssm-1.2.2-cp312-cp312-linux_x86_64.whl ✓
[bootstrap] Phase 3: verifying mamba fast path (symbol + CUDA op)...
[bootstrap]   ABI fingerprint: GPU=Tesla T4 sm75 | CUDA=12.8 | PyTorch=2.10.0+cu128 | Python=cp312
[bootstrap] FATAL: Mamba fast path verification FAILED: cannot import name
    'SampleDecoderOnlyOutput' from 'transformers.generation'
    (/usr/local/lib/python3.12/dist-packages/transformers/generation/__init__.py)
[bootstrap]        Exiting now (no slow-path fallback — 500s/step is unusable).
```

**Root cause:** The Session 5 shim only patched `GreedySearchDecoderOnlyOutput`. `mamba_ssm 1.2.2`'s `utils/generation.py` imports additional removed names. Rather than patch one name per session, the shim was rewritten to cover the full set of 10 class names removed in `transformers>=4.44` (5 decoder-only + 5 encoder-decoder variants).

**Fix applied (2026-04-14):** Replaced the Phase 1.5 block in `_bootstrap()` with a dict-driven loop that patches all 10 removed names in one pass. See BLUEPRINT Part 0.2 for the exact replacement block.

**Both sm75 wheels confirmed on Hub — bootstrap Phase 2 is now <30s.**

---

## Wheel Build Session 5 — sm75 (Kaggle T4, 2026-04-14)
**Script:** `jamba_coconut_finetune.py` (bootstrap, git+https fix applied)
**Status:** 🔴 FAILED — `GreedySearchDecoderOnlyOutput` removed from `transformers>=4.44`

**ABI:** PyTorch=2.10.0+cu128 | Python=cp312 | CUDA=12.8

**mamba_ssm wheel:** ✅ Built from git+https source, uploaded to Hub for sm75.

**Bootstrap Phase 3 failure (verbatim):**
```
[bootstrap] FATAL: Mamba fast path verification FAILED: cannot import name
    'GreedySearchDecoderOnlyOutput' from 'transformers.generation'
    (/usr/local/lib/python3.12/dist-packages/transformers/generation/__init__.py)
[bootstrap]     Exiting now (no slow-path fallback — 500s/step is unusable).
```

**Fix applied:** Phase 1.5 shim (single-name; superseded by Session 6 comprehensive shim).

**Hub wheel status after this session:**
- `causal_conv1d-1.6.1-cp312-cp312-linux_x86_64-sm75.whl` ✅ on Hub
- `mamba_ssm-1.2.2-cp312-cp312-linux_x86_64-sm75.whl` ✅ on Hub
- `causal_conv1d-1.6.1-cp312-cp312-linux_x86_64-sm100.whl` ✅ on Hub (Session 3)

---

## Wheel Build Session 4 — sm75 (Kaggle T4, 2026-04-14)
**Status:** 🔴 FAILED — mamba_ssm PyPI sdist is a 35kB stub (missing CUDA source files)

**ABI:** PyTorch=2.10.0+cu128 | Python=cp312

**causal_conv1d:** Built from source and uploaded to Hub for sm75 ✓

**mamba_ssm source build failure (verbatim key excerpt):**
```
ninja: error: '/tmp/.../csrc/selective_scan/selective_scan.cpp', needed by
  '.../csrc/selective_scan_interface.o', missing and no known rule to make it
[bootstrap] FATAL: Source build failed for mamba-ssm==1.2.2.
```

**Fix applied:** Source-build spec changed to `git+https://github.com/state-spaces/mamba.git@v1.2.2` in both scripts.

---

## Wheel Build Session 3 + Bootstrap Attempt (Kaggle sm100, 2026-04-13)
**Status:** 🟡 PARTIAL — causal_conv1d ✅  mamba_ssm ❌ (404 on Hub)

**Bootstrap phase 2 (verbatim):**
```
[bootstrap]   Downloaded causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl ...
[bootstrap]   Installed causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl ✓
[bootstrap] FATAL: Could not download mamba_ssm-1.2.2-cp312-cp312-linux_x86_64.whl:
         404 Client Error. Entry Not Found.
```

---

## Stage 3 — Hub Wheel Session (Kaggle T4, 2026-04-13)
**Status:** 🔴 FAILED — mamba-ssm 2.3.1 API mismatch + causal_conv1d arch mismatch

**Key events (verbatim):**
```
63.9s  Successfully installed mamba-ssm-2.3.1
95.8s  FATAL: cannot import name 'selective_state_update' from
       'mamba_ssm.ops.selective_scan_interface'
```

---

## Stage 3 — Smoke Test Attempt 2 (Kaggle, 2026-04-13)
**Status:** 🔴 CRASHED — bitsandbytes too old; mamba_ssm source build failed

**Dataset confirmed:** Train: 36,906  Val: 1,940  median_steps=10  max=16  → `--max_stage=10`

---

## Stage 3 — Smoke Test (Kaggle Dual T4, 2026-04-11)
**Status:** ✅ COMPLETED — 3 stages ran. Two training bugs found.

**Key metrics (verbatim):**
```
trainable params: 26,851,328 || all params: 3,056,191,360 || trainable%: 0.8786
d_model=2560  layers=28
Stage 2/2  S2E0: ce=1.464  gn=36.926               ← EXPLODING GRAD
[val] s=2 ep=0 val_ce=0.0000 val_acc=0.0000         ← SEQ LEN BUG
```

**Fixes codified:** `--max_seq_len 1024`, `--max_grad_norm 0.3`

---

## Stage 1 Pre-training — Session 6 (Final)
**Status:** ✅ COMPLETE (graceful timeout)
```
Tokens processed: 704,544,768   Last val CE: 5.324
Hub: checkpoint-0021000 (commit=d70d2c49)
```

---

## Stage 0 — Viability Gate ✅ ALL GATES PASSED
```
G1 CE < 3.5        final CE = 2.0034   PASS ✓
G2 UWR > 0.1       mean UWR = 0.573    PASS ✓
G3 gnorm < 10.0    max = 4.0312        PASS ✓
G4 VRAM Δ < 1.0GB  Δ = 0.000 GB       PASS ✓
Total time: 3.4 min   Peak VRAM: 2.07 GB
```
