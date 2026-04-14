# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**
**Purpose: session outcomes, key metrics, bugs found/fixed, generation samples.**
**Blueprint holds decisions/status. This file holds evidence.**

---

## Kaggle Utils Session 8 — sm75 (Kaggle T4, 2026-04-14)
**Script synced from repo commit:** `5fcd2fde5ae8529014d8539d67eeb7a60270ce67`
**Status:** 🔴 FAILED — verifier advanced past import-path bug, then failed on bad test input shape

**Notebook smoke-test output (verbatim):**
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
[bootstrap]   Wheel bases: ['causal_conv1d-1.6.1-cp312-cp312-linux_x86_64', 'mamba_ssm-1.2.2-cp312-cp312-linux_x86_64']
[bootstrap] FATAL: Mamba fast path verification FAILED: weight must have shape (dim, width)
[bootstrap]        Exiting now (no slow-path fallback — 500s/step is unusable).
```

**Root cause:** The prior patch correctly fixed the `selective_state_update` import path, but `_bootstrap_verify_fast_path()` still called `causal_conv1d_fn` with weight shape `(dim, 1, width)`. The public `causal_conv1d_fn` interface expects weight shape `(dim, width)`. This was therefore a verifier bug, not yet evidence of a broken sm75 wheel.

**Hardening applied after Session 8:**
1. Fixed `causal_conv1d_fn` verifier input shape to `(dim, width)`.
2. Expanded verifier from a single conv call to a composite real-op smoke test covering:
   - `causal_conv1d_fn`
   - `causal_conv1d_update`
   - `selective_scan_fn`
   - `selective_state_update`
   - `mamba_inner_fn`
3. Added **Phase 2.5 auto-heal rebuild**:
   - if Hub-wheel install succeeds but composite verify fails,
   - purge cached `mamba_ssm` / `causal_conv1d` modules,
   - rebuild both wheels from source once,
   - reinstall, and re-verify in the same session.
4. Added explicit Phase 1 install for `einops` and `safetensors`.
5. Broadened removed-generation-name alias propagation beyond `transformers.generation` to reduce namespace edge cases.

**Interpretation:** Session 8 eliminated the earlier false blocker. The next run with the hardened script should tell us whether the sm75 Hub wheels are actually valid, or whether Phase 2.5 must heal them from source.

---

## Kaggle Utils Session 7 — sm75 (Kaggle T4, 2026-04-14)
**Status:** 🔴 FAILED — verifier import path wrong for `selective_state_update`

**Notebook smoke-test output (verbatim key excerpt):**
```
[bootstrap] Shim: patched 10 removed transformers.generation names ✓
[bootstrap]   Downloaded mamba_ssm-1.2.2-cp312-cp312-linux_x86_64-sm75.whl ✓
[bootstrap]   Installed mamba_ssm-1.2.2-cp312-cp312-linux_x86_64.whl ✓
[bootstrap] FATAL: Mamba fast path verification FAILED: cannot import name 'selective_state_update' from 'mamba_ssm.ops.selective_scan_interface'
```

**Root cause:** Jamba imports `selective_scan_fn` and `mamba_inner_fn` from `mamba_ssm.ops.selective_scan_interface`, but imports `selective_state_update` from `mamba_ssm.ops.triton.selective_state_update`. Our verifier used the wrong module path.

**Fix applied:** verifier import path corrected.

---

## Wheel Build Session 6 — sm75 (Kaggle T4, 2026-04-14)
**Script:** `jamba_coconut_finetune.py` (comprehensive shim applied)
**Status:** 🟡 PATCH READY — smoke test pending

**New failure observed (verbatim):**
```
[bootstrap] FATAL: Mamba fast path verification FAILED: cannot import name
    'SampleDecoderOnlyOutput' from 'transformers.generation'
```

**Root cause:** single-name shim insufficient.

**Fix applied:** comprehensive 10-alias transformers generation shim.

---

## Wheel Build Session 5 — sm75 (Kaggle T4, 2026-04-14)
**Status:** 🔴 FAILED — `GreedySearchDecoderOnlyOutput` removed from `transformers>=4.44`

---

## Wheel Build Session 4 — sm75 (Kaggle T4, 2026-04-14)
**Status:** 🔴 FAILED — mamba_ssm PyPI sdist is a 35kB stub (missing CUDA source files)

**Fix applied:** switched source build spec to `git+https://github.com/state-spaces/mamba.git@v1.2.2`.

---

## Stage 3 — Smoke Test (Kaggle Dual T4, 2026-04-11)
**Status:** ✅ COMPLETED — 3 stages ran. Two training bugs found.

**Key metrics (verbatim):**
```
trainable params: 26,851,328 || all params: 3,056,191,360 || trainable%: 0.8786
d_model=2560  layers=28
Stage 2/2  S2E0: ce=1.464  gn=36.926
[val] s=2 ep=0 val_ce=0.0000 val_acc=0.0000
```

**Fixes codified:** `--max_seq_len 1024`, `--max_grad_norm 0.3`
