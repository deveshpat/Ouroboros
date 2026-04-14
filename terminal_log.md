# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**
**Purpose: session outcomes, key metrics, bugs found/fixed, generation samples.**
**Blueprint holds decisions/status. This file holds evidence.**

---

## Wheel Build Session 5 — sm75 (Kaggle T4, 2026-04-14)
**Script:** `jamba_coconut_finetune.py` (bootstrap, git+https fix applied)
**Status:** 🔴 FAILED — new blocker: `GreedySearchDecoderOnlyOutput` removed from `transformers>=4.44`

**GPU confirmed: sm_75 (Tesla T4)**
**ABI:** PyTorch=2.10.0+cu128 | Python=cp312 | CUDA=12.8
**Session duration:** 2661s (~44 min)

**mamba_ssm wheel:** ✅ Built from git+https source, uploaded to Hub for sm75.

**Bootstrap Phase 3 failure (verbatim):**
```
[bootstrap]   Installed mamba-ssm-1.2.2-cp312-cp312-linux_x86_64.whl ✓
[bootstrap] Phase 3: verifying mamba fast path (symbol + CUDA op)...
[bootstrap]   ABI fingerprint: GPU=Tesla T4 sm75 | CUDA=12.8 | PyTorch=2.10.0+cu128 | Python=cp312
[bootstrap] FATAL: Mamba fast path verification FAILED: cannot import name
    'GreedySearchDecoderOnlyOutput' from 'transformers.generation'
    (/usr/local/lib/python3.12/dist-packages/transformers/generation/__init__.py)
[bootstrap]     Exiting now (no slow-path fallback — 500s/step is unusable).
```

**Root cause (confirmed):**
`mamba_ssm==1.2.2` internally imports `GreedySearchDecoderOnlyOutput` (in `mamba_ssm/utils/generation.py`). This class was removed in `transformers>=4.44`. Bootstrap installs `transformers>=4.54.0` for Jamba — the import fails before any CUDA op is even attempted. The built wheel itself is correct; the missing symbol is a pure-Python API removal.

**Fix applied (2026-04-14):**
Add a Phase 1.5 compatibility shim in `_bootstrap()` (in `jamba_coconut_finetune.py` only — `build_wheels_kaggle.py` unaffected). Shim backfills `GreedySearchDecoderOnlyOutput` as an alias for `GenerateDecoderOnlyOutput` (the replacement class in modern transformers) so mamba_ssm 1.2.2 imports cleanly:

```python
    # Phase 1.5: transformers / mamba_ssm compatibility shim
    try:
        import transformers.generation as _tg_mod
        _tg_mod.GreedySearchDecoderOnlyOutput   # already present — nothing to do
    except AttributeError:
        from transformers.generation.utils import GenerateDecoderOnlyOutput as _GDO
        _tg_mod.GreedySearchDecoderOnlyOutput = _GDO
        print("[bootstrap] Shim: GreedySearchDecoderOnlyOutput -> GenerateDecoderOnlyOutput ✓")
```

**Hub wheel status after this session:**
- `causal_conv1d-1.6.1-cp312-cp312-linux_x86_64-sm75.whl` ✅ on Hub
- `mamba_ssm-1.2.2-cp312-cp312-linux_x86_64-sm75.whl` ✅ on Hub
- `causal_conv1d-1.6.1-cp312-cp312-linux_x86_64-sm100.whl` ✅ on Hub (Session 3)

**Next action:** Apply Phase 1.5 shim, re-run smoke test. Both wheels now cached on Hub — next session bootstrap time <30s for sm75.

---

## Wheel Build Session 4 — sm75 (Kaggle T4, 2026-04-14)
**Script:** `jamba_coconut_finetune.py` (bootstrap source-build path)
**Status:** 🔴 FAILED — mamba_ssm PyPI sdist is a 35kB stub (missing CUDA source files)

**GPU confirmed: sm_75 (Tesla T4)**
**ABI:** PyTorch=2.10.0+cu128 | Python=cp312

**causal_conv1d:** Built from source and uploaded to Hub for sm75 ✓

**mamba_ssm source build failure (verbatim key excerpt):**
```
[bootstrap]   mamba_ssm-1.2.2-cp312-cp312-linux_x86_64-sm75.whl not on Hub
    (RemoteEntryNotFoundError). Compiling from source...
[bootstrap]   Building mamba-ssm==1.2.2 (TORCH_CUDA_ARCH_LIST=7.5+PTX) ...
ninja: error: '/tmp/.../csrc/selective_scan/selective_scan.cpp', needed by
  '.../csrc/selective_scan_interface.o', missing and no known rule to make it
[bootstrap] FATAL: Source build failed for mamba-ssm==1.2.2.
```

**Root cause:** mamba-ssm 1.2.2 PyPI sdist is a 35kB stub — CUDA source files absent from PyPI release.

**Fix applied (2026-04-14):** Source-build spec changed from `"mamba-ssm==1.2.2"` to `"git+https://github.com/state-spaces/mamba.git@v1.2.2"` in both `jamba_coconut_finetune.py` `_bootstrap()` and `build_wheels_kaggle.py`.

---

## Wheel Build Session 3 + Bootstrap Attempt (Kaggle sm_100, 2026-04-13)
**Status:** 🟡 PARTIAL — causal_conv1d ✅  mamba_ssm ❌ (404 on Hub)

**GPU confirmed: sm_100 (Blackwell B100)**

**Bootstrap phase 2 (verbatim):**
```
[bootstrap]   Downloading causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl ...
[bootstrap]   Installed causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl ✓
[bootstrap]   Downloading mamba_ssm-1.2.2-cp312-cp312-linux_x86_64.whl ...
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

**Key events (verbatim):**
```
1795.4s [data] train: 36906 samples -> data/coconut_v1/train.jsonl
1797.8s [data] val:   1940 samples  -> data/coconut_v1/val.jsonl
1797.8s [data] stats.json written. median_steps=10  recommended --max_stage=10
1806.3s <|lat|> token id: 65536  vocab: 65537
1806.3s [WARN] mamba CUDA kernels unavailable: No module named 'mamba_ssm'
1806.9s ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes>=0.46.1
```

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

**Generation samples (verbatim):**
```
Q: What is the capital of Japan?
A: The answer is straightforward: the capital is Tokyo.  [k_actual=1 uwr=0.506]
Q: What is the capital of Japan?
A: ,aemic for Japan. </think> about the question...Wait is a term for furniture.  [k_actual=2 uwr=0.296]
Mean UWR: 0.290
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
