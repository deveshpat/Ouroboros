# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**
**Purpose: session outcomes, key metrics, bugs found/fixed, generation samples.**
**Blueprint holds decisions/status. This file holds evidence.**

---

## Wheel Build Session 4 — sm75 (Kaggle T4, 2026-04-14)
**Script:** `jamba_coconut_finetune.py` (bootstrap source-build path)
**Status:** 🔴 FAILED — mamba_ssm PyPI sdist is a 35kB stub (missing CUDA source files)

**GPU confirmed: sm_75 (Tesla T4)**
**ABI:** PyTorch=2.10.0+cu128 | Python=cp312

**causal_conv1d:** Built from source and uploaded to Hub for sm75 ✓

**mamba_ssm source build failure (verbatim key excerpt from image 8/9):**
```
[bootstrap]   mamba_ssm-1.2.2-cp312-cp312-linux_x86_64-sm75.whl not on Hub (RemoteEntryNotFoundError). Compiling from source...
[bootstrap]   Building mamba-ssm==1.2.2 (TORCH_CUDA_ARCH_LIST=7.5+PTX) ...
Downloading mamba-ssm-1.2.2.tar.gz (35 kB)
ninja: error: '/tmp/.../csrc/selective_scan/selective_scan.cpp', needed by
  '.../csrc/selective_scan_interface.o', missing and no known rule to make it
RuntimeError: Error compiling objects for extension
subprocess.CalledProcessError: Command '['ninja', '-v', '-j', '4']' returned non-zero exit status 1.
ERROR: Failed building wheel for mamba-ssm
[bootstrap] FATAL: Source build failed for mamba-ssm==1.2.2.
            Run build_wheels_kaggle.py manually and capture stderr: python build_wheels_kaggle.py 2>&1 | tee build.log
```

**Root cause (confirmed definitively):**
The mamba-ssm 1.2.2 PyPI sdist is a 35kB stub. It contains only Python metadata and setup.py — the CUDA source files (`selective_scan.cpp`, `selective_scan_interface.cu`, etc.) were never included in the PyPI release. They exist only on GitHub at the v1.2.2 tag. This is why every source-build attempt fails: there is literally nothing to compile.

**Fix applied (2026-04-14):**
- `jamba_coconut_finetune.py` `_bootstrap()`: mamba_ssm source-build pip spec changed from `"mamba-ssm==1.2.2"` to `"git+https://github.com/state-spaces/mamba.git@v1.2.2"`
- `build_wheels_kaggle.py`: `MAMBA_SSM_VERSION` changed from `"mamba-ssm==1.2.2"` to `"git+https://github.com/state-spaces/mamba.git@v1.2.2"`

**Kaggle GPU quota remaining:** 9h 36m of 30h (image 11 — ~20h 24m consumed on wheel debugging).

**Next action:** Run `build_wheels_kaggle.py` once on a fresh GPU session with the fix. This will build from full GitHub source, upload sm75 wheel to Hub. After that, bootstrap will download the wheel in <30s.

---

## Wheel Build Session 3 + Bootstrap Attempt (Kaggle sm_100, 2026-04-13)
**Scripts:** `build_wheels_kaggle.py` (build) → `jamba_coconut_finetune.py` (bootstrap)
**Status:** 🟡 PARTIAL — causal_conv1d ✅  mamba_ssm ❌ (404 on Hub)

**GPU confirmed: sm_100 (Blackwell B100)**

**causal_conv1d ptxas output (verbatim sample):**
```
ptxas info    : Compiling entry function '..._bwd_kernel...' for 'sm_100'
ptxas info    : Used 168 registers, used 1 barriers, 37728 bytes smem
ptxas info    : Compile time = 460.955 ms
```

**Bootstrap phase 2 (verbatim):**
```
[bootstrap]   Downloading causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl ...
[bootstrap]   Installed causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl ✓
[bootstrap]   Downloading mamba_ssm-1.2.2-cp312-cp312-linux_x86_64.whl ...
[bootstrap] FATAL: Could not download mamba_ssm-1.2.2-cp312-cp312-linux_x86_64.whl:
         404 Client Error. Entry Not Found.
```

**Root cause (now understood):** mamba_ssm build failed silently in build session — the PyPI sdist was used, which is a 35kB stub.

---

## Stage 3 — Hub Wheel Session (Kaggle T4, 2026-04-13)
**Status:** 🔴 FAILED — mamba-ssm 2.3.1 API mismatch + causal_conv1d arch mismatch

**ABI fingerprint (verbatim):**
```
GPU=Tesla T4 sm_75 | CUDA=12.8 | PyTorch=2.10.0+cu128 | Python=cp312
```

**Key events (verbatim):**
```
63.9s  Successfully installed mamba-ssm-2.3.1
95.8s  FATAL: cannot import name 'selective_state_update' from
       'mamba_ssm.ops.selective_scan_interface'
```

**Root cause:** mamba-ssm 2.x moved `selective_state_update` to Triton path; breaks `_bootstrap_verify_fast_path()`.

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

## Post-Smoke-Test Audit (2026-04-12)
**Status:** ✅ Root causes identified. Patches ready.

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
-- Generation @ step 2 stage=1 --
Q: What is the capital of Japan?
A: The answer is straightforward: the capital is Tokyo.  [k_actual=1 uwr=0.506]
-- Generation @ step 3 stage=2 --
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
