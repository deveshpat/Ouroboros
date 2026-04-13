# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**
**Purpose: session outcomes, key metrics, bugs found/fixed, generation samples.**
**Blueprint holds decisions/status. This file holds evidence.**

---

## Wheel Build Session 3 + Bootstrap Attempt (Kaggle sm_100, 2026-04-13)
**Scripts:** `build_wheels_kaggle.py` (build) → `jamba_coconut_finetune.py` (bootstrap)
**Status:** 🟡 PARTIAL — causal_conv1d ✅  mamba_ssm ❌ (404 on Hub)

**GPU confirmed: sm_100 (Blackwell B100)** — not sm_120 as previously documented.

**causal_conv1d ptxas output (verbatim sample, build phase ~1506s):**
```
ptxas info    : Compiling entry function '..._bwd_kernel...' for 'sm_100'
ptxas info    : Used 168 registers, used 1 barriers, 37728 bytes smem
...
ptxas info    : Compile time = 460.955 ms
```
All causal_conv1d kernels compiled for `sm_100`. Build succeeded and was uploaded to Hub.

**Bootstrap phase 2 (verbatim):**
```
1568.5s  [bootstrap] Phase 2: installing Hub wheels...
1568.8s  [bootstrap]   Downloading causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl ...
1573.1s  Processing /tmp/ouroboros_wheels/causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl
1593.0s  Successfully installed causal-conv1d-1.6.1
1593.0s  [bootstrap]   Installed causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl ✓
1593.0s  [bootstrap]   Downloading mamba_ssm-1.2.2-cp312-cp312-linux_x86_64.whl ...
1593.0s  [bootstrap] FATAL: Could not download mamba_ssm-1.2.2-cp312-cp312-linux_x86_64.whl:
         404 Client Error.
         Entry Not Found for url: https://huggingface.co/WeirdRunner/Ouroboros/resolve/main/
         mamba_ssm-1.2.2-cp312-cp312-linux_x86_64.whl.
1593.0s  [bootstrap]        Upload a compatible wheel with build_wheels_kaggle.py.
```

**Root cause:** mamba_ssm-1.2.2 build failed silently in the build session (~3600 lines truncated).
Wheel was never uploaded. causal_conv1d wheel for sm_100 is now correctly on Hub.

**What changed vs prior sessions:**
- GPU arch corrected: **sm_100** (B100), not sm_120 (B200) as logged 2026-04-12
- causal_conv1d now correctly built and on Hub for this arch ✅
- mamba_ssm remains the sole unresolved blocker ❌

**Next action:** Re-run `build_wheels_kaggle.py` with `--verbose` stderr capture for mamba_ssm.
See Blueprint Part 0.1 for exact command.

---

## Stage 3 — Hub Wheel Session (Kaggle T4, 2026-04-13)
**Script:** `jamba_coconut_finetune.py` (Hub-wheel bootstrap)
**Status:** 🔴 FAILED — two root causes now confirmed with certainty

**ABI fingerprint (verbatim, line 52):**
```
GPU=Tesla T4 sm_75 | CUDA=12.8 | PyTorch=2.10.0+cu128 | Python=cp312
```

**Key events (verbatim):**
```
34.1s  [bootstrap] Phase 1: pure-Python deps...
63.9s  Successfully installed mamba-ssm-2.3.1
64.3s  [bootstrap]   ABI fingerprint: GPU=Tesla T4 sm_75 | CUDA=12.8 | PyTorch=2.10.0+cu128 | Python=cp312
95.8s  FATAL: cannot import name 'selective_state_update' from
       'mamba_ssm.ops.selective_scan_interface'
       (/usr/local/lib/python3.12/dist-packages/mamba_ssm/ops/selective_scan_interface.py)
```

**Root cause 1 — Python API breaking change (confirmed):**
`selective_state_update` deleted from `mamba_ssm.ops.selective_scan_interface` in 2.x.
transformers Jamba still checks the 1.x path. Hub has 2.3.1 → broken regardless of GPU.

**Root cause 2 — causal_conv1d arch mismatch (confirmed, masked by root cause 1):**
Hub wheel built on Blackwell. PTX cannot JIT downward to sm_75 (T4).

---

## Stage 3 — Smoke Test Attempt 2 (Kaggle, 2026-04-13)
**Script:** `jamba_coconut_finetune.py`
**Status:** 🔴 CRASHED — bitsandbytes version too old; mamba_ssm source build failed again

**Key events (verbatim):**
```
13.2s  [bootstrap] Installing dependencies (skips already-installed)...
1795.4s [data] train: 36906 samples -> data/coconut_v1/train.jsonl
1797.8s [data] val:   1940 samples  -> data/coconut_v1/val.jsonl
1797.8s [data] stats.json written. median_steps=10  recommended --max_stage=10
1801.1s Loaded 190 train / 10 val from data/coconut_v1
1801.1s Step stats: median=10 mean=10.42 max=16
1806.3s <|lat|> token id: 65536  vocab: 65537
1806.3s flash-attn not installed: falling back to eager attention
1806.3s [WARN] mamba CUDA kernels unavailable: No module named 'mamba_ssm'
1806.3s Slow PyTorch path forced (~500s/step).
1806.9s ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes:
        `pip install -U bitsandbytes>=0.46.1`
```

**Dataset confirmed (first successful Hub download):**
- Train: 36,906 samples  Val: 1,940 samples
- median_steps=10  mean=10.42  max=16
- **Confirmed `--max_stage=10` for production runs**

---

## Post-Smoke-Test Audit (2026-04-12)
**Status:** ✅ Root causes identified. Patches ready.

Three-layer mamba fast path failure — see Blueprint Part 0.2 for full blocker table.

---

## Stage 3 — Smoke Test (Kaggle Dual T4, 2026-04-11)
**Script:** `jamba_coconut_finetune.py`  **Status:** ✅ COMPLETED — all 3 stages ran without crashing. Two training bugs found.

**Key metrics (verbatim):**
```
trainable params: 26,851,328 || all params: 3,056,191,360 || trainable%: 0.8786
d_model=2560  layers=28
<|lat|> token id: 65536  vocab: 65537

Stage 0/2  (CoT warmup)   Epochs: 1  Steps/epoch: 24  Total: 24
Stage 1/2  1 latent pass  Epochs: 1  Steps/epoch: 1   Total: 1
  S1E0: 100%  ce=0.872  gn=0.954
  [val] s=1 ep=0 val_ce=0.0000 val_acc=0.0000   ← BUG

Stage 2/2  2 latent pass  Epochs: 1  Steps/epoch: 1   Total: 1
  S2E0: 100%  ce=1.464  gn=36.926               ← EXPLODING GRAD
  [val] s=2 ep=0 val_ce=0.0000 val_acc=0.0000   ← BUG
```

**Generation samples (verbatim):**
```
-- Generation @ step 2 stage=1 --
Q: What is 15 + 27?
A: The user is asking for the sum of 15 and 27...  [k_actual=1 uwr=0.613]
Q: What is the capital of Japan?
A: The answer is straightforward: the capital is Tokyo.  [k_actual=1 uwr=0.506]
Mean UWR: 0.592

-- Generation @ step 3 stage=2 --
Q: What is the capital of Japan?
A: ,aemic for Japan. </think> about the question...Wait is a term for furniture.  [k_actual=2 uwr=0.296]
Q: Explain what a neural network is in simple terms.
A: , </think>  Okay </think>  <think>  <think>  the  the  the  [k_actual=2 uwr=0.074]
Mean UWR: 0.290
```

**Root causes:** val=0.0 → `build_sample_at_stage` returns None (seq > 512). gn=36.9 → latent injection at k=2 with 1 step destabilises gradients.
**Script runtime:** 3768.7s (~63 min).

---

## Wheel Build Session 2 (Kaggle, 2026-04-12)
**Script:** `build_wheels_kaggle.py` (mamba-ssm==1.2.2)  **Status:** ✅ causal_conv1d built. ✗ mamba_ssm build failed.

**Critical discovery — GPU is Blackwell (sm_120), not T4:**
```
ptxas: Compiling ... for 'sm_120'
causal_conv1d-1.6.1: built and uploaded ✓   (254 MB)
mamba_ssm==1.2.2:    build failed silently ✗
```

---

## Wheel Build Session 1 (Kaggle Dual T4, 2026-04-12)
**Script:** `build_wheels_kaggle.py`  **Status:** ✅ Built + uploaded. ✗ Fast path blocked (wrong version).

**Root cause confirmed:** `mamba_ssm 2.3.1` — `selective_state_update` moved to Triton path.
**Fix:** Pin `mamba-ssm==1.2.2`.

**Verification (verbatim):**
```
mamba_ssm.ops.selective_scan_interface.selective_scan_fn: OK
mamba_ssm.ops.selective_scan_interface.selective_state_update: None — ABI mismatch (FAIL)
✗ 1 symbol(s) missing.
```

---

## Stage 2 SFT — Session 9 (Single GPU, 3 epochs attempt)
**wandb run:** `comic-planet-8`  **Status:** 🔴 DEGENERATE — val_acc collapsed

**Key metrics (verbatim):**
```
train/ce:         2.17731
gen/mean_uwr:     0.08077   ← BELOW viability threshold (0.10)
```
**Duration:** 14,472s (~4 hours). ~800/9,840 steps before timeout.

---

## Stage 2 SFT — Session 8 (DDP, OOM at step 250)
**Status:** 🔴 OOM at first val step
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 9.25 GiB.
GPU 1: 14.56 GiB total, 433.81 MiB free
```

---

## Stage 2 SFT — Sessions 5–7 (DDP, NCCL watchdog)
**Status:** 🔴 All killed by NCCL watchdog (val+gen > 600s timeout)

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
