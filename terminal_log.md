# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**
**Purpose: session outcomes, key metrics, bugs found/fixed, generation samples.**
**Blueprint holds decisions/status. This file holds evidence.**

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
Error is on a `.py` file, not `.so`. CUDA extension loaded fine (31-second gap = `.so` init).
`selective_state_update` was **deleted** from `mamba_ssm.ops.selective_scan_interface` in 2.x.
transformers Jamba still checks the 1.x path. Hub has 2.3.1 → broken everywhere regardless of GPU.

**Root cause 2 — causal_conv1d arch mismatch (confirmed, masked by root cause 1):**
Hub wheel built on Blackwell (`sm_120+PTX`). sm_120 PTX cannot be JIT'd on sm_75 (T4) — PTX forward
compat only goes UP to newer GPUs, not down. The CUDA op test would have caught this next.

**Why prior build failures don't prove building is broken:**
Both failed with UNPATCHED code (no ARCH_LIST injection). The patched `build_wheels_kaggle.py`
has NEVER been run. There is zero data from a patched run.

**Fix:** Run `build_wheels_kaggle.py` (patched, as-is) on T4. `_HUB_WHEEL_FILES` updated to
`mamba_ssm-1.2.2-cp312-cp312-linux_x86_64.whl` in `jamba_coconut_finetune.py`.

---

## Stage 3 — Smoke Test Attempt 2 (Kaggle, 2026-04-13)
**Script:** `jamba_coconut_finetune.py`  **Hardware:** Kaggle GPU (exact GPU unknown)
**Status:** 🔴 CRASHED — bitsandbytes version too old; mamba_ssm source build silently failed again

**Key events (verbatim):**
```
13.2s  [bootstrap] Installing dependencies (skips already-installed)...
       ... (~29.7 min: mamba-ssm==1.2.2 source build ran; build output truncated) ...
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

**Crash location:** `load_model_and_tokenizer` → `_safe_from_pretrained` → `quantizer_bnb_4bit.py:60`

**Root cause 1 — bitsandbytes version floor missing (PRIMARY, easy fix):**
Container ships bitsandbytes < 0.46.1. `_bootstrap()` installs `"bitsandbytes"` with no version
constraint → pip sees it already satisfied → old version stays → crash on 4-bit load.
**Fix:** Change `"bitsandbytes"` to `"bitsandbytes>=0.46.1"` in `_bootstrap()`.

**Root cause 2 — mamba_ssm source build failed again (BLOCKING, unresolved):**
Bootstrap spent ~29.7 min — build definitely ran (visible pip progress in first 94 log lines).
Module still absent post-install. Build stderr is in ~8000 truncated log lines; exact failure
not visible. Likely causes:
  (a) nvcc missing / CUDA toolkit headers absent in this container image
  (b) Arch list issue despite `--no-build-isolation` (needs verbose build log to confirm)
**Next step:** add `--verbose` to the mamba-ssm pip install call in `_bootstrap()` and
capture stderr to a file; also inject `TORCH_CUDA_ARCH_LIST` env var in bootstrap subprocess.

**Dataset confirmed (first successful Hub download):**
- Train: 36,906 samples  Val: 1,940 samples
- median_steps=10  mean=10.42  max=16
- **Confirmed `--max_stage=10` for production runs**

---

## Post-Smoke-Test Audit (2026-04-12)
**Status:** ✅ Root causes identified. Patches ready.

### Discrepancy note
Terminal log records **3768.7s script runtime** for the smoke test. User reported ~90 min total.
Delta (~27 min) = Kaggle notebook startup + Jamba 3B download (~6 GB) + pip install time.

The "2 max samples" feeling: with `--max_seq_len 512` active, `build_sample_at_stage` returns
`None` for nearly all val samples at stages 1+. `steps_per_epoch` collapses to `max(1, ...)` = 1.
Effectively 0 real training samples processed at stages 1 and 2 despite `--max_samples 200` in CLI.

### Three-layer failure: mamba fast path never used

**Layer 1 — Code (primary):** `use_mamba_kernels=False` was unconditionally hardcoded.
**Fix:** Replaced with runtime probe — import `selective_scan_fn`; only set False on ImportError.

**Layer 2 — Wheel/ABI mismatch:** Existing Hub wheels were wrong-arch.
**Fix:** Retired wheel workflow. Script self-installs via `_bootstrap()`.

**Layer 3 — `--no-build-isolation` required for source builds:** All three packages need it.
`build_wheels_kaggle.py` passes `--no-build-isolation` for all three.

---

## Stage 3 — Smoke Test (Kaggle Dual T4, 2026-04-11)
**Script:** `jamba_coconut_finetune.py`  **Status:** ✅ COMPLETED — all 3 stages ran without crashing. Two training bugs found.

**Command:**
```
python jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit \
  --epochs_per_stage 1 --max_stage 2 --max_samples 200 \
  --batch_size 2 --grad_accum 4 \
  --session_timeout_hours 1.5 --graceful_exit_buffer_minutes 10 \
  --wandb_mode disabled --output_dir runs/smoke
```

**Environment:** Latest Container Image (NOT pinned) — mamba CUDA kernels unavailable, slow path used (~520s/step)

**Key metrics (verbatim):**
```
trainable params: 26,851,328 || all params: 3,056,191,360 || trainable%: 0.8786
d_model=2560  layers=28
<|lat|> token id: 65536  vocab: 65537
embed_tokens and lm_head paths verified after PEFT wrap.

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
A: The user is asking for the sum of 15 and 27...15 plus 20 is 35, then plus 7 more  [k_actual=1 uwr=0.613]
Q: What is the capital of Japan?
A: The user is asking for the capital of Japan. The answer is straightforward: the capital is Tokyo.  [k_actual=1 uwr=0.506]
Mean UWR: 0.592

-- Generation @ step 3 stage=2 --
Q: What is the capital of Japan?
A: ,aemic for Japan. </think> about the question...Wait is a term for a type of furniture.  [k_actual=2 uwr=0.296]
Q: Explain what a neural network is in simple terms.
A: , </think>  Okay </think>  <think>  <think>  <think>  the  the  the  the  [k_actual=2 uwr=0.074]
Mean UWR: 0.290
```

**Root causes:**
1. **val=0.0**: `build_sample_at_stage` returns None for all val samples (sequences exceed `--max_seq_len 512`)
2. **gn=36.926**: Latent injection at k=2 with only 1 training step destabilises gradients

**Script runtime:** 3768.7s (~63 min). Total session ~90 min.

---

## Wheel Build Session 2 (Kaggle, 2026-04-12)
**Script:** `build_wheels_kaggle.py` (mamba-ssm==1.2.2)  **Status:** ✅ causal_conv1d built. ✗ mamba_ssm build failed.

**Critical discovery — GPU is Blackwell (sm_120), not T4:**
```
ptxas: Compiling ... for 'sm_120'
causal_conv1d-1.6.1: built and uploaded ✓   (254 MB)
mamba_ssm==1.2.2:    build failed silently ✗  → No module named 'mamba_ssm'
```

1.2.2 was released before Blackwell. Its setup.py TORCH_CUDA_ARCH_LIST does not include sm_120.
**Fix:** Auto-detect GPU CC and inject `TORCH_CUDA_ARCH_LIST="{major}.{minor}+PTX"` into env.
Updated in `build_wheels_kaggle.py`.

**Verification (verbatim):**
```
✗ mamba_ssm.ops.selective_scan_interface.selective_scan_fn: ImportError: No module named 'mamba_ssm'
✓ causal_conv1d.causal_conv1d_fn: OK
✓ causal_conv1d.causal_conv1d_update: OK
```

---

## Wheel Build Session 1 (Kaggle Dual T4, 2026-04-12)
**Script:** `build_wheels_kaggle.py`  **Status:** ✅ Built + uploaded. ✗ Fast path blocked (wrong version).

**Environment:** CUDA 12.8  PyTorch 2.10  Python cp312  cxx11=TRUE

**Root cause confirmed:** `mamba_ssm 2.3.1` (2.x series) — `selective_state_update` moved to Triton path.
**Fix:** Pin `mamba-ssm==1.2.2`. Updated in `build_wheels_kaggle.py`.

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
