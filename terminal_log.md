# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**
**Purpose: session outcomes, key metrics, bugs found/fixed, generation samples.**
**Blueprint holds decisions/status. This file holds evidence.**

---

## Session 11 — Dual T4 DDP, Full Curriculum Run (2026-04-15) 🟡 IN PROGRESS (resumed)
**Script:** `jamba_coconut_finetune.py`
**Command:**
```bash
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit \
  --epochs_per_stage 3 --max_stage 10 --batch_size 2 --grad_accum 8 \
  --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \
  --output_dir runs/stage3_curriculum
```
**Status:** 🟡 Stage 0/10 — epoch 0 at step 629/2307 when timed out. Checkpoint saved. Auto-resume on next session.

**Training metrics (verbatim log — key steps only):**
```
  step=    20 s=0 ep=0 ce=0.5169 gn=0.3009
  step=    40 s=0 ep=0 ce=0.4260 gn=0.2113
  step=    60 s=0 ep=0 ce=0.4534 gn=0.1474
  step=   100 s=0 ep=0 ce=0.4658 gn=0.1921
  step=   200 s=0 ep=0 ce=0.4209 gn=0.2124
  step=   300 s=0 ep=0 ce=0.3806 gn=0.1900
  step=   400 s=0 ep=0 ce=0.3117 gn=0.1618
  step=   500 s=0 ep=0 ce=0.3676 gn=0.1210
  step=   580 s=0 ep=0 ce=0.3011 gn=0.1557
  step=   600 s=0 ep=0 ce=0.3405 gn=0.1657
  step=   620 s=0 ep=0 ce=0.3428 gn=0.2226
```

**Step time observed: ~60s/step** on Dual T4, full 36,906-sample dataset, stage 0 batched forward.

**Timeout and checkpoint (verbatim):**
```
S0E0:  27%|██▏     | 629/2307 [10:40:33<27:54:04, 59.86s/it, ce=0.361, gn=0.166]
  [timeout] 10.68h elapsed - 19.4 min remaining (< 20 min buffer).
  [timeout] saving emergency checkpoint at step 629 ...
  [ckpt] saved -> runs/stage3_curriculum/stage_0/checkpoint-0000629  acc=None  ce=None

================================================================
  [timeout] Session budget exhausted - checkpoint saved.
  Re-run the same command with the same --output_dir to auto-resume.
================================================================
```

**Analysis:** CE loss healthy — trending from 0.517 → 0.343 over 629 steps with no gradient explosions (gn stable at 0.11–0.30). However, at 60s/step, Stage 0 at 3 epochs = ~115h total. Critical throughput decision required before next session (see Blueprint Part 0.1 action #5).

---

## Session 10 — Dual T4 DDP, Profile/Smoke Test (2026-04-15) ✅ COMPLETE
**Script:** `jamba_coconut_finetune.py` (with Phase 2.5/2.6 swap + batched stage-0 forward)
**Output dir:** `runs/profile_dual_t4`
**Status:** ✅ Stage 0/0 COMPLETE — 12 steps, val_acc=0.4000

**Bootstrap confirmation (from screenshot — verbatim):**
```
[bootstrap] Shim: patched 10 removed transformers.generation names ✓
[bootstrap] Kernel export shim: mamba_ssm.selective_state_update ✓
[bootstrap] Mamba fast path: ACTIVE ✓
```
Phase 2.5/2.6 swap **confirmed** — generation shim now runs before kernel export patch. No WARNING this session.

**Training (from screenshot):**
```
Stage 0/0  (CoT warmup)
Epochs: 1  Steps/epoch: 12  Total: 12
```
Step times: 23–37s/it (average ~30s/step) on Dual T4, batched stage-0 forward, 200 samples.

**Validation (verbatim):**
```
[val] s=0 ep=0 val_ce=0.4253 val_acc=0.4000
[ckpt] best -> runs/profile_dual_t4/stage_0/best  acc=0.4  ce=0.4252639559711584
```

**Generation callback (verbatim, from screenshot):**
```
-- Generation @ step 12 stage=0 --
Q: What is 15 + 27?
A: ...15 + 27 = 42...  [k_actual=0 uwr=0.484]
Q: Write a Python function that returns the factorial of n.
A: ...The factorial of n is the product of all positive integers up to n...  [k_actual=0 uwr=0.590]
Q: What is the capital of Japan?
A: ...The capital of Japan is Tokyo...  [k_actual=0 uwr=0.377]
Q: Explain what a neural network is in simple terms.
A: ...We need to define a neural network in simple terms...  [k_actual=0 uwr=0.618]
Q: Solve for x: 3x + 6 = 21.
A: ...3x = 21 - 6 = 15. Then divide by 3: x = 15/3 = 5...  [k_actual=0 uwr=0.623]
Mean UWR: 0.538
```

All 5 generations factually correct. val_acc=0.4000 after only 12 steps indicates fast learning.

**Stage completion (verbatim):**
```
[stage] Stage 0 done. Best acc=0.4000. Loading best ckpt before advancing.
[resume] step=12 epoch=0 stage_k=0 val_acc=0.4
Curriculum complete. Stages: [0]  Global steps: 12
```

**Key finding:** Batched stage-0 forward gives ~30s/step on Dual T4 with 200 samples (vs 113s/step single T4 without fix). Full dataset (36,906 samples) shows 60s/step — confirms dataset I/O contributes ~30s overhead per step at this scale.

---

## Session 9 — sm75 (Kaggle Single T4, 2026-04-14) ✅ FIRST SUCCESSFUL SMOKE TEST
**Script:** `jamba_coconut_finetune.py` (all Session 8 fixes applied)
**Status:** ✅ Stage 0 COMPLETE — Stage 1 timeout (as expected at 1.5h budget)

**Bootstrap output (verbatim):**
```
[bootstrap] Phase 2: arch-aware Hub wheel install...
[bootstrap]   GPU arch: sm75  (TORCH_CUDA_ARCH_LIST=7.5+PTX if build needed)
[bootstrap]   Downloaded causal_conv1d-1.6.1-cp312-cp312-linux_x86_64-sm75.whl ✓
[bootstrap]   Downloaded mamba_ssm-1.2.2-cp312-cp312-linux_x86_64-sm75.whl ✓
[bootstrap] WARNING: kernel export shim failed: cannot import name 'GreedySearchDecoderOnlyOutput' from 'transformers.generation'
[bootstrap] Shim: patched 10 removed transformers.generation names ✓
[bootstrap] Mamba fast path: ACTIVE ✓ — ~5s/step expected.
```
Note: WARNING present here (Phase 2.5 before 2.6). Fixed in Session 10.

**Training output (verbatim):**
```
trainable params: 26,851,328 || all params: 3,056,191,360 || trainable%: 0.8786
d_model=2560  layers=28

Stage 0/2  (CoT warmup)
Epochs: 1  Steps/epoch: 12  Total: 12

S0E0: 100%|████████████████| 12/12 [22:06<00:00, 110.54s/it, ce=0.615, gn=0.224]
  [val] s=0 ep=0 val_ce=0.5430 val_acc=0.2000
  [best] stage=0 new best acc=0.2000
```

**Step time observed: ~113s/step** (single T4, no batching fix, Stage 0).

**Stage 1 timeout (expected):**
```
  [timeout] 1.40h elapsed - 6.0 min remaining (< 20 min buffer).
  [timeout] saving emergency checkpoint at step 12 ...
  Session budget exhausted - checkpoint saved.
```

---

## Sessions 4–8 — sm75 (Kaggle T4, 2026-04-14)
- Session 8: `causal_conv1d_fn` weight shape `(dim, 1, width)` → fixed to `(dim, width)` ✅
- Session 7: `selective_state_update` wrong import path → fixed to `mamba_ssm.ops.triton.selective_state_update` ✅
- Session 6: single-name generation shim insufficient → comprehensive 10-alias shim ✅
- Session 5: `GreedySearchDecoderOnlyOutput` missing → led to single-name shim (insufficient)
- Session 4: mamba_ssm PyPI sdist is 35kB stub → fixed by `git+https://github.com/state-spaces/mamba.git@v1.2.2`

---

## Stage 3 — Early Smoke Test (Kaggle Dual T4, 2026-04-11)
**Status:** ✅ Pipeline verified — training bugs found and codified

```
trainable params: 26,851,328 || all params: 3,056,191,360 || trainable%: 0.8786
Stage 2/2  S2E0: ce=1.464  gn=36.926
[val] s=2 ep=0 val_ce=0.0000 val_acc=0.0000
```

**Fixes codified:** `--max_seq_len 1024`, `--max_grad_norm 0.3`
