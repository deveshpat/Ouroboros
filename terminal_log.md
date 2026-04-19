# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**
**Purpose: session outcomes, key metrics, bugs found/fixed, generation samples.**

---

## Session 15 — Hub+Prune Confirmed, Stage 1 Complete, Stage 2 In Progress (2026-04-18 → 2026-04-19) 🟡 ACTIVE

**Command (Cell 5, with `--push_to_hub` added):**
```bash
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit \
  --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 \
  --batch_size 4 --grad_accum 8 \
  --val_batch_size 2 \
  --val_skip_buffer_minutes 60 \
  --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \
  --push_to_hub \
  --output_dir runs/stage3_curriculum
```

**Startup hub sync (verbatim):**
```
  [startup] Found 5 local checkpoint(s). Uploading to Hub before pruning...
  [startup]   stage_0/best  ✓
  [startup]   stage_0/checkpoint-0000894  ✓
  [startup]   stage_0/checkpoint-0001154  ✓
  [startup]   stage_1/checkpoint-0001338  ✓
  [startup]   stage_1/checkpoint-0002293  (resume)  ✓
  [startup]   pruned stage_0/checkpoint-0000894
  [startup]   pruned stage_0/checkpoint-0001154
  [startup]   pruned stage_1/checkpoint-0001338
  [startup] Sync+prune complete. Pruned 3 checkpoint(s) locally.
```

**FP16 / resume confirmation (verbatim):**
```
  [GPU] Tesla T4  cc=sm75  VRAM=16GB  amp_dtype=float16
  [resume] step=2293 epoch=0 stage_k=1 val_acc=None
  Resuming stage from epoch=0 step_in_epoch=1138 global_step=2293
```

**Stage 1 completion (verbatim):**
```
S1E0: 100%|█████████████████| 15/15 [10:21<00:00, 41.46s/it, ce=0.372, gn=0.575]
  [val] s=1 ep=0 val_ce=0.4912 val_acc=0.0444
  [ckpt] saved -> runs/stage3_curriculum/stage_1/checkpoint-0002308  acc=0.044444444444444446  ce=0.4911867433400155
  [ckpt] best -> runs/stage3_curriculum/stage_1/best  acc=0.044444444444444446  ce=0.4911867433400155
  [best] stage=1 new best acc=0.0444
```

**Stage 1 generation @ step 2308 (verbatim):**
```
  Q: What is 15 + 27?
  A: 15 + 27 = 42 So, the sum of 15 and 27 is 42. ...  [k_actual=1 uwr=0.441]
  Q: Write a Python function that returns the factorial of n.
  A: ```python def factorial(n): ...```  [k_actual=1 uwr=0.487]
  Q: What is the capital of Japan?
  A: The capital of Japan is Tokyo. ...  [k_actual=1 uwr=0.461]
  Q: Explain what a neural network is in simple terms.
  A: It's like having a large number of interconnected nodes ...  [k_actual=1 uwr=0.455]
  Q: Solve for x: 3x + 6 = 21.
  A: Subtract 6 from both sides: 3x = 15 Divide both sides by 3: x = 5 ...  [k_actual=1 uwr=0.556]
  Mean UWR: 0.480
```

**Stage 2 start (verbatim):**
```
  [stage] Stage 1 done. Best acc=0.0444. Loading best ckpt before advancing.
  Stage 2/10  2 latent pass(es)
  Epochs: 1  Steps/epoch: 1154  Total: 1154
S2E0:  49%|████▉     | 564/1154 [8:04:51<9:22:07, 57.17s/it, ce=0.565, gn=0.876]
```

**Stage 2 step time: ~57s/step** (vs model prediction of ~46s; ~11.5s per latent pass empirically).

**Stage 2 step logs (verbatim, selected):**
```
  step=  2320 s=2 ep=0 ce=0.4721 gn=0.5595
  step=  2360 s=2 ep=0 ce=0.5179 gn=0.4136
  step=  2400 s=2 ep=0 ce=0.3645 gn=0.6076
  step=  2500 s=2 ep=0 ce=0.4663 gn=0.4861
  step=  2600 s=2 ep=0 ce=0.6568 gn=0.6353
  step=  2620 s=2 ep=0 ce=0.5691 gn=1.5024
  step=  2720 s=2 ep=0 ce=0.5018 gn=1.7615
  step=  2740 s=2 ep=0 ce=0.6196 gn=1.9030
  step=  2860 s=2 ep=0 ce=0.5273 gn=1.1574
```
*Note: gn values are pre-clip norms; `max_grad_norm=0.3` clips all. CE not diverging — training healthy.*

**Issues identified:**
- `save_checkpoint` hub upload uses wrong remote path: `runs/stage3/checkpoint-0002308` instead of `runs/stage3/stage_1/checkpoint-0002308` (missing stage subdir). Non-critical — startup sync corrects next session.

---

## Session 14 — Stage 1 Resumed, FP16 Confirmed (2026-04-18) ✅ COMPLETE

**Status:** FP16 patch CONFIRMED. Resumed from `checkpoint-0001338` (step_in_epoch=183). Stage 1 in progress; timed out at step 2293.

**FP16 + resume (verbatim):**
```
  [GPU] Tesla T4  cc=sm75  VRAM=16GB  amp_dtype=float16
  [resume] step=1338 epoch=0 stage_k=1 val_acc=None
  Resuming stage from epoch=0 step_in_epoch=183 global_step=1338
```

**Step time (verbatim tqdm):**
```
S1E0:   4%|▌            | 39/970 [26:51<10:41:10, 41.32s/it, ce=0.389, gn=0.200]
```
**Confirmed: ~41s/step at k=1** (~4× speedup over pre-patch 162s/step).

**Timeout save (verbatim):**
```
  [timeout] saving emergency checkpoint at step 2293 ...
  [ckpt] saved -> runs/stage3_curriculum/stage_1/checkpoint-0002293  acc=None  ce=None
```

**Issues identified this session:** `--push_to_hub` never passed; `prune_epoch_checkpoints` only after val → fixed in `AGENT_PROMPT_hub_prune_fix.md`.

---

## Session 13 — Stage 0 Val Complete, Stage 1 Started (2026-04-17 → 2026-04-18) ✅ COMPLETE

**Status:** NCCL fix confirmed. Stage 0 val succeeded. Stage 1 pre-FP16 patch (~162s/step). Timed out at step 1338.

**Stage 0 val (verbatim):**
```
  [val] s=0 ep=0 val_ce=0.4041 val_acc=0.0222
  [ckpt] best -> runs/stage3_curriculum/stage_0/best  acc=0.022222222222222223  ce=0.4040850892827772
```

**Stage 1 pre-patch step time (mean from 8 intervals): 162s ± 3s**

---

## Session 12 — Stage 0 Training Complete, Val NCCL Crash (2026-04-17) ✅ RESOLVED

**Stage 0 final metrics (verbatim):**
```
S0E0: 100%|████████████| 260/260 [9:54:14<00:00, 137.13s/it, ce=0.385, gn=0.111]
  [ckpt] saved -> runs/stage3_curriculum/stage_0/checkpoint-0001154  acc=None  ce=None
```
NCCL watchdog killed val. Fix: `timedelta(hours=4)` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=14400`.

---

## Session 10 — Dual T4 DDP Profile Run (2026-04-15) ✅ COMPLETE

```
[bootstrap] Mamba fast path: ACTIVE ✓
[val] s=0 ep=0 val_ce=0.4253 val_acc=0.4000
```

---

## Sessions 4–9 — Kernel / Bootstrap Debugging (2026-04-14) ✅

- Session 9: First successful Stage 0 smoke test. ~113s/step single T4 (pre-batching).
- Session 8: `causal_conv1d_fn` weight shape fix ✅
- Session 7: `selective_state_update` import path fix ✅
- Session 6: 10-alias generation shim ✅

---

## Stage 3 — Early Smoke Test (2026-04-11)

```
trainable params: 26,851,328 || all params: 3,056,191,360 || trainable%: 0.8786
Stage 2/2  S2E0: ce=1.464  gn=36.926
```
Fixes: `--max_seq_len 1024`, `--max_grad_norm 0.3`
