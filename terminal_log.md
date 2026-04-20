# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**

---

## Session 16 — DiLoCo Stage 2 Round 0 Active, Workers A+B Confirmed (2026-04-20) 🟡 IN PROGRESS

**Context:** First live DiLoCo run. Worker C quota exhausted; A+B sufficient (min_workers=2).
Bugs fixed in single pass by ChatGPT o3 extended thinking (brief pasted into project chat):
- `_SCRIPT_START` moved to module import (true wall-clock timeout)
- DiLoCo sharding subtracts `total_samples_seen[stage_k]` before A/B/C split
- Pre-val guard: only runs for truly new stages (`is_new_stage` check)
- DDP-safe Hub auto-resume: rank 0 resolves, broadcasts path to all ranks; `.hub_resume/` cleanup deferred

**Worker B (weirdrunner007) — verbatim:**
```
2026-04-20 19:15:48
================================================================
  Stage 2/10  2 latent pass(es)
  Epochs: 1  Steps/epoch: 159  Total: 159
================================================================
S2E0:   0%|                                             | 0/159 [00:00<?, ?it/s]
S2E0:  18%|██▍           | 28/159 [24:11<2:02:00, 55.88s/it, ce=0.567, gn=0.928]
  step=    20 s=2 ep=0 ce=0.5731 gn=1.1896
```

**Worker A (weirdrunner) — verbatim:**
```
2026-04-20 19:28:22
================================================================
  Stage 2/10  2 latent pass(es)
  Epochs: 1  Steps/epoch: 159  Total: 159
================================================================
S2E0:   0%|                                             | 0/159 [00:00<?, ?it/s]
S2E0:   9%|█▎            | 15/159 [12:08<1:59:39, 49.86s/it, ce=0.492, gn=1.049]
```

**Key observations:**
- Steps/epoch = 159 confirms remainder-sharding is working correctly (blueprint target: ~158 steps/worker)
- CE values (0.49–0.57) are within Stage 2 historical range (0.36–0.64 in Session 15)
- gn values healthy and pre-clip; clipping at 0.3 correctly active

---

## Session 15 — Hub+Prune Confirmed, Stage 1 Complete, Stage 2 59% (2026-04-18 → 2026-04-19) ✅ COMPLETE (timed out)

**Command:**
```bash
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit \
  --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 \
  --batch_size 4 --grad_accum 8 --val_batch_size 2 \
  --val_skip_buffer_minutes 60 \
  --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \
  --push_to_hub --output_dir runs/stage3_curriculum
```

**Hub+prune (verbatim):**
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

**Stage 1 completion (verbatim):**
```
S1E0: 100%|█████████████████| 15/15 [10:21<00:00, 41.46s/it, ce=0.372, gn=0.575]
  [val] s=1 ep=0 val_ce=0.4912 val_acc=0.0444
  [best] stage=1 new best acc=0.0444
```

**Stage 1 generation @ step 2308 (verbatim):**
```
  Q: What is 15 + 27?    A: 15 + 27 = 42 ...  [k_actual=1 uwr=0.441]
  Q: factorial of n.     A: def factorial(n): ...  [k_actual=1 uwr=0.487]
  Q: Capital of Japan?   A: The capital of Japan is Tokyo. ...  [k_actual=1 uwr=0.461]
  Q: Neural network?     A: It's like having a large number of interconnected nodes ...  [k_actual=1 uwr=0.455]
  Q: 3x + 6 = 21.        A: x = 5 ...  [k_actual=1 uwr=0.556]
  Mean UWR: 0.480
```

**Stage 2 start + timeout (verbatim):**
```
  Stage 2/10  2 latent pass(es)
  Epochs: 1  Steps/epoch: 1154  Total: 1154
S2E0:  59%|█████▉    | 679/1154 [9:43:06<6:55:29, 52.48s/it, ce=0.636, gn=1.069]
  [timeout] 10.68h elapsed - 19.5 min remaining (< 20 min buffer).
  [timeout] saving emergency checkpoint at step 2987 ...
  [ckpt] saved -> runs/stage3_curriculum/stage_2/checkpoint-0002987  acc=None  ce=None
  [hub] uploaded runs/stage3/checkpoint-0002987 -> WeirdRunner/Ouroboros
```

**Stage 2 selected step logs (verbatim):**
```
  step=  2320 s=2 ep=0 ce=0.4721 gn=0.5595
  step=  2400 s=2 ep=0 ce=0.3645 gn=0.6076
  step=  2500 s=2 ep=0 ce=0.4663 gn=0.4861
  step=  2620 s=2 ep=0 ce=0.5691 gn=1.5024
  step=  2740 s=2 ep=0 ce=0.6196 gn=1.9030
  step=  2860 s=2 ep=0 ce=0.5273 gn=1.1574
```
*gn values are pre-clip; max_grad_norm=0.3 clipping correctly active. CE not diverging.*

**Bugs identified post-session (fixed before Session 16):**
1. `save_checkpoint` uploads to `runs/stage3/checkpoint-XXXXXXX` missing `stage_{k}/` subdir → fixed.
2. Timeout clock started too late (inside `main()` after model load) → `_SCRIPT_START` moved to module import.
3. DiLoCo sharding ignored `total_samples_seen` remainder → fixed with pre-subtraction.
4. Hub auto-resume not DDP-coordinated → rank 0 resolves, broadcasts, deferred cleanup.
5. Auto pre-val on resumed partial stages → guarded by `is_new_stage`.

---

## Session 14 — Stage 1 Resumed, FP16 Confirmed (2026-04-18) ✅ COMPLETE

FP16 patch confirmed. Resumed from checkpoint-0001338. Timed out at step 2293.

```
  [GPU] Tesla T4  cc=sm75  VRAM=16GB  amp_dtype=float16
S1E0:   4%|▌  | 39/970 [26:51<10:41:10, 41.32s/it, ce=0.389, gn=0.200]
  [timeout] saving emergency checkpoint at step 2293 ...
```
**Confirmed: ~41s/step at k=1** (~4× speedup over pre-patch 162s/step).

---

## Session 13 — Stage 0 Val Complete, Stage 1 Started (2026-04-17 → 2026-04-18) ✅ COMPLETE

```
  [val] s=0 ep=0 val_ce=0.4041 val_acc=0.0222
  [ckpt] best -> runs/stage3_curriculum/stage_0/best  acc=0.022222...  ce=0.4040...
```
Stage 1 pre-FP16 patch: **162s ± 3s/step**. Timed out at step 1338.

---

## Session 12 — Stage 0 Training Complete, Val NCCL Crash (2026-04-17) ✅ RESOLVED

```
S0E0: 100%|████████████| 260/260 [9:54:14<00:00, 137.13s/it, ce=0.385, gn=0.111]
```
NCCL watchdog killed val. Fix: `timedelta(hours=4)` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`.

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
