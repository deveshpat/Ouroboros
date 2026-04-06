# terminal_log.md — Project Ouroboros
**All verified terminal outputs, newest first.**

---

> **Naming note:** Scripts were renamed for consistency.
> `phase1_viability_gate.py` → `viability_gate.py`
> `train_sft_phase2.py` → `train_sft.py`
> `Ouroboros_Blueprint_v3.md` → `BLUEPRINT.md`

---

## Stage 2 SFT — Session 2 (full dataset mix, DDP, FAILED)
**Script:** `train_sft.py`
**Date:** 2026-04-06
**Hardware:** Kaggle Dual T4 (world_size=2)
**Status:** ❌ FAILED — multiple cascading bugs; all Stage 2 local checkpoints deleted

**Critical failures:**
1. Resume downloaded 18 Stage 1 Hub checkpoints into output_dir before finding local Stage 2 checkpoint
2. Hub downloads contaminated output_dir → prune deleted ALL Stage 2 checkpoints (kept Stage 1 downloads)
3. Optimizer state carried over at near-minimum LR (1.89e-05) → 75 spikes in 717 steps
4. max_seq_len=1024 filtered 97% of Stratos, 92% of OpenR1-Math, 99.9% of OpenR1-Code
5. Hub upload failed ("not a directory") because checkpoint was pruned before upload

**Startup:**
```
2026-04-06 03:37:42   Stage 2 SFT - Project Ouroboros
2026-04-06 03:37:42   dataset_mix   : full
2026-04-06 03:37:42   world_size    : 2  (DDP auto-enabled)
2026-04-06 03:37:42   per_gpu_batch : 1
```

**Dataset counts at max_seq_len=1024:**
```
kept 515 / target 16710   (too_long=16195)  ← Stratos: 97% filtered!
kept 11140 / target 11140 (too_long=18)     ← MetaMathQA: ok
kept 8355 / target 8355   (too_long=306)    ← OpenHermes: ok
kept 843 / target 11140   (too_long=92890)  ← OpenR1-Math: 92% filtered!
kept 9 / target 8355      (too_long=11663)  ← OpenR1-Code: 99.9% filtered!
Total mixed samples: 20862
```

**Resume log — catastrophic cascade:**
```
2026-04-06 04:54:02   [resume] local_stage2_latest=2979  hub_latest=21000
2026-04-06 04:54:02   [hub]  downloading checkpoint-0021000 ...  ← should have stopped here
2026-04-06 04:54:08   [hub]  downloading checkpoint-0020000 ...
...  (18 Hub checkpoints downloaded into output_dir)
2026-04-06 04:56:09   [resume] corrupt checkpoint-0004000: CUDA out of memory
2026-04-06 04:56:16   [resume] corrupt checkpoint-0003000: CUDA out of memory
2026-04-06 04:56:16   [resume] checkpoint-0002979  restored optimizer/model state at step=2979,
                       but dataset-defining args changed; resetting epoch/sample cursor.
```

**Data change handled wrong — optimizer NOT reset:**
```
2026-04-06 04:59:01    2980     3.8943      0.3458          -         -   3.2812   1.89e-05
                                                                                    ↑ LR near minimum, should be ~1e-4
```

**Generation — catastrophic (MetaMathQA number tokens dominating):**
```
  Q: What is 15 + 27?
  A: 100000000000000000000000000000000000000000000000000000000000000000000000000000
  Q: What is the capital of Japan?
  A: 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000
```

**Prune cascade (deleted ALL Stage 2 checkpoints):**
```
[ckpt] pruned -> checkpoint-0002000  ← Stage 2
[ckpt] pruned -> checkpoint-0002500  ← Stage 2
[ckpt] pruned -> checkpoint-0002979  ← Stage 2 FINAL
[ckpt] pruned -> checkpoint-0003000  ← just saved, deleted before Hub upload
...
[ckpt] pruned -> checkpoint-0018000  ← Stage 1 Hub downloads
(kept: 0019000, 0020000, 0021000 — all Stage 1)
```

**Hub upload failures:**
```
[hub] upload failed for checkpoint-0003000: Provided path: '...' is not a directory
[hub] upload failed for checkpoint-0003500: Provided path: '...' is not a directory
[hub] upload failed for checkpoint-0003717: Provided path: '...' is not a directory
```

**Final state:**
```
2026-04-06 05:19:30   Total time   : 20.5 min
2026-04-06 05:19:30   Peak VRAM    : 14.30 GB
2026-04-06 05:19:30   Spike count  : 75
2026-04-06 05:19:30   Final val CE : 5.7135  (answer tokens only)
2026-04-06 05:19:30   Status       : val_ce >= 1.5 - extend training or check data quality
```

**Checkpoint status after Session 2:**
- Local: checkpoint-0019000, 0020000, 0021000 (all Stage 1, useless for Stage 2)
- Hub: checkpoint-0002979 confirmed (commit=8981b950) ← only valid Stage 2 checkpoint

---

## Stage 2 SFT — Session 1 (nano, stratos-only, from checkpoint-0021000)
**Script:** `train_sft.py`
**Date:** 2026-04-05
**Hardware:** Kaggle T4 (single GPU)
**Status:** ✅ COMPLETE — step 2979/2979 — val_ce=4.9153, gate NOT met

**⚠ Key finding:** Session 1 trained on stratos WITHOUT reasoning chains. The 16,710 samples
fitting in 1024 tokens is only possible if reasoning was stripped (sessions 1 code may have
returned empty reasoning from `_parse_assistant_blob`). No `<think>` tags ever appeared in
generation throughout all 2979 steps, confirming the model never learned the reasoning format.

**Stage 1→2 transfer confirmed:**
```
  [resume] candidate summary: local_latest=none  hub_latest=21000
  [hub]  downloading checkpoint-0021000 ...
  [init]   checkpoint-0021000  loaded Stage 1 weights; resetting optimizer/scheduler for Stage 2.
```
✅

**Startup:**
```
2026-04-05 16:53:58   preset        : nano
2026-04-05 16:53:58   seq_len       : 1024
2026-04-05 16:53:58   batch x accum : 2 x 8 = 16
2026-04-05 16:53:58   dataset_mix   : stratos
2026-04-05 17:02:21   16710 samples kept, 0 skipped
2026-04-05 17:02:21   train: 15875  val: 835
2026-04-05 17:02:23   92.5M parameters  (preset=nano)
2026-04-05 17:02:23   Schedule  : cosine warmup=100 total=2979  (3 epochs x 993 steps/epoch)
```

**Val CE trajectory — plateau, gate never reached:**
```
step  250  val_ce=5.2199  val_acc=0.2367  mean_uwr=0.156
step  500  val_ce=5.0644  val_acc=0.2514  mean_uwr=0.108
step  750  val_ce=4.9821  val_acc=0.2578  mean_uwr=0.107
step 1000  val_ce=4.9480  val_acc=0.2606  mean_uwr=0.114
step 1250  val_ce=4.9280  val_acc=0.2618  mean_uwr=0.100
step 1500  val_ce=4.9232  val_acc=0.2621  mean_uwr=0.087
step 1750  val_ce=4.9202  val_acc=0.2622  mean_uwr=0.098
step 2000  val_ce=4.9172  val_acc=0.2624  mean_uwr=0.086
step 2250  val_ce=4.9162  val_acc=0.2624  mean_uwr=0.090
step 2500  val_ce=4.9153  val_acc=0.2624  mean_uwr=0.088
step 2750  val_ce=4.9154  val_acc=0.2623  mean_uwr=0.087
step 2979  val_ce=4.9153  val_acc=0.2624  mean_uwr=0.085
```

**Training log (selected):**
```
2026-04-05 17:02:35       1     3.9773      0.2948          -         -   1.9141   2.00e-06    2.440     2442
2026-04-05 17:11:28     100     3.6698      0.3556          -         -   1.5234   1.00e-04    2.440     3021
2026-04-05 17:20:26     200     3.3553      0.3774          -         -   0.9102   9.97e-05    2.440     3026
2026-04-05 18:37:50    1000     2.9958      0.4151     4.9821    0.2578   0.8047   8.00e-05    2.440     2851
2026-04-05 20:15:11    2000     2.8142      0.4366     4.9172    0.2622   0.8594   3.33e-05    2.440     2822
2026-04-05 21:48:54    2960     2.9100      0.4437     4.9154    0.2623   0.9648   1.00e-05    2.440     2810
```

**Generation sample @ step 2979 (unchanged since step 250 — no format learning):**
```
  Q: What is 15 + 27?
  A: So, the problem is that the problem is a problem. The problem is that the problem is a problem...
     uwr=0.079
  Q: What is the capital of Japan?
  A: The question is about the question. The question is asking for the question...
     uwr=0.079
  Mean UWR: 0.085
```

**Completion:**
```
2026-04-05 21:52:38   Total steps  : 2979
2026-04-05 21:52:38   Total time   : 290.2 min
2026-04-05 21:52:38   Peak VRAM    : 6.59 GB
2026-04-05 21:52:38   Final val CE : 4.9153  (answer tokens only)
2026-04-05 21:52:38   Final val acc: 0.2624
2026-04-05 21:52:38   Status       : val_ce >= 1.5 - extend training or check data quality
2026-04-05 21:52:38   [hub] uploaded  checkpoint-0002979 (commit=8981b950)
```

**Hub checkpoints (Stage 2 Session 1):** 0000500 (1612c426), 0001000 (1149879b), 0001500 (145294c9), 0002000 (3748d098), 0002500 (7220ea27), **0002979 (8981b950) ← final**

---

## Stage 2 SFT — Patch Verification (Code Audit, no run)
**Script:** `train_sft.py`  **Date:** 2026-04-03

**Bug 1 — compute_val_ce live-weight restore:** ✅ FIXED
**Bug 2 — load_latest_checkpoint direct path handling:** ✅ FIXED
**Bug 3 — collate prompt masking:** ✅ FIXED (before any real Stage 2 run)
**Issue 3 — Header string:** ✅ FIXED
**Issue 4 — Multi-dataset mixing:** ✅ IMPLEMENTED

---

## Stage 1 — Pre-training, Kaggle Dual T4 (Session 6, resumed from checkpoint-0014902)
**Script:** `pretrain.py`
**Date:** 2026-04-05
**Hardware:** Kaggle Dual T4 (2× T4 16 GB, DDP auto-launched, world_size=2)
**Status:** ✅ COMPLETE (graceful timeout exit) — steps 14902→21501, tokens 488M→705M

**⚠ Val CE rose 5.2767 (step 16000) → 5.3241 (step 21500) — +0.047 nats. Rising, not flat.**
**⚠ Spike density tripled in steps 18000–21500 (~15+ events). shuffle_buffer=20000 insufficient.**

**Val CE summary:**
```
step 16000  val_ce=5.2767  mean_uwr=0.163  ← only improvement this session
step 17000  val_ce=5.2883  mean_uwr=0.272  ← rising begins
step 18000  val_ce=5.3052  mean_uwr=0.161  ← dense spikes start
step 20000  val_ce=5.3135  mean_uwr=0.081
step 21000  val_ce=5.3190  mean_uwr=0.099
step 21500  val_ce=5.3241  mean_uwr=0.266
```

**Selected training log:**
```
2026-04-05 05:55:31   16500     4.1851     5.2767     4.1577   0.4961   5.10e-04    2.035       5778
2026-04-05 08:13:01   [spike] step=17843  raw=4.8913  ema=4.1343   ← 3-step cluster
2026-04-05 08:13:06   [spike] step=17844  raw=4.7847  ema=4.1409
2026-04-05 08:13:12   [spike] step=17845  raw=4.6801  ema=4.1462
2026-04-05 09:04:04   [spike] step=18344  raw=5.1457  ema=4.1429   ← large
2026-04-05 11:45:01   [spike] step=19936  raw=5.2361  ema=4.0927   ← large
2026-04-05 13:32:53   21000     4.3295     5.3152     4.0865   0.9219   4.59e-04    2.035       5769
2026-04-05 14:23:52   21500     4.0554     5.3190     4.0796   0.5703   4.52e-04    2.035       5784
```

**Graceful timeout exit:**
```
2026-04-05 14:27:36   [timeout] 11.77h elapsed — 13.7 min remaining (< 15 min buffer).
2026-04-05 14:27:38   [ckpt] saved  -> runs/stage1/checkpoint-0021501
2026-04-05 14:27:38   Tokens processed   : 704,544,768 / 2,000,000,000
2026-04-05 14:27:38   Last val CE        : 5.324081295402125
```

**Checkpoint status:** Local: checkpoint-0021501 (local only). Hub: checkpoint-0021000 (commit=d70d2c49) ← safe resume point.

---

## Stage 1 — Pre-training, Kaggle Dual T4 (Session 5, resumed from checkpoint-0002000)
**Script:** `pretrain.py`  **Date:** 2026-04-03
**Status:** ✅ COMPLETE — steps 2000→14902

```
  [resume] local   checkpoint-0002000  step=2000  epoch=0  tokens=65,536,000  val_ce=5.564212733394234
  epoch 0  offset=228  skipping=64000 chunks
  [spike] step=8797  raw=5.7361  ema=4.2857
  9000     4.1912     5.2900     4.2486   0.3691   5.73e-04    2.035       5814
```

---

## Stage 1 — Pre-training, Kaggle Dual T4 (Session 4, steps 1–1700)
**Script:** `pretrain.py`
**Status:** ✅ COMPLETE — no checkpoint saved (Bug 5); Session 5 restarted → checkpoint-0002000

```
[smoke] epoch_offset=7  [smoke] step 20  loss=7.3500  [smoke] All checks passed.
[ddp] detected 2 CUDA devices; launching single-node DDP with global batch_size=8 (4 per GPU).
      1    11.9824   1.9766   6.00e-06    2.035       4222
   1000     4.9713   5.8478   0.4590   6.00e-04    2.035       5719
  [hub] upload failed: Client error '401 Unauthorized' ← ⚠ Bug 5
```

---

## Stage 2 SFT — Dry-run (nano, 300 samples, 100 steps)
**Script:** `train_sft_phase2.py` (now: `train_sft.py`)  **Date:** Session 3
**Result:** Pipeline verified. EMA generation degenerate at step 100 (expected at decay=0.999). Corrected to `--ema_decay 0.995`.
```
      1    11.9931   3.7031   4.00e-06    1.576     1662
    100     4.9913    11.9826   1.8594   2.00e-05    1.576     3412
  Total time: 2.3 min  Peak VRAM: 3.62 GB
```

---

## Stage 0 — Viability Gate
**Script:** `viability_gate.py`  **Date:** Session 2  **Result:** ALL GATES PASSED
```
  G1_ce_converged        CE < 3.5       final CE = 2.0034   PASS ✓
  G2_generation_coherent UWR > 0.1      mean UWR = 0.573    PASS ✓
  G3_grad_norm_stable    gnorm < 10.0   max = 4.0312        PASS ✓
  G4_vram_stable         VRAM Δ < 1.0GB Δ = 0.000 GB        PASS ✓
  Total time: 3.4 min  Peak VRAM: 2.07 GB  Steps: 300
```

---

## Stage 0 — Baseline Architecture Smoke Test
**Script:** `baseline_trm_mamba.py`  **Date:** Session 1  **Result:** PASSED
```
parameters : 92,477,440 (92.5 M)   initial loss : 11.9904   backward : OK
grad norms : total=6.4242           All checks passed. Baseline architecture is healthy.
```
