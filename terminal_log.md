# terminal_log.md — Project Ouroboros
**All verified terminal outputs, newest first.**

---

> **Naming note:** Scripts were renamed for consistency.
> `phase1_viability_gate.py` → `viability_gate.py`
> `train_sft_phase2.py` → `train_sft.py`
> `Ouroboros_Blueprint_v3.md` → `BLUEPRINT.md`
> Logs below use original script names as they appeared at run time.

---

## Stage 1 — Pre-training, Kaggle Dual T4 (Session 6, resumed from checkpoint-0014902)
**Script:** `pretrain.py`
**Date:** 2026-04-05
**Hardware:** Kaggle Dual T4 (2× T4 16 GB, DDP auto-launched, world_size=2)
**Status:** 🟡 IN PROGRESS — log captured through step 15900 / 61,036 (26.1%)

**Clean resume confirmed:**
```
  [resume] local   checkpoint-0014902  step=14902  epoch=0  tokens=488,308,736  val_ce=5.289637256266823
```
✅

**Tokenizer warning (harmless):**
```
Token indices sequence length is longer than the specified maximum sequence length for
this model (133809 > 131072). Running this sequence through the model will result in
indexing errors
```
This is emitted by the HuggingFace tokenizer's own warning mechanism during dataset
streaming — it refers to a raw document length, not a model input. Model inputs are
packed at `chunk_size=1024` and are never oversized. No action required.

**Startup:**
```
2026-04-05 02:40:55 
2026-04-05 02:40:55 ========================================================================
2026-04-05 02:40:55   Stage 1 Pre-training - Project Ouroboros
2026-04-05 02:40:55 ========================================================================
2026-04-05 02:40:55   dataset          : HuggingFaceFW/fineweb-edu / sample-10BT
2026-04-05 02:40:55   tokenizer        : Qwen/Qwen2.5-0.5B  vocab=151,665
2026-04-05 02:40:55   preset           : nano
2026-04-05 02:40:55   model            : d_model=512  groups=1  heads=8/4
2026-04-05 02:40:55   chunk_size       : 1024
2026-04-05 02:40:55   batch x accum    : 8 global x 4
2026-04-05 02:40:55   world_size       : 2  (DDP auto-enabled)
2026-04-05 02:40:55   per_gpu_batch    : 4
2026-04-05 02:40:55   tokens / step    : 32,768
2026-04-05 02:40:55   token_budget     : 2,000,000,000
2026-04-05 02:40:55   total_steps      : 61,036
2026-04-05 02:40:55   dtype            : torch.bfloat16
2026-04-05 02:40:55   device           : cuda:0
2026-04-05 02:40:55   output_dir       : runs/stage1
2026-04-05 02:40:55   push_to_hub      : True
2026-04-05 02:40:55   timeout          : 12.0h  (buffer=15 min)
2026-04-05 02:40:55 ========================================================================
2026-04-05 02:40:57 
2026-04-05 02:40:57 Model parameters : 92,477,440 (92.5 M)
```

**Val + generation callbacks:**
```
2026-04-05 03:26:57   [val] step=15000  val_ce=5.2792

2026-04-05 03:26:57   -- Generation @ step 15000 (live weights) --
2026-04-05 03:26:58   P: The capital of France is
2026-04-05 03:26:58   C:  the town of L'O'Héon, the capital of the town of L'O'Héon, in the town of L'O'Héon. The town is the town of L'O'Héon, and the town of L'O'H'O'O'H'O'O'H'O'O'H'O
2026-04-05 03:26:58      uwr=0.393
2026-04-05 03:27:00   P: In mathematics, a prime number is
2026-04-05 03:27:00   C:  a number of numbers that are written in a number of different ways. For example, a number is a number that is written in a number of ways. For example, a numbe
2026-04-05 03:27:00      uwr=0.123
2026-04-05 03:27:01   P: def factorial(n):
2026-04-05 03:27:01     """Return n!."""
2026-04-05 03:27:01     if n
2026-04-05 03:27:01   C:  is a function that is not a function of the function, it is a function that is used to define the function of the function. For example, if the function is a f
2026-04-05 03:27:01      uwr=0.168
2026-04-05 03:27:03   P: Neural networks learn by
2026-04-05 03:27:03   C:  the time they are born. This is a very important step in the development of the brain. It is a very important part of the brain's development. It is a very imp
2026-04-05 03:27:03      uwr=0.292
2026-04-05 03:27:04   P: The French Revolution began in
2026-04-05 03:27:04   C:  1799, and the French Revolution was a period of great prosperity. The French Revolution was a period of great prosperity, and the French Revolution was a perio
2026-04-05 03:27:04      uwr=0.124
2026-04-05 03:27:04   Mean UWR: 0.220

2026-04-05 04:17:32   [val] step=15500  val_ce=5.2799

2026-04-05 04:17:32   -- Generation @ step 15500 (live weights) --
2026-04-05 04:17:34   P: The capital of France is
2026-04-05 04:17:34   C:  to build a new capital, the capital of the country, which is to be used as a capital. The capital is to be used as a capital, and the capital is to be used as
2026-04-05 04:17:34      uwr=0.165
2026-04-05 04:17:35   P: In mathematics, a prime number is
2026-04-05 04:17:35   C:  a number of numbers that are in the same order as the number of numbers in the same number. For example, the number of digits in the number of digits in the nu
2026-04-05 04:17:35      uwr=0.127
2026-04-05 04:17:37   P: def factorial(n):
2026-04-05 04:17:37     """Return n!."""
2026-04-05 04:17:37     if n
2026-04-05 04:17:37   C:  2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n 2, n
2026-04-05 04:17:37      uwr=0.033
2026-04-05 04:17:38   P: Neural networks learn by
2026-04-05 04:17:38   C:  means of a series of different mechanisms that are used to create a single system. The first is the system of the system of the brain. The second is the system
2026-04-05 04:17:38      uwr=0.185
2026-04-05 04:17:40   P: The French Revolution began in
2026-04-05 04:17:40   C:  1778, and the French revolution was a period of the revolution. The French revolution was a period of the revolution, and the revolution was the revolution of
2026-04-05 04:17:40      uwr=0.133
2026-04-05 04:17:40   Mean UWR: 0.129
```

**Training log (steps 14950–15900):**
```
2026-04-05 03:18:25   14950     4.1160          -     4.1650   0.4512   5.25e-04    2.035        706
2026-04-05 03:23:06   15000     4.3443          -     4.1642   0.4648   5.25e-04    2.035       5816
2026-04-05 03:29:47   [spike] step=15030  raw=4.8237  ema=4.1789
2026-04-05 03:31:37   15050     4.2771     5.2792     4.1668   0.4141   5.24e-04    2.035       3209
2026-04-05 03:36:18   15100     4.2057     5.2792     4.1648   0.3945   5.24e-04    2.035       5833
2026-04-05 03:41:00   15150     4.2546     5.2792     4.1605   0.4375   5.23e-04    2.035       5803
2026-04-05 03:45:43   15200     4.2229     5.2792     4.1516   0.4160   5.23e-04    2.035       5796
2026-04-05 03:47:07   [spike] step=15215  raw=4.7100  ema=4.1549
2026-04-05 03:50:26   15250     4.2361     5.2792     4.1536   0.4453   5.22e-04    2.035       5791
2026-04-05 03:55:09   15300     4.1254     5.2792     4.1561   0.4043   5.22e-04    2.035       5792
2026-04-05 03:59:52   15350     4.0473     5.2792     4.1481   0.3789   5.21e-04    2.035       5783
2026-04-05 04:04:35   15400     4.1413     5.2792     4.1615   0.4160   5.21e-04    2.035       5791
2026-04-05 04:09:17   15450     4.2836     5.2792     4.1639   0.4199   5.21e-04    2.035       5795
2026-04-05 04:14:00   15500     4.1576     5.2792     4.1583   0.4922   5.20e-04    2.035       5790
2026-04-05 04:22:13   15550     4.2562     5.2799     4.1567   0.4609   5.20e-04    2.035       3325
2026-04-05 04:26:54   15600     4.2673     5.2799     4.1618   0.3926   5.19e-04    2.035       5837
2026-04-05 04:31:36   15650     4.1992     5.2799     4.1574   0.4141   5.19e-04    2.035       5808
2026-04-05 04:36:18   15700     4.0449     5.2799     4.1487   0.4023   5.18e-04    2.035       5810
2026-04-05 04:41:00   15750     4.2621     5.2799     4.1494   0.5000   5.18e-04    2.035       5804
2026-04-05 04:42:36   [spike] step=15767  raw=4.7158  ema=4.1564
2026-04-05 04:44:24   [spike] step=15786  raw=4.6814  ema=4.1619
2026-04-05 04:44:30   [spike] step=15787  raw=4.8845  ema=4.1691
2026-04-05 04:45:43   15800     4.3277     5.2799     4.1686   0.7266   5.17e-04    2.035       5800
2026-04-05 04:50:26   15850     4.1247     5.2799     4.1648   0.4570   5.17e-04    2.035       5794
2026-04-05 04:53:15   [spike] step=15880  raw=4.7188  ema=4.1635
2026-04-05 04:55:08   15900     4.1067     5.2799     4.1620   0.4395   5.16e-04    2.035       5798
```
*(Log ends here — session still running)*

**Checkpoint status (end of captured log):**
- Local (keep_last=3): checkpoint-0013000 pruned; checkpoint-0015000 saved and uploaded (commit=cc7891dc)
- Hub: checkpoint-0015000 confirmed uploaded

---

## Stage 1 — Pre-training, Kaggle Dual T4 (Session 5, resumed from checkpoint-0002000)
**Script:** `pretrain.py`
**Date:** 2026-04-03
**Hardware:** Kaggle Dual T4 (2× T4 16 GB, DDP auto-launched, world_size=2)
**Status:** 🟡 IN PROGRESS — log captured through step 9000 / 61,036 (run still ongoing)

**Bug 5 fix confirmed:**
```
  [resume] local   checkpoint-0002000  step=2000  epoch=0  tokens=65,536,000  val_ce=5.564212733394234
```
Clean local resume from disk. ✅

**Startup:**
```
2026-04-03 08:27:48 
2026-04-03 08:27:48 ========================================================================
2026-04-03 08:27:48   Stage 1 Pre-training - Project Ouroboros
2026-04-03 08:27:48 ========================================================================
2026-04-03 08:27:48   dataset          : HuggingFaceFW/fineweb-edu / sample-10BT
2026-04-03 08:27:48   tokenizer        : Qwen/Qwen2.5-0.5B  vocab=151,665
2026-04-03 08:27:48   preset           : nano
2026-04-03 08:27:48   model            : d_model=512  groups=1  heads=8/4
2026-04-03 08:27:48   chunk_size       : 1024
2026-04-03 08:27:48   batch x accum    : 8 global x 4
2026-04-03 08:27:48   world_size       : 2  (DDP auto-enabled)
2026-04-03 08:27:48   per_gpu_batch    : 4
2026-04-03 08:27:48   tokens / step    : 32,768
2026-04-03 08:27:48   token_budget     : 2,000,000,000
2026-04-03 08:27:48   total_steps      : 61,036
2026-04-03 08:27:48   dtype            : torch.bfloat16
2026-04-03 08:27:48   device           : cuda:0
2026-04-03 08:27:48   output_dir       : runs/stage1
2026-04-03 08:27:48   push_to_hub      : True
2026-04-03 08:27:48 ========================================================================
2026-04-03 08:27:50 
2026-04-03 08:27:50 Model parameters : 92,477,440 (92.5 M)
2026-04-03 08:27:50 
2026-04-03 08:27:50   Building val buffer (2,000,000 tokens) ...
2026-04-03 08:28:05   Val buffer: 2,000,000 tokens from 1,887 docs
2026-04-03 08:28:06   [resume] local   checkpoint-0002000  step=2000  epoch=0  tokens=65,536,000  val_ce=5.564212733394234
2026-04-03 08:28:07    step   train_ce     val_ce       smth    gnorm         lr     VRAM      tok/s
2026-04-03 08:28:07 --------------------------------------------------------------------------------
2026-04-03 08:28:07   epoch 0  offset=228  skipping=64000 chunks
```

**Training log (steps 8500–9000):**
```
2026-04-03 19:35:45   -- Generation @ step 8500 (live weights) --
2026-04-03 19:35:47   P: The capital of France is
2026-04-03 19:35:47   C:  the capital of the country. The capital is the capital of the country. The capital is the capital of the country. The capital is the capital of the country. Th
2026-04-03 19:35:47      uwr=0.056
2026-04-03 19:35:48   P: In mathematics, a prime number is
2026-04-03 19:35:48   C:  a number. The number of numbers is the number of numbers. The number of numbers is the number of numbers. The number of numbers is the number of numbers. The n
2026-04-03 19:35:48      uwr=0.083
2026-04-03 19:35:50   P: def factorial(n):
2026-04-03 19:35:50     """Return n!."""
2026-04-03 19:35:50     if n
2026-04-03 19:35:50   C: . 1
2026-04-03 19:35:50 * 100% of the time
...
2026-04-03 19:35:50      uwr=0.119
2026-04-03 19:35:51   P: Neural networks learn by
2026-04-03 19:35:51   C:  means of the Internet. The Internet is a network of networks that connect to the Internet. The Internet is a network of networks that connect to the Internet.
2026-04-03 19:35:51      uwr=0.135
2026-04-03 19:35:53   P: The French Revolution began in
2026-04-03 19:35:53   C:  1780, when the French government was forced to take over the French and French colonies. The French were forced to leave the French colonies in 1799, and the F
2026-04-03 19:35:53      uwr=0.239
2026-04-03 19:35:53   Mean UWR: 0.126
2026-04-03 19:40:34    8550     4.4062     5.2900     4.2943   0.4258   5.75e-04    2.035       3190
2026-04-03 19:45:16    8600     3.9206     5.2900     4.2877   0.9844   5.75e-04    2.035       5816
2026-04-03 19:49:58    8650     4.1947     5.2900     4.2884   0.3691   5.75e-04    2.035       5811
2026-04-03 19:54:40    8700     4.2052     5.2900     4.2722   0.3574   5.74e-04    2.035       5808
2026-04-03 19:59:22    8750     4.3004     5.2900     4.2747   0.3809   5.74e-04    2.035       5819
2026-04-03 20:03:47   [spike] step=8797  raw=5.7361  ema=4.2857   ← ⚠ cluster start
2026-04-03 20:04:03    8800     4.3443     5.2900     4.2871   0.7773   5.74e-04    2.035       5819
2026-04-03 20:08:28   [spike] step=8847  raw=4.8565  ema=4.2874   ← ⚠ consecutive
2026-04-03 20:08:45    8850     4.2235     5.2900     4.2874   0.6406   5.74e-04    2.035       5811
2026-04-03 20:13:27   [spike] step=8900  raw=4.8290  ema=4.2898   ← ⚠ consecutive
2026-04-03 20:13:27    8900     4.8290     5.2900     4.2898   0.6250   5.73e-04    2.035       5820
2026-04-03 20:18:08    8950     4.1701     5.2900     4.2706   0.3730   5.73e-04    2.035       5820
2026-04-03 20:22:50    9000     4.1912     5.2900     4.2486   0.3691   5.73e-04    2.035       5814
```

*(Log ends here — re-run pending)*

---

## Stage 2 SFT — Patch Verification (Code Audit, no run)
**Script:** `train_sft.py`
**Date:** 2026-04-03
**Method:** Static code audit of the submitted `train_sft.py` file.

**Bug 1 — compute_val_ce live-weight restore:** ✅ FIXED
**Bug 2 — load_latest_checkpoint direct path handling:** ✅ FIXED
**Bug 3 — collate prompt masking:** 🔴 NOT FIXED (must fix before Stage 2 run)
**Issue 3 — Header string:** ✅ FIXED
**Issue 4 — Multi-dataset mixing:** ✅ IMPLEMENTED

---

## Stage 1 — Pre-training, Kaggle Dual T4 (Session 4, steps 1–1700)
**Script:** `pretrain.py`
**Date:** Session 4
**Hardware:** Kaggle Dual T4 (2× T4 16 GB, DDP auto-launched, world_size=2)
**Status:** Session ended at step 1700; resumed in Session 5

**⚠ No checkpoint saved (Bug 5):** Hub 401 at step 1000 caused `save_checkpoint` to
abandon `.tmp` without renaming to final. Bug 5 was patched before Session 5.
Session 5 started from scratch (step=0) since no local checkpoint existed from Session 4.

**Smoke test output (runs automatically before main loop):**
```
  Building val buffer (512 tokens) ...
  Val buffer: 512 tokens from 2 docs
[smoke] epoch_offset=7
[smoke] step  1  loss=8.2671
[smoke] step 10  loss=8.1193
[smoke] step 20  loss=7.3500
[smoke] val_ce computed: 7.9517
  [ckpt] saved  -> /tmp/stage1_smoke_.../checkpoint-0000020
  [resume] local  checkpoint-0000020  step=20  tokens=2,560
[smoke] checkpoint saved and reloaded cleanly
[smoke] All checks passed - launching main training loop
```

**DDP auto-launch:**
```
[ddp] detected 2 CUDA devices; launching single-node DDP with global batch_size=8 (4 per GPU).
```

**Training log (steps 1–1700, Session 4):**
```
      1    11.9824          -    11.9824   1.9766   6.00e-06    2.035       4222
     50     9.2922          -    11.4559   2.2656   1.53e-04    2.035       6040
    100     7.0954          -     9.9666   0.6797   3.03e-04    2.035       5859
    ...
   1000     4.9713     6.3811     5.1369   0.4590   6.00e-04    2.035       5719
  [val] step=1000  val_ce=5.8478
  [hub] upload failed for checkpoint-0001000: Client error '401 Unauthorized' ← ⚠ Bug 5
   1500     4.8855     5.8478     4.9102   0.4805   5.99e-04    2.035       5702
  [val] step=1500  val_ce=5.6810
   1700     5.0407     5.6810     4.8453   0.4668   5.99e-04    2.035       5735
```
*(Session 4 ended at step 1700 — no checkpoint saved due to Bug 5)*

---

## Stage 2 SFT — Dry-run (nano, 300 samples, 100 steps)
**Script:** `train_sft_phase2.py` (now: `train_sft.py`)
**Date:** Session 3
**Result:** Pipeline verified. EMA generation degenerate at step 100 (expected at decay=0.999). Corrected to `--ema_decay 0.995`.

```
  preset=nano  seq_len=512  batch×accum=2×4=8  lr=0.0002  warmup=100

   Step   Train CE     Val CE    GNorm         LR     VRAM    Tok/s
────────────────────────────────────────────────────────────────────
      1    11.9931          -   3.7031   4.00e-06    1.576     1662
     40     9.6128          -   2.7969   8.20e-05    1.576     3731
  [val] step=50  val_ce=11.9826
    100     4.9913    11.9826   1.8594   2.00e-05    1.576     3412
  [val] step=100  val_ce=11.9646
  [ckpt] saved  → runs/phase2/checkpoint-0000100

  Total time: 2.3 min  Peak VRAM: 3.62 GB  Final val CE: 11.9646
```

---

## Stage 0 — Viability Gate
**Script:** `viability_gate.py`   **Date:** Session 2   **Result:** ALL GATES PASSED

```
  G1_ce_converged        CE < 3.5       final CE = 2.0034   PASS ✓
  G2_generation_coherent UWR > 0.1      mean UWR = 0.573    PASS ✓
  G3_grad_norm_stable    gnorm < 10.0   max = 4.0312        PASS ✓
  G4_vram_stable         VRAM Δ < 1.0GB Δ = 0.000 GB        PASS ✓
  Total time: 3.4 min  Peak VRAM: 2.07 GB  Steps: 300
```

---

## Stage 0 — Baseline Architecture Smoke Test
**Script:** `baseline_trm_mamba.py`   **Date:** Session 1   **Result:** PASSED

```
parameters : 92,477,440 (92.5 M)   initial loss : 11.9904   backward : OK
grad norms : total=6.4242           All checks passed. Baseline architecture is healthy.
```
