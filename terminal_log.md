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
**Status:** ✅ COMPLETE (graceful timeout exit) — steps 14902→21501, tokens 488M→705M

**⚠ Val CE rose from 5.2767 (step 16000) to 5.3241 (step 21500) — +0.047 nats. Rising, not flat.**
**⚠ Spike density tripled in steps 18000–21500 (~15+ events). shuffle_buffer=20000 insufficient for this data shard.**

**Clean resume confirmed:**
```
  [resume] local   checkpoint-0014902  step=14902  epoch=0  tokens=488,308,736  val_ce=5.289637256266823
```
✅

**Val CE + generation summary (complete):**
```
step 15000  val_ce=5.2792  mean_uwr=0.220
step 15500  val_ce=5.2799  mean_uwr=0.129
step 16000  val_ce=5.2767  mean_uwr=0.163  ← only improvement this session
step 16500  val_ce=5.2788  mean_uwr=0.307
step 17000  val_ce=5.2883  mean_uwr=0.272  ← rising begins
step 17500  val_ce=5.2925  mean_uwr=0.447
step 18000  val_ce=5.3052  mean_uwr=0.161  ← dense spikes start
step 18500  val_ce=5.3079  mean_uwr=0.280
step 19000  val_ce=5.3105  mean_uwr=0.292
step 19500  val_ce=5.3129  mean_uwr=0.134
step 20000  val_ce=5.3135  mean_uwr=0.081
step 20500  val_ce=5.3152  mean_uwr=0.344
step 21000  val_ce=5.3190  mean_uwr=0.099
step 21500  val_ce=5.3241  mean_uwr=0.266
```

**Selected training log with spikes annotated:**
```
2026-04-05 03:18:25   14950     4.1160          -     4.1650   0.4512   5.25e-04    2.035        706
2026-04-05 03:23:06   15000     4.3443          -     4.1642   0.4648   5.25e-04    2.035       5816
2026-04-05 03:29:47   [spike] step=15030  raw=4.8237  ema=4.1789
2026-04-05 03:47:07   [spike] step=15215  raw=4.7100  ema=4.1549
2026-04-05 04:14:00   15500     4.1576     5.2792     4.1583   0.4922   5.20e-04    2.035       5790
2026-04-05 04:42:36   [spike] step=15767  raw=4.7158  ema=4.1564
2026-04-05 04:44:24   [spike] step=15786  raw=4.6814  ema=4.1619
2026-04-05 04:44:30   [spike] step=15787  raw=4.8845  ema=4.1691
2026-04-05 04:53:15   [spike] step=15880  raw=4.7188  ema=4.1635
2026-04-05 05:04:34   16000     4.1732     5.2799     4.1677   0.4141   5.15e-04    2.035       5795
2026-04-05 05:14:47   [spike] step=16068  raw=4.7410  ema=4.1540
2026-04-05 05:53:09   [spike] step=16475  raw=5.5527  ema=4.1674   ← large spike
2026-04-05 05:55:31   16500     4.1851     5.2767     4.1577   0.4961   5.10e-04    2.035       5778
2026-04-05 06:09:45   [spike] step=16614  raw=4.7115  ema=4.1560
2026-04-05 06:46:13   17000     4.1732     5.2788     4.1289   0.5000   5.05e-04    2.035       5769
2026-04-05 07:49:07   [spike] step=17590  raw=4.9689  ema=4.1164
2026-04-05 08:13:01   [spike] step=17843  raw=4.8913  ema=4.1343   ← 3-step cluster start
2026-04-05 08:13:06   [spike] step=17844  raw=4.7847  ema=4.1409
2026-04-05 08:13:12   [spike] step=17845  raw=4.6801  ema=4.1462
2026-04-05 08:27:51   18000     4.4503     5.2925     4.1206   0.4492   4.94e-04    2.035       5780
2026-04-05 08:42:25   [spike] step=18115  raw=4.9637  ema=4.1359
2026-04-05 08:46:11   [spike] step=18155  raw=4.6484  ema=4.1341
2026-04-05 08:47:48   [spike] step=18172  raw=4.6760  ema=4.1363
2026-04-05 08:55:56   [spike] step=18258  raw=4.6397  ema=4.1252
2026-04-05 09:04:04   [spike] step=18344  raw=5.1457  ema=4.1429   ← large
2026-04-05 09:09:50   [spike] step=18405  raw=5.1301  ema=4.1451   ← large
2026-04-05 09:14:05   [spike] step=18450  raw=4.6893  ema=4.1426
2026-04-05 09:23:17   [spike] step=18509  raw=4.8420  ema=4.1376
2026-04-05 09:23:23   [spike] step=18510  raw=4.8002  ema=4.1442
2026-04-05 09:29:06   [spike] step=18572  raw=5.0423  ema=4.1312   ← large
2026-04-05 10:09:29   19000     4.1597     5.3052     4.0977   0.5117   4.82e-04    2.035       5771
2026-04-05 10:58:47   [spike] step=19483  raw=5.1208  ema=4.1207   ← large
2026-04-05 11:29:50   [spike] step=19775  raw=4.7476  ema=4.1051
2026-04-05 11:45:01   [spike] step=19936  raw=5.2361  ema=4.0927   ← large
2026-04-05 11:51:05   20000     4.0818     5.3129     4.0760   0.4043   4.71e-04    2.035       5773
2026-04-05 12:16:09   [spike] step=20226  raw=4.6460  ema=4.0820
2026-04-05 12:19:11   [spike] step=20258  raw=4.8589  ema=4.0897
2026-04-05 12:23:33   [spike] step=20304  raw=4.5883  ema=4.0861
2026-04-05 13:22:05   [spike] step=20886  raw=4.6380  ema=4.0712
2026-04-05 13:32:41   [spike] step=20998  raw=4.6908  ema=4.0823
2026-04-05 13:32:53   21000     4.3295     5.3152     4.0865   0.9219   4.59e-04    2.035       5769
2026-04-05 13:53:59   [spike] step=21184  raw=4.6594  ema=4.0861
2026-04-05 14:01:33   [spike] step=21264  raw=4.8866  ema=4.1154
2026-04-05 14:13:28   [spike] step=21390  raw=4.9710  ema=4.1150
2026-04-05 14:23:52   21500     4.0554     5.3190     4.0796   0.5703   4.52e-04    2.035       5784
```

**Graceful timeout exit:**
```
2026-04-05 14:27:36   [timeout] 11.77h elapsed — 13.7 min remaining (< 15 min buffer).
2026-04-05 14:27:36   [timeout] Saving emergency checkpoint at step 21501 (local only) ...
2026-04-05 14:27:38   [ckpt] saved  -> runs/stage1/checkpoint-0021501
2026-04-05 14:27:38   [ckpt] pruned -> checkpoint-0019000
2026-04-05 14:27:38   [timeout] Emergency checkpoint saved.
2026-04-05 14:27:38 ========================================================================
2026-04-05 14:27:38   [timeout] Session budget exhausted — graceful exit
2026-04-05 14:27:38 ========================================================================
2026-04-05 14:27:38   Wall time elapsed  : 11.77h / 12.0h
2026-04-05 14:27:38   Steps completed    : 21,501 / 61,036
2026-04-05 14:27:38   Tokens processed   : 704,544,768 / 2,000,000,000
2026-04-05 14:27:38   Last val CE        : 5.324081295402125
2026-04-05 14:27:38   Checkpoint saved   : runs/stage1/checkpoint-0021501  (local only)
2026-04-05 14:27:38 ========================================================================
```

**Checkpoint status (end of session):**
- Local: checkpoint-0021501 (local only — Hub push did not occur before timeout)
- Hub: checkpoint-0021000 confirmed (commit=d70d2c49) ← safe resume point if local lost
- Pruned: checkpoint-0019000

---

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
