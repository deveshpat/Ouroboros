# terminal_log.md — Project Ouroboros
**All verified terminal outputs, newest first.**

---

> **Naming note:** Scripts were renamed for consistency.
> `phase1_viability_gate.py` → `viability_gate.py`
> `train_sft_phase2.py` → `train_sft.py`
> `Ouroboros_Blueprint_v3.md` → `BLUEPRINT.md`

---

## Stage 2 SFT — Session 1 (nano, stratos-only, from checkpoint-0021000)
**Script:** `train_sft.py`
**Date:** 2026-04-05
**Hardware:** Kaggle T4 (single GPU)
**Status:** 🟡 IN PROGRESS — step ~1020 / 2979 (34%); val CE declining, generation still degenerate

**Stage 1→2 transfer confirmed:**
```
  [resume] candidate summary: local_latest=none  hub_latest=21000
  [hub]  downloading checkpoint-0021000 ...
  [init]   checkpoint-0021000  loaded Stage 1 weights; resetting optimizer/scheduler for Stage 2.
```
✅

**Startup:**
```
2026-04-05 16:53:57   Stage 2 SFT - Project Ouroboros
2026-04-05 16:53:58   preset        : nano
2026-04-05 16:53:58   seq_len       : 1024
2026-04-05 16:53:58   batch x accum : 2 x 8 = 16
2026-04-05 16:53:58   lr            : 0.0001  warmup=100
2026-04-05 16:53:58   dtype         : torch.bfloat16
2026-04-05 16:53:58   dataset_mix   : stratos
2026-04-05 16:53:58   answer_only_ce: True
2026-04-05 16:53:58   push_to_hub   : True
2026-04-05 16:54:00   vocab: 151,665  pad_token: '<|endoftext|>'
2026-04-05 17:02:21   16710 samples kept, 0 skipped
2026-04-05 17:02:21   train: 15875  val: 835
2026-04-05 17:02:23   92.5M parameters  (preset=nano)
2026-04-05 17:02:23   Schedule  : cosine warmup=100 total=2979  (3 epochs x 993 steps/epoch)
```

**Val CE + generation summary:**
```
step  250  val_ce=5.2199  val_acc=0.2367  mean_uwr=0.156
step  500  val_ce=5.0644  val_acc=0.2514  mean_uwr=0.108
step  750  val_ce=4.9821  val_acc=0.2578  mean_uwr=0.107
step 1000  val_ce=4.9480  val_acc=0.2606  mean_uwr=0.114
```

**Training log:**
```
2026-04-05 17:02:35       1     3.9773      0.2948          -         -   1.9141   2.00e-06    2.440     2442
2026-04-05 17:11:28     100     3.6698      0.3556          -         -   1.5234   1.00e-04    2.440     3021
2026-04-05 17:20:26     200     3.3553      0.3774          -         -   0.9102   9.97e-05    2.440     3026
2026-04-05 17:26:36   [val] step=250  val_ce=5.2199  val_acc=0.2367
2026-04-05 17:49:08     500     3.0003      0.4257          -         -   0.8242   9.58e-05    2.440     2912
2026-04-05 17:50:49   [val] step=500  val_ce=5.0644  val_acc=0.2514
2026-04-05 17:50:59   [ckpt] saved  -> runs/stage2/checkpoint-0000500
2026-04-05 17:51:11   [hub] uploaded  checkpoint-0000500 (commit=1612c426)
2026-04-05 18:37:50    1000     2.9958      0.4151     4.9821    0.2578   0.8047   8.00e-05    2.440     2851
2026-04-05 18:39:31   [val] step=1000  val_ce=4.9480  val_acc=0.2606
2026-04-05 18:39:40   [ckpt] saved  -> runs/stage2/checkpoint-0001000
2026-04-05 18:39:52   [hub] uploaded  checkpoint-0001000 (commit=1149879b)
2026-04-05 18:41:39    1020     2.8206      0.4344     4.9480    0.2606   0.8750   7.92e-05    2.440     2796
```

**Generation sample @ step 1000 (representative — same pattern since step 250):**
```
  Q: What is 15 + 27?
  A: So, let's take a look at the problem. The problem is that the problem is a problem...
     uwr=0.120
  Q: What is the capital of Japan?
  A: The main idea is that the main idea is to find the best possible...
     uwr=0.119
  Mean UWR: 0.114
```
*No `<think>` tags observed. Repetitive but coherent English. Expected at 34% of training.*

**⚠ Deviations from blueprint:**
- `dataset_mix=stratos` only — 16,710 samples vs full recommended mix
- Val CE trajectory (~0.27 nats drop over 1000 steps) requires inflection in epoch 2 to reach < 1.5 by step 2979; monitor step 1500–2000

---

## Stage 1 — Pre-training, Kaggle Dual T4 (Session 6, resumed from checkpoint-0014902)
**Script:** `pretrain.py`
**Date:** 2026-04-05
**Hardware:** Kaggle Dual T4 (2× T4 16 GB, DDP auto-launched, world_size=2)
**Status:** ✅ COMPLETE (graceful timeout exit) — steps 14902→21501, tokens 488M→705M

**⚠ Val CE rose 5.2767 (step 16000) → 5.3241 (step 21500) — +0.047 nats. Rising, not flat.**
**⚠ Spike density tripled in steps 18000–21500 (~15+ events). shuffle_buffer=20000 insufficient.**

**Clean resume confirmed:**
```
  [resume] local   checkpoint-0014902  step=14902  epoch=0  tokens=488,308,736  val_ce=5.289637256266823
```
✅

**Val CE summary:**
```
step 15000  val_ce=5.2792  mean_uwr=0.220
step 16000  val_ce=5.2767  mean_uwr=0.163  ← only improvement this session
step 17000  val_ce=5.2883  mean_uwr=0.272  ← rising begins
step 18000  val_ce=5.3052  mean_uwr=0.161  ← dense spikes start
step 19000  val_ce=5.3105  mean_uwr=0.292
step 20000  val_ce=5.3135  mean_uwr=0.081
step 21000  val_ce=5.3190  mean_uwr=0.099
step 21500  val_ce=5.3241  mean_uwr=0.266
```

**Selected training log with spikes annotated:**
```
2026-04-05 05:55:31   16500     4.1851     5.2767     4.1577   0.4961   5.10e-04    2.035       5778
2026-04-05 06:46:13   17000     4.1732     5.2788     4.1289   0.5000   5.05e-04    2.035       5769
2026-04-05 08:13:01   [spike] step=17843  raw=4.8913  ema=4.1343   ← 3-step cluster
2026-04-05 08:13:06   [spike] step=17844  raw=4.7847  ema=4.1409
2026-04-05 08:13:12   [spike] step=17845  raw=4.6801  ema=4.1462
2026-04-05 08:27:51   18000     4.4503     5.2925     4.1206   0.4492   4.94e-04    2.035       5780
2026-04-05 09:04:04   [spike] step=18344  raw=5.1457  ema=4.1429   ← large
2026-04-05 09:09:50   [spike] step=18405  raw=5.1301  ema=4.1451   ← large
2026-04-05 09:29:06   [spike] step=18572  raw=5.0423  ema=4.1312   ← large
2026-04-05 10:09:29   19000     4.1597     5.3052     4.0977   0.5117   4.82e-04    2.035       5771
2026-04-05 11:45:01   [spike] step=19936  raw=5.2361  ema=4.0927   ← large
2026-04-05 11:51:05   20000     4.0818     5.3129     4.0760   0.4043   4.71e-04    2.035       5773
2026-04-05 13:32:53   21000     4.3295     5.3152     4.0865   0.9219   4.59e-04    2.035       5769
2026-04-05 14:23:52   21500     4.0554     5.3190     4.0796   0.5703   4.52e-04    2.035       5784
```

**Graceful timeout exit:**
```
2026-04-05 14:27:36   [timeout] 11.77h elapsed — 13.7 min remaining (< 15 min buffer).
2026-04-05 14:27:38   [ckpt] saved  -> runs/stage1/checkpoint-0021501
2026-04-05 14:27:38   [ckpt] pruned -> checkpoint-0019000
2026-04-05 14:27:38   Wall time elapsed  : 11.77h / 12.0h
2026-04-05 14:27:38   Steps completed    : 21,501 / 61,036
2026-04-05 14:27:38   Tokens processed   : 704,544,768 / 2,000,000,000
2026-04-05 14:27:38   Last val CE        : 5.324081295402125
```

**Checkpoint status:** Local: checkpoint-0021501 (local only). Hub: checkpoint-0021000 (commit=d70d2c49) ← safe resume point.

---

## Stage 2 SFT — Patch Verification (Code Audit, no run)
**Script:** `train_sft.py`  **Date:** 2026-04-03

**Bug 1 — compute_val_ce live-weight restore:** ✅ FIXED
**Bug 2 — load_latest_checkpoint direct path handling:** ✅ FIXED
**Bug 3 — collate prompt masking:** 🔴 NOT FIXED (must fix before Stage 2 run)
**Issue 3 — Header string:** ✅ FIXED
**Issue 4 — Multi-dataset mixing:** ✅ IMPLEMENTED

---

## Stage 1 — Pre-training, Kaggle Dual T4 (Session 5, resumed from checkpoint-0002000)
**Script:** `pretrain.py`  **Date:** 2026-04-03
**Status:** ✅ COMPLETE — steps 2000→14902

**Clean resume:**
```
  [resume] local   checkpoint-0002000  step=2000  epoch=0  tokens=65,536,000  val_ce=5.564212733394234
  Model parameters : 92,477,440 (92.5 M)
  Val buffer: 2,000,000 tokens from 1,887 docs
  epoch 0  offset=228  skipping=64000 chunks
```

**Log snippet (steps 8500–9000):**
```
2026-04-03 19:35:53   Mean UWR: 0.126  (@ step 8500)
2026-04-03 20:03:47   [spike] step=8797  raw=5.7361  ema=4.2857
2026-04-03 20:08:28   [spike] step=8847  raw=4.8565  ema=4.2874
2026-04-03 20:13:27   [spike] step=8900  raw=4.8290  ema=4.2898
2026-04-03 20:22:50    9000     4.1912     5.2900     4.2486   0.3691   5.73e-04    2.035       5814
```
*(Session ran to step 14902; log captured only through step 9000)*

---

## Stage 1 — Pre-training, Kaggle Dual T4 (Session 4, steps 1–1700)
**Script:** `pretrain.py`
**Status:** ✅ COMPLETE — no checkpoint saved (Bug 5); Session 5 restarted from scratch → checkpoint-0002000

**Smoke test:**
```
[smoke] epoch_offset=7
[smoke] step 20  loss=7.3500
[smoke] val_ce computed: 7.9517
[smoke] checkpoint saved and reloaded cleanly
[smoke] All checks passed - launching main training loop
[ddp] detected 2 CUDA devices; launching single-node DDP with global batch_size=8 (4 per GPU).
```

**Training log (steps 1–1700):**
```
      1    11.9824   1.9766   6.00e-06    2.035       4222
   1000     4.9713   5.8478   0.4590   6.00e-04    2.035       5719
  [hub] upload failed: Client error '401 Unauthorized' ← ⚠ Bug 5
   1700     5.0407   5.6810   0.4668   5.99e-04    2.035       5735
```

---

## Stage 2 SFT — Dry-run (nano, 300 samples, 100 steps)
**Script:** `train_sft_phase2.py` (now: `train_sft.py`)  **Date:** Session 3
**Result:** Pipeline verified. EMA generation degenerate at step 100 (expected at decay=0.999). Corrected to `--ema_decay 0.995`.

```
      1    11.9931   3.7031   4.00e-06    1.576     1662
    100     4.9913    11.9826   1.8594   2.00e-05    1.576     3412
  [val] step=100  val_ce=11.9646
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
