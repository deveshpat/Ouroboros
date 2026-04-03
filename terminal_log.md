# terminal_log.md — Project Ouroboros
**All verified terminal outputs, newest first.**

---

> **Naming note:** Scripts were renamed for consistency.
> `phase1_viability_gate.py` → `viability_gate.py`
> `train_sft_phase2.py` → `train_sft.py`
> `Ouroboros_Blueprint_v3.md` → `BLUEPRINT.md`
> Logs below use original script names as they appeared at run time.

---

## Stage 1 — Pre-training, Kaggle Dual T4 (Session 5, resumed from checkpoint-0002000)
**Script:** `pretrain.py`
**Date:** Session 5 (2026-04-03)
**Hardware:** Kaggle Dual T4 (2× T4 16 GB, DDP auto-launched, world_size=2)
**Status:** 🟡 IN PROGRESS — log captured at step 3500 / 61,036

**Bug 5 fix confirmed:**
```
  [resume] local   checkpoint-0002000  step=2000  epoch=0  tokens=65,536,000  val_ce=5.5642
```
Clean local resume from disk — no Hub needed, no data loss from session interrupt. ✅

**Training log (steps 2000–3500):**
```
   step   train_ce     val_ce       smth    gnorm         lr     VRAM      tok/s
--------------------------------------------------------------------------------
  epoch 0  offset=228  skipping=64000 chunks

   2050     4.5941          -     4.8104   0.7031   5.99e-04    2.035       3169
   2100     4.5080          -     4.7605   0.3906   5.99e-04    2.035       5809
   2150     4.7421          -     4.7493   0.4121   5.99e-04    2.035       5808
   2200     4.7011          -     4.7553   0.4277   5.99e-04    2.035       5802
   2250     4.6945          -     4.7333   0.4043   5.98e-04    2.035       5806
  [spike] step=2299  raw=5.3932  ema=4.7166
   2300     5.0453          -     4.7199   0.9805   5.98e-04    2.035       5815
   2350     4.5325          -     4.7081   0.4570   5.98e-04    2.035       5810
   2400     4.4693          -     4.6923   0.8672   5.98e-04    2.035       5810
   2450     4.6070          -     4.6939   0.5508   5.98e-04    2.035       5808
  [spike] step=2479  raw=5.2191  ema=4.6873
   2500     4.5709          -     4.6795   0.4023   5.98e-04    2.035       5808
  [val] step=2500  val_ce=5.4796
```

**Generation @ step 2500 (~82M tokens seen):**
```
  P: The capital of France is
  C:  the only one that is the most important of the world. The world is the world's most important...
     uwr=0.202
  P: In mathematics, a prime number is
  C:  a simple one. The first thing that is important is that the basic principle of the word is the value...
     uwr=0.193
  P: def factorial(n):  """Return n!."""  if n
  C:  . 1000000000000000000000000000000000000000000000000000000000000000000000...
     uwr=1.000   (digit loop, expected)
  P: Neural networks learn by
  C:  using the same technology as the internet. The internet is a very common type of internet connection...
     uwr=0.222
  P: The French Revolution began in
  C:  the 19th century, when the French Revolution was the first to be a part of the Soviet Union...
     uwr=0.260
  Mean UWR: 0.375
```

```
   2550     4.8075     5.4796     4.6841   0.6641   5.98e-04    2.035       3181
   2600     4.5635     5.4796     4.6661   0.3965   5.98e-04    2.035       5802
   2650     4.5249     5.4796     4.6582   0.4473   5.98e-04    2.035       5806
   2700     4.4584     5.4796     4.6462   0.5391   5.98e-04    2.035       5810
   2750     4.7246     5.4796     4.6456   0.3809   5.98e-04    2.035       5815
  [spike] step=2772  raw=5.1745  ema=4.6504
   2800     4.5656     5.4796     4.6321   0.3770   5.98e-04    2.035       5810
   2850     4.5634     5.4796     4.6257   0.3691   5.97e-04    2.035       5810
   2900     4.5406     5.4796     4.6220   0.3730   5.97e-04    2.035       5810
  [spike] step=2923  raw=5.1878  ema=4.6173
   2950     4.4513     5.4796     4.6141   0.5586   5.97e-04    2.035       5805
   3000     4.4762     5.4796     4.6025   0.4414   5.97e-04    2.035       5823
  [val] step=3000  val_ce=5.4154
```

**Generation @ step 3000 (~98M tokens seen):**
```
  P: The capital of France is
  C:  the most important part of the economy. The main purpose of the economy is to make the economy...
     uwr=0.255
  P: In mathematics, a prime number is
  C:  the number of times the number of times the number of people in a given group is 1...
     uwr=0.152
  P: def factorial(n):  """Return n!."""  if n
  C:  . 1000000000000000000000000000000000000000000000000000000000000000000000...
     uwr=1.000   (digit loop, expected)
  P: Neural networks learn by
  C:  the time they are given. The data is then collected from the data and then sent to the data...
     uwr=0.160
  P: The French Revolution began in
  C:  1822, and the French Revolution was the first to be replaced by the French Revolution...
     uwr=0.150
  Mean UWR: 0.343
```

**Checkpoint-0003000 saved and Hub-synced:**
```
  [ckpt] saved  -> runs/stage1/checkpoint-0003000
  [hub] uploading checkpoint-0003000 -> WeirdRunner/Ouroboros ...
  [hub] uploaded  checkpoint-0003000 (commit=5e2ba2b8)
```
Local-first + Hub fire-and-forget confirmed working. ✅

```
   3050     4.5598     5.4154     4.5969   0.4785   5.97e-04    2.035       3070
   3100     4.7398     5.4154     4.5867   0.4180   5.97e-04    2.035       5804
  [spike] step=3112  raw=5.4127  ema=4.5942
  [spike] step=3146  raw=5.2242  ema=4.6168
   3150     4.6520     5.4154     4.6142   0.4648   5.97e-04    2.035       5801
  [spike] step=3160  raw=5.2053  ema=4.6183
   3200     4.4035     5.4154     4.6143   0.3926   5.97e-04    2.035       5801
   3250     4.7021     5.4154     4.6059   0.6797   5.97e-04    2.035       5807
   3300     4.7683     5.4154     4.5951   0.3867   5.97e-04    2.035       5806
   3350     4.4831     5.4154     4.5798   0.4121   5.97e-04    2.035       5803
   3400     4.1759     5.4154     4.5612   0.4941   5.96e-04    2.035       5802
   3450     4.5199     5.4154     4.5482   0.4004   5.96e-04    2.035       5800
  [spike] step=3475  raw=5.8600  ema=4.5615   ← ⚠ cluster start
  [spike] step=3476  raw=5.8159  ema=4.5740   ← ⚠ consecutive
  [spike] step=3477  raw=5.6536  ema=4.5848   ← ⚠ consecutive
   3500     4.4163     5.4154     4.5772   0.5078   5.96e-04    2.035       5799
  [val] step=3500  val_ce=5.3622
```

**Generation @ step 3500 (~115M tokens seen):**
```
  P: The capital of France is
  C:  a major factor in the development of the country. The government of the country is a major factor...
     uwr=0.217
  P: In mathematics, a prime number is
  C:  a number of numbers. The number of numbers is the number of numbers in a number...
     uwr=0.088   ← repetitive, spike cluster effect
  P: def factorial(n):  """Return n!."""  if n
  C: is ll l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l l
     uwr=0.025   ← degenerate (letter loop), spike cluster effect; expected to recover
  P: Neural networks learn by
  C:  the way, and the network is a network of networks that can be used to communicate with the network...
     uwr=0.178
  P: The French Revolution began in
  C:  the 19th century, and the French Revolution was the result of the revolution of the 19th century...
     uwr=0.267
  Mean UWR: 0.155   ← ⚠ below previous values; correlated with spike cluster 3475–3477
```

*(Log ends here — run still in progress)*

---

**Loss curve summary (full run to step 3500):**

| Step | Train CE | Smoothed | Val CE | Tokens Seen | Notes |
|---|---|---|---|---|---|
| 1 | 11.98 | 11.98 | — | 32k | Random init |
| 500 | 5.46 | 5.78 | 6.38 | 16.4M | Phrases forming |
| 1000 | 4.97 | 5.14 | 5.85 | 32.8M | Real sentences |
| 1500 | 4.89 | 4.91 | 5.68 | 49.2M | Coherent prose |
| 2000 | — | — | 5.56 | 65.5M | Resumed (ckpt-2000) |
| 2500 | 4.57 | 4.68 | 5.48 | 82.0M | Consistent drop |
| 3000 | 4.48 | 4.60 | 5.42 | 98.3M | Hub sync working |
| 3500 | 4.42 | 4.58 | 5.36 | 114.7M | Spike cluster ⚠ |

**Key observations:**
- Val CE declining every checkpoint: 6.38 → 5.85 → 5.68 → 5.56 → 5.48 → 5.42 → 5.36 ✅
- VRAM flat at 2.035 GB throughout — zero graph retention ✅
- Spike rate: 12 spikes / 3500 steps = 0.34% — well within 10% threshold ✅
- Spike cluster at 3475–3477 (3 consecutive) is the first cluster seen; isolated spikes
  before this were all single-step. Watch for recurrence. If another cluster appears,
  consider increasing `--shuffle_buffer` to 20000.
- UWR dip at step 3500 is a lagging indicator from the spike cluster and expected to
  recover at step 4000. Val CE (5.36) did not spike — primary signal is healthy.
- Code prompt loops (digit/letter) are expected — FineWeb-Edu has negligible code content.

---

## Stage 1 — Pre-training, Kaggle Dual T4 (Session 4, steps 1–1700)
**Script:** `pretrain.py`
**Date:** Session 4
**Hardware:** Kaggle Dual T4 (2× T4 16 GB, DDP auto-launched, world_size=2)
**Run type:** Full 2B-token run — `token_budget=2_000_000_000`, `total_steps=61,036`
**Status:** 🟡 Session ended at step 1700; resumed in Session 5

**⚠ No checkpoint saved (Bug 5):** Hub 401 at step 1000 caused `save_checkpoint` to
abandon `.tmp` without renaming to final. Bug 5 was patched before Session 5.
Session 5 started from scratch (step=0) since no local checkpoint existed from Session 4.

---

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

**Startup + val buffer:**
```
========================================================================
  Stage 1 Pre-training - Project Ouroboros
========================================================================
  dataset          : HuggingFaceFW/fineweb-edu / sample-10BT
  tokenizer        : Qwen/Qwen2.5-0.5B  vocab=151,665
  preset           : nano
  model            : d_model=512  groups=1  heads=8/4
  chunk_size       : 1024
  batch x accum    : 8 global x 4
  world_size       : 2  (DDP auto-enabled)
  per_gpu_batch    : 4
  tokens / step    : 32,768
  token_budget     : 2,000,000,000
  total_steps      : 61,036
  dtype            : torch.bfloat16
  device           : cuda:0
  output_dir       : runs/stage1
  push_to_hub      : True
========================================================================

Model parameters : 92,477,440 (92.5 M)

  Building val buffer (2,000,000 tokens) ...
  Val buffer: 2,000,000 tokens from 1,887 docs
  [resume] No checkpoint found - starting from scratch.
  epoch 0  offset=228  skipping=0 chunks
```

**Training log (step / train_ce / val_ce / smth / gnorm / lr / VRAM / tok/s):**
```
   step   train_ce     val_ce       smth    gnorm         lr     VRAM      tok/s
--------------------------------------------------------------------------------
      1    11.9824          -    11.9824   1.9766   6.00e-06    2.035       4222
     50     9.2922          -    11.4559   2.2656   1.53e-04    2.035       6040
    100     7.0954          -     9.9666   0.6797   3.03e-04    2.035       5859
    150     7.0693          -     8.7018   1.0938   4.53e-04    2.035       5812
    200     6.2197          -     7.7876   0.7617   6.00e-04    2.035       5794
    250     6.1138          -     7.1607   0.6172   6.00e-04    2.035       5793
    300     6.1652          -     6.7050   0.7773   6.00e-04    2.035       5794
    350     5.7886          -     6.3685   1.0625   6.00e-04    2.035       5796
    400     5.6378          -     6.1231   0.6211   6.00e-04    2.035       5783
    450     5.6214          -     5.9260   0.7383   6.00e-04    2.035       5807
    500     5.4631          -     5.7808   0.5664   6.00e-04    2.035       5805
  [val] step=500  val_ce=6.3811
```

**Generation @ step 500 (~16.4M tokens seen):**
```
  -- Generation @ step 500 (live weights) --
  P: The capital of France is
  C:  the first to be the first of the 1980s. The first was the first of the 1980s...
     uwr=0.292
  P: In mathematics, a prime number is
  C:  a major factor. The first is a great deal of time, but it is a great deal of time...
     uwr=0.231
  P: def factorial(n):  """Return n!."""  if n
  C:  = 10000000000000000000000000000000000000000000000000000000000000000000000...
     uwr=1.000   (code prompt: digit loop, expected — FineWeb-Edu has no code)
  P: Neural networks learn by
  C:  the other. The first step is to make a new approach to the other...
     uwr=0.333
  P: The French Revolution began in
  C:  1980. The first time was the first to be the first to be the first...
     uwr=0.070
  Mean UWR: 0.385
```

```
    550     5.5842     6.3811     5.6558   0.5352   6.00e-04    2.035       3515
    600     5.4063     6.3811     5.5648   0.5078   6.00e-04    2.035       5860
    650     5.1124     6.3811     5.4716   0.5898   6.00e-04    2.035       5807
    700     5.2329     6.3811     5.4048   0.6016   6.00e-04    2.035       5817
    750     5.1832     6.3811     5.3621   0.4316   6.00e-04    2.035       5813
    800     5.2054     6.3811     5.3224   0.5195   6.00e-04    2.035       5766
    850     5.0932     6.3811     5.2711   0.5664   6.00e-04    2.035       5731
    900     4.9945     6.3811     5.2250   0.4375   6.00e-04    2.035       5737
    950     5.0298     6.3811     5.1857   0.4277   6.00e-04    2.035       5724
   1000     4.9713     6.3811     5.1369   0.4590   6.00e-04    2.035       5719
  [val] step=1000  val_ce=5.8478
```

**Generation @ step 1000 (~32.8M tokens seen):**
```
  -- Generation @ step 1000 (live weights) --
  P: The capital of France is
  C:  a major part of the country. It is a national and political state...
     uwr=0.159
  P: In mathematics, a prime number is
  C:  a standard of the same language, and the number of students in the 1990s...
     uwr=0.365
  P: def factorial(n):  """Return n!."""  if n
  C:  = 10000000000000000000000000000000000000000000000000000000000000000000000...
     uwr=1.000   (still looping)
  P: Neural networks learn by
  C:  the human brain. The brain is a complex, complex, and complex system...
     uwr=0.161
  P: The French Revolution began in
  C:  1917, when the 1910s were the first to be used in the 1990s...
     uwr=0.420
  Mean UWR: 0.421
```

**⚠ Hub 401 at step 1000 — checkpoint NOT saved to disk:**
```
  [hub] upload failed for checkpoint-0001000: Client error '401 Unauthorized'
  Invalid username or password.
  [warn] step 1000: Hub sync failed; checkpoint not finalized.
```

```
   1050     5.0967     5.8478     5.1125   0.5820   6.00e-04    2.035       3471
   1100     5.1161     5.8478     5.0923   0.4473   6.00e-04    2.035       5779
  [spike] step=1148  raw=5.6166  ema=5.0871     (isolated, normal)
   1150     5.1219     5.8478     5.0904   0.6680   6.00e-04    2.035       5725
   1200     4.9278     5.8478     5.0599   0.4395   6.00e-04    2.035       5727
   1250     4.9201     5.8478     5.0361   0.4473   6.00e-04    2.035       5719
   1300     4.8458     5.8478     5.0072   0.4414   6.00e-04    2.035       5720
   1350     5.0672     5.8478     4.9811   0.4980   6.00e-04    2.035       5733
   1400     4.6678     5.8478     4.9578   0.4395   5.99e-04    2.035       5727
   1450     4.8361     5.8478     4.9292   0.4531   5.99e-04    2.035       5706
   1500     4.8855     5.8478     4.9102   0.4805   5.99e-04    2.035       5702
  [val] step=1500  val_ce=5.6810
```

**Generation @ step 1500 (~49.2M tokens seen):**
```
  -- Generation @ step 1500 (live weights) --
  P: The capital of France is
  C:  a great example of a new country. The city is a city of the city...
     uwr=0.139
  P: In mathematics, a prime number is
  C:  a 30% chance of a 200-year-old, 20-year-old girl, who is a member of the U.S....
     uwr=0.280
  P: def factorial(n):  """Return n!."""  if n
  C: .d.100000000000000000000000000000000000000000000000000000000000000000000000...
     uwr=1.000   (still looping)
  P: Neural networks learn by
  C:  using a new device. The device is also used to provide a variety of information...
     uwr=0.344
  P: The French Revolution began in
  C:  1960, and the United States was a major source of the war...
     uwr=0.229
  Mean UWR: 0.398
```

```
   1550     4.8263     5.6810     4.8900   0.4395   5.99e-04    2.035       3490
   1600     4.8256     5.6810     4.8595   0.5430   5.99e-04    2.035       5746
  [spike] step=1611  raw=5.5458  ema=4.8666     (isolated, normal)
   1650     4.8607     5.6810     4.8580   0.4863   5.99e-04    2.035       5754
   1700     5.0407     5.6810     4.8453   0.4668   5.99e-04    2.035       5735
```
*(Log ends here — run still in progress at step 1700 / 61,036)*

---

**Loss curve summary (steps 1–1700):**

| Step | Train CE | Smoothed | Val CE | Tokens Seen | Notes |
|---|---|---|---|---|---|
| 1 | 11.98 | 11.98 | — | 32k | Random init |
| 500 | 5.46 | 5.78 | 6.38 | 16.4M | Coherent phrases |
| 1000 | 4.97 | 5.14 | 5.85 | 32.8M | Real sentences |
| 1500 | 4.89 | 4.91 | 5.68 | 49.2M | Coherent prose |
| 1700 | 5.04 | 4.85 | 5.68 | 55.7M | In progress |

**Overall assessment:**
- Loss trajectory is healthy. At 55.7M of 2,000M tokens (~2.8%) the model has already dropped CE by 7.1 nats.
- Mean UWR stable 0.38–0.42; well above the 0.05 Stage 1 success threshold.
- VRAM perfectly flat at 2.035 GB — no graph retention.
- Throughput ~5800 tok/s on Dual T4 bfloat16. Full run ETA: ~9.4h from launch.
- Code prompt digit loop is expected — FineWeb-Edu has negligible code content.
- Two isolated spikes; spike rate 0.12%, within acceptable threshold.
- **CRITICAL: no checkpoint saved yet.** Fix Bug 5 immediately or the session may be lost.

---

## Stage 2 SFT — Dry-run (nano, 300 samples, 100 steps)
**Script:** `train_sft_phase2.py` (now: `train_sft.py`)
**Date:** Session 3
**Result:** Pipeline verified. EMA generation degenerate at step 100 (expected at decay=0.999). Corrected to `--ema_decay 0.995`.

**⚠ Bugs identified post-run:**
1. `compute_val_ce` — does not restore live weights after EMA eval (Bug 1, HIGH)
2. `load_latest_checkpoint` — does not handle direct checkpoint paths (Bug 2, MEDIUM)
3. `collate` — applies loss to all tokens including user prompt (Bug 3, HIGH)

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
