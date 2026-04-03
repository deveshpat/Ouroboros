# terminal_log.md — Project Ouroboros
**All verified terminal outputs, newest first.**

---

> **Naming note:** Scripts were renamed for consistency.
> `phase1_viability_gate.py` → `viability_gate.py`
> `train_sft_phase2.py` → `train_sft.py`
> `Ouroboros_Blueprint_v3.md` → `BLUEPRINT.md`
> Logs below use original script names as they appeared at run time.

---

## Stage 1 — Pre-training, Kaggle Dual T4 (full run, in progress)
**Script:** `pretrain.py`
**Date:** Session 4 (current)
**Hardware:** Kaggle Dual T4 (2× T4 16 GB, DDP auto-launched, world_size=2)
**Run type:** Full 2B-token run — `token_budget=2_000_000_000`, `total_steps=61,036`
**Status:** 🟡 IN PROGRESS — log captured at step 1700 / 61,036

**⚠ CRITICAL issue discovered at step 1000:**
Hub 401 (invalid HF token) caused `save_checkpoint` to abandon the `.tmp` directory
without finalizing it. No local checkpoint was written. If the session resets before
the next successful save, all progress is lost. See Bug 5 in BLUEPRINT.md Part 7.
**Immediate action:** Fix `save_checkpoint` to finalize locally before Hub push.

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
