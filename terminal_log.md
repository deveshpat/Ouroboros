# terminal_log.md — Project Ouroboros
**All verified terminal outputs, newest first.**

---

## Stage 2 SFT — Session 8 (DDP, OOM at step 250)
**Script:** `train_sft.py` (DDP v2)  **Date:** 2026-04-08  **Hardware:** Kaggle Dual T4  
**Status:** 🔴 OOM at first val step — `train_sft_fixes_prompt.md` pending

**What worked:** DDP training ran cleanly. Checkpoint resume detected fingerprint mismatch (full→cached), correctly loaded weights from Hub ckpt-0003500, reset optimizer, started from step 0. Loss descended 3.68→2.57 over 250 steps.

**Training snippet (S8):**
```
      1     3.6784      0.3993          -         -   1.2656   1.20e-05    1.527     4599
     20     3.3819      0.4156          -         -   1.2969   1.26e-04    4.131     4577
    100     3.1499      0.4089          -         -   1.4531   3.00e-04    4.345     4362
    200     2.6602      0.4589          -         -   0.9648   2.99e-04    4.345     4236
    240     2.6970      0.4450          -         -   1.0547   2.99e-04    1.546     4165
```

**Crash (verbatim):**
```
[rank1]: File "train_sft.py", line 1164, in compute_val_metrics_distributed
[rank1]:   shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size).float()
[rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 9.25 GiB.
         GPU 1: 14.56 GiB total, 433.81 MiB free, 13.77 GiB allocated by PyTorch
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 9.25 GiB.
         GPU 0: 14.56 GiB total, 3.63 GiB free, 10.62 GiB allocated by PyTorch
```

**Root cause:** `val_batch_size=16`, `max_seq_len=2048`, `vocab=151,680` → logit tensor `16 × 2047 × 151680 × 4 bytes = 9.25 GiB`. Fix: `val_batch_size=2` default + `torch.cuda.empty_cache()` before val.

**Resume fingerprint log (correct behaviour):**
```
[resume] runs/stage2/checkpoint-0003500  loaded model weights (step=3500), 
         but data stream changed — resetting step/optimizer/scheduler for new data.
[resume] saved_data={'dataset_mix': 'full', ...}
[resume] new_data  ={'dataset_mix': 'cached', ...}
[resume] optimizer/scheduler reset; training starts fresh from step 0.
```

Hub checkpoint saved: `checkpoint-0000250` (partial, from graceful exit at OOM).

---

## Stage 2 SFT — Sessions 5–7 (DDP, NCCL watchdog kills)
**Status:** 🔴 All killed by NCCL watchdog — root cause: combined val+gen time exceeded 600s

| Session | Crash point | Last val_ce | Root cause |
|---|---|---|---|
| S5 | step ~3700, SIGKILL (exit 137) | 5.62 | val(557s)+gen(90s) > 600s NCCL timeout |
| S6 | step 3750, SIGABRT | 5.63 | Same |
| S7 | step 3522, SIGABRT | ~3.24 | Rewrite not applied before run |

**Fix applied:** `stage2_rewrite_prompt.md` → val capped to 500 samples (batch=16, ~10s), `gen_every=500`, NCCL timeout raised to 1800s.

---

## Stage 2 SFT — Sessions 1–4

| Session | Outcome |
|---|---|
| S1 | max_seq_len=1024, val_ce=4.92 plateau. Hub: ckpt-0002979. |
| S2 | Bugs 6–10, val_ce=5.71. All local Stage 2 ckpts deleted by prune bug. |
| S3–S4 | Dry-runs only. Patches verified. |

---

## Stage 1 Pre-training — Session 6 (Kaggle Dual T4)
**Status:** ✅ COMPLETE (graceful timeout)
```
Tokens processed: 704,544,768   Last val CE: 5.324
Hub: checkpoint-0021000 (commit=d70d2c49)  ← SFT starting point
```

---

## Stage 0 — Viability Gate  ✅ ALL GATES PASSED
```
G1 CE < 3.5        final CE = 2.0034   PASS ✓
G2 UWR > 0.1       mean UWR = 0.573    PASS ✓
G3 gnorm < 10.0    max = 4.0312        PASS ✓
G4 VRAM Δ < 1.0GB  Δ = 0.000 GB       PASS ✓
Total time: 3.4 min   Peak VRAM: 2.07 GB
```

## Stage 0 — Baseline Smoke Test  ✅ PASSED
```
parameters: 92,477,440 (92.5M)   initial loss: 11.9904   All checks passed.
```
