# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**
**Purpose: session outcomes, key metrics, bugs found/fixed, generation samples.**

---

## Session 14 — Stage 1 Resumed, FP16 Confirmed (2026-04-18) 🟡 ACTIVE
**Script:** `jamba_coconut_finetune.py`
**Command (from kaggle-utils.ipynb Cell 5 — identical to Session 13):**
```bash
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit \
  --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 \
  --batch_size 4 --grad_accum 8 \
  --val_batch_size 2 \
  --val_skip_buffer_minutes 60 \
  --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \
  --output_dir runs/stage3_curriculum
```
**Status:** FP16 patch CONFIRMED working. Resumed from `checkpoint-0001338` (step_in_epoch=183). Stage 1 in progress.

**FP16 confirmation (verbatim):**
```
  [GPU] Tesla T4  cc=sm75  VRAM=16GB  amp_dtype=float16
  device=cuda:0 amp_dtype=float16
```

**Resume confirmation (verbatim):**
```
  [resume] discovered latest checkpoint: runs/stage3_curriculum/stage_1/checkpoint-0001338
  [resume] step=1338 epoch=0 stage_k=1 val_acc=None
  Resuming stage from epoch=0 step_in_epoch=183 global_step=1338
```

**Step time post-FP16 (verbatim tqdm snapshot):**
```
S1E0:   4%|▌            | 39/970 [26:51<10:41:10, 41.32s/it, ce=0.389, gn=0.200]
```
**Confirmed step time: ~41s/step** (down from ~162s/step pre-patch). **~4× end-to-end speedup** (beats conservative 1.5–2.5× prediction).

**Step logs (verbatim):**
```
  step=  1340 s=1 ep=0 ce=0.4357 gn=0.1727
  step=  1360 s=1 ep=0 ce=0.3632 gn=0.1486
```

**Estimated Stage 1 completion:** ~970 remaining steps × 41s ≈ 11.1h → 1–2 sessions.

**Issues identified this session:**
- `--push_to_hub` never passed in command → hub upload never fired (Stage 0 best never pushed)
- `prune_epoch_checkpoints` only called after successful val; timeout sessions skip it entirely; also scoped per-stage only → disk will fill up
- Fix tracked in `AGENT_PROMPT_hub_prune_fix.md`

---

## Session 13 — Stage 0 Val Complete, Stage 1 In Progress (2026-04-17 → 2026-04-18) ✅ COMPLETE
**Script:** `jamba_coconut_finetune.py`
**Command (verbatim from kaggle-utils.ipynb Cell 5 — the actually executed command):**
```bash
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit \
  --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 \
  --batch_size 4 --grad_accum 8 \
  --val_batch_size 2 \
  --val_skip_buffer_minutes 60 \
  --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \
  --output_dir runs/stage3_curriculum
```
> ⚠️ Note: `--val_skip_buffer_minutes 60` (not 120 as previously noted in Blueprint). `--no-gen_every_stage` was NOT passed; generation ran at Stage 0 end (confirmed in log). Blueprint corrected accordingly.

**Status:** NCCL fix confirmed working. Stage 0 val succeeded (DDP, all ranks). Stage 1 in progress (pre-FP16 patch, ~162s/step). Session ended at step 1338, timeout triggered.

**Model config confirmed (verbatim):**
```
  <|lat|> token id: 65536  vocab: 65537
  trainable params: 26,851,328 || all params: 3,056,191,360 || trainable%: 0.8786
  d_model=2560  layers=28
  amp_dtype=bfloat16    ← ⚠️  BF16 on sm75 (T4) — uses FP32 compute paths (pre-patch)
```

**Stage 0 val result (verbatim):**
```
  [val] s=0 ep=0 val_ce=0.4041 val_acc=0.0222
  [ckpt] saved -> runs/stage3_curriculum/stage_0/checkpoint-0001154  acc=0.022222222222222223  ce=0.4040850892827772
  [ckpt] best -> runs/stage3_curriculum/stage_0/best  acc=0.022222222222222223  ce=0.4040850892827772
  [best] stage=0 new best acc=0.0222
```

**Stage 0 generation @ step 1154 (selected sample):**
```
  Q: Solve for x: 3x + 6 = 21.
  A: To solve for x, we need to isolate it on one side of the equation. First, subtract 6 from both sides ...
     3x = 15  Next, divide both sides ...  [k_actual=0 uwr=0.358]
  Mean UWR: 0.433
```

**Stage 1 step times, pre-patch (verbatim — 8 consecutive 20-step intervals):**
```
  step=  1160 s=1 ep=0 ce=0.4871 gn=0.2722   ← 21:51:55
  step=  1180 s=1 ep=0 ce=0.3932 gn=0.1418   ← 22:44:50
  step=  1200 s=1 ep=0 ce=0.4809 gn=0.1980   ← 23:40:16
  step=  1220 s=1 ep=0 ce=0.4597 gn=0.1449   ← 00:34:26
  step=  1240 s=1 ep=0 ce=0.4930 gn=0.1338   ← 01:27:51
  step=  1260 s=1 ep=0 ce=0.5882 gn=0.1786   ← 02:20:48
  step=  1280 s=1 ep=0 ce=0.4068 gn=0.1259   ← 03:13:51
  step=  1300 s=1 ep=0 ce=0.4123 gn=0.1405   ← 04:08:34
```
**Mean pre-patch Stage 1 step time: 162s ± 3s**

**Session timeout (verbatim):**
```
S1E0:  16%|█▎      | 184/1154 [8:14:47<47:35:22, 176.62s/it, ce=0.425, gn=0.135]
  [timeout] 10.69h elapsed - 18.4 min remaining (< 20 min buffer).
  [timeout] saving emergency checkpoint at step 1338 ...
  [ckpt] saved -> runs/stage3_curriculum/stage_1/checkpoint-0001338  acc=None  ce=None
```

---

## Session 12 — Stage 0 Training Complete, Val NCCL Crash (2026-04-17) ✅ RESOLVED
**Status:** Stage 0 training COMPLETE at step 1154/1154. Checkpoint-0001154 saved. Val killed by NCCL watchdog. Fixed in Session 13.

**Training metrics (final 260 steps):**
```
  step=  1000 s=0 ep=0 ce=0.4889 gn=0.1063
  step=  1060 s=0 ep=0 ce=0.3693 gn=0.1310
  step=  1120 s=0 ep=0 ce=0.5040 gn=0.1780
  step=  1140 s=0 ep=0 ce=0.3823 gn=0.1360
S0E0: 100%|████████████| 260/260 [9:54:14<00:00, 137.13s/it, ce=0.385, gn=0.111]
  [ckpt] saved -> runs/stage3_curriculum/stage_0/checkpoint-0001154  acc=None  ce=None
```

**NCCL crash root cause (verbatim):**
```
[rank1]: Watchdog caught collective operation timeout: WorkNCCL(...Timeout(ms)=3600000) ran for
3600001 milliseconds before timing out.
```
Fix: `timedelta(hours=4)` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=14400`.

---

## Session 11 — Stage 0 Partial Run (2026-04-15) 🟡 SUPERSEDED
**Status:** Epoch 0 at step 629/2307. Superseded by Session 12.

```
step=   620 s=0 ep=0 ce=0.3405 gn=0.1657
S0E0:  27%|██▏| 629/2307 [10:40:33<27:54:04, 59.86s/it, ce=0.361, gn=0.166]
  [timeout] 10.68h elapsed - 19.4 min remaining (< 20 min buffer).
  [ckpt] saved -> runs/stage3_curriculum/stage_0/checkpoint-0000629  acc=None  ce=None
```

---

## Session 10 — Dual T4 DDP Profile Run (2026-04-15) ✅ COMPLETE
**Status:** Stage 0/0 COMPLETE — 12 steps, val_acc=0.4000

```
[bootstrap] Mamba fast path: ACTIVE ✓
[val] s=0 ep=0 val_ce=0.4253 val_acc=0.4000
[ckpt] best -> runs/profile_dual_t4/stage_0/best  acc=0.4  ce=0.4252639559711584
```

---

## Sessions 4–9 — Kernel / Bootstrap Debugging (2026-04-14) ✅
- Session 9: First successful Stage 0 smoke test. ~113s/step single T4 (pre-batching fix).
- Session 8: `causal_conv1d_fn` weight shape fix ✅
- Session 7: `selective_state_update` import path fix ✅
- Session 6: 10-alias generation shim ✅
- Session 5: `GreedySearchDecoderOnlyOutput` missing
- Session 4: mamba_ssm PyPI sdist is 35kB stub → `git+https://` fix

---

## Stage 3 — Early Smoke Test (2026-04-11)
```
trainable params: 26,851,328 || all params: 3,056,191,360 || trainable%: 0.8786
Stage 2/2  S2E0: ce=1.464  gn=36.926
[val] s=2 ep=0 val_ce=0.0000 val_acc=0.0000
```
**Fixes codified:** `--max_seq_len 1024`, `--max_grad_norm 0.3`
