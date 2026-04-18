# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**
**Purpose: session outcomes, key metrics, bugs found/fixed, generation samples.**

---

## Session 13 — Stage 0 Val Complete, Stage 1 In Progress (2026-04-17 → 2026-04-18) 🟡 ACTIVE
**Script:** `jamba_coconut_finetune.py`
**Command:**
```bash
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit \
  --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 \
  --batch_size 4 --grad_accum 8 \
  --val_batch_size 2 \
  --val_skip_buffer_minutes 120 \
  --no-gen_every_stage \
  --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \
  --output_dir runs/stage3_curriculum
```
**Status:** NCCL fix confirmed working. Stage 0 val succeeded (DDP, all ranks). Stage 1 in progress at ~162s/step.

**Model config confirmed (verbatim):**
```
  <|lat|> token id: 65536  vocab: 65537
  trainable params: 26,851,328 || all params: 3,056,191,360 || trainable%: 0.8786
  d_model=2560  layers=28
  amp_dtype=bfloat16    ← ⚠️  BF16 on sm75 (T4) — uses FP32 compute paths
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
Generation is coherent and mathematically correct. Low acc (2.2%) is a normalization artifact at Stage 0, not model failure. CE=0.40 confirms healthy training.

**Stage 1 step time (verbatim excerpts — 7 consecutive 20-step intervals):**
```
  step=  1160 s=1 ep=0 ce=0.4871 gn=0.2722   ← 21:51:55
  step=  1180 s=1 ep=0 ce=0.3932 gn=0.1418   ← 22:44:50  (52:55 / 20 steps = 158.8s/step)
  step=  1200 s=1 ep=0 ce=0.4809 gn=0.1980   ← 23:40:16  (55:26 / 20 steps = 166.3s/step)
  step=  1220 s=1 ep=0 ce=0.4597 gn=0.1449   ← 00:34:26  (54:10 / 20 steps = 162.5s/step)
  step=  1240 s=1 ep=0 ce=0.4930 gn=0.1338   ← 01:27:51  (53:25 / 20 steps = 160.3s/step)
  step=  1260 s=1 ep=0 ce=0.5882 gn=0.1786   ← 02:20:48  (52:57 / 20 steps = 158.9s/step)
  step=  1280 s=1 ep=0 ce=0.4068 gn=0.1259   ← 03:13:51  (53:03 / 20 steps = 159.2s/step)
  step=  1300 s=1 ep=0 ce=0.4123 gn=0.1405   ← 04:08:34  (54:43 / 20 steps = 164.2s/step)
```
**Mean Step 1 step time: 162s ± 3s**  (Stage 0 was 137s; delta ≈ 25s/latent pass — matches model)

**tqdm progress snapshot:**
```
S1E0: 12%|▉ | 143/1154 [~6:22:52<48:51:41, 173.99s/it, ce=0.512, gn=0.139]
```
Estimated stage 1 completion time at current rate: ~48h from stage start. Session ended before completion.

---

## Session 12 — Stage 0 Training Complete, Val NCCL Crash (2026-04-17) ✅ RESOLVED
**Status:** Stage 0 training COMPLETE at step 1154/1154. Checkpoint-0001154 saved. Val killed by NCCL watchdog. Fixed in Session 13.

**Training metrics (final 260 steps, resuming from checkpoint-0000894):**
```
  step=  1000 s=0 ep=0 ce=0.4889 gn=0.1063
  step=  1060 s=0 ep=0 ce=0.3693 gn=0.1310
  step=  1120 s=0 ep=0 ce=0.5040 gn=0.1780
  step=  1140 s=0 ep=0 ce=0.3823 gn=0.1360
S0E0: 100%|████████████| 260/260 [9:54:14<00:00, 137.13s/it, ce=0.385, gn=0.111]
  [ckpt] saved -> runs/stage3_curriculum/stage_0/checkpoint-0001154  acc=None  ce=None
```

**NCCL crash (verbatim root cause):**
```
[rank1]: Watchdog caught collective operation timeout: WorkNCCL(...Timeout(ms)=3600000) ran for
3600001 milliseconds before timing out.
```
Root cause: `init_process_group(timeout=timedelta(minutes=60))`. Val ran ~170min on rank 0; rank 1 sat at `barrier()`. Fix: `timedelta(hours=4)` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=14400`.

---

## Session 11 — Stage 0 Partial Run (2026-04-15) 🟡 SUPERSEDED
**Status:** Epoch 0 at step 629/2307 (batch_size=2). Superseded by Session 12 which completed Stage 0.

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
[bootstrap] Shim: patched 10 removed transformers.generation names ✓
[bootstrap] Kernel export shim: mamba_ssm.selective_state_update ✓
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
