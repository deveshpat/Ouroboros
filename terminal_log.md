# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**
**Purpose: session outcomes, key metrics, bugs found/fixed, generation samples.**

---

## Session 12 — Stage 0 Training Complete, Val NCCL Crash (2026-04-17) 🔴 NEEDS FIX
**Script:** `jamba_coconut_finetune.py`
**Command:**
```bash
torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit \
  --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 \
  --batch_size 4 --grad_accum 8 \
  --val_batch_size 2 \
  --no-gen_every_stage \
  --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \
  --output_dir runs/stage3_curriculum
```
**Status:** Stage 0 training COMPLETE at step 1154/1154. Checkpoint-0001154 saved. Val killed by NCCL watchdog. Training preserved.

**Timeline reconstruction:**
- Session start: 06:18:47
- Model load: ~1h3m
- Training (260 steps @ 137s/step): 9h54m
- Training end / pre-val checkpoint saved: ~16:12
- NCCL watchdog fires on rank 1: 17:14:43 (exactly 3600s after val started)
- Process killed: 17:15:47

**Training metrics (verbatim — resuming from checkpoint-0000894):**
```
  step=   900 s=0 ep=0 ce=0.3795 gn=0.1209
  step=   920 s=0 ep=0 ce=0.3363 gn=0.1123
  step=   940 s=0 ep=0 ce=0.4156 gn=0.1561
  step=   960 s=0 ep=0 ce=0.3658 gn=0.1002
  step=   980 s=0 ep=0 ce=0.3707 gn=0.1416
  step=  1000 s=0 ep=0 ce=0.4889 gn=0.1063
  step=  1020 s=0 ep=0 ce=0.3771 gn=0.1229
  step=  1040 s=0 ep=0 ce=0.5007 gn=0.1409
  step=  1060 s=0 ep=0 ce=0.3693 gn=0.1310
  step=  1080 s=0 ep=0 ce=0.4147 gn=0.1264
  step=  1100 s=0 ep=0 ce=0.4418 gn=0.1430
  step=  1120 s=0 ep=0 ce=0.5040 gn=0.1780
  step=  1140 s=0 ep=0 ce=0.3823 gn=0.1360
S0E0: 100%|████████████| 260/260 [9:54:14<00:00, 137.13s/it, ce=0.385, gn=0.111]
```

**Pre-val checkpoint saved (verbatim):**
```
  [ckpt] saved -> runs/stage3_curriculum/stage_0/checkpoint-0001154  acc=None  ce=None
```

**NCCL crash (verbatim):**
```
[rank1]:[E417 17:14:43.399819004 ProcessGroupNCCL.cpp:688] [Rank 1] Watchdog caught
collective operation timeout: WorkNCCL(SeqNum=58725, OpType=ALLREDUCE, NumelIn=1,
NumelOut=1, Timeout(ms)=3600000) ran for 3600001 milliseconds before timing out.
...
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
jamba_coconut_finetune.py FAILED
Root Cause [0]: rank 1 exitcode: -6 (SIGABRT)
[1]:            rank 0 exitcode: -15 (SIGTERM)
```

**Root cause analysis:**
- Val runs only on rank 0 (`if is_main:`). Rank 1 blocks at `barrier()`.
- `init_process_group(timeout=timedelta(minutes=60))` → NCCL watchdog fires after 3600s.
- Val estimated duration: ~130min CE loop (1940 samples × val_batch_size=1 × 4s) + ~42min accuracy decode (50 samples, no KV cache) = ~170min. Far exceeds 60-min timeout.
- `check_timeout()` uses `graceful_exit_buffer_minutes=20`; 46min remained → val not skipped. But val needs 170min.

**Fixes required (see AGENT_PROMPT_nccl_val_fix.md):**
1. `timedelta(minutes=60)` → `timedelta(hours=4)` in `init_process_group`
2. `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC = 4*3600` env var
3. New `--val_skip_buffer_minutes 120` argument
4. Pre-val skip check uses new threshold instead of `graceful_exit_buffer_minutes`
5. Same guard applied to gen callback

**Resume plan for next session:** checkpoint-0001154 has `step_in_epoch=1153`, `steps_per_epoch=1154`. `start_step = 1154`, `remaining_steps = 0`. Step loop skipped. Val runs directly with fixes applied (~98min with val_batch_size=2). If successful, stage advances to Stage 1.

---

## Session 11 — Stage 0 Partial Run (2026-04-15) 🟡 SUPERSEDED
**Status:** Epoch 0 at step 629/2307 (batch_size=2). Superseded by Session 12 which completed Stage 0.

**Key metrics:**
```
step=    20 s=0 ep=0 ce=0.5169 gn=0.3009
step=   620 s=0 ep=0 ce=0.3405 gn=0.1657
S0E0:  27%|██▏| 629/2307 [10:40:33<27:54:04, 59.86s/it, ce=0.361, gn=0.166]
  [timeout] 10.68h elapsed - 19.4 min remaining (< 20 min buffer).
  [ckpt] saved -> runs/stage3_curriculum/stage_0/checkpoint-0000629  acc=None  ce=None
```

---

## Session 10 — Dual T4 DDP Profile Run (2026-04-15) ✅ COMPLETE
**Status:** Stage 0/0 COMPLETE — 12 steps, val_acc=0.4000

**Bootstrap confirmation:**
```
[bootstrap] Shim: patched 10 removed transformers.generation names ✓
[bootstrap] Kernel export shim: mamba_ssm.selective_state_update ✓
[bootstrap] Mamba fast path: ACTIVE ✓
```

**Validation:**
```
[val] s=0 ep=0 val_ce=0.4253 val_acc=0.4000
[ckpt] best -> runs/profile_dual_t4/stage_0/best  acc=0.4  ce=0.4252639559711584
```

**Generation (5 prompts, all factually correct, mean UWR=0.538)**

---

## Session 9 — sm75 Single T4, First Successful Smoke Test (2026-04-14) ✅
**Status:** Stage 0 COMPLETE — Stage 1 timeout (expected at 1.5h budget)

**Step time: ~113s/step** (single T4, no batching fix, Stage 0).

---

## Sessions 4–8 — sm75 Kernel / Bootstrap Debugging (2026-04-14)
- Session 8: `causal_conv1d_fn` weight shape `(dim, 1, width)` → fixed to `(dim, width)` ✅
- Session 7: `selective_state_update` wrong import path → fixed ✅
- Session 6: single-name generation shim insufficient → 10-alias shim ✅
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
