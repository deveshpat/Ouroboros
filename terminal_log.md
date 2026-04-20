# terminal_log.md — Project Ouroboros
**Newest first. Verbatim excerpts only — no hallucinated numbers.**

---

## Session 16 — DiLoCo Stage 2 Round 0 (2026-04-20) ✅ TRAINING COMPLETE / ⚠️ COORDINATOR CRASHED

**Context:** First live DiLoCo run. Workers A+B completed 159/159 steps each, weights on HF Hub.
Coordinator triggered manually (GitHub signal push failed due to bad GITHUB_TOKEN scope).
Coordinator ran, found both workers, aggregated on CPU, then crashed at save step — `numpy` not installed.
Fix deployed to `diloco_coordinator.yml`. Re-trigger pending.

---

### Worker A (weirdrunner) — verbatim

```
S2E0: 100%|█████████████| 159/159 [2:08:26<00:00, 48.47s/it, ce=0.593, gn=0.874]
  step=    20 s=2 ep=0 ce=0.4321 gn=0.8874
  step=    40 s=2 ep=0 ce=0.5493 gn=0.8912
  step=    60 s=2 ep=0 ce=0.4998 gn=0.9167
  step=    80 s=2 ep=0 ce=0.5836 gn=1.1541
  step=   100 s=2 ep=0 ce=0.4660 gn=0.6701
  step=   120 s=2 ep=0 ce=0.5587 gn=1.3681
  step=   140 s=2 ep=0 ce=0.6616 gn=0.7871
  [diloco] WARNING: GitHub signal push failed: 403 {"message":"Resource not accessible by personal access token",...}
  [diloco] Worker A done. stage=2 round=0 samples_seen=5060
```

### Worker B (weirdrunner007) — verbatim

```
S2E0: 100%|█████████████| 159/159 [2:20:23<00:00, 52.98s/it, ce=0.462, gn=0.828]
  step=    20 s=2 ep=0 ce=0.5731 gn=1.1896
  step=    40 s=2 ep=0 ce=0.5661 gn=0.7973
  step=    60 s=2 ep=0 ce=0.5267 gn=0.6795
  step=    80 s=2 ep=0 ce=0.5345 gn=0.8094
  step=   100 s=2 ep=0 ce=0.6022 gn=1.1261
  step=   120 s=2 ep=0 ce=0.5403 gn=1.1760
  step=   140 s=2 ep=0 ce=0.4631 gn=0.9939
  [diloco] WARNING: GitHub signal push failed: 401 {"message": "Bad credentials",...}
  [diloco] Worker B done. stage=2 round=0 samples_seen=5059
```

**Step timing:** Worker A 48.47s/step | Worker B 52.98s/step | Average ~51s/step.

---

### Coordinator Run #2 — verbatim crash (2026-04-20T22:58)

```
[coordinator] Reading round state...
[coordinator] stage=2 round=0
[coordinator] Worker A: 5060 samples ready
[coordinator] Worker B: 5059 samples ready
[coordinator] Worker C: not ready (status=None)
[coordinator] Loading anchor weights...
[coordinator] Loading worker weights...
[coordinator] Aggregating on CPU...
Traceback (most recent call last):
  File "diloco_coordinator.py", line 496, in <module>
    main()
  File "diloco_coordinator.py", line 411, in main
    save_and_upload_anchor(
  File "diloco_coordinator.py", line 179, in save_and_upload_anchor
    save_file(new_weights, str(weights_path))
  File "safetensors/torch.py", line 468, in _tobytes
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

**Root cause:** `safetensors.torch.save_file` has a Rust backend but its PyTorch tensor serialization path calls `import numpy`. `numpy` was not in the pip install list.

**State after crash:** `round_state.json` still at `{stage_k: 2, round_n: 0}` (crash before state update). All worker weights intact on HF Hub. Safe to re-trigger after deploying fix.

**Fix applied to `diloco_coordinator.yml`:**
- `pip install numpy` added
- `torch` → CPU-only wheel (`--index-url https://download.pytorch.org/whl/cpu`)
- `timeout-minutes: 30` (was 15)
- `workflow_dispatch:` trigger added

---

## Session 15 — Hub+Prune Confirmed, Stage 1 Complete, Stage 2 59% (2026-04-18 → 2026-04-19) ✅ COMPLETE (timed out)

**Stage 1 completion:**
```
S1E0: 100%|█████████████████| 15/15 [10:21<00:00, 41.46s/it, ce=0.372, gn=0.575]
  [val] s=1 ep=0 val_ce=0.4912 val_acc=0.0444
  [best] stage=1 new best acc=0.0444
```

**Stage 2 timeout:**
```
S2E0:  59%|█████▉    | 679/1154 [9:43:06<6:55:29, 52.48s/it, ce=0.636, gn=1.069]
  [timeout] 10.68h elapsed - 19.5 min remaining (< 20 min buffer).
  [ckpt] saved -> runs/stage3_curriculum/stage_2/checkpoint-0002987
```

---

## Session 14 — Stage 1 Resumed, FP16 Confirmed (2026-04-18) ✅ COMPLETE

```
  [GPU] Tesla T4  cc=sm75  VRAM=16GB  amp_dtype=float16
S1E0:   4%|▌  | 39/970 [26:51<10:41:10, 41.32s/it, ce=0.389, gn=0.200]
  [timeout] saving emergency checkpoint at step 2293 ...
```
Confirmed: ~41s/step at k=1.

---

## Session 13 — Stage 0 Val Complete, Stage 1 Started (2026-04-17 → 2026-04-18) ✅ COMPLETE

```
  [val] s=0 ep=0 val_ce=0.4041 val_acc=0.0222
  [ckpt] best -> runs/stage3_curriculum/stage_0/best  acc=0.022222...  ce=0.4040...
```
Stage 1 pre-FP16 patch: 162s ± 3s/step. Timed out at step 1338.

---

## Session 12 — Stage 0 Training Complete, Val NCCL Crash (2026-04-17) ✅ RESOLVED

```
S0E0: 100%|████████████| 260/260 [9:54:14<00:00, 137.13s/it, ce=0.385, gn=0.111]
```
NCCL watchdog killed val. Fix: `timedelta(hours=4)` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`.

---

## Session 10 — Dual T4 DDP Profile Run (2026-04-15) ✅ COMPLETE

```
[bootstrap] Mamba fast path: ACTIVE ✓
[val] s=0 ep=0 val_ce=0.4253 val_acc=0.4000
```

---

## Sessions 4–9 — Kernel / Bootstrap Debugging (2026-04-14) ✅

- Session 9: First successful Stage 0 smoke test. ~113s/step single T4 (pre-batching).
- Session 8: `causal_conv1d_fn` weight shape fix ✅
- Session 7: `selective_state_update` import path fix ✅
- Session 6: 10-alias generation shim ✅

---

## Stage 3 — Early Smoke Test (2026-04-11)

```
trainable params: 26,851,328 || all params: 3,056,191,360 || trainable%: 0.8786
Stage 2/2  S2E0: ce=1.464  gn=36.926
```
Fixes: `--max_seq_len 1024`, `--max_grad_norm 0.3`
