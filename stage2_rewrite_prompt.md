# Stage 2 `train_sft.py` Rewrite — Coding Agent Prompt
## Project Ouroboros / SERF Framework

> **Self-contained. Feed this entire file to a coding agent.**
> Do not rewrite the file from scratch. Apply only the changes enumerated below,
> in the order listed. After each change, verify the modification is complete before
> proceeding to the next.

---

## Background & Failure Record

Session 5 (S5) died at step 3750 with a hard NCCL watchdog kill:

```
WorkNCCL(SeqNum=13274, OpType=ALLREDUCE, NumelIn=1, NumelOut=1,
  Timeout(ms)=600000) ran for 600003 milliseconds before timing out.
```

Root cause chain:
- Step 3750 triggers both `val_every=250` AND `gen_every=250` simultaneously
- Rank 0 runs `compute_val_metrics()` (2761 samples ÷ batch_size=2 = 1380 forward passes ≈ 557s)
- Rank 0 then runs `run_generation_callback()` (~90s)
- Rank 1 sits idle at `dist.barrier()` the entire time — 647s total
- NCCL watchdog fires at 600s → rank 1 SIGABRT → crash

This document fixes all confirmed open bugs:

| Bug | Description | Severity |
|---|---|---|
| 13 | Save order wrong: `val→gen→save` instead of `save→val→gen` | CRITICAL |
| 14 | Emergency save calls `push_to_hub=args.push_to_hub` | CRITICAL |
| 15 | DDP barrier deadlock on graceful timeout exit | CRITICAL |
| 16 | Default `lr=1e-4` cannot break number-loop attractor | FUNCTIONAL |
| 17 | DRY: three Hub utility functions re-implemented locally | MEDIUM |
| 18 | `gen_every=250` adds 90s to every val window | MINOR |
| 19 | NCCL watchdog (600s) killed by combined val+gen > 600s | CRITICAL |

---

## Part 0 — Prerequisites & Constraints

- **Do NOT rewrite the file from scratch.** Apply surgical edits only.
- **Do NOT change the public API** (CLI argument names, checkpoint format, `run_training` signature).
- **Do NOT modify** `baseline_trm_mamba.py`, `training_utils.py`, or any other file.
- The canonical utility functions in `training_utils.py` must be used directly;
  do not re-implement them.
- All changes must be compatible with both single-GPU and DDP (world_size=2) execution.
- Target hardware: Kaggle Dual T4 (2× 16GB VRAM, NCCL backend).

---

## Change 1 — NCCL Timeout Environment Variable (Bug 19 Sub-fix A)

**Location:** `run_training()` function, immediately before the
`dist.init_process_group(...)` call.

**Action:** Add one line:

```python
os.environ.setdefault("NCCL_TIMEOUT", "1800")  # 30 min; kills genuine hangs fast
```

This must appear **before** `dist.init_process_group(...)`. The existing
`os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` line
already demonstrates the correct pattern.

**Why 1800s:** With val capped at 500 samples × batch_size=16 (≈10s) and gen_every=500
(≈90s every other save), the maximum time rank 1 waits at a barrier is well under 5
minutes. 1800s catches genuine hangs while giving 30× headroom.

---

## Change 2 — Rank 0 Loads Data, Broadcasts to Rank 1 (User Request)

**Problem:** Both DDP ranks currently call `load_mixed_dataset()` or
`load_and_tokenize()` independently. This doubles HuggingFace cache reads and doubles
tokenization time (25–45 min total becomes 45–90 min if the cache is cold).

**Location:** Inside `run_training()`, after tokenizer setup, where `all_samples` is
currently populated. Find this block:

```python
if args.dataset_mix == "full":
    all_samples = load_mixed_dataset(tokenizer, args.max_samples, args.max_seq_len)
else:
    all_samples = load_and_tokenize(args.dataset_name, tokenizer, args.max_samples, args.max_seq_len)
if not all_samples:
    sys.exit("No samples loaded. Check dataset connectivity and --max_samples.")
```

**Replace with:**

```python
# ── Data loading: rank 0 only, then broadcast ─────────────────────────────
if is_main_process(rank):
    if args.dataset_mix == "full":
        all_samples = load_mixed_dataset(tokenizer, args.max_samples, args.max_seq_len)
    else:
        all_samples = load_and_tokenize(args.dataset_name, tokenizer, args.max_samples, args.max_seq_len)
    if not all_samples:
        sys.exit("No samples loaded. Check dataset connectivity and --max_samples.")
else:
    all_samples = None

if distributed:
    # broadcast_object_list handles arbitrary Python objects (list of dicts with Tensors)
    # via pickle. For 55k samples at avg 512 tokens each ≈ 225 MB — well within T4 RAM.
    objects = [all_samples]
    dist.broadcast_object_list(objects, src=0)
    all_samples = objects[0]
elif all_samples is None:
    sys.exit("No samples loaded.")
```

**Important:** The `dist.barrier()` implicit in `broadcast_object_list` ensures rank 1
cannot proceed into training until rank 0 has finished loading. No separate barrier
is needed here.

---

## Change 3 — Val Sample Cap and Batch Size (Bug 19 Sub-fix B)

### 3a — Add `--val_max_samples` CLI argument

**Location:** `parse_args()` function, after the existing `--val_fraction` argument.

**Add:**

```python
parser.add_argument(
    "--val_max_samples",
    type=int,
    default=500,
    help=(
        "Cap validation set to this many samples for speed. "
        "500 samples at val_batch_size=16 ≈ 10s on T4. "
        "Set to -1 to use the full val set (slow under DDP)."
    ),
)
parser.add_argument(
    "--val_batch_size",
    type=int,
    default=16,
    help="Micro-batch size for validation forward passes (no gradients; can be larger than --batch_size).",
)
```

### 3b — Apply cap inside `run_training()`

**Location:** After `train_samples, val_samples = split_train_val_samples(...)`.

**Add immediately after that line:**

```python
if args.val_max_samples > 0 and len(val_samples) > args.val_max_samples:
    # Deterministic subsample: always take the first N after the split permutation.
    val_samples = val_samples[:args.val_max_samples]
    if is_main_process(rank):
        print(f"  val capped to {len(val_samples)} samples (--val_max_samples={args.val_max_samples})")
```

### 3c — Thread `val_batch_size` through `compute_val_metrics` calls

There are two call sites for `compute_val_metrics` inside `run_training()`. Both currently
pass `args.batch_size`. Change both to pass `args.val_batch_size` instead:

```python
# Before (both call sites):
last_val_ce, last_val_acc = compute_val_metrics(
    raw_model, ema, val_samples, pad_id, device, dtype,
    args.batch_size,       # ← change this
    config.vocab_size,
)

# After:
last_val_ce, last_val_acc = compute_val_metrics(
    raw_model, ema, val_samples, pad_id, device, dtype,
    args.val_batch_size,   # ← 16 by default
    config.vocab_size,
)
```

**Expected impact:** 500 samples ÷ 16 = 32 forward passes ≈ 10s on T4.
Previously: 2761 ÷ 2 = 1380 passes ≈ 557s.

---

## Change 4 — Fix Callback Ordering (Bug 13)

**Problem:** Current per-step order is `val_every → gen_every → save_every`.
A timeout firing during generation means the checkpoint for that step is never written.

**Location:** Inside the main training loop `while step < total_steps:`, find the three
`if step % args.X_every == 0` blocks and their surrounding timeout check and save block.

**Required order after fix:**

```
1. timeout check + broadcast
2. (if timeout) emergency save (NO Hub push) + break
3. save_every  ← MUST come first among the three callbacks
4. val_every
5. gen_every   ← MUST come last; most expensive callback
```

Concretely, reorder the blocks so they appear in this sequence:

```python
# ── Timeout check ─────────────────────────────────────────────────────────
check_timeout()
if broadcast_timeout():
    # ... emergency save block (see Change 5) ...
    break

# ── 1. Checkpoint (always before slow callbacks) ──────────────────────────
if step % args.save_every == 0 or step == total_steps:
    if is_main_process(rank):
        save_checkpoint(...)
        last_saved_step = step
    if distributed:
        dist.barrier()

# ── 2. Validation CE (fast after Change 3) ───────────────────────────────
if step % args.val_every == 0 or step == total_steps:
    if is_main_process(rank):
        ...
    if distributed:
        dist.barrier()

# ── 3. Generation (slow; runs half as often after Change 6) ──────────────
if step % args.gen_every == 0 or step == total_steps:
    if is_main_process(rank):
        ...
    if distributed:
        dist.barrier()
```

---

## Change 5 — Emergency Save Must Not Push to Hub (Bug 14)

**Location:** The timeout handler block inside `run_training()`, where
`save_checkpoint(...)` is called after `broadcast_timeout()` returns True.

**Find the emergency save call (it currently passes `push_to_hub=args.push_to_hub`).**

**Change ONLY this specific `save_checkpoint` call** to pass `push_to_hub=False`:

```python
# Emergency save — NO Hub push; local disk only.
save_checkpoint(
    output_dir=output_dir,
    step=step,
    model=raw_model,
    ema=ema,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler if dtype == torch.float16 else None,
    config=config,
    val_ce=last_val_ce,
    keep_last=args.keep_last,
    epoch=current_epoch,
    samples_seen=samples_seen,
    sft_config=sanitize_args_for_serialization(args),
    push_to_hub=False,           # ← CRITICAL: never push on emergency save
    hf_repo_id=args.hf_repo_id,
    hf_token=hf_token,
    hf_stage_subdir=args.hf_stage_subdir,
)
```

---

## Change 6 — DDP Barrier on Graceful Exit (Bug 15)

**Location:** The same timeout handler block, immediately after the emergency save,
where `dist.barrier()` is currently called unconditionally.

**Replace:**

```python
if distributed:
    dist.barrier()
break
```

**With:**

```python
# Do NOT call dist.barrier() here. Rank 0 just finished a slow emergency save.
# Rank 1 may be multiple steps ahead. The NCCL timeout (Change 1) will catch
# any genuine hangs. Both ranks will reach the `break` via their own
# `broadcast_timeout()` check within the next optimizer step.
break
```

That is: **delete the `dist.barrier()` call entirely from the timeout break path.**

The `broadcast_timeout()` function already propagates the timeout flag to all ranks at
the top of each optimizer step. Rank 1 will break on its next iteration without needing
a barrier here.

---

## Change 7 — Default LR and Other Hyperparameter Defaults (Bug 16, 18)

**Location:** `parse_args()` function.

Make the following default value changes only (do not change argument names or help text):

```python
# Bug 16: lr too low to break number-loop attractor
parser.add_argument("--lr",           type=float, default=3e-4)   # was 1e-4
parser.add_argument("--warmup_steps", type=int,   default=50)     # was 100
parser.add_argument("--ema_decay",    type=float, default=0.99)   # was 0.995

# Bug 18: gen_every too frequent under DDP
parser.add_argument("--gen_every",    type=int,   default=500)    # was 250

# Larger graceful exit buffer to absorb Hub push + val latency
parser.add_argument(
    "--graceful_exit_buffer_minutes",
    type=float,
    default=15.0,    # was 7.0 (not enough for Hub push + DDP barrier)
    ...
)
```

---

## Change 8 — Remove Local Re-implementations of Utility Functions (Bug 17)

`train_sft.py` currently contains three functions that shadow canonical implementations
in `training_utils.py`. These must be **deleted** and replaced with imports.

### 8a — Delete these three functions from `train_sft.py`:

1. `list_remote_checkpoint_names(...)` — the local version has a different signature
   (`hf_repo_id, hf_token, remote_prefix, prefer_irregular_steps`)
2. `download_checkpoint_from_hub(...)` — the local version has a different parameter
   order and uses `HfFileSystem` instead of `snapshot_download`
3. `sync_checkpoint_to_hub(...)` — the local version is blocking (no `run_as_future`)

### 8b — Update the import block

**Find the existing import from `training_utils`:**

```python
from training_utils import (
    ModelEMA,
    autocast_context,
    build_adamw_optimizer,
    checkpoint_step_from_name,
    cleanup_temporary_checkpoints,
    cosine_with_warmup,
    download_checkpoint_from_hub,
    ema_scope,
    list_local_checkpoints,
    list_remote_checkpoint_names,
    pad_vocab_size,
    resolve_hf_token,
    set_seed,
    sync_checkpoint_to_hub,
    try_load_state,
    vram_gb,
)
```

This import already lists all three functions. Simply **delete the three local function
definitions** — the import line does not change.

### 8c — Audit all call sites

After deleting the local definitions, audit every call to these three functions and
confirm the argument signatures match the `training_utils.py` versions:

**`list_remote_checkpoint_names`** in `training_utils.py` signature:
```python
def list_remote_checkpoint_names(
    repo_id: str,
    token: Optional[str],
    remote_prefix: Optional[str] = None,
    prefer_irregular_steps: bool = False,
) -> List[str]:
```

**`download_checkpoint_from_hub`** in `training_utils.py` signature:
```python
def download_checkpoint_from_hub(
    checkpoint_name: str,
    output_dir: Path,
    repo_id: str,
    token: Optional[str],
    remote_prefix: Optional[str] = None,
) -> Optional[Path]:
```

**`sync_checkpoint_to_hub`** in `training_utils.py` signature:
```python
def sync_checkpoint_to_hub(
    checkpoint_dir: Path,
    repo_id: str,
    token: Optional[str],
    timeout_seconds: float = HF_UPLOAD_TIMEOUT_SECONDS,
    remote_prefix: Optional[str] = None,
) -> bool:
```

Update every call site in `train_sft.py` (in `load_latest_checkpoint` and
`save_checkpoint`) to match these signatures. The key differences:
- `repo_id` not `hf_repo_id` as parameter name (positional, so order matters)
- `sync_checkpoint_to_hub` takes `remote_prefix` not `hf_stage_subdir` as kwarg name

---

## Part 3 — Verification Checklist

### Dry-run (no GPU required for syntax, single GPU for smoke test):

```bash
python train_sft.py \
  --preset nano \
  --max_seq_len 512 \
  --max_samples 100 \
  --max_steps 10 \
  --val_every 5 \
  --gen_every 10 \
  --save_every 5 \
  --val_max_samples 50 \
  --val_batch_size 16 \
  --wandb_mode disabled \
  --dataset_mix stratos \
  --output_dir runs/stage2_test
```

- [ ] No import errors
- [ ] `NCCL_TIMEOUT=1800` set in environment before `init_process_group` (check with `os.environ.get`)
- [ ] Only rank 0 prints dataset loading messages
- [ ] Training log shows `val capped to 50 samples` message
- [ ] Callback order in log: `[ckpt] saved` appears before `[val]` appears before generation
- [ ] `--lr` default is `3e-4` (verify with `python train_sft.py --help`)
- [ ] `--gen_every` default is `500`
- [ ] `--graceful_exit_buffer_minutes` default is `15.0`
- [ ] No `NameError` or `TypeError` from the removed local utility functions
- [ ] Checkpoint saved with `"stage": "sft"` key

### DDP dry-run (Dual T4):

```bash
python train_sft.py \
  --preset nano \
  --max_seq_len 512 \
  --max_samples 200 \
  --max_steps 20 \
  --val_every 10 \
  --gen_every 20 \
  --save_every 10 \
  --val_max_samples 50 \
  --val_batch_size 16 \
  --batch_size 4 \
  --wandb_mode disabled \
  --dataset_mix stratos \
  --output_dir runs/stage2_ddp_test
```

- [ ] Both ranks start training (two `[rank0]` / `[rank1]` NCCL init lines visible)
- [ ] Dataset loading messages appear only once (rank 0 only)
- [ ] `broadcast_object_list` completes without hanging
- [ ] Val takes < 30s for 50 samples at batch_size=16
- [ ] No NCCL timeout errors
- [ ] Clean exit (no SIGABRT, no barrier deadlock)

### Full S6 run command (after all checks pass):

```bash
python train_sft.py \
  --preset nano \
  --max_seq_len 2048 \
  --dataset_mix full \
  --num_epochs 3 \
  --batch_size 2 \
  --grad_accum 16 \
  --lr 3e-4 \
  --warmup_steps 50 \
  --ema_decay 0.99 \
  --val_max_samples 500 \
  --val_batch_size 16 \
  --output_dir runs/stage2 \
  --push_to_hub \
  --hf_token $HF_TOKEN \
  --wandb_project ouroboros-stage2 \
  --save_every 500 \
  --val_every 250 \
  --gen_every 500 \
  --session_timeout_hours 11.5 \
  --graceful_exit_buffer_minutes 15
```

**Expected timing per optimizer step:** ~9.5s (unchanged).  
**Expected val timing:** 500 ÷ 16 = 32 forward passes ≈ 10s.  
**Expected gen timing:** ~90s every 500 steps ≈ 9.5 min intervals.  
**Maximum rank 1 barrier wait:** val (10s) + gen (90s) = 100s << 1800s NCCL timeout.

---

## Part 4 — Change Summary Table

| # | Location | Type | Description |
|---|---|---|---|
| 1 | `run_training()`, before `init_process_group` | Add 1 line | `NCCL_TIMEOUT=1800` |
| 2 | `run_training()`, data loading block | Replace ~5 lines with ~15 lines | Rank 0 loads, `broadcast_object_list` to rank 1 |
| 3a | `parse_args()` | Add 2 arguments | `--val_max_samples`, `--val_batch_size` |
| 3b | `run_training()`, after split | Add 4 lines | Cap val_samples to `val_max_samples` |
| 3c | `run_training()`, 2 call sites | Change 1 arg each | Pass `val_batch_size` instead of `batch_size` to `compute_val_metrics` |
| 4 | `run_training()`, main loop | Reorder 3 blocks | `save → val → gen` |
| 5 | `run_training()`, timeout handler | Change 1 kwarg | `push_to_hub=False` in emergency save |
| 6 | `run_training()`, timeout handler | Delete 2 lines | Remove `dist.barrier()` from break path |
| 7 | `parse_args()` | Change 5 default values | `lr`, `warmup_steps`, `ema_decay`, `gen_every`, `graceful_exit_buffer_minutes` |
| 8a | Module level | Delete 3 functions | Remove locally-defined Hub utilities |
| 8b | Import block | No change needed | Functions already imported from `training_utils` |
| 8c | All call sites | Update signatures | Match `training_utils.py` parameter names/order |
