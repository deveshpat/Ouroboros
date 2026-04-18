# AGENT_PROMPT_hub_prune_fix.md
**Status: PENDING — apply before next session**

---

## Context

Two production bugs identified after Sessions 13–14:

1. **Hub upload never fires.** `save_checkpoint` checks `args.push_to_hub` which defaults to `False`. The flag was never added to the `torchrun` command. The Stage 0 best checkpoint was never pushed to Hub.

2. **Disk fills silently.** `prune_epoch_checkpoints` is (a) only called after a successful validation — timeout sessions skip it entirely — and (b) scoped to a single stage directory. Kaggle `/kaggle/working/` ≈ 20GB. Each LoRA checkpoint ≈ 1.5–2GB. After 3–4 stages of timeout saves with no pruning, training dies with a disk-full error.

**Fix strategy for next session:** On startup, before training begins, push all local checkpoints to Hub then prune locally — keeping only the checkpoint we are about to resume from (plus all `best/` dirs).

---

## Change 1 — New startup sync function in `jamba_coconut_finetune.py`

Add the following function **after** `find_latest_resume_checkpoint` and **before** `main`:

```python
def startup_hub_sync_and_prune(
    output_dir: Path,
    resume_path: Optional[Path],
    hf_token: str,
    hf_repo_id: str,
    hf_stage_subdir: str,
) -> None:
    """
    Called once at session start (rank 0 only, before training).
    1. Upload every local checkpoint that exists to Hub (best + numbered).
    2. Delete all local numbered checkpoints EXCEPT the one we are resuming from.
       Always preserve best/ dirs.

    This prevents Kaggle disk overflow across sessions.
    Safe to call even if Hub upload fails — prune still runs locally.
    """
    if not _is_main_process():
        return

    all_ckpts: List[Tuple[Path, bool]] = []  # (path, is_resume)
    if output_dir.exists():
        for stage_dir in sorted(output_dir.iterdir()):
            if _parse_stage_dir_name(stage_dir.name) is None or not stage_dir.is_dir():
                continue
            for ckpt in sorted(stage_dir.iterdir()):
                if not ckpt.is_dir():
                    continue
                if not (ckpt / "training_state.pt").exists():
                    continue
                is_resume = (resume_path is not None and ckpt.resolve() == resume_path.resolve())
                all_ckpts.append((ckpt, is_resume))

    if not all_ckpts:
        print("  [startup] No local checkpoints found; nothing to sync/prune.")
        return

    print(f"  [startup] Found {len(all_ckpts)} local checkpoint(s). Uploading to Hub before pruning...")
    for ckpt, is_resume in all_ckpts:
        stage_dir_name = ckpt.parent.name
        remote_prefix = f"{hf_stage_subdir.strip('/')}/{stage_dir_name}"
        ok = _hub_upload_checkpoint(ckpt, hf_repo_id, hf_token, remote_prefix=remote_prefix)
        status = "✓" if ok else "✗ (upload failed — keeping locally)"
        print(f"  [startup]   {stage_dir_name}/{ckpt.name}  {status}")

    # Prune: delete numbered checkpoints that are NOT the resume checkpoint.
    # Always keep best/ dirs regardless.
    pruned = 0
    for ckpt, is_resume in all_ckpts:
        if ckpt.name == "best":
            continue  # always keep best/
        if is_resume:
            continue  # keep the checkpoint we are about to resume from
        shutil.rmtree(ckpt, ignore_errors=True)
        print(f"  [startup]   pruned {ckpt.parent.name}/{ckpt.name}")
        pruned += 1

    print(f"  [startup] Sync+prune complete. Pruned {pruned} checkpoint(s) locally.")
```

---

## Change 2 — Call `startup_hub_sync_and_prune` in `main()`

In `main()`, locate the block that resolves `resume_path` and sets `hf_token`. After `resume_path` is finalized and before `load_model_and_tokenizer`, add:

```python
    # ── Session-start Hub sync + local prune ─────────────────────────────────
    if hf_token and getattr(args, "push_to_hub", False) and is_main:
        startup_hub_sync_and_prune(
            output_dir=output_dir,
            resume_path=resume_path,
            hf_token=hf_token,
            hf_repo_id=getattr(args, "hf_repo_id", "WeirdRunner/Ouroboros"),
            hf_stage_subdir=getattr(args, "hf_stage_subdir", "runs/stage3"),
        )
    barrier()  # ensure prune completes before rank 1 tries to read checkpoints
```

The `barrier()` call is critical — rank 1 must not attempt to read a checkpoint that rank 0 is still uploading or has just pruned.

---

## Change 3 — Update `kaggle-utils.ipynb` Cell 5 command

Replace the current Cell 5 torchrun command with:

```bash
!torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit \
  --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 \
  --batch_size 4 --grad_accum 8 \
  --val_batch_size 2 \
  --val_skip_buffer_minutes 60 \
  --no-gen_every_stage \
  --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \
  --push_to_hub \
  --output_dir runs/stage3_curriculum
```

Key changes vs. previous command:
- Added `--push_to_hub` (was missing; caused silent no-upload)
- Added `--no-gen_every_stage` (make explicit; gen ran unintentionally in sessions 13/14)

---

## Verification checklist (run at next session start)

```
[startup] Found N local checkpoint(s). Uploading to Hub before pruning...
[startup]   stage_0/checkpoint-0001154  ✓
[startup]   stage_0/best  ✓
[startup]   stage_1/checkpoint-0001338  ✓   ← resume checkpoint (kept locally)
[startup]   pruned stage_0/checkpoint-0001154
[startup] Sync+prune complete. Pruned 1 checkpoint(s) locally.
```

After startup, `du -sh /kaggle/working/runs/` should show only:
- `stage_0/best/`
- `stage_1/checkpoint-0001338/`  ← resume point
- Any `best/` dirs from prior stages

---

## DRY cleanup (low-risk, same PR)

While in the file, also clean up the following **dead or redundant code**:

### 2a. Merge duplicate token resolvers

`_bootstrap_resolve_token()` and `_resolve_hf_token()` are byte-for-byte identical in logic. They exist separately because bootstrap runs before huggingface_hub is importable. This is valid — **do not merge**. Document with a comment instead:

```python
# NOTE: _bootstrap_resolve_token() and _resolve_hf_token() are intentionally
# separate. The bootstrap version runs before any third-party imports; the
# main version runs after. Keep both.
```

### 2b. Remove redundant fast-path verification in `load_model_and_tokenizer`

The call to `_bootstrap_verify_fast_path()` inside `load_model_and_tokenizer` is redundant — `_bootstrap()` already verified the fast path and would have `sys.exit(1)` on failure. Remove these lines (keeping the `_patch_transformers_jamba_fast_path_globals()` calls and the `_probe_jamba_runtime_fast_path()` call, which are still needed):

```python
# REMOVE this block from load_model_and_tokenizer:
_mamba_fast_path = False
if device.type == "cuda":
    try:
        _bootstrap_verify_fast_path()        # ← redundant; bootstrap already verified
        _mamba_fast_path = True
        if is_main:
            print("  mamba CUDA kernels: OK — fast path ACTIVE ...")
    except Exception as _kern_exc:
        load_kwargs["use_mamba_kernels"] = False
        ...
```

Replace with the simpler:

```python
_mamba_fast_path = device.type == "cuda"
if not _mamba_fast_path:
    load_kwargs["use_mamba_kernels"] = False
elif is_main:
    print("  mamba CUDA kernels: fast path ACTIVE (verified at bootstrap)")
```

> ⚠️ Keep all `_patch_transformers_jamba_fast_path_globals()` and `_probe_jamba_runtime_fast_path()` calls unchanged — those are still needed post-model-load.

### 2c. Note on latent inject pattern repetition

The latent loop pattern appears three times: `_forward_batched_latent`, `evaluate_stage`, and `run_generation_callback`. Refactoring this into a shared helper is the largest DRY opportunity in the file but carries meaningful risk (different return shapes, with/without halt gate, teacher-forced vs. autoregressive). **Defer to after Stage 3 is complete** and mark as a tech-debt item.
