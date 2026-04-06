# Stage 2 Patch Prompt — Project Ouroboros
## Fix all bugs that caused the full-dataset Session 2 catastrophic failure

**File to modify: `train_sft.py` only. Do NOT rewrite from scratch.**

---

## Bug Inventory (all confirmed from terminal logs)

### Bug A — Resume ignores local_stage2_latest and downloads Hub checkpoints (CRITICAL)
The log shows `local_stage2_latest=2979` was detected correctly, but the code then
downloaded 18 Hub Stage 1 checkpoints (steps 21000 down to 3000) before finding the
local one. Root cause: candidates are sorted by `(step, priority)` descending, so
Hub step=21000 outranks local step=2979 despite `priority=0 < 1`.

### Bug B — Hub-downloaded checkpoints placed inside output_dir (CRITICAL)
`download_checkpoint_from_hub` used `local_root` (derived from `output_dir`) as the
download target. Steps 3000–21000 were downloaded into `/kaggle/working/runs/stage2/`.

### Bug C — Prune deleted ALL Stage 2 checkpoints (CRITICAL, caused by Bug B)
`save_checkpoint` prune logic scanned `output_dir`, saw checkpoints 2000–21000,
applied `keep_last=3`, and kept steps 19000/20000/21000 (Stage 1 Hub downloads).
ALL Stage 2 checkpoints (2000, 2500, 2979) were deleted. Hub upload then failed
with "not a directory" because the just-saved checkpoint was also pruned.

### Bug D — Optimizer/scheduler NOT reset when dataset changes (CRITICAL)
When `data_changed=True`, the code reset epoch/samples_seen but kept the optimizer
state (momentum adapted to stratos answer-only distribution) and the scheduler at
LR~1e-05 (cosine minimum from step 2979 of the old schedule). New data saw near-zero
LR and stale momentum → 75 loss spikes in 717 steps. Generation degraded to pure
number loops ("100000000000000000000000...").

### Bug E — VRAM OOM from loading too many checkpoint tensors (caused by Bug A)
Downloading and `torch.load(..., map_location=device)` for 18×740MB checkpoints
filled GPU memory. Checkpoints 4000 and 3000 failed with CUDA OOM.

### Bug F — max_seq_len=1024 filters 97% of reasoning datasets (data quality)
Stratos: 515/16710 kept; OpenR1-Math: 843/11140; OpenR1-Code: 9/11672. The whole
point of Stage 2 is to teach reasoning via `<think>` chains. At 1024 tokens only
the shortest (least informative) samples survive. **The stratos-only Session 1 also
never produced `<think>` tags — confirmed that reasoning was not being learned.
Increase to max_seq_len=2048. T4 VRAM at seq_len=1024 was 2.44 GB; at 2048 with
same batch_size/grad_accum expect ~5–6 GB — well within 16 GB T4 headroom.**

---

## Required Changes

### Change 1 — Fix `load_latest_checkpoint`: local Stage 2 first, Hub in temp dir

Replace the entire `load_latest_checkpoint` function with the version below.

Key behavioral changes:
1. Try ALL local Stage 2 checkpoints first. If any succeed, return immediately. No Hub downloads.
2. Only attempt Hub downloads if zero local Stage 2 checkpoints exist.
3. Hub downloads go to `output_dir / ".hub_resume"` (a temp dir), never into `output_dir` itself.
4. Delete `.hub_resume` after a successful load.
5. Return a 4th value `reset_optimizer: bool`. True when data changed.

```python
def load_latest_checkpoint(
    output_dir: Path,
    model: BaselineTRMMamba,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    device: torch.device,
    push_to_hub: bool = False,
    hf_repo_id: str = "WeirdRunner/Ouroboros",
    hf_token: Optional[str] = None,
    verbose: bool = True,
    current_sft_config: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, int, bool]:
    """Load the newest valid Stage 2 checkpoint.

    Returns (step, epoch, samples_seen, reset_optimizer).
    reset_optimizer=True when data stream changed or Stage 1 was loaded.
    """

    def _classify_state(state: Dict[str, Any]) -> str:
        if _looks_like_pretrain_checkpoint(state) and "sft_config" not in state:
            return "stage1"
        if "sft_config" in state:
            return "stage2"
        return "unknown"

    def _restore_stage1(state: Dict[str, Any], label: str) -> Tuple[int, int, int, bool]:
        model.load_state_dict(state["model_state_dict"])
        if state.get("ema"):
            ema.load_state_dict(state["ema"])
        elif state.get("ema_backbone_state_dict"):
            _load_ema_shadow_from_alias(ema, state["ema_backbone_state_dict"])
        if verbose:
            print(
                f"  [init]   {label}  loaded Stage 1 weights; "
                "resetting optimizer/scheduler for Stage 2."
            )
        return 0, 0, 0, True  # reset_optimizer=True

    def _restore_stage2(state: Dict[str, Any], label: str) -> Optional[Tuple[int, int, int, bool]]:
        if any(key not in state for key in ("model_state_dict", "optimizer", "scheduler")):
            if verbose:
                print(f"  [resume] incomplete Stage 2 checkpoint {label} - skipping")
            return None

        epoch = int(state.get("epoch", 0))
        samples_seen = int(state.get("samples_seen", 0))
        model.load_state_dict(state["model_state_dict"])
        if state.get("ema"):
            ema.load_state_dict(state["ema"])
        elif state.get("ema_backbone_state_dict"):
            _load_ema_shadow_from_alias(ema, state["ema_backbone_state_dict"])

        saved_cfg = dict(state.get("sft_config") or {})
        saved_fingerprint = dict(state.get("data_fingerprint") or build_data_fingerprint(saved_cfg))
        current_fingerprint = build_data_fingerprint(current_sft_config or {})
        data_changed = bool(current_sft_config) and (saved_fingerprint != current_fingerprint)

        if data_changed:
            # Data stream changed: keep weights + EMA, but reset optimizer/scheduler/step.
            # Do NOT load optimizer/scheduler state — caller rebuilds them fresh.
            if verbose:
                print(
                    f"  [resume] {label}  loaded model weights (step={state.get('step', 0)}), "
                    "but data stream changed — resetting step/optimizer/scheduler for new data."
                )
                print(f"  [resume] saved_data={saved_fingerprint}")
                print(f"  [resume] new_data  ={current_fingerprint}")
            return 0, 0, 0, True  # step=0, reset_optimizer=True
        else:
            # Same data: full resume including optimizer state.
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if scaler and state.get("scaler"):
                scaler.load_state_dict(state["scaler"])
            step = int(state.get("step", 0))
            if verbose:
                print(
                    f"  [resume] {label}  step={step}  epoch={epoch}  "
                    f"samples_seen={samples_seen}  val_ce={state.get('val_ce')}"
                )
            return step, epoch, samples_seen, False  # reset_optimizer=False

    # ── Resolve search root ───────────────────────────────────────────────────
    search_root = Path(output_dir)
    direct_candidates: List[Path] = []

    if search_root.is_file() and search_root.name == "training_state.pt":
        direct_candidates.append(search_root.parent)
        search_root = search_root.parent.parent if search_root.parent.name.startswith("checkpoint-") else search_root.parent
    elif (search_root / "training_state.pt").exists():
        direct_candidates.append(search_root)
        if search_root.name.startswith("checkpoint-"):
            search_root = search_root.parent
    elif search_root.name.startswith("checkpoint-"):
        search_root = search_root.parent

    local_root = search_root if search_root.exists() else search_root.parent

    # ── Collect local candidates ──────────────────────────────────────────────
    local_stage2: List[Dict[str, Any]] = []
    local_stage1_fallback: Optional[Dict[str, Any]] = None
    local_stage1_label: Optional[str] = None
    seen_paths: set[str] = set()

    for candidate in direct_candidates + list_local_checkpoints(local_root):
        candidate_str = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if candidate_str in seen_paths:
            continue
        seen_paths.add(candidate_str)
        state = try_load_state(candidate, device)
        if state is None or "model_state_dict" not in state:
            continue
        kind = _classify_state(state)
        step = int(state.get("step", checkpoint_step_from_name(candidate.name)))
        if kind == "stage2":
            local_stage2.append({"step": step, "label": candidate.name, "state": state})
        elif kind == "stage1" and local_stage1_fallback is None:
            local_stage1_fallback = state
            local_stage1_label = candidate.name

    # Sort local Stage 2 candidates newest first
    local_stage2.sort(key=lambda r: r["step"], reverse=True)

    if verbose:
        local_s2 = local_stage2[0]["step"] if local_stage2 else None
        print(
            f"  [resume] local Stage 2 candidates: {len(local_stage2)} "
            f"(newest step={local_s2 if local_s2 is not None else 'none'})"
        )

    # ── PRIORITY 1: try local Stage 2 checkpoints ─────────────────────────────
    for record in local_stage2:
        try:
            result = _restore_stage2(record["state"], record["label"])
            if result is not None:
                return result
        except Exception as exc:
            if verbose:
                print(f"  [resume] failed to restore local {record['label']}: {exc} - skipping")

    # ── PRIORITY 2: Hub Stage 2 checkpoints (only if no local Stage 2 found) ──
    # Downloads go to a SEPARATE temp dir, NOT inside output_dir.
    hub_resume_dir = local_root / ".hub_resume"
    if hf_token:
        if verbose:
            print("  [resume] no local Stage 2 found; checking Hub for Stage 2 checkpoints...")
        remote_names = list_remote_checkpoint_names(hf_repo_id, hf_token)
        # Only download up to 3 candidates to avoid VRAM exhaustion.
        candidates_tried = 0
        for ckpt_name in remote_names:
            if candidates_tried >= 3:
                if verbose:
                    print("  [resume] reached Hub download limit (3); stopping Hub search.")
                break
            if verbose:
                print(f"  [hub]  checking {ckpt_name} ...")
            hub_resume_dir.mkdir(parents=True, exist_ok=True)
            downloaded = download_checkpoint_from_hub(ckpt_name, hub_resume_dir, hf_repo_id, hf_token)
            if downloaded is None:
                continue
            candidates_tried += 1
            state = try_load_state(downloaded, device)
            if state is None or "model_state_dict" not in state:
                continue
            kind = _classify_state(state)
            if kind == "stage2":
                try:
                    result = _restore_stage2(state, ckpt_name)
                    if result is not None:
                        # Clean up temp dir
                        shutil.rmtree(hub_resume_dir, ignore_errors=True)
                        return result
                except Exception as exc:
                    if verbose:
                        print(f"  [resume] failed to restore Hub {ckpt_name}: {exc} - skipping")
            elif kind == "stage1" and local_stage1_fallback is None:
                # Remember as Stage 1 fallback but keep looking for Stage 2
                local_stage1_fallback = state
                local_stage1_label = ckpt_name

    # Clean up temp dir if nothing useful was found there
    if hub_resume_dir.exists():
        shutil.rmtree(hub_resume_dir, ignore_errors=True)

    # ── PRIORITY 3: Stage 1 fallback ─────────────────────────────────────────
    if local_stage1_fallback is not None:
        if verbose:
            print(f"  [resume] no Stage 2 checkpoint found; using Stage 1 fallback from {local_stage1_label}.")
        return _restore_stage1(local_stage1_fallback, local_stage1_label or "stage1")

    if verbose:
        print("  [resume] No checkpoint found - starting from scratch.")
    return 0, 0, 0, True
```

---

### Change 2 — Update `run_training` to handle `reset_optimizer` return value

In `run_training`, find the block that calls `load_latest_checkpoint` (there are two: one for
distributed rank 0, one for non-distributed). Both currently unpack 3 values:
```python
start_step, start_epoch, samples_seen = load_latest_checkpoint(...)
```

Replace ALL occurrences of this unpacking pattern with 4 values, and add the optimizer/scheduler
reset logic immediately after the non-distributed (or rank-0 distributed) call:

```python
start_step, start_epoch, samples_seen, need_opt_reset = load_latest_checkpoint(
    resume_search,
    raw_model,
    ema,
    optimizer,
    scheduler,
    scaler,
    device,
    push_to_hub=args.push_to_hub,
    hf_repo_id=args.hf_repo_id,
    hf_token=hf_token,
    verbose=True,
    current_sft_config=sanitize_args_for_serialization(args),
)

if need_opt_reset:
    # Rebuild optimizer and scheduler fresh (model weights were already loaded above).
    optimizer, fused_enabled = build_adamw_optimizer(
        model=raw_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
        prefer_fused=True,
    )
    scheduler = cosine_with_warmup(optimizer, args.warmup_steps, total_steps, args.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    if is_main_process(rank):
        print(f"  [resume] optimizer/scheduler reset; training starts fresh from step 0.")
```

Do the same for the distributed rank-0 branch and the non-distributed branch. The verbose=False
(rank > 0) branch does NOT need the reset because DDP models sync weights via broadcast, and
optimizer state will be rebuilt.

**For the distributed case (rank > 0 branch):** unpack the 4th value but ignore it:
```python
start_step, start_epoch, samples_seen, _ = load_latest_checkpoint(
    ..., verbose=False, ...
)
```

---

### Change 3 — Add `PYTORCH_CUDA_ALLOC_CONF` at the top of `run_training`

Near the top of `run_training`, before any model or tensor creation, add:
```python
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

---

### Change 4 — Change default `max_seq_len` to 2048

In `parse_args`, change:
```python
parser.add_argument("--max_seq_len", type=int, default=1024)
```
to:
```python
parser.add_argument("--max_seq_len", type=int, default=2048)
```

Also update the startup banner in `run_training` to print the actual value:
```python
print(f"  seq_len       : {args.max_seq_len}")
```
(It already does this — just verify it's printing `args.max_seq_len` not a hardcoded value.)

---

### Change 5 — Fix save_checkpoint to not prune Hub-downloaded checkpoints

The prune logic in `save_checkpoint` should skip the `.hub_resume` subdirectory and
any checkpoints that were not saved by the current training run. The cleanest fix is
already achieved by Change 1 (Hub downloads no longer land in `output_dir`). But add
an explicit guard so it skips `.hub_resume`:

In the prune block inside `save_checkpoint`, replace:
```python
existing = sorted(
    [
        entry
        for entry in output_dir.iterdir()
        if entry.is_dir()
        and entry.name.startswith("checkpoint-")
        and not entry.name.endswith(".tmp")
        and checkpoint_step_from_name(entry.name) >= 0
    ],
    key=lambda entry: checkpoint_step_from_name(entry.name),
)
```
with:
```python
existing = sorted(
    [
        entry
        for entry in output_dir.iterdir()
        if entry.is_dir()
        and entry.name.startswith("checkpoint-")
        and not entry.name.endswith(".tmp")
        and not entry.name.startswith(".hub_")
        and checkpoint_step_from_name(entry.name) >= 0
    ],
    key=lambda entry: checkpoint_step_from_name(entry.name),
)
```

---

## Verification Checklist (dry-run before real training)

```bash
python train_sft.py \
  --preset nano \
  --max_seq_len 2048 \
  --max_samples 100 \
  --max_steps 10 \
  --val_every 5 \
  --gen_every 5 \
  --save_every 5 \
  --wandb_mode disabled \
  --dataset_mix full \
  --output_dir runs/stage2_test
```

Confirm:
- [ ] `load_latest_checkpoint` prints "no local Stage 2 found; checking Hub" (no local checkpoints yet)
- [ ] Hub search downloads ≤ 3 checkpoints, each to `runs/stage2_test/.hub_resume/`, NOT `runs/stage2_test/`
- [ ] After loading Stage 2 checkpoint-0002979 from Hub: prints "data stream changed — resetting step/optimizer/scheduler"
- [ ] Training begins at step=0 with LR ~0 (warmup start)
- [ ] `runs/stage2_test/.hub_resume/` is deleted after load
- [ ] Checkpoint saved to `runs/stage2_test/checkpoint-0000005`
- [ ] Prune only deletes checkpoints inside `runs/stage2_test/`, not `.hub_resume`

---

## Run Command (after patches verified)

```bash
python train_sft.py \
  --preset nano \
  --resume_from runs/stage2  \
  --max_seq_len 2048 \
  --dataset_mix full \
  --num_epochs 3 \
  --batch_size 2 \
  --grad_accum 16 \
  --lr 1e-4 \
  --warmup_steps 100 \
  --ema_decay 0.995 \
  --output_dir runs/stage2 \
  --push_to_hub \
  --hf_token $HF_TOKEN \
  --wandb_project ouroboros-stage2 \
  --save_every 500 \
  --val_every 250 \
  --gen_every 250 \
  --session_timeout_hours 11.5 \
  --graceful_exit_buffer_minutes 7
```

Expected behavior:
- Resume finds no local Stage 2 checkpoints
- Downloads checkpoint-0002979 from Hub to `.hub_resume/`
- Detects data_changed (stratos → full), resets optimizer/scheduler/step
- Trains 3 epochs × ceil(~20000/32) = ~1875 steps with lr=1e-4, warmup=100
- At seq_len=2048: Stratos keeps ~60-70% samples, OpenR1-Math keeps ~30%, OpenR1-Code keeps ~15%
- Generation should show `<think>` tags appearing by step 500-1000

**Expected data counts at max_seq_len=2048 (approximate):**
- Stratos: ~10000-12000 (60-70% of 16710)
- MetaMathQA: ~11000
- OpenHermes: ~8000
- OpenR1-Math: ~3000-5000
- OpenR1-Code: ~500-1500
- Total: ~35000-38000 samples

---

## Additional Notes for Agent

- Do NOT change `training_utils.py` — all changes are in `train_sft.py`
- Do NOT change `baseline_trm_mamba.py`, `pretrain.py`, or any other file
- The `_restore_stage2` and `_restore_stage1` closures must be defined INSIDE `load_latest_checkpoint` (they close over `model`, `ema`, `optimizer`, `scheduler`, `scaler`, `verbose`)
- The `shutil` import is already present in `train_sft.py`
- `hub_resume_dir.mkdir(parents=True, exist_ok=True)` must be called before `download_checkpoint_from_hub`
- The Hub download limit of 3 candidates prevents VRAM exhaustion; the newest Stage 2 checkpoint on Hub is checkpoint-0002979 which will be found as the 1st Hub candidate (it's the newest Stage 2 step)
- After all changes: `mypy`-equivalent sanity — the return type of `load_latest_checkpoint` changes from `Tuple[int, int, int]` to `Tuple[int, int, int, bool]`. Update the type annotation in the function signature accordingly.
