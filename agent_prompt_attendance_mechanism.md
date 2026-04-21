# Agent Prompt: Worker Attendance & Timeout Mechanism

> Feed this entire file to the coding agent. It modifies three files:
> `diloco_coordinator.py`, `jamba_coconut_finetune.py`, `.github/workflows/diloco_coordinator.yml`.
> Source of truth for implementation intent is this document.

---

## Problem

The coordinator stalls indefinitely when a triggered worker never responds (quota exhausted, kernel failure, etc.). There is no deadline logic. `triggered_workers` is written once and the coordinator exits early on every subsequent run until all listed workers post a status — which never happens if a worker is quota-dead.

---

## Solution Overview: 3-Mode Lifecycle

| Mode | `mode` field | `triggered_workers` | `attendance_workers` | Meaning |
|---|---|---|---|---|
| Normal | `"diloco"` / `"solo"` | workers that train | `[]` | All active workers training as before |
| Mixed | `"diloco"` / `"solo"` | workers that train | workers being pinged | Some workers timed out; rest train normally |
| Waiting | `"waiting"` | `[]` | all credentialed workers | All quota exhausted; round frozen until workers signal presence |

**Attendance round**: A worker in `attendance_workers` skips training entirely. It loads the anchor, uploads a `status.json` with `samples_seen=0`, and pushes a GitHub signal. This proves its Kaggle quota is active. The coordinator promotes it to `triggered_workers` starting the next round.

**Waiting mode**: `round_n` is frozen. Coordinator re-dispatches attendance pings on every run (manual `workflow_dispatch` or an incoming worker signal). Once any worker responds, it is promoted, round_n advances, training resumes. Self-healing: no config changes needed when quota renews.

---

## New `round_state.json` Schema Fields

Add exactly two fields. All existing fields are unchanged.

```json
{
  "attendance_workers": ["C"],       // list of worker IDs in ping-only mode; default []
  "triggered_at": 1776760000.0       // unix float: when workers were last dispatched
}
```

`triggered_at` is written on EVERY dispatch (training workers + attendance workers together). It is reset to `time.time()` each time workers are triggered — including re-dispatch in waiting mode.

---

## File 1: `diloco_coordinator.py`

### 1.1 New CLI argument

Add to `parse_args()`:

```python
parser.add_argument(
    "--worker_timeout_hours",
    type=float,
    default=13.0,
    help=(
        "Hours after triggered_at before a non-responsive worker is demoted to "
        "attendance_workers. 13h = Kaggle 12h hard wall + 1h grace. Default 13.0."
    ),
)
```

### 1.2 New state read at the top of `main()`

Immediately after the existing `state = hub_download_json(...)` block, add:

```python
current_mode = state.get("mode", "diloco")
triggered_at = float(state.get("triggered_at", 0.0))
attendance_workers_prev = list(state.get("attendance_workers", []))
worker_timeout_s = args.worker_timeout_hours * 3600.0
is_round_timed_out = triggered_at > 0 and (time.time() - triggered_at) > worker_timeout_s
```

Also move the `kaggle_creds`, `credentialed`, and `force_ids` computation to this same location (immediately after state fields are read), since the waiting-mode early path in 1.3 requires these variables to already be in scope.

### 1.3 Waiting-mode early path

Insert this block BEFORE the W&B init and BEFORE `collect_ready_workers`. If `current_mode == "waiting"`, the coordinator takes a separate, simpler path:

```python
if current_mode == "waiting":
    # Check if any attendance workers have responded since last dispatch
    responded_in_waiting = collect_ready_workers(
        args.repo_id, args.hf_token, stage_k, round_n,
        expected_workers=attendance_workers_prev,
    )
    responded_ids = {str(w.get("worker_id", "")) for w in responded_in_waiting}
    still_absent = [w for w in attendance_workers_prev if w not in responded_ids]

    if not responded_ids:
        if not is_round_timed_out:
            print("[coordinator] Waiting mode: no responses yet, standing by.")
            return
        # Timed out with no responses — re-dispatch attendance to all
        print(f"[coordinator] Waiting mode: re-dispatching attendance to {attendance_workers_prev}")
        new_state = {**state, "triggered_at": time.time()}
        hub_upload_json(args.repo_id, ROUND_STATE_PATH, new_state, args.hf_token,
                        message=f"Waiting mode: re-dispatch attendance round={round_n}")
        if not args.skip_trigger and not args.dry_run:
            trigger_kaggle_workers(
                kaggle_creds,
                active_workers=attendance_workers_prev,
                notebook_path=Path(args.kaggle_notebook_path),
            )
        print("[coordinator] Done (waiting mode re-dispatch).")
        return

    # Workers responded — promote them, advance round
    print(f"[coordinator] Waiting mode exit: promoting {list(responded_ids)}")
    # Aggregate is skipped (samples_seen=0 for all attendance workers)
    total_samples_seen[str(stage_k)] = stage_samples_seen  # unchanged
    next_round_n = round_n + 1
    next_stage_k = stage_k
    projected_shards_next = _compute_projected_shards(
        total_samples=args.total_train_samples,
        stage_k=next_stage_k,
        round_n=next_round_n,
        seed=seed,
        total_samples_seen=stage_samples_seen,
    )
    eligible_for_training = [w for w in credentialed if w in responded_ids]
    next_mode, next_active_workers = _determine_round_mode(
        projected_shards=projected_shards_next,
        credentialed_workers=eligible_for_training,
        min_shard_samples=args.min_shard_samples,
        force_worker_ids=force_ids,
    )
    next_attendance_workers = still_absent
    if not next_active_workers:
        next_mode = "waiting"
        next_attendance_workers = attendance_workers_prev  # everyone back to attendance

    new_state = {
        **state,
        "stage_k": next_stage_k,
        "round_n": next_round_n,
        "mode": next_mode,
        "triggered_workers": next_active_workers,
        "attendance_workers": next_attendance_workers,
        "projected_shards": projected_shards_next,
        "total_samples_seen": total_samples_seen,
        "last_updated": time.time(),
        "triggered_at": time.time(),
        "last_round_workers": list(responded_ids),
        "last_round_samples": 0,
        "seed": seed,
    }
    hub_upload_json(args.repo_id, ROUND_STATE_PATH, new_state, args.hf_token,
                    message=f"Waiting mode resolved: stage={next_stage_k} round={next_round_n} mode={next_mode}")
    print(f"[coordinator] round_state updated: stage={next_stage_k} round={next_round_n} mode={next_mode}")
    if not args.skip_trigger and next_active_workers:
        trigger_kaggle_workers(
            kaggle_creds,
            active_workers=next_active_workers + next_attendance_workers,
            notebook_path=Path(args.kaggle_notebook_path),
        )
    print("[coordinator] Done (waiting mode resolved).")
    return
```

### 1.4 Replace the `missing_workers` early-return block

Current code (inside the `if expected_workers:` block):
```python
if missing_workers:
    print(f"[coordinator] Waiting for workers to finish this round: {missing_workers}")
    return
```

Replace with:
```python
if missing_workers:
    if not is_round_timed_out:
        print(f"[coordinator] Waiting for workers to finish this round: {missing_workers}")
        return
    # Past timeout — classify and continue
    newly_demoted = [w for w in missing_workers if w not in attendance_workers_prev]
    still_absent  = [w for w in missing_workers if w in attendance_workers_prev]
    if newly_demoted:
        print(f"[coordinator] Timed out (>{args.worker_timeout_hours}h): {newly_demoted} — demoting to attendance")
    if still_absent:
        print(f"[coordinator] Still absent after attendance: {still_absent} — retrying")
```

### 1.5 Compute next `attendance_workers` list

Add this block AFTER the aggregation section (after `save_and_upload_anchor`) and BEFORE the stage-advance check. It computes who goes into `next_attendance_workers` for the following round:

```python
ready_ids = {str(w.get("worker_id", "")) for w in ready_workers}

# Workers promoted from attendance (responded with samples_seen=0 this round)
attendance_promoted = [w for w in attendance_workers_prev if w in ready_ids]
if attendance_promoted:
    print(f"[coordinator] Attendance workers responded, promoting next round: {attendance_promoted}")

# Build next round's attendance list
newly_demoted = [
    w for w in (expected_workers or [])
    if w not in ready_ids and w not in attendance_workers_prev
] if is_round_timed_out else []

still_attending = [w for w in attendance_workers_prev if w not in ready_ids]
next_attendance_workers = list(set(newly_demoted + still_attending))
```

### 1.6 Adjust `_determine_round_mode` call to exclude attendance workers

When computing `next_mode` and `next_active_workers` for the next round, exclude `next_attendance_workers` from the eligible pool:

```python
eligible_for_training = [w for w in credentialed if w not in next_attendance_workers]
next_mode, next_active_workers = _determine_round_mode(
    projected_shards=projected_shards_next,
    credentialed_workers=eligible_for_training,
    min_shard_samples=args.min_shard_samples,
    force_worker_ids=force_ids,
)
```

### 1.7 Waiting mode detection

After determining `next_mode` and `next_active_workers`, add:

```python
# If all credentialed workers ended up in attendance with no trainers → waiting mode
if not next_active_workers and next_attendance_workers:
    print("[coordinator] All workers absent — entering waiting mode. Coordinator idles until workers signal presence.")
    next_mode = "waiting"
    # Do NOT advance round_n in waiting mode (re-checked on next trigger)
    next_round_n = round_n + 1 if not stage_complete else 0
    # Override: freeze round when waiting
    if next_mode == "waiting":
        next_round_n = round_n
        next_stage_k = stage_k
```

Note: the stage-complete check runs before this point; if the stage already closed, waiting mode is irrelevant.

### 1.8 Write `attendance_workers` and `triggered_at` into `round_state.json`

Update the `new_state` dict that is passed to `hub_upload_json` to include the two new fields:

```python
new_state = {
    "stage_k": next_stage_k,
    "round_n": next_round_n,
    "mode": next_mode,
    "triggered_workers": next_active_workers,
    "attendance_workers": next_attendance_workers,    # NEW
    "projected_shards": projected_shards_next,
    "anchor_path": ANCHOR_PREFIX,
    "total_samples_seen": total_samples_seen,
    "completed_stages": completed_stages,
    "last_updated": time.time(),
    "triggered_at": time.time() if (next_active_workers or next_attendance_workers) else 0.0,  # NEW
    "last_round_workers": [w["worker_id"] for w in contributing_workers] if contributing_workers else [],
    "last_round_samples": sum(w.get("samples_seen", 0) for w in contributing_workers) if contributing_workers else 0,
    "seed": seed,
}
```

### 1.9 Trigger both training and attendance workers together

Update the `trigger_kaggle_workers` call at the end of `main()`:

```python
all_workers_to_trigger = next_active_workers + [
    w for w in next_attendance_workers if w not in next_active_workers
]
if args.skip_trigger:
    print("[coordinator] --skip_trigger set. Skipping worker trigger.")
elif not all_workers_to_trigger:
    print("[coordinator] No workers to trigger (stage complete or waiting with no dispatch needed).")
else:
    print(f"[coordinator] Triggering training: {next_active_workers}  attendance: {next_attendance_workers}")
    trigger_kaggle_workers(
        kaggle_creds,
        active_workers=all_workers_to_trigger,
        notebook_path=Path(args.kaggle_notebook_path),
    )
```

### 1.10 `dry_run` output update

Add `attendance_workers` and `triggered_at` to the dry-run print block:

```python
print(f"  next_attendance_workers={next_attendance_workers}")
print(f"  worker_timeout_hours={args.worker_timeout_hours}")
```

---

## File 2: `jamba_coconut_finetune.py`

### 2.1 Attendance check in `run_diloco_worker()`

Add this block IMMEDIATELY AFTER `diloco_read_round_state()` is called and `stage_k`/`round_n`/`anchor_path` are extracted — BEFORE the shard computation and BEFORE loading the anchor:

```python
# ── Attendance mode check ─────────────────────────────────────────────────
attendance_workers = round_state.get("attendance_workers", [])
is_attendance_only = args.diloco_worker_id in attendance_workers

if is_attendance_only:
    if _is_main_process():
        print(
            f"  [diloco] Worker {args.diloco_worker_id} in attendance mode — "
            f"signaling presence (no training this round)."
        )
        # Download anchor to confirm Hub access / quota is active
        diloco_download_anchor(model, hf_token, args.diloco_state_repo, anchor_path, device)

        # Upload passthrough status (samples_seen=0 signals "present, no work done")
        _attend_dir = (
            output_dir / "diloco_worker_upload"
            / f"worker_{args.diloco_worker_id}_attend_{stage_k}_{round_n}"
        )
        _attend_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(_attend_dir))
        diloco_upload_worker_state(
            adapter_dir=_attend_dir,
            worker_id=args.diloco_worker_id,
            stage_k=stage_k,
            round_n=round_n,
            samples_seen=0,
            hf_token=hf_token,
            repo_id=args.diloco_state_repo,
        )
        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token and args.diloco_signal_repo:
            diloco_push_signal(
                args.diloco_worker_id, stage_k, round_n,
                github_token, args.diloco_signal_repo,
            )
        else:
            print("  [diloco] No GITHUB_TOKEN — coordinator must be triggered manually.")
        print(
            f"  [diloco] Worker {args.diloco_worker_id} attendance done. "
            f"stage={stage_k} round={round_n}"
        )
    barrier()
    return {
        "stage_k": stage_k,
        "round_n": round_n,
        "samples_seen": 0,
        "global_step": 0,
        "timeout_triggered": False,
        "val_budget_triggered": False,
        "stages": [stage_k],
    }
```

No other changes to `jamba_coconut_finetune.py`. The existing empty-shard passthrough path remains unchanged (safety net for computed-zero shards).

---

## File 3: `.github/workflows/diloco_coordinator.yml`

Add `--worker_timeout_hours 13.0` to the `python diloco_coordinator.py` run command:

```yaml
run: |
  python diloco_coordinator.py \
    --hf_token          "$HF_TOKEN" \
    --repo_id           "WeirdRunner/Ouroboros" \
    --outer_lr          0.7 \
    --worker_timeout_hours 13.0 \
    --wandb_key         "$WANDB_KEY" \
    ...
```

---

## One-Time Manual Fix (before deploying this patch)

Before pushing this patch, manually update `diloco_state/round_state.json` on `WeirdRunner/Ouroboros` to remove Worker C from `triggered_workers`:

```json
{
  "stage_k": 2,
  "round_n": 4,
  "triggered_workers": ["A", "B"],
  "attendance_workers": ["C"],
  "triggered_at": 0,
  ...
}
```

Setting `triggered_at: 0` means the coordinator will not apply a timeout on the first run — it waits for A and B normally. Worker C is already in `attendance_workers` so it will be pinged this round and promoted once it responds.

---

## Verification Checklist

After deploying:

1. **Normal timeout path**: trigger a round, don't run one worker. After 13h the coordinator should print `Timed out (>13h): [X] — demoting to attendance` and proceed with the other worker. Check `attendance_workers` in the new `round_state.json`.

2. **Attendance promotion**: the demoted worker runs its attendance ping (Cell 5 → attendance check at top of `run_diloco_worker`), uploads status with `samples_seen=0`, pushes signal. Next coordinator run should print `Attendance workers responded, promoting next round: [X]` and include X in `triggered_workers`.

3. **Waiting mode entry**: let ALL workers time out. Coordinator should write `mode: "waiting"`, `triggered_workers: []`, `attendance_workers: [all]`, `round_n` frozen. Confirm in HF Hub.

4. **Waiting mode exit**: trigger `workflow_dispatch` (or let an attendance signal fire). Coordinator should print `Waiting mode exit: promoting [...]` and write a new `round_state.json` with `mode: diloco` and advanced `round_n`.

5. **round_n freeze**: while `mode: "waiting"`, confirm `round_n` does not increment across multiple coordinator runs.

6. No-trigger gap awareness: When workers exhaust quota mid-round without pushing signals, the coordinator receives no automatic trigger. One manual `workflow_dispatch` after ≥13h is required to initiate the timeout path. After that single dispatch, the system is fully self-healing. No `round_state.json` edits are needed.
