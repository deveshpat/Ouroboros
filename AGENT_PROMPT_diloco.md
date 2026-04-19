# AGENT_PROMPT_diloco.md
# DiLoCo Parallel Training — Implementation Instructions

## Overview

Implement **DiLoCo-style distributed training** across three Kaggle accounts (Workers A/B/C) using HuggingFace Hub as shared storage and **GitHub Actions as the CPU-only coordinator**. This replaces the sequential relay with true parallelism, achieving ~3× throughput.

**Do not rewrite the existing training loop.** Add a clean opt-in `--diloco_mode` code path that wraps the existing `main()` logic.

---

## Architecture

```
┌─────────────┐     upload weights     ┌─────────────────────┐
│  Worker A   │ ──────────────────────▶│                     │
│  (Kaggle)   │ ◀── download anchor ── │   HF Hub            │
└─────────────┘                        │  (WeirdRunner/      │
                                        │   Ouroboros)        │
┌─────────────┐     upload weights     │                     │
│  Worker B   │ ──────────────────────▶│  diloco_state/      │
│  (Kaggle)   │ ◀── download anchor ── │    round_state.json │
└─────────────┘                        │    anchor/          │
                                        │    worker_{id}/     │
┌─────────────┐     upload weights     │                     │
│  Worker C   │ ──────────────────────▶└─────────────────────┘
│  (Kaggle)   │ ◀── download anchor ──         ▲  │
└─────────────┘                                │  │ push/pull
                                                │  ▼
                ┌──────────────────────────────────────┐
                │  GitHub Actions (coordinator)        │
                │  Triggered by: signal push to repo   │
                │  Runtime: ~5 min, CPU-only           │
                │  - Download worker weights from Hub  │
                │  - Weighted average of deltas        │
                │  - Upload new anchor to Hub          │
                │  - Update round_state.json           │
                │  - Trigger next Kaggle sessions      │
                └──────────────────────────────────────┘
```

### Worker lifecycle per round
1. Read `round_state.json` from Hub → get `stage_k`, `round_n`, anchor location
2. Download anchor adapter weights from Hub
3. Load anchor into model (replacing current adapter weights)
4. Determine shard: `indices = get_shard(train_samples, worker_id, stage_k, round_n)`
5. Train on shard for `min(shard_steps, steps_remaining_in_session)` steps
6. Upload weights + status JSON to Hub
7. Push signal file to GitHub repo → triggers coordinator
8. **Exit immediately** (do not wait for coordinator)

### Coordinator lifecycle per round (GitHub Actions)
1. Fetch all `worker_status_{id}.json` files from Hub
2. Find rounds where ≥ `min_workers` (default 2) have `status: done`
3. Download those workers' adapter weights
4. Compute weighted average (weighted by `samples_seen`)
5. Upload new anchor to Hub
6. Advance stage if `total_samples_seen_this_stage >= len(train_samples)` across all workers all rounds
7. Update `round_state.json`
8. Trigger next Kaggle worker sessions via Kaggle API

---

## File Changes

### 1. `jamba_coconut_finetune.py` — additions only

#### New CLI flags (add to `parse_args()`):
```python
parser.add_argument("--diloco_mode", action="store_true",
    help="Enable DiLoCo parallel training mode.")
parser.add_argument("--diloco_worker_id", default=None, choices=["A", "B", "C"],
    help="This worker's identity. Required when --diloco_mode is set.")
parser.add_argument("--diloco_outer_lr", type=float, default=0.7,
    help="Outer SGD learning rate for DiLoCo aggregation. Default: 0.7 (DiLoCo paper).")
parser.add_argument("--diloco_min_workers", type=int, default=2,
    help="Minimum workers needed for coordinator to aggregate (default: 2 of 3).")
parser.add_argument("--diloco_state_repo", default="WeirdRunner/Ouroboros",
    help="HF Hub repo used as shared state store.")
parser.add_argument("--diloco_signal_repo", default="deveshpat/Ouroboros",
    help="GitHub repo to push coordinator trigger signals to.")
parser.add_argument("--diloco_run_val", action="store_true",
    help="Run val pass before training begins (used by first worker of a new stage).")
```

#### New function: `diloco_get_shard()`
```python
def diloco_get_shard(
    train_samples: List[Dict],
    worker_id: str,          # "A", "B", or "C"
    stage_k: int,
    round_n: int,
    seed: int,
) -> List[Dict]:
    """
    Deterministic shard assignment. Every worker with the same args gets the same shard.
    Shard is 1/3 of the dataset, non-overlapping, covering the full dataset across 3 workers.
    Round-robin across rounds so different samples are seen each round.
    """
    worker_idx = {"A": 0, "B": 1, "C": 2}[worker_id]
    n = len(train_samples)
    rng = random.Random(seed + stage_k * 100_003 + round_n * 7)
    indices = list(range(n))
    rng.shuffle(indices)
    shard_size = n // 3
    start = worker_idx * shard_size
    end = start + shard_size if worker_idx < 2 else n   # last shard gets remainder
    shard_indices = indices[start:end]
    return [train_samples[i] for i in shard_indices]
```

#### New function: `diloco_read_round_state()`
```python
def diloco_read_round_state(hf_token: str, repo_id: str) -> Dict[str, Any]:
    """
    Download and parse diloco_state/round_state.json from Hub.
    Returns default state if file doesn't exist (first run).
    """
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="diloco_state/round_state.json",
            token=hf_token,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        # Default: start at stage 0, round 0
        return {
            "stage_k": 0,
            "round_n": 0,
            "anchor_path": "diloco_state/anchor",
            "total_samples_seen": {},   # {stage_k: int}
            "completed_stages": [],
        }
```

#### New function: `diloco_upload_worker_state()`
```python
def diloco_upload_worker_state(
    adapter_dir: Path,          # local path to saved adapter_model/
    worker_id: str,
    stage_k: int,
    round_n: int,
    samples_seen: int,
    hf_token: str,
    repo_id: str,
) -> None:
    """
    Upload worker adapter weights and status to Hub.
    Paths:
      diloco_state/workers/{worker_id}/round_{round_n:04d}_stage_{stage_k}/adapter_model.safetensors
      diloco_state/workers/{worker_id}/status.json
    """
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    remote_prefix = f"diloco_state/workers/{worker_id}/round_{round_n:04d}_stage_{stage_k}"

    # Upload adapter weights
    for fname in ["adapter_model.safetensors", "adapter_config.json"]:
        fpath = adapter_dir / fname
        if fpath.exists():
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=f"{remote_prefix}/{fname}",
                repo_id=repo_id,
                token=hf_token,
                commit_message=f"Worker {worker_id} round {round_n} stage {stage_k}",
            )

    # Upload status
    import tempfile, json as _json
    status = {
        "worker_id": worker_id,
        "stage_k": stage_k,
        "round_n": round_n,
        "samples_seen": samples_seen,
        "status": "done",
        "timestamp": time.time(),
        "weights_path": remote_prefix,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        _json.dump(status, tf)
        tmp_path = tf.name
    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=f"diloco_state/workers/{worker_id}/status.json",
        repo_id=repo_id,
        token=hf_token,
        commit_message=f"Worker {worker_id} status update",
    )
```

#### New function: `diloco_download_anchor()`
```python
def diloco_download_anchor(
    model,                  # PEFT model — adapter weights will be replaced in-place
    hf_token: str,
    repo_id: str,
    anchor_path: str,       # e.g. "diloco_state/anchor"
    device: torch.device,
) -> None:
    """
    Download anchor adapter weights from Hub and load them into the model in-place.
    Falls back silently if no anchor exists (first round uses random init).
    """
    from huggingface_hub import hf_hub_download
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    try:
        dl_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{anchor_path}/adapter_model.safetensors",
            token=hf_token,
        )
        weights = load_file(dl_path, device=str(device))
        set_peft_model_state_dict(model, weights)
        if _is_main_process():
            print(f"  [diloco] Loaded anchor weights from {anchor_path}")
    except Exception as exc:
        if _is_main_process():
            print(f"  [diloco] No anchor found at {anchor_path} ({exc}); using current weights.")
```

#### New function: `diloco_push_signal()`
```python
def diloco_push_signal(
    worker_id: str,
    stage_k: int,
    round_n: int,
    github_token: str,
    github_repo: str,     # e.g. "deveshpat/Ouroboros"
) -> None:
    """
    Push a signal file to GitHub to trigger the coordinator GitHub Action.
    File: signals/worker_{id}_stage_{k}_round_{n}.json
    Uses GitHub API directly (no git clone needed).
    """
    import base64, requests

    signal_path = f"signals/worker_{worker_id}_stage_{stage_k}_round_{round_n}.json"
    content = json.dumps({
        "worker_id": worker_id,
        "stage_k": stage_k,
        "round_n": round_n,
        "timestamp": time.time(),
    })
    encoded = base64.b64encode(content.encode()).decode()

    url = f"https://api.github.com/repos/{github_repo}/contents/{signal_path}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Check if file exists (to get sha for update)
    resp = requests.get(url, headers=headers)
    payload = {
        "message": f"Worker {worker_id} done: stage {stage_k} round {round_n}",
        "content": encoded,
    }
    if resp.status_code == 200:
        payload["sha"] = resp.json()["sha"]

    resp = requests.put(url, headers=headers, json=payload)
    if resp.status_code in (200, 201):
        if _is_main_process():
            print(f"  [diloco] Signal pushed to GitHub: {signal_path}")
    else:
        if _is_main_process():
            print(f"  [diloco] WARNING: GitHub signal push failed: {resp.status_code} {resp.text[:200]}")
```

#### Modifications to `main()`:
At the very start of `main()`, after loading model/tokenizer, add:

```python
if args.diloco_mode:
    assert args.diloco_worker_id is not None, "--diloco_worker_id required with --diloco_mode"
    assert hf_token, "HF token required for DiLoCo mode"

    round_state = diloco_read_round_state(hf_token, args.diloco_state_repo)
    stage_k = round_state["stage_k"]
    round_n  = round_state["round_n"]

    if is_main:
        print(f"  [diloco] Worker {args.diloco_worker_id} | stage={stage_k} round={round_n}")

    # Load anchor weights
    diloco_download_anchor(model, hf_token, args.diloco_state_repo,
                           round_state["anchor_path"], device)

    # Optionally run val before training (first worker of new stage)
    if args.diloco_run_val and val_samples:
        val_ce, val_acc = evaluate_stage(model, val_samples, tokenizer,
                                          lat_token_id, stage_k, device, args)
        if is_main:
            print(f"  [diloco] Pre-training val: stage={stage_k} ce={val_ce:.4f} acc={val_acc:.4f}")
            # Log to Hub + W&B as usual

    # Replace train_samples with this worker's shard
    train_shard = diloco_get_shard(
        train_samples, args.diloco_worker_id, stage_k, round_n, args.seed
    )
    if is_main:
        print(f"  [diloco] Shard size: {len(train_shard)} samples")

    # Run one epoch on the shard (existing training loop, single stage_k)
    # ... (call the existing single-stage training code with train_shard)
    # Track samples_seen during training loop

    # After training:
    adapter_tmp = output_dir / "diloco_worker_upload" / "adapter_model"
    adapter_tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_tmp.parent))

    if is_main:
        diloco_upload_worker_state(
            adapter_dir=adapter_tmp,
            worker_id=args.diloco_worker_id,
            stage_k=stage_k,
            round_n=round_n,
            samples_seen=samples_seen_this_round,   # tracked in training loop
            hf_token=hf_token,
            repo_id=args.diloco_state_repo,
        )

        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token and args.diloco_signal_repo:
            diloco_push_signal(
                args.diloco_worker_id, stage_k, round_n,
                github_token, args.diloco_signal_repo
            )
        else:
            print("  [diloco] No GITHUB_TOKEN — coordinator must be triggered manually.")

    print(f"  [diloco] Worker {args.diloco_worker_id} done. Exiting.")
    return  # Exit immediately; do not wait for coordinator
```

---

### 2. `diloco_coordinator.py` — new standalone script (CPU-only)

This script runs in GitHub Actions. It must have **zero CUDA dependencies**.

```python
#!/usr/bin/env python3
"""
DiLoCo Coordinator — CPU-only weight aggregation.
Runs in GitHub Actions after receiving worker signals.

Usage:
    python diloco_coordinator.py \
        --hf_token $HF_TOKEN \
        --repo_id WeirdRunner/Ouroboros \
        --min_workers 2 \
        --outer_lr 0.7 \
        --kaggle_username $KAGGLE_USERNAME \
        --kaggle_key $KAGGLE_KEY
"""
import argparse, json, os, sys, tempfile, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token", required=True)
    p.add_argument("--repo_id", default="WeirdRunner/Ouroboros")
    p.add_argument("--min_workers", type=int, default=2)
    p.add_argument("--outer_lr", type=float, default=0.7)
    p.add_argument("--kaggle_username", default=None)
    p.add_argument("--kaggle_key", default=None)
    p.add_argument("--total_train_samples", type=int, default=36906)
    return p.parse_args()


def hub_download_json(repo_id: str, path: str, token: str) -> Optional[Dict]:
    from huggingface_hub import hf_hub_download
    try:
        local = hf_hub_download(repo_id=repo_id, filename=path, token=token)
        with open(local) as f:
            return json.load(f)
    except Exception:
        return None


def hub_upload_json(repo_id: str, path: str, data: Dict, token: str, message: str) -> None:
    from huggingface_hub import HfApi
    import tempfile
    api = HfApi(token=token)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(data, tf, indent=2)
        tmp = tf.name
    api.upload_file(path_or_fileobj=tmp, path_in_repo=path,
                    repo_id=repo_id, token=token, commit_message=message)
    os.unlink(tmp)


def load_adapter_weights_cpu(repo_id: str, weights_path: str, token: str) -> Dict:
    """Load safetensors adapter weights to CPU tensors."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    local = hf_hub_download(
        repo_id=repo_id,
        filename=f"{weights_path}/adapter_model.safetensors",
        token=token,
    )
    return load_file(local, device="cpu")   # CPU only — no CUDA needed


def weighted_average_deltas(
    anchor_weights: Dict,
    worker_weights: List[Dict],
    worker_samples: List[int],
    outer_lr: float,
) -> Dict:
    """
    DiLoCo outer update:
      pseudo_grad_i = anchor - worker_i
      outer_grad = weighted_mean(pseudo_grad_i, weights=samples_i)
      new_anchor = anchor - outer_lr * outer_grad

    All operations on CPU tensors.
    """
    import torch
    total_samples = sum(worker_samples)
    assert total_samples > 0

    new_weights = {}
    for key in anchor_weights:
        anchor_tensor = anchor_weights[key].float()
        outer_grad = torch.zeros_like(anchor_tensor)
        for w_weights, n_samples in zip(worker_weights, worker_samples):
            if key in w_weights:
                delta = anchor_tensor - w_weights[key].float()
                outer_grad += delta * (n_samples / total_samples)
        new_weights[key] = (anchor_tensor - outer_lr * outer_grad).to(anchor_weights[key].dtype)

    return new_weights


def save_and_upload_anchor(
    new_weights: Dict,
    anchor_adapter_config: Dict,
    repo_id: str,
    token: str,
    message: str,
) -> None:
    from huggingface_hub import HfApi
    from safetensors.torch import save_file
    api = HfApi(token=token)

    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = Path(tmpdir) / "adapter_model.safetensors"
        config_path  = Path(tmpdir) / "adapter_config.json"
        save_file(new_weights, str(weights_path))
        with open(config_path, "w") as f:
            json.dump(anchor_adapter_config, f, indent=2)

        for fname in ["adapter_model.safetensors", "adapter_config.json"]:
            api.upload_file(
                path_or_fileobj=str(Path(tmpdir) / fname),
                path_in_repo=f"diloco_state/anchor/{fname}",
                repo_id=repo_id,
                token=token,
                commit_message=message,
            )
    print(f"  [coordinator] New anchor uploaded: {message}")


def trigger_kaggle_workers(kaggle_username: str, kaggle_key: str) -> None:
    """
    Trigger all three worker Kaggle notebooks via Kaggle API.
    Notebooks must be pre-configured with 'Run on API trigger' enabled.
    Kernel slugs: {username}/ouroboros-worker-a, worker-b, worker-c
    """
    if not kaggle_username or not kaggle_key:
        print("  [coordinator] No Kaggle credentials — workers must be triggered manually.")
        return

    for worker_id in ["a", "b", "c"]:
        slug = f"{kaggle_username}/ouroboros-worker-{worker_id}"
        url = f"https://www.kaggle.com/api/v1/kernels/{slug}/run"
        resp = requests.post(
            url,
            auth=(kaggle_username, kaggle_key),
            json={},
        )
        if resp.status_code == 200:
            print(f"  [coordinator] Triggered worker {worker_id.upper()}: {slug}")
        else:
            print(f"  [coordinator] WARNING: Failed to trigger {slug}: "
                  f"{resp.status_code} {resp.text[:200]}")


def main():
    args = parse_args()

    # Install deps if not present (GitHub Actions environment)
    os.system("pip install -q huggingface_hub safetensors torch requests")

    print("[coordinator] Reading round state...")
    state = hub_download_json(args.repo_id, "diloco_state/round_state.json", args.hf_token)
    if state is None:
        print("[coordinator] No round_state.json found. Nothing to aggregate.")
        sys.exit(0)

    stage_k = state["stage_k"]
    round_n  = state["round_n"]
    total_samples_seen = state.get("total_samples_seen", {})
    completed_stages   = state.get("completed_stages", [])

    print(f"[coordinator] stage={stage_k} round={round_n}")

    # Collect worker statuses
    ready_workers = []
    for wid in ["A", "B", "C"]:
        status = hub_download_json(
            args.repo_id,
            f"diloco_state/workers/{wid}/status.json",
            args.hf_token,
        )
        if (status is not None
                and status.get("stage_k") == stage_k
                and status.get("round_n") == round_n
                and status.get("status") == "done"):
            ready_workers.append(status)
            print(f"  [coordinator] Worker {wid}: {status['samples_seen']} samples ✓")
        else:
            print(f"  [coordinator] Worker {wid}: not ready (status={status})")

    if len(ready_workers) < args.min_workers:
        print(f"[coordinator] Only {len(ready_workers)}/{args.min_workers} workers ready. Exiting.")
        sys.exit(0)

    # Load anchor weights
    print("[coordinator] Loading anchor weights...")
    anchor_weights = load_adapter_weights_cpu(
        args.repo_id, "diloco_state/anchor", args.hf_token
    )

    # Load anchor adapter config (for re-upload)
    from huggingface_hub import hf_hub_download
    config_path = hf_hub_download(
        repo_id=args.repo_id,
        filename="diloco_state/anchor/adapter_config.json",
        token=args.hf_token,
    )
    with open(config_path) as f:
        anchor_adapter_config = json.load(f)

    # Load worker weights
    print("[coordinator] Loading worker weights...")
    worker_weights_list = []
    worker_samples_list = []
    for status in ready_workers:
        w = load_adapter_weights_cpu(args.repo_id, status["weights_path"], args.hf_token)
        worker_weights_list.append(w)
        worker_samples_list.append(status["samples_seen"])

    # Compute weighted average
    print("[coordinator] Aggregating (CPU)...")
    new_anchor = weighted_average_deltas(
        anchor_weights,
        worker_weights_list,
        worker_samples_list,
        args.outer_lr,
    )

    # Upload new anchor
    save_and_upload_anchor(
        new_anchor,
        anchor_adapter_config,
        args.repo_id,
        args.hf_token,
        message=f"DiLoCo anchor: stage {stage_k} round {round_n} "
                f"({len(ready_workers)} workers, "
                f"{sum(worker_samples_list)} samples)",
    )

    # Update stage progress
    stage_key = str(stage_k)
    current_stage_samples = total_samples_seen.get(stage_key, 0)
    current_stage_samples += sum(worker_samples_list)
    total_samples_seen[stage_key] = current_stage_samples
    print(f"[coordinator] Stage {stage_k} progress: "
          f"{current_stage_samples}/{args.total_train_samples} samples seen")

    # Check stage completion (one full epoch across all workers)
    stage_complete = current_stage_samples >= args.total_train_samples
    next_stage_k = stage_k
    next_round_n = round_n + 1
    if stage_complete:
        print(f"[coordinator] Stage {stage_k} COMPLETE. Advancing to stage {stage_k + 1}.")
        completed_stages.append(stage_k)
        next_stage_k = stage_k + 1
        next_round_n = 0

    # Update round_state.json
    new_state = {
        "stage_k": next_stage_k,
        "round_n": next_round_n,
        "anchor_path": "diloco_state/anchor",
        "total_samples_seen": total_samples_seen,
        "completed_stages": completed_stages,
        "last_updated": time.time(),
        "last_round_workers": [s["worker_id"] for s in ready_workers],
        "last_round_samples": sum(worker_samples_list),
    }
    hub_upload_json(
        args.repo_id,
        "diloco_state/round_state.json",
        new_state,
        args.hf_token,
        message=f"Round state update: stage {next_stage_k} round {next_round_n}",
    )
    print(f"[coordinator] round_state.json updated: stage={next_stage_k} round={next_round_n}")

    # Trigger next worker sessions
    trigger_kaggle_workers(args.kaggle_username, args.kaggle_key)
    print("[coordinator] Done.")


if __name__ == "__main__":
    main()
```

---

### 3. `.github/workflows/diloco_coordinator.yml` — new file

```yaml
name: DiLoCo Coordinator

on:
  push:
    paths:
      - 'signals/worker_*.json'   # Triggered when any worker pushes a signal

jobs:
  coordinate:
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install deps
        run: pip install huggingface_hub safetensors torch requests

      - name: Run coordinator
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
        run: |
          python diloco_coordinator.py \
            --hf_token "$HF_TOKEN" \
            --repo_id "WeirdRunner/Ouroboros" \
            --min_workers 2 \
            --outer_lr 0.7 \
            --kaggle_username "$KAGGLE_USERNAME" \
            --kaggle_key "$KAGGLE_KEY" \
            --total_train_samples 36906
```

**GitHub Secrets to configure** (repo Settings → Secrets → Actions):
- `HF_TOKEN` — HuggingFace write token
- `KAGGLE_USERNAME` — Kaggle account username (any account works)
- `KAGGLE_KEY` — Kaggle API key

---

### 4. New Kaggle worker notebooks

Create three separate notebooks: `ouroboros-worker-a`, `ouroboros-worker-b`, `ouroboros-worker-c`.

Cell 5 (the training command) changes per notebook — only `--diloco_worker_id` differs:

```bash
# Worker A notebook
!torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
  --data_dir data/coconut_v1 --use_4bit \
  --stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 \
  --batch_size 4 --grad_accum 8 \
  --val_batch_size 2 \
  --val_skip_buffer_minutes 60 \
  --session_timeout_hours 12.0 --graceful_exit_buffer_minutes 20 \
  --diloco_mode \
  --diloco_worker_id A \
  --diloco_outer_lr 0.7 \
  --diloco_state_repo WeirdRunner/Ouroboros \
  --diloco_signal_repo deveshpat/Ouroboros \
  --push_to_hub \
  --output_dir runs/diloco
```

For the **first worker of a new stage** (automatically detected when `round_n == 0`), add:
```bash
  --diloco_run_val \
```

The `--diloco_run_val` flag should be auto-set in code when `round_state["round_n"] == 0` to avoid manual tracking. Hard-code this logic in `main()`.

---

## Hub State Layout

```
WeirdRunner/Ouroboros/
  diloco_state/
    round_state.json              ← coordinator writes after each aggregation
    anchor/
      adapter_model.safetensors  ← latest aggregated weights (starting point for all workers)
      adapter_config.json
    workers/
      A/
        status.json              ← worker A's latest round status
        round_0000_stage_2/
          adapter_model.safetensors
          adapter_config.json
      B/
        status.json
        round_0000_stage_2/
          adapter_model.safetensors
          adapter_config.json
      C/
        status.json
        round_0000_stage_2/ ...
  signals/                       ← GitHub trigger files (tiny, push by workers)
    worker_A_stage_2_round_0.json
    worker_B_stage_2_round_0.json
    ...
```

---

## Bootstrap: First Run Setup

Before workers start, run this **once** to initialize the Hub state and set the anchor to the current best checkpoint:

```python
# bootstrap_diloco.py — run once from any machine with Hub write access
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file
import json, os

HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "WeirdRunner/Ouroboros"
EXISTING_CHECKPOINT = "runs/stage3/stage_2/checkpoint-0002987"  # current best on Hub

api = HfApi(token=HF_TOKEN)

# Copy existing checkpoint adapter weights as the initial DiLoCo anchor
for fname in ["adapter_model.safetensors", "adapter_config.json"]:
    api.upload_file(
        # Download from existing location and re-upload as anchor
        # (or point directly if already local)
        path_or_fileobj=...,  # download first via hf_hub_download
        path_in_repo=f"diloco_state/anchor/{fname}",
        repo_id=REPO_ID,
        token=HF_TOKEN,
        commit_message="Initialize DiLoCo anchor from checkpoint-0002987",
    )

# Write initial round_state.json
initial_state = {
    "stage_k": 2,           # Resume from Stage 2 (59% done)
    "round_n": 0,
    "anchor_path": "diloco_state/anchor",
    "total_samples_seen": {"2": 679 * 32},  # approx samples from sequential pre-DiLoCo
    "completed_stages": [0, 1],
    "last_updated": 0,
}
# ... upload as diloco_state/round_state.json
```

---

## Validation Rules

- **Stage advancement:** triggered when `total_samples_seen[stage_k] >= 36906` (full dataset)
- **Val:** runs automatically when `round_n == 0` (start of new stage). Coordinator is responsible; workers skip val unless `--diloco_run_val` is set.
- **Minimum workers:** 2 of 3. If only 1 worker is available for 3 consecutive rounds, print a warning but continue (single-worker rounds are valid, just slower).
- **Outer LR:** 0.7 (DiLoCo paper default). Reduce to 0.5 if loss spikes on stage advance.
- **No gradient checkpointing state mismatch:** The adapter weights are all that's synced. Base model is frozen. No optimizer state is transferred across workers — each worker uses a fresh optimizer per round (this is correct DiLoCo behavior; inner optimizer state is local and discarded after each round).

---

## Testing Plan

**Phase 1 — Proof of concept (Stage 2 remaining, right now):**
1. Run `bootstrap_diloco.py` to set `checkpoint-0002987` as anchor
2. Start Workers A, B, C simultaneously on Stage 2 remaining (~475 steps / 3 ≈ 158 steps each, ~2.3h)
3. All three upload, push signals
4. Coordinator GH Action fires, aggregates, uploads new anchor, triggers Stage 3
5. If val looks sane (within 10% of sequential val_acc), proceed
6. **Fallback:** if anything breaks, existing `checkpoint-0002987` is untouched; return to sequential relay

**Phase 2 — Full DiLoCo from Stage 3 onward:**
All future stages use DiLoCo. The `jamba_coconut_finetune.py` sequential path remains intact for fallback.
