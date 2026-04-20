#!/usr/bin/env python3
"""
DiLoCo Coordinator - CPU-only weight aggregation.
Runs in GitHub Actions after receiving worker signals.

Usage:
    python diloco_coordinator.py \
        --hf_token "$HF_TOKEN" \
        --repo_id WeirdRunner/Ouroboros \
        --min_workers 2 \
        --outer_lr 0.7 \
        --wandb_key "$WANDB_KEY" \
        --wandb_project "ouroboros-stage3-jamba" \
        --kaggle_username_a "$KAGGLE_USERNAME_A" \
        --kaggle_key_a "$KAGGLE_KEY_A" \
        --kaggle_username_b "$KAGGLE_USERNAME_B" \
        --kaggle_key_b "$KAGGLE_KEY_B" \
        --kaggle_username_c "$KAGGLE_USERNAME_C" \
        --kaggle_key_c "$KAGGLE_KEY_C"
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


ROUND_STATE_PATH = "diloco_state/round_state.json"
ANCHOR_PREFIX = "diloco_state/anchor"
WORKER_IDS = ["A", "B", "C"]
# Maps worker ID -> owning Kaggle username and kernel slug.
# All three accounts run their own copy of kaggle-utils (no separate notebooks).
# Slug format: {owner_username}/kaggle-utils
WORKER_KAGGLE_SLUGS: Dict[str, Tuple[str, str]] = {
    "A": ("weirdrunner", "weirdrunner/kaggle-utils"),
    "B": ("weirdrunner007", "weirdrunner007/kaggle-utils"),
    "C": ("weirdrunner008", "weirdrunner008/kaggle-utils"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only DiLoCo coordinator")
    parser.add_argument("--hf_token", required=True)
    parser.add_argument("--repo_id", default="WeirdRunner/Ouroboros")
    parser.add_argument("--min_workers", type=int, default=2)
    parser.add_argument("--outer_lr", type=float, default=0.7)
    # Per-worker Kaggle credentials (each account can only trigger its own notebook)
    parser.add_argument(
        "--kaggle_username_a",
        default=None,
        help="Kaggle username for Worker A. Required to auto-trigger worker A.",
    )
    parser.add_argument("--kaggle_key_a", default=None, help="Kaggle API key for Worker A.")
    parser.add_argument("--kaggle_username_b", default=None, help="Kaggle username for Worker B.")
    parser.add_argument("--kaggle_key_b", default=None, help="Kaggle API key for Worker B.")
    parser.add_argument("--kaggle_username_c", default=None, help="Kaggle username for Worker C.")
    parser.add_argument("--kaggle_key_c", default=None, help="Kaggle API key for Worker C.")
    # W&B
    parser.add_argument(
        "--wandb_key",
        default=None,
        help="W&B API key. If omitted, coordinator skips W&B logging.",
    )
    parser.add_argument("--wandb_project", default="ouroboros-stage3-jamba")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--total_train_samples", type=int, default=36906)
    return parser.parse_args()



def hub_download_json(repo_id: str, path: str, token: str) -> Optional[Dict]:
    from huggingface_hub import hf_hub_download

    try:
        local = hf_hub_download(repo_id=repo_id, filename=path, token=token)
        with open(local, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None



def hub_upload_json(repo_id: str, path: str, data: Dict, token: str, message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(data, tf, indent=2)
        tmp = tf.name
    try:
        api.upload_file(
            path_or_fileobj=tmp,
            path_in_repo=path,
            repo_id=repo_id,
            token=token,
            commit_message=message,
        )
    finally:
        Path(tmp).unlink(missing_ok=True)



def hub_download_text(repo_id: str, path: str, token: str) -> str:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id=repo_id, filename=path, token=token)
    return Path(local).read_text(encoding="utf-8")



def load_adapter_weights_cpu(repo_id: str, weights_path: str, token: str) -> Dict:
    """Load safetensors adapter weights to CPU tensors."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    local = hf_hub_download(
        repo_id=repo_id,
        filename=f"{weights_path}/adapter_model.safetensors",
        token=token,
    )
    return load_file(local, device="cpu")



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

    All operations run on CPU tensors.
    """
    import torch

    total_samples = sum(worker_samples)
    if total_samples <= 0:
        raise ValueError("total_samples must be > 0 for aggregation")

    new_weights = {}
    for key in anchor_weights:
        anchor_tensor = anchor_weights[key].float()
        outer_grad = torch.zeros_like(anchor_tensor)
        for weights, n_samples in zip(worker_weights, worker_samples):
            if key not in weights:
                continue
            delta = anchor_tensor - weights[key].float()
            outer_grad += delta * (float(n_samples) / float(total_samples))
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
        tmp_path = Path(tmpdir)
        weights_path = tmp_path / "adapter_model.safetensors"
        config_path = tmp_path / "adapter_config.json"
        save_file(new_weights, str(weights_path))
        config_path.write_text(json.dumps(anchor_adapter_config, indent=2), encoding="utf-8")

        for fname in ["adapter_model.safetensors", "adapter_config.json"]:
            api.upload_file(
                path_or_fileobj=str(tmp_path / fname),
                path_in_repo=f"{ANCHOR_PREFIX}/{fname}",
                repo_id=repo_id,
                token=token,
                commit_message=message,
            )
    print(f"[coordinator] New anchor uploaded: {message}")



def _trigger_single_worker(worker_id: str, username: str, key: str, slug: str) -> bool:
    """
    Trigger a Kaggle kernel run via pull → re-push using the official SDK.

    Kaggle has NO standalone 'run' endpoint. The only programmatic way to
    trigger a new run of an existing notebook is to push a new version.
    kaggle.api.kernels_pull() downloads the current code + kernel-metadata.json;
    kaggle.api.kernels_push() re-uploads it unchanged, which creates a new
    version and starts it immediately.

    Per-worker credentials are injected via KAGGLE_USERNAME / KAGGLE_KEY env vars
    (takes precedence over ~/.kaggle/kaggle.json per official SDK behaviour).
    A fresh KaggleApi instance is created per worker so credentials don't bleed.
    """
    import os
    import tempfile

    prev = {k: os.environ.get(k) for k in ("KAGGLE_USERNAME", "KAGGLE_KEY")}
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Downloads notebook source + kernel-metadata.json (preserves GPU/internet flags)
            api.kernels_pull(slug, path=tmpdir, metadata=True, quiet=True)
            # Re-push identical content → new version → run starts automatically
            api.kernels_push(tmpdir)

        print(f"[coordinator] Triggered Worker {worker_id}: {slug}")
        return True
    except Exception as exc:
        print(f"[coordinator] WARNING: Failed to trigger {slug}: {exc}")
        return False
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def trigger_kaggle_workers(
    kaggle_creds: Dict[str, Tuple[Optional[str], Optional[str]]],
) -> None:
    """
    Trigger each worker's Kaggle notebook using that worker's own credentials.
    Kaggle's /run API does not exist; cross-account triggers also return 403.
    Each worker therefore has its own credential pair stored as GitHub secrets.
    """
    for worker_id in WORKER_IDS:
        username, key = kaggle_creds.get(worker_id, (None, None))
        _, slug = WORKER_KAGGLE_SLUGS[worker_id]

        if not username or not key:
            print(
                f"[coordinator] No credentials for Worker {worker_id} ({slug}) - "
                "skipping automatic trigger. Start this worker manually."
            )
            continue

        _trigger_single_worker(worker_id, username, key, slug)


def collect_ready_workers(repo_id: str, token: str, stage_k: int, round_n: int) -> List[Dict]:
    ready: List[Dict] = []
    for worker_id in WORKER_IDS:
        status = hub_download_json(
            repo_id,
            f"diloco_state/workers/{worker_id}/status.json",
            token,
        )
        if (
            status is not None
            and int(status.get("stage_k", -1)) == stage_k
            and int(status.get("round_n", -1)) == round_n
            and status.get("status") == "done"
        ):
            ready.append(status)
            print(f"[coordinator] Worker {worker_id}: {int(status['samples_seen'])} samples ready")
        else:
            print(f"[coordinator] Worker {worker_id}: not ready (status={status})")
    ready.sort(key=lambda item: item.get("worker_id", ""))
    return ready



def maybe_update_insufficient_worker_streak(
    *,
    state: Dict,
    repo_id: str,
    token: str,
    ready_workers: List[Dict],
    min_workers: int,
) -> Optional[List[Dict]]:
    """
    Validation rule from the prompt:
      - prefer min_workers workers
      - after 3 consecutive single-worker rounds, warn and continue with 1 worker
    """
    streak = int(state.get("insufficient_worker_rounds", 0))
    if len(ready_workers) >= min_workers:
        state["insufficient_worker_rounds"] = 0
        return ready_workers

    if len(ready_workers) != 1:
        print(
            f"[coordinator] Only {len(ready_workers)}/{min_workers} workers ready. "
            "Waiting for more workers."
        )
        return None

    streak += 1
    state["insufficient_worker_rounds"] = streak
    if streak < 3:
        print(
            f"[coordinator] Only 1/{min_workers} workers ready. "
            f"Deferring aggregation (streak={streak}/3)."
        )
        hub_upload_json(
            repo_id,
            ROUND_STATE_PATH,
            {**state, "last_updated": time.time()},
            token,
            message=(
                f"Track insufficient workers: stage {int(state['stage_k'])} "
                f"round {int(state['round_n'])} streak {streak}"
            ),
        )
        return None

    print(
        "[coordinator] WARNING: only one worker has been available for 3 consecutive rounds. "
        "Proceeding with single-worker aggregation."
    )
    state["insufficient_worker_rounds"] = 0
    return ready_workers



def main() -> None:
    args = parse_args()

    print("[coordinator] Reading round state...")
    state = hub_download_json(args.repo_id, ROUND_STATE_PATH, args.hf_token)
    if state is None:
        print("[coordinator] No round_state.json found. Nothing to aggregate.")
        return

    stage_k = int(state.get("stage_k", 0))
    round_n = int(state.get("round_n", 0))
    total_samples_seen = {str(k): int(v) for k, v in dict(state.get("total_samples_seen", {})).items()}
    completed_stages = [int(x) for x in state.get("completed_stages", [])]

    print(f"[coordinator] stage={stage_k} round={round_n}")

    # - W&B coordinator run --------------------------------------------------
    coordinator_wandb_run = None
    try:
        if args.wandb_key:
            try:
                import wandb
                wandb.login(key=args.wandb_key, relogin=True)
                coordinator_wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    id=f"diloco-coordinator-s{stage_k}",
                    resume="allow",
                    name=f"Coordinator | Stage {stage_k}",
                    config={
                        "stage_k": stage_k,
                        "outer_lr": args.outer_lr,
                        "min_workers": args.min_workers,
                        "total_train": args.total_train_samples,
                    },
                    mode="online",
                )
            except Exception as _we:
                print(f"[coordinator] W&B init failed: {_we}")

        ready_workers = collect_ready_workers(args.repo_id, args.hf_token, stage_k, round_n)
        allowed_workers = maybe_update_insufficient_worker_streak(
            state=state,
            repo_id=args.repo_id,
            token=args.hf_token,
            ready_workers=ready_workers,
            min_workers=args.min_workers,
        )
        if allowed_workers is None:
            return

        print("[coordinator] Loading anchor weights...")
        anchor_weights = load_adapter_weights_cpu(args.repo_id, ANCHOR_PREFIX, args.hf_token)
        anchor_adapter_config = json.loads(
            hub_download_text(args.repo_id, f"{ANCHOR_PREFIX}/adapter_config.json", args.hf_token)
        )

        print("[coordinator] Loading worker weights...")
        worker_weights_list = []
        worker_samples_list = []
        for status in allowed_workers:
            worker_weights_list.append(
                load_adapter_weights_cpu(args.repo_id, status["weights_path"], args.hf_token)
            )
            worker_samples_list.append(int(status["samples_seen"]))

        print("[coordinator] Aggregating on CPU...")
        new_anchor = weighted_average_deltas(
            anchor_weights,
            worker_weights_list,
            worker_samples_list,
            args.outer_lr,
        )

        save_and_upload_anchor(
            new_anchor,
            anchor_adapter_config,
            args.repo_id,
            args.hf_token,
            message=(
                f"DiLoCo anchor: stage {stage_k} round {round_n} "
                f"({len(allowed_workers)} workers, {sum(worker_samples_list)} samples)"
            ),
        )

        stage_key = str(stage_k)
        current_stage_samples = int(total_samples_seen.get(stage_key, 0)) + sum(worker_samples_list)
        total_samples_seen[stage_key] = current_stage_samples
        print(
            f"[coordinator] Stage {stage_k} progress: "
            f"{current_stage_samples}/{args.total_train_samples} samples seen"
        )

        stage_complete = current_stage_samples >= args.total_train_samples
        next_stage_k = stage_k
        next_round_n = round_n + 1
        if stage_complete:
            print(f"[coordinator] Stage {stage_k} COMPLETE. Advancing to stage {stage_k + 1}.")
            if stage_k not in completed_stages:
                completed_stages.append(stage_k)
            completed_stages = sorted(set(completed_stages))
            next_stage_k = stage_k + 1
            next_round_n = 0

        if coordinator_wandb_run is not None:
            import wandb
            wandb.log(
                {
                    "coordinator/round": round_n,
                    "coordinator/workers_aggregated": len(allowed_workers),
                    "coordinator/samples_this_round": sum(worker_samples_list),
                    "coordinator/total_samples_stage": current_stage_samples,
                    "coordinator/stage_complete": int(stage_complete),
                    "coordinator/pct_stage_done": round(
                        current_stage_samples / max(args.total_train_samples, 1) * 100, 1
                    ),
                },
                step=round_n,
            )

        new_state = {
            "stage_k": next_stage_k,
            "round_n": next_round_n,
            "anchor_path": ANCHOR_PREFIX,
            "total_samples_seen": total_samples_seen,
            "completed_stages": completed_stages,
            "insufficient_worker_rounds": 0,
            "last_updated": time.time(),
            "last_round_workers": [status["worker_id"] for status in allowed_workers],
            "last_round_samples": sum(worker_samples_list),
        }
        hub_upload_json(
            args.repo_id,
            ROUND_STATE_PATH,
            new_state,
            args.hf_token,
            message=f"Round state update: stage {next_stage_k} round {next_round_n}",
        )
        print(f"[coordinator] round_state.json updated: stage={next_stage_k} round={next_round_n}")

        kaggle_creds: Dict[str, Tuple[Optional[str], Optional[str]]] = {
            "A": (args.kaggle_username_a, args.kaggle_key_a),
            "B": (args.kaggle_username_b, args.kaggle_key_b),
            "C": (args.kaggle_username_c, args.kaggle_key_c),
        }
        trigger_kaggle_workers(kaggle_creds)

        if coordinator_wandb_run is not None:
            import wandb
            wandb.finish()
            coordinator_wandb_run = None
        print("[coordinator] Done.")
    finally:
        if coordinator_wandb_run is not None:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
