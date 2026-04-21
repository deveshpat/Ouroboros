#!/usr/bin/env python3
"""
DiLoCo Coordinator - CPU-only weight aggregation.
Runs in GitHub Actions after receiving worker signals.

Usage:
    python diloco_coordinator.py \
        --hf_token "$HF_TOKEN" \
        --repo_id WeirdRunner/Ouroboros \
        --min_shard_samples 32 \
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
import shutil
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


DEFAULT_KAGGLE_NOTEBOOK_PATH = Path(__file__).resolve().with_name("kaggle-utils.ipynb")


def _compute_projected_shards(
    total_samples: int,
    stage_k: int,
    round_n: int,
    seed: int,
    total_samples_seen: int,
    worker_ids: List[str] = None,
) -> Dict[str, int]:
    """
    Deterministically compute each worker's projected shard size for the
    upcoming round. Uses identical partition logic to diloco_get_shard()
    in jamba_coconut_finetune.py.

    Returns dict: {worker_id: projected_shard_size}
    """
    if worker_ids is None:
        worker_ids = WORKER_IDS

    worker_index = {"A": 0, "B": 1, "C": 2}
    remaining = max(total_samples - int(total_samples_seen), 0)

    result: Dict[str, int] = {}
    for wid in worker_ids:
        idx = worker_index.get(wid, 0)
        n_parts = 3  # always 3-way partition for determinism, even if C is inactive
        base = remaining // n_parts
        remainder = remaining % n_parts
        width = base + (1 if idx < remainder else 0)
        result[wid] = max(int(width), 0)

    return result



def _determine_round_mode(
    projected_shards: Dict[str, int],
    credentialed_workers: List[str],
    min_shard_samples: int,
    force_worker_ids: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """
    Determine the coordination mode and which workers to trigger.

    Returns:
        mode: "complete" | "solo" | "diloco"
        active_workers: list of worker IDs to trigger
    """
    if force_worker_ids:
        # Manual override: trigger exactly the specified workers, no threshold check
        active = [w for w in force_worker_ids if w in credentialed_workers]
        if not active:
            return "complete", []
        mode = "solo" if len(active) == 1 else "diloco"
        return mode, active

    active = [
        w for w in credentialed_workers
        if projected_shards.get(w, 0) >= min_shard_samples
    ]

    if not active:
        return "complete", []
    if len(active) == 1:
        return "solo", active
    return "diloco", active


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only DiLoCo coordinator")
    parser.add_argument("--hf_token", required=True)
    parser.add_argument("--repo_id", default="WeirdRunner/Ouroboros")
    parser.add_argument(
        "--min_shard_samples",
        type=int,
        default=32,
        help=(
            "Minimum projected samples a worker must have to be triggered. "
            "Default 32 = one optimizer step (batch_size=4 × grad_accum=8). "
            "Workers below this threshold are skipped. "
            "If total remaining < min_shard_samples, stage is declared complete."
        ),
    )
    parser.add_argument(
        "--skip_trigger",
        action="store_true",
        help="Aggregate previous round only. Do not trigger next workers. "
             "For use when workers were started manually.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the round plan (projected shards, mode, active workers) "
             "without aggregating or triggering anything.",
    )
    parser.add_argument(
        "--force_worker_ids",
        default=None,
        help="Comma-separated worker IDs to force-trigger regardless of shard size, "
             "e.g. 'A,B'. Bypasses min_shard_samples check. "
             "Useful for quota-exhausted scenarios.",
    )
    parser.add_argument("--outer_lr", type=float, default=0.7)
    parser.add_argument(
        "--worker_timeout_hours",
        type=float,
        default=13.0,
        help=(
            "Hours after triggered_at before a non-responsive worker is demoted to "
            "attendance_workers. 13h = Kaggle 12h hard wall + 1h grace. Default 13.0."
        ),
    )
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
    parser.add_argument(
        "--kaggle_notebook_path",
        default=str(DEFAULT_KAGGLE_NOTEBOOK_PATH),
        help="Absolute or repo-relative path to the Kaggle notebook that should be pushed to auto-trigger workers.",
    )
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


def _build_kaggle_kernel_metadata(*, slug: str, notebook_filename: str) -> Dict[str, object]:
    title = slug.split("/", 1)[-1]
    return {
        "id": slug,
        "title": title,
        "code_file": notebook_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,          # ← ADD: pins T4, not just any GPU
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": ["weirdrunner007/ouroboros-cache"],  # ← ADD: attaches cache
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "keywords": [],
    }


def _stage_local_kaggle_kernel(notebook_path: Path, slug: str, staging_dir: Path) -> Path:
    if not notebook_path.exists():
        raise FileNotFoundError(
            f"Notebook source not found at {notebook_path}. "
            "Auto-trigger requires a local kaggle-utils.ipynb checkout."
        )

    staged_notebook = staging_dir / notebook_path.name
    shutil.copy2(notebook_path, staged_notebook)
    metadata_path = staging_dir / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps(
            _build_kaggle_kernel_metadata(slug=slug, notebook_filename=staged_notebook.name),
            indent=2,
        ),
        encoding="utf-8",
    )
    return staged_notebook


def _trigger_single_worker(
    worker_id: str,
    username: str,
    key: str,
    slug: str,
    notebook_path: Path,
) -> bool:
    """
    Trigger a Kaggle kernel by pushing the repo-tracked notebook with generated
    metadata, instead of pulling the live kernel back from Kaggle first.

    Why this path is safer:
      - the coordinator logs show `kaggle kernels pull` failing with
        `Permission 'kernels.get' was denied`, which blocks the old pull→push flow
        even when the worker has already completed successfully.
      - `kaggle kernels push` is the supported CLI path for updating and running a
        kernel; staging the checked-in notebook locally avoids the fragile readback
        permission entirely.
    """
    import os
    import subprocess
    import tempfile

    if not username or not key:
        print(f"[coordinator] No credentials for Worker {worker_id} — skipping trigger.")
        return False

    expected_owner, _ = WORKER_KAGGLE_SLUGS[worker_id]
    if username.strip().lower() != expected_owner.lower():
        print(
            f"[coordinator] WARNING: Worker {worker_id} expects Kaggle owner "
            f"{expected_owner}, but received username {username!r}. Trigger may fail."
        )

    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key
    env.pop("KAGGLE_CONFIG_DIR", None)

    def _run_kaggle(args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["kaggle"] + args,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _stage_local_kaggle_kernel(notebook_path, slug, tmp_path)

            push_args = ["kernels", "push", "-p", str(tmp_path)]
            push = _run_kaggle(push_args)
            if push.returncode != 0:
                err = (push.stderr or push.stdout or "").strip()
                print(f"[coordinator] WARNING: kernels push failed for Worker {worker_id} ({slug}): {err}")
                return False

        out = (push.stdout or push.stderr or "").strip()
        print(f"[coordinator] Triggered Worker {worker_id}: {slug}  ({out})")
        return True

    except subprocess.TimeoutExpired:
        print(f"[coordinator] WARNING: kaggle CLI timed out for Worker {worker_id} ({slug})")
        return False
    except FileNotFoundError as exc:
        print(f"[coordinator] WARNING: Auto-trigger prerequisites missing for {slug}: {exc}")
        return False
    except Exception as exc:
        print(f"[coordinator] WARNING: Failed to trigger {slug}: {exc}")
        return False


def trigger_kaggle_workers(
    kaggle_creds: Dict[str, Tuple[Optional[str], Optional[str]]],
    *,
    active_workers: List[str],
    notebook_path: Path,
) -> None:
    """
    Trigger only the specified active_workers using their Kaggle credentials.
    """
    for worker_id in active_workers:
        username, key = kaggle_creds.get(worker_id, (None, None))
        _, slug = WORKER_KAGGLE_SLUGS[worker_id]

        if not username or not key:
            print(
                f"[coordinator] No credentials for Worker {worker_id} ({slug}) - "
                "skipping automatic trigger. Start this worker manually."
            )
            continue

        _trigger_single_worker(
            worker_id,
            username,
            key,
            slug,
            notebook_path=notebook_path,
        )


def collect_ready_workers(
    repo_id: str,
    token: str,
    stage_k: int,
    round_n: int,
    expected_workers: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Collect workers whose status.json marks them done for this stage/round.
    Only checks expected_workers if provided (from round_state.triggered_workers).
    Falls back to checking all WORKER_IDS for backward compatibility.
    Workers with samples_seen=0 are included (empty-shard passthrough).
    """
    check_ids = expected_workers if expected_workers else WORKER_IDS
    ready: List[Dict] = []
    for worker_id in check_ids:
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
            samples = int(status.get("samples_seen", 0))
            print(f"[coordinator] Worker {worker_id}: {samples} samples ready")
        else:
            print(f"[coordinator] Worker {worker_id}: not ready (status={status})")
    ready.sort(key=lambda item: item.get("worker_id", ""))
    return ready


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
    # Workers that were triggered last round — only check these for ready status
    expected_workers = state.get("triggered_workers", None)
    seed = int(state.get("seed", 42))
    current_mode = state.get("mode", "diloco")
    triggered_at = float(state.get("triggered_at", 0.0))
    attendance_workers_prev = list(state.get("attendance_workers", []))
    worker_timeout_s = args.worker_timeout_hours * 3600.0
    is_round_timed_out = triggered_at > 0 and (time.time() - triggered_at) > worker_timeout_s

    kaggle_creds: Dict[str, Tuple[Optional[str], Optional[str]]] = {
        "A": (args.kaggle_username_a, args.kaggle_key_a),
        "B": (args.kaggle_username_b, args.kaggle_key_b),
        "C": (args.kaggle_username_c, args.kaggle_key_c),
    }
    credentialed = [w for w in WORKER_IDS if kaggle_creds[w][0] and kaggle_creds[w][1]]

    force_ids: Optional[List[str]] = None
    if args.force_worker_ids:
        force_ids = [w.strip().upper() for w in args.force_worker_ids.split(",") if w.strip()]

    print(f"[coordinator] stage={stage_k} round={round_n} mode={current_mode}")

    stage_samples_seen = int(total_samples_seen.get(str(stage_k), 0))
    projected_shards = _compute_projected_shards(
        total_samples=args.total_train_samples,
        stage_k=stage_k,
        round_n=round_n,
        seed=seed,
        total_samples_seen=stage_samples_seen,
    )
    remaining = max(args.total_train_samples - stage_samples_seen, 0)
    print(f"[coordinator] Remaining samples for stage {stage_k}: {remaining}")
    print(f"[coordinator] Projected shards: {projected_shards}")

    eligible_for_training_now = [w for w in credentialed if w not in attendance_workers_prev]
    next_mode, next_active_workers = _determine_round_mode(
        projected_shards=projected_shards,
        credentialed_workers=eligible_for_training_now,
        min_shard_samples=args.min_shard_samples,
        force_worker_ids=force_ids,
    )
    next_attendance_workers = list(attendance_workers_prev)
    print(f"[coordinator] Next round mode: {next_mode}  active workers: {next_active_workers}")

    if args.dry_run:
        print("[coordinator] DRY RUN — no aggregation or triggering.")
        print(f"  stage_k={stage_k} round_n={round_n}")
        print(f"  remaining={remaining} min_shard_samples={args.min_shard_samples}")
        print(f"  projected_shards={projected_shards}")
        print(f"  next_mode={next_mode} next_active_workers={next_active_workers}")
        print(f"  next_attendance_workers={next_attendance_workers}")
        print(f"  worker_timeout_hours={args.worker_timeout_hours}")
        return

    if current_mode == "waiting":
        responded_in_waiting = collect_ready_workers(
            args.repo_id,
            args.hf_token,
            stage_k,
            round_n,
            expected_workers=attendance_workers_prev,
        )
        responded_ids = {str(w.get("worker_id", "")) for w in responded_in_waiting}
        still_absent = [w for w in attendance_workers_prev if w not in responded_ids]

        if not responded_ids:
            if not is_round_timed_out:
                print("[coordinator] Waiting mode: no responses yet, standing by.")
                return
            print(f"[coordinator] Waiting mode: re-dispatching attendance to {attendance_workers_prev}")
            new_state = {**state, "triggered_at": time.time()}
            hub_upload_json(
                args.repo_id,
                ROUND_STATE_PATH,
                new_state,
                args.hf_token,
                message=f"Waiting mode: re-dispatch attendance round={round_n}",
            )
            if not args.skip_trigger and not args.dry_run:
                trigger_kaggle_workers(
                    kaggle_creds,
                    active_workers=attendance_workers_prev,
                    notebook_path=Path(args.kaggle_notebook_path),
                )
            print("[coordinator] Done (waiting mode re-dispatch).")
            return

        responded_list = sorted(responded_ids)
        print(f"[coordinator] Waiting mode exit: promoting {responded_list}")
        total_samples_seen[str(stage_k)] = stage_samples_seen
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
            next_attendance_workers = attendance_workers_prev

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
            "last_round_workers": responded_list,
            "last_round_samples": 0,
            "seed": seed,
        }
        hub_upload_json(
            args.repo_id,
            ROUND_STATE_PATH,
            new_state,
            args.hf_token,
            message=f"Waiting mode resolved: stage={next_stage_k} round={next_round_n} mode={next_mode}",
        )
        print(f"[coordinator] round_state updated: stage={next_stage_k} round={next_round_n} mode={next_mode}")
        if not args.skip_trigger and next_active_workers:
            trigger_kaggle_workers(
                kaggle_creds,
                active_workers=next_active_workers + next_attendance_workers,
                notebook_path=Path(args.kaggle_notebook_path),
            )
        print("[coordinator] Done (waiting mode resolved).")
        return

    # ── W&B init ─────────────────────────────────────────────────────────────
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
                        "min_shard_samples": args.min_shard_samples,
                        "total_train": args.total_train_samples,
                    },
                    mode="online",
                )
            except Exception as _we:
                print(f"[coordinator] W&B init failed: {_we}")

        # ── Collect ready workers from previous round ────────────────────────
        ready_workers = collect_ready_workers(
            args.repo_id, args.hf_token, stage_k, round_n,
            expected_workers=expected_workers,
        )

        if expected_workers:
            ready_ids = {str(w.get("worker_id", "")) for w in ready_workers}
            missing_workers = [w for w in expected_workers if w not in ready_ids]
            if missing_workers:
                if not is_round_timed_out:
                    print(f"[coordinator] Waiting for workers to finish this round: {missing_workers}")
                    return
                newly_demoted = [w for w in missing_workers if w not in attendance_workers_prev]
                still_absent = [w for w in missing_workers if w in attendance_workers_prev]
                if newly_demoted:
                    print(
                        f"[coordinator] Timed out (>{args.worker_timeout_hours}h): {newly_demoted} — demoting to attendance"
                    )
                if still_absent:
                    print(f"[coordinator] Still absent after attendance: {still_absent} — retrying")

        # Filter to workers that actually did work (samples_seen > 0) for aggregation
        contributing_workers = [w for w in ready_workers if int(w.get("samples_seen", 0)) > 0]

        if not contributing_workers:
            # No work was done this round (can happen on very first coordinator run
            # before any workers have trained, or on a stage advance)
            print("[coordinator] No contributing workers found. Proceeding to trigger planning.")
        else:
            # ── Aggregate ───────────────────────────────────────────────────
            mode_this_round = state.get("mode", "diloco")
            print("[coordinator] Loading anchor weights...")
            anchor_weights = load_adapter_weights_cpu(args.repo_id, ANCHOR_PREFIX, args.hf_token)
            anchor_adapter_config = json.loads(
                hub_download_text(args.repo_id, f"{ANCHOR_PREFIX}/adapter_config.json", args.hf_token)
            )
            print("[coordinator] Loading worker weights...")
            worker_weights_list = []
            worker_samples_list = []
            for status in contributing_workers:
                worker_weights_list.append(
                    load_adapter_weights_cpu(args.repo_id, status["weights_path"], args.hf_token)
                )
                worker_samples_list.append(int(status["samples_seen"]))

            print("[coordinator] Aggregating on CPU...")
            if len(contributing_workers) == 1 or mode_this_round == "solo":
                print(f"[coordinator] Solo mode: promoting Worker {contributing_workers[0]['worker_id']} weights directly.")
                new_anchor = worker_weights_list[0]
            else:
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
                    f"({len(contributing_workers)} workers, {sum(worker_samples_list)} samples, mode={mode_this_round})"
                ),
            )

            # ── Update stage sample counts ───────────────────────────────────
            stage_key = str(stage_k)
            current_stage_samples = stage_samples_seen + sum(worker_samples_list)
            total_samples_seen[stage_key] = current_stage_samples
            print(
                f"[coordinator] Stage {stage_k} progress: "
                f"{current_stage_samples}/{args.total_train_samples} samples seen"
            )

            if coordinator_wandb_run is not None:
                import wandb
                wandb.log(
                    {
                        "coordinator/round": round_n,
                        "coordinator/workers_aggregated": len(contributing_workers),
                        "coordinator/samples_this_round": sum(worker_samples_list),
                        "coordinator/total_samples_stage": current_stage_samples,
                        "coordinator/mode": mode_this_round,
                        "coordinator/pct_stage_done": round(
                            current_stage_samples / max(args.total_train_samples, 1) * 100, 1
                        ),
                    },
                    step=round_n,
                )

        ready_ids = {str(w.get("worker_id", "")) for w in ready_workers}

        attendance_promoted = [w for w in attendance_workers_prev if w in ready_ids]
        if attendance_promoted:
            print(f"[coordinator] Attendance workers responded, promoting next round: {attendance_promoted}")

        newly_demoted = [
            w for w in (expected_workers or [])
            if w not in ready_ids and w not in attendance_workers_prev
        ] if is_round_timed_out else []

        still_attending = [w for w in attendance_workers_prev if w not in ready_ids]
        next_attendance_workers = [
            w for w in WORKER_IDS if w in set(newly_demoted + still_attending)
        ]

        # ── Re-check remaining after aggregation ─────────────────────────────
        final_stage_samples = int(total_samples_seen.get(str(stage_k), stage_samples_seen))
        remaining_after = max(args.total_train_samples - final_stage_samples, 0)

        # Recompute mode for next round based on updated sample counts
        projected_shards_next = _compute_projected_shards(
            total_samples=args.total_train_samples,
            stage_k=stage_k,
            round_n=round_n + 1,
            seed=seed,
            total_samples_seen=final_stage_samples,
        )
        completion_mode, _ = _determine_round_mode(
            projected_shards=projected_shards_next,
            credentialed_workers=credentialed,
            min_shard_samples=args.min_shard_samples,
            force_worker_ids=force_ids,
        )
        eligible_for_training = [w for w in credentialed if w not in next_attendance_workers]
        next_mode, next_active_workers = _determine_round_mode(
            projected_shards=projected_shards_next,
            credentialed_workers=eligible_for_training,
            min_shard_samples=args.min_shard_samples,
            force_worker_ids=force_ids,
        )

        # ── Stage advance check ───────────────────────────────────────────────
        stage_complete = (
            final_stage_samples >= args.total_train_samples
            or completion_mode == "complete"
        )
        next_stage_k = stage_k
        next_round_n = round_n + 1
        if stage_complete:
            print(
                f"[coordinator] Stage {stage_k} COMPLETE ({final_stage_samples}/{args.total_train_samples} samples). Advancing to stage {stage_k + 1}."
            )
            if stage_k not in completed_stages:
                completed_stages.append(stage_k)
            completed_stages = sorted(set(completed_stages))
            next_stage_k = stage_k + 1
            next_round_n = 0
            projected_shards_next = _compute_projected_shards(
                total_samples=args.total_train_samples,
                stage_k=next_stage_k,
                round_n=0,
                seed=seed,
                total_samples_seen=0,
            )
            eligible_for_training = [w for w in credentialed if w not in next_attendance_workers]
            next_mode, next_active_workers = _determine_round_mode(
                projected_shards=projected_shards_next,
                credentialed_workers=eligible_for_training,
                min_shard_samples=args.min_shard_samples,
                force_worker_ids=force_ids,
            )

            if coordinator_wandb_run is not None:
                import wandb
                wandb.log({"coordinator/stage_complete": 1}, step=round_n)

        if not stage_complete and not next_active_workers and next_attendance_workers:
            print("[coordinator] All workers absent — entering waiting mode. Coordinator idles until workers signal presence.")
            next_mode = "waiting"
            next_round_n = round_n
            next_stage_k = stage_k

        # ── Write round_state.json ────────────────────────────────────────────
        new_state = {
            "stage_k": next_stage_k,
            "round_n": next_round_n,
            "mode": next_mode,
            "triggered_workers": next_active_workers,
            "attendance_workers": next_attendance_workers,
            "projected_shards": projected_shards_next,
            "anchor_path": ANCHOR_PREFIX,
            "total_samples_seen": total_samples_seen,
            "completed_stages": completed_stages,
            "last_updated": time.time(),
            "triggered_at": time.time() if (next_active_workers or next_attendance_workers) else 0.0,
            "last_round_workers": [w["worker_id"] for w in contributing_workers] if contributing_workers else [],
            "last_round_samples": sum(w.get("samples_seen", 0) for w in contributing_workers) if contributing_workers else 0,
            "seed": seed,
        }
        hub_upload_json(
            args.repo_id,
            ROUND_STATE_PATH,
            new_state,
            args.hf_token,
            message=f"Round state: stage={next_stage_k} round={next_round_n} mode={next_mode}",
        )
        print(f"[coordinator] round_state.json updated: stage={next_stage_k} round={next_round_n} mode={next_mode}")

        # ── Trigger next workers ──────────────────────────────────────────────
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
