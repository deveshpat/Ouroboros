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
import base64
import json
import os
import shutil
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import requests

from ouroboros.diloco.aggregation import (
    ANCHOR_PREFIX,
    aggregate_worker_updates,
    load_adapter_weights_cpu,
    save_and_upload_anchor,
    weighted_average_deltas,
)
from ouroboros.diloco.dispatch import (
    WORKER_KAGGLE_SLUGS,
    _first_nonempty_text,
    _build_kaggle_kernel_metadata,
    _build_worker_dispatch_cell,
    _build_worker_runtime_env,
    _encode_runtime_env_payload,
    _stage_local_kaggle_kernel,
    _trigger_single_worker,
    trigger_kaggle_workers,
)
from ouroboros.diloco.state import (
    WORKER_IDS,
    _compute_projected_shards,
    _determine_round_mode,
    _mode_from_active_workers,
    _ordered_unique_worker_ids,
    _partition_ready_workers,
    _reconcile_post_dispatch_state,
)
from ouroboros.workflow_validation import CPU_SMOKE_MODE, workflow_validation_remote_paths


T = TypeVar("T")

ROUND_STATE_PATH = "diloco_state/round_state.json"
DEFAULT_KAGGLE_NOTEBOOK_PATH = Path(__file__).resolve().parents[2] / "kaggle-utils.ipynb"
DEFAULT_IO_RETRIES = 3
DEFAULT_IO_RETRY_BASE_DELAY_S = 1.5
DILOCO_TERMINAL_STAGE = 10
DILOCO_RUN_MODE = "diloco"
DGAC_ANCHOR_EVAL_RUN_MODE = "dgac-anchor-eval"
DGAC_TRAIN_RUN_MODE = "dgac-train"


def _retry_io(
    label: str,
    fn: Callable[[], T],
    *,
    attempts: int = DEFAULT_IO_RETRIES,
    base_delay_s: float = DEFAULT_IO_RETRY_BASE_DELAY_S,
    swallow: bool = False,
    default: Optional[T] = None,
) -> Optional[T]:
    """Retry transient coordinator I/O with exponential backoff."""
    last_exc: Optional[Exception] = None
    attempts = max(int(attempts), 1)
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - coordinator must keep going on transient I/O errors
            last_exc = exc
            if attempt >= attempts:
                if swallow:
                    print(
                        f"[coordinator] {label} failed after {attempts} attempts: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    return default
                raise
            delay = base_delay_s * (2 ** (attempt - 1))
            print(
                f"[coordinator] {label} failed (attempt {attempt}/{attempts}): "
                f"{type(exc).__name__}: {exc}. Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
    if swallow:
        return default
    assert last_exc is not None
    raise last_exc




























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
        "--workflow_validate",
        default=None,
        help=(
            "Optional workflow validation mode. Use 'cpu-smoke' to run a read-only "
            "GitHub Actions -> Kaggle -> Hub validation instead of mutating round_state."
        ),
    )
    parser.add_argument(
        "--workflow_validation_run_id",
        default=None,
        help="Stable id for remote workflow validation artifacts. Defaults to GitHub run id when available.",
    )
    parser.add_argument(
        "--workflow_validation_timeout_s",
        type=float,
        default=900.0,
        help="Seconds to wait for remote CPU-smoke status artifacts after Kaggle dispatch.",
    )
    parser.add_argument(
        "--workflow_validation_poll_s",
        type=float,
        default=10.0,
        help="Seconds between remote CPU-smoke status checks.",
    )
    parser.add_argument(
        "--worker_timeout_hours",
        type=float,
        default=13.0,
        help=(
            "Hours after triggered_at before a non-responsive worker is demoted to "
            "attendance_workers. 13h = Kaggle 12h hard wall + 1h grace. Default 13.0."
        ),
    )
    parser.add_argument(
        "--attendance_join_grace_minutes",
        type=float,
        default=5.0,
        help=(
            "Minutes to wait in waiting mode after the first attendance response "
            "before promoting a partial attendance set. Default 5.0."
        ),
    )
    parser.add_argument(
        "--kaggle_run_mode",
        default=os.environ.get("OUROBOROS_KAGGLE_RUN_MODE", DILOCO_RUN_MODE),
        choices=[DILOCO_RUN_MODE, DGAC_ANCHOR_EVAL_RUN_MODE, DGAC_TRAIN_RUN_MODE],
        help=(
            "Kaggle notebook launch mode. Use 'dgac-anchor-eval' to push one "
            "GPU eval-only notebook for the terminal DiLoCo anchor; use "
            "'dgac-train' to launch Phase 3.4 DGAC from the terminal anchor. "
            "Both modes skip DiLoCo round_state reads/mutations."
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

    def _download() -> Dict:
        local = hf_hub_download(repo_id=repo_id, filename=path, token=token)
        with open(local, encoding="utf-8") as f:
            return json.load(f)

    return _retry_io(
        f"Download JSON {path}",
        _download,
        swallow=True,
        default=None,
    )



def hub_upload_json(repo_id: str, path: str, data: Dict, token: str, message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(data, tf, indent=2)
        tmp = tf.name
    try:
        _retry_io(
            f"Upload JSON {path}",
            lambda: api.upload_file(
                path_or_fileobj=tmp,
                path_in_repo=path,
                repo_id=repo_id,
                token=token,
                commit_message=message,
            ),
        )
    finally:
        Path(tmp).unlink(missing_ok=True)



def hub_download_text(repo_id: str, path: str, token: str) -> str:
    from huggingface_hub import hf_hub_download

    def _download() -> str:
        local = hf_hub_download(repo_id=repo_id, filename=path, token=token)
        return Path(local).read_text(encoding="utf-8")

    result = _retry_io(f"Download text {path}", _download)
    assert result is not None
    return result


























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


def _positive_ready_worker_ids(statuses: List[Dict]) -> set[str]:
    """Workers with useful current-round training output."""
    return {
        str(status.get("worker_id", "")).upper()
        for status in statuses
        if int(status.get("samples_seen", 0)) > 0
    }


def _terminal_stage_state(
    *,
    state: Dict[str, Any],
    stage_k: int,
    completed_stages: List[int],
    total_samples_seen: Dict[str, int],
    seed: int,
    last_round_workers: List[str],
    last_round_samples: int,
) -> Dict[str, Any]:
    completed = sorted(set([int(stage) for stage in completed_stages] + [stage_k]))
    return {
        **state,
        "stage_k": stage_k,
        "round_n": 0,
        "mode": "terminal",
        "triggered_workers": [],
        "attendance_workers": [],
        "projected_shards": {},
        "anchor_path": ANCHOR_PREFIX,
        "total_samples_seen": total_samples_seen,
        "completed_stages": completed,
        "dgac_manual_gate": True,
        "last_updated": time.time(),
        "triggered_at": 0.0,
        "dispatch_failures": [],
        "last_round_workers": last_round_workers,
        "last_round_samples": last_round_samples,
        "seed": seed,
    }


def _print_dgac_manual_gate_message(stage_k: int) -> None:
    print(
        f"[coordinator] Stage {stage_k} is terminal for DiLoCo. "
        "DGAC is ready for manual quality review; no stage-11 DiLoCo dispatch will run."
    )
    print(
        "[coordinator] DGAC manual gate: review final stage-10 anchor, run CPU-smoke if needed, "
        "then launch DGAC explicitly."
    )



def _workflow_validation_mode_from_args(args: argparse.Namespace) -> Optional[str]:
    mode = _first_nonempty_text(
        getattr(args, "workflow_validate", None),
        os.environ.get("OUROBOROS_WORKFLOW_VALIDATE"),
    )
    return mode.lower() if mode else None


def _workflow_validation_run_id_from_args(args: argparse.Namespace) -> str:
    explicit = _first_nonempty_text(
        getattr(args, "workflow_validation_run_id", None),
        os.environ.get("OUROBOROS_WORKFLOW_VALIDATION_RUN_ID"),
    )
    if explicit:
        return explicit
    github_run = _first_nonempty_text(os.environ.get("GITHUB_RUN_ID"))
    github_attempt = _first_nonempty_text(os.environ.get("GITHUB_RUN_ATTEMPT"))
    if github_run and github_attempt:
        return f"{github_run}-{github_attempt}"
    if github_run:
        return github_run
    return f"local-{int(time.time())}"


def _build_kaggle_creds(args: argparse.Namespace) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    return {
        "A": (args.kaggle_username_a, args.kaggle_key_a),
        "B": (args.kaggle_username_b, args.kaggle_key_b),
        "C": (args.kaggle_username_c, args.kaggle_key_c),
    }


def _workflow_validation_worker_ids(args: argparse.Namespace) -> List[str]:
    if args.force_worker_ids:
        requested = _ordered_unique_worker_ids(
            [w.strip().upper() for w in str(args.force_worker_ids).split(",") if w.strip()]
        )
        if requested:
            return requested
    # CPU smoke should be a one-click validation path. Defaulting to Worker A keeps
    # it deterministic and avoids accidentally pushing three notebooks when the
    # operator only wanted to prove the end-to-end plumbing.
    return ["A"]


def _kaggle_eval_worker_ids(args: argparse.Namespace) -> List[str]:
    if args.force_worker_ids:
        requested = _ordered_unique_worker_ids(
            [w.strip().upper() for w in str(args.force_worker_ids).split(",") if w.strip()]
        )
        if requested:
            return requested
    # One anchor eval is sufficient; default to Worker A for a deterministic one-click path.
    return ["A"]


def _kaggle_dgac_worker_ids(args: argparse.Namespace) -> List[str]:
    if args.force_worker_ids:
        requested = _ordered_unique_worker_ids(
            [w.strip().upper() for w in str(args.force_worker_ids).split(",") if w.strip()]
        )
        if requested:
            # DGAC is a single training job, not a DiLoCo quorum. Honor the first
            # requested worker only to avoid duplicate jobs racing Hub checkpoint writes.
            return [requested[0]]
    # DGAC is a single training job, not a DiLoCo quorum. Default to Worker A.
    return ["A"]


def run_kaggle_anchor_eval(args: argparse.Namespace) -> None:
    worker_ids = _kaggle_eval_worker_ids(args)
    kaggle_creds = _build_kaggle_creds(args)
    print(
        "[kaggle-eval] Dispatching DGAC anchor eval-only notebook "
        f"workers={worker_ids} repo={args.repo_id}"
    )
    if args.dry_run:
        print("[kaggle-eval] DRY RUN — no Kaggle dispatch.")
        return

    dispatch_results = trigger_kaggle_workers(
        kaggle_creds,
        active_workers=worker_ids,
        notebook_path=Path(args.kaggle_notebook_path),
        coordinator_args=args,
    )
    if any(dispatch_results.get(worker_id) != "success" for worker_id in worker_ids):
        raise RuntimeError(
            "DGAC anchor eval-only dispatch failed: "
            f"{dispatch_results}"
        )
    print(
        "[kaggle-eval] Dispatch accepted by Kaggle. Monitor the Kaggle kernel "
        "and W&B eval_only/* metrics; this mode does not mutate round_state."
    )


def run_kaggle_dgac_train(args: argparse.Namespace) -> None:
    worker_ids = _kaggle_dgac_worker_ids(args)
    kaggle_creds = _build_kaggle_creds(args)
    print(
        "[kaggle-dgac] Dispatching Phase 3.4 DGAC training notebook "
        f"workers={worker_ids} repo={args.repo_id}"
    )
    if args.dry_run:
        print("[kaggle-dgac] DRY RUN — no Kaggle dispatch.")
        return

    dispatch_results = trigger_kaggle_workers(
        kaggle_creds,
        active_workers=worker_ids,
        notebook_path=Path(args.kaggle_notebook_path),
        coordinator_args=args,
    )
    if any(dispatch_results.get(worker_id) != "success" for worker_id in worker_ids):
        raise RuntimeError(
            "DGAC training dispatch failed: "
            f"{dispatch_results}"
        )
    print(
        "[kaggle-dgac] Dispatch accepted by Kaggle. Monitor the Kaggle kernel, "
        "W&B train/val/gen metrics, and Hub runs/stage3_dgac checkpoints; "
        "this mode does not mutate DiLoCo round_state."
    )


def _is_matching_cpu_smoke_status(status: Optional[Dict], *, worker_id: str, run_id: str) -> bool:
    if not isinstance(status, dict):
        return False
    return (
        str(status.get("worker_id", "")).upper() == worker_id
        and str(status.get("validation_mode", "")).lower() == CPU_SMOKE_MODE
        and str(status.get("validation_run_id", "")) == run_id
        and status.get("status") == "done"
    )


def _poll_cpu_smoke_validation_status(
    *,
    repo_id: str,
    token: str,
    worker_ids: List[str],
    run_id: str,
    timeout_s: float,
    poll_s: float,
) -> Dict[str, Dict]:
    deadline = time.time() + max(float(timeout_s), 0.0)
    poll_s = max(float(poll_s), 0.1)
    latest: Dict[str, Optional[Dict]] = {}
    while True:
        ready: Dict[str, Dict] = {}
        for worker_id in worker_ids:
            status_path, _ = workflow_validation_remote_paths(run_id=run_id, worker_id=worker_id)
            status = hub_download_json(repo_id, status_path, token)
            latest[worker_id] = status
            if _is_matching_cpu_smoke_status(status, worker_id=worker_id, run_id=run_id):
                ready[worker_id] = status
            else:
                print(f"[workflow-validate] Waiting for Worker {worker_id} status at {status_path}: {status}")
        if len(ready) == len(worker_ids):
            return ready
        if time.time() >= deadline:
            raise TimeoutError(
                "Timed out waiting for CPU-smoke validation status artifacts: "
                f"run_id={run_id} latest={latest}"
            )
        time.sleep(poll_s)


def run_workflow_validation(args: argparse.Namespace) -> None:
    mode = _workflow_validation_mode_from_args(args)
    if mode != CPU_SMOKE_MODE:
        raise ValueError(f"Unsupported workflow validation mode: {mode!r}")

    worker_ids = _workflow_validation_worker_ids(args)
    run_id = _workflow_validation_run_id_from_args(args)
    kaggle_creds = _build_kaggle_creds(args)
    print(
        "[workflow-validate] Starting CPU-smoke validation "
        f"run_id={run_id} workers={worker_ids} repo={args.repo_id}"
    )
    if args.dry_run:
        print("[workflow-validate] DRY RUN — no Kaggle dispatch or polling.")
        return

    dispatch_results = trigger_kaggle_workers(
        kaggle_creds,
        active_workers=worker_ids,
        notebook_path=Path(args.kaggle_notebook_path),
        coordinator_args=args,
    )
    if any(dispatch_results.get(worker_id) != "success" for worker_id in worker_ids):
        raise RuntimeError(
            "CPU-smoke validation dispatch failed; refusing to pass end-to-end gate: "
            f"{dispatch_results}"
        )

    ready = _poll_cpu_smoke_validation_status(
        repo_id=args.repo_id,
        token=args.hf_token,
        worker_ids=worker_ids,
        run_id=run_id,
        timeout_s=args.workflow_validation_timeout_s,
        poll_s=args.workflow_validation_poll_s,
    )
    print(
        "[workflow-validate] CPU-smoke validation verified via remote Hub artifacts: "
        f"{sorted(ready)}"
    )


def main() -> None:
    args = parse_args()

    if _workflow_validation_mode_from_args(args):
        run_workflow_validation(args)
        return

    kaggle_run_mode = getattr(args, "kaggle_run_mode", DILOCO_RUN_MODE)
    if kaggle_run_mode == DGAC_ANCHOR_EVAL_RUN_MODE:
        run_kaggle_anchor_eval(args)
        return
    if kaggle_run_mode == DGAC_TRAIN_RUN_MODE:
        run_kaggle_dgac_train(args)
        return

    print("[coordinator] Reading round state...")
    state = hub_download_json(args.repo_id, ROUND_STATE_PATH, args.hf_token)
    if state is None:
        print("[coordinator] No round_state.json found. Nothing to aggregate.")
        return

    stage_k = int(state.get("stage_k", 0))
    round_n = int(state.get("round_n", 0))
    total_samples_seen = {str(k): int(v) for k, v in dict(state.get("total_samples_seen", {})).items()}
    completed_stages = [int(x) for x in state.get("completed_stages", [])]
    # Workers that were triggered last round — only check these for ready status.
    # Sanitize/normalize to defend against duplicated, lowercase, or overlapping IDs.
    expected_workers = _ordered_unique_worker_ids(state.get("triggered_workers"))
    seed = int(state.get("seed", 42))
    current_mode = state.get("mode", "diloco")
    triggered_at = float(state.get("triggered_at", 0.0))

    if current_mode == "terminal" or bool(state.get("dgac_manual_gate")):
        print(f"[coordinator] stage={stage_k} round={round_n} mode={current_mode}")
        _print_dgac_manual_gate_message(min(stage_k, DILOCO_TERMINAL_STAGE))
        return

    attendance_workers_prev = [
        w for w in _ordered_unique_worker_ids(state.get("attendance_workers"))
        if w not in set(expected_workers)
    ]
    worker_timeout_s = args.worker_timeout_hours * 3600.0
    is_round_timed_out = triggered_at > 0 and (time.time() - triggered_at) > worker_timeout_s

    kaggle_creds = _build_kaggle_creds(args)
    credentialed = [w for w in WORKER_IDS if kaggle_creds[w][0] and kaggle_creds[w][1]]

    force_ids: Optional[List[str]] = None
    if args.force_worker_ids:
        force_ids = _ordered_unique_worker_ids(
            [w.strip().upper() for w in args.force_worker_ids.split(",") if w.strip()]
        ) or None

    print(f"[coordinator] stage={stage_k} round={round_n} mode={current_mode}")
    if attendance_workers_prev:
        print(f"[coordinator] Attendance workers: {attendance_workers_prev}")

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
        responded_ids = {str(w.get("worker_id", "")).upper() for w in responded_in_waiting}
        still_absent = [w for w in attendance_workers_prev if w not in responded_ids]

        if not responded_ids:
            if triggered_at <= 0:
                print(
                    "[coordinator] Waiting mode: no confirmed dispatch timestamp yet; "
                    "attempting attendance dispatch now."
                )
                new_state = {
                    **state,
                    "triggered_at": time.time(),
                    "last_updated": time.time(),
                    "dispatch_failures": [],
                }
                hub_upload_json(
                    args.repo_id,
                    ROUND_STATE_PATH,
                    new_state,
                    args.hf_token,
                    message=f"Waiting mode: initial attendance dispatch round={round_n}",
                )
                if not args.skip_trigger and not args.dry_run:
                    dispatch_results = trigger_kaggle_workers(
                        kaggle_creds,
                        active_workers=attendance_workers_prev,
                        notebook_path=Path(args.kaggle_notebook_path),
                        coordinator_args=args,
                    )
                    corrected_state = _reconcile_post_dispatch_state(
                        state=new_state,
                        planned_active_workers=[],
                        planned_attendance_workers=attendance_workers_prev,
                        dispatch_results=dispatch_results,
                    )
                    if corrected_state is not None:
                        hub_upload_json(
                            args.repo_id,
                            ROUND_STATE_PATH,
                            corrected_state,
                            args.hf_token,
                            message=(
                                f"Waiting mode dispatch reconcile: stage={stage_k} round={round_n} "
                                f"attendance={corrected_state['attendance_workers']}"
                            ),
                        )
                print("[coordinator] Done (waiting mode initial dispatch).")
                return
            if not is_round_timed_out:
                print("[coordinator] Waiting mode: no responses yet, standing by.")
                return
            print(f"[coordinator] Waiting mode: re-dispatching attendance to {attendance_workers_prev}")
            new_state = {
                **state,
                "triggered_at": time.time(),
                "last_updated": time.time(),
                "dispatch_failures": [],
            }
            hub_upload_json(
                args.repo_id,
                ROUND_STATE_PATH,
                new_state,
                args.hf_token,
                message=f"Waiting mode: re-dispatch attendance round={round_n}",
            )
            if not args.skip_trigger and not args.dry_run:
                dispatch_results = trigger_kaggle_workers(
                    kaggle_creds,
                    active_workers=attendance_workers_prev,
                    notebook_path=Path(args.kaggle_notebook_path),
                    coordinator_args=args,
                )
                corrected_state = _reconcile_post_dispatch_state(
                    state=new_state,
                    planned_active_workers=[],
                    planned_attendance_workers=attendance_workers_prev,
                    dispatch_results=dispatch_results,
                )
                if corrected_state is not None:
                    hub_upload_json(
                        args.repo_id,
                        ROUND_STATE_PATH,
                        corrected_state,
                        args.hf_token,
                        message=(
                            f"Waiting mode re-dispatch reconcile: round={round_n} "
                            f"attendance={corrected_state['attendance_workers']}"
                        ),
                    )
            print("[coordinator] Done (waiting mode re-dispatch).")
            return

        attendance_grace_s = max(float(args.attendance_join_grace_minutes), 0.0) * 60.0
        waiting_elapsed_s = (time.time() - triggered_at) if triggered_at > 0 else 0.0
        all_attendance_responded = not still_absent
        grace_expired = triggered_at > 0 and waiting_elapsed_s >= attendance_grace_s
        if not all_attendance_responded and not grace_expired:
            print(
                "[coordinator] Waiting mode: attendance responders received "
                f"{sorted(responded_ids)}, still waiting for {still_absent} "
                f"within {args.attendance_join_grace_minutes:g}m join grace."
            )
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
            "dispatch_failures": [],
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
            dispatch_results = trigger_kaggle_workers(
                kaggle_creds,
                active_workers=next_active_workers + next_attendance_workers,
                notebook_path=Path(args.kaggle_notebook_path),
                coordinator_args=args,
            )
            corrected_state = _reconcile_post_dispatch_state(
                state=new_state,
                planned_active_workers=next_active_workers,
                planned_attendance_workers=next_attendance_workers,
                dispatch_results=dispatch_results,
            )
            if corrected_state is not None:
                hub_upload_json(
                    args.repo_id,
                    ROUND_STATE_PATH,
                    corrected_state,
                    args.hf_token,
                    message=(
                        f"Waiting mode resolved dispatch reconcile: stage={next_stage_k} "
                        f"round={next_round_n} mode={corrected_state['mode']}"
                    ),
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
        workers_to_check = _ordered_unique_worker_ids(expected_workers, attendance_workers_prev, force_ids)
        ready_statuses = collect_ready_workers(
            args.repo_id,
            args.hf_token,
            stage_k,
            round_n,
            expected_workers=workers_to_check if workers_to_check else None,
        )
        ready_workers, attendance_ready_workers = _partition_ready_workers(
            ready_statuses,
            expected_workers=expected_workers,
            attendance_workers=attendance_workers_prev,
        )
        ready_ids = _positive_ready_worker_ids(ready_workers)
        ready_or_attendance_ids = ready_ids | _positive_ready_worker_ids(attendance_ready_workers)
        attendance_ready_ids = {str(w.get("worker_id", "")) for w in attendance_ready_workers}
        zero_sample_active = sorted(
            str(w.get("worker_id", "")).upper()
            for w in ready_workers
            if int(w.get("samples_seen", 0)) <= 0
        )
        if zero_sample_active:
            print(
                "[coordinator] Ignoring zero-sample active completions as training output: "
                f"{zero_sample_active}"
            )
        if attendance_ready_ids:
            print(
                f"[coordinator] Attendance responses received: {sorted(attendance_ready_ids)}"
            )

        if expected_workers:
            missing_workers = [w for w in expected_workers if w not in ready_ids]
            if missing_workers:
                force_dispatch_workers: List[str] = []
                force_already_done: List[str] = []
                force_unavailable: List[str] = []
                if force_ids:
                    for worker_id in force_ids:
                        if worker_id in expected_workers:
                            continue
                        if worker_id in ready_or_attendance_ids:
                            force_already_done.append(worker_id)
                            continue
                        if worker_id not in credentialed:
                            force_unavailable.append(worker_id)
                            continue
                        force_dispatch_workers.append(worker_id)

                if force_unavailable:
                    print(
                        "[coordinator] Force repair skipped unavailable workers: "
                        f"{force_unavailable}"
                    )

                force_added_workers = _ordered_unique_worker_ids(
                    force_already_done,
                    force_dispatch_workers,
                )
                if force_added_workers:
                    repaired_active_workers = _ordered_unique_worker_ids(
                        expected_workers,
                        force_added_workers,
                    )
                    repaired_attendance_workers = [
                        w for w in attendance_workers_prev
                        if w not in set(repaired_active_workers)
                    ]
                    repaired_state = {
                        **state,
                        "mode": _mode_from_active_workers(repaired_active_workers),
                        "triggered_workers": repaired_active_workers,
                        "attendance_workers": repaired_attendance_workers,
                        "last_updated": time.time(),
                        "triggered_at": time.time(),
                        "dispatch_failures": [],
                    }
                    hub_upload_json(
                        args.repo_id,
                        ROUND_STATE_PATH,
                        repaired_state,
                        args.hf_token,
                        message=(
                            f"Force repair dispatch: stage={stage_k} round={round_n} "
                            f"active={repaired_active_workers}"
                        ),
                    )
                    if force_already_done:
                        print(
                            "[coordinator] Force repair counted already-done workers without re-dispatch: "
                            f"{force_already_done}"
                        )
                    if force_dispatch_workers:
                        print(
                            "[coordinator] Force repair dispatching missing workers: "
                            f"{force_dispatch_workers}; preserving active workers: {expected_workers}"
                        )
                    if force_dispatch_workers and not args.skip_trigger:
                        dispatch_results = trigger_kaggle_workers(
                            kaggle_creds,
                            active_workers=force_dispatch_workers,
                            notebook_path=Path(args.kaggle_notebook_path),
                            coordinator_args=args,
                        )
                        corrected_state = _reconcile_post_dispatch_state(
                            state=repaired_state,
                            planned_active_workers=repaired_active_workers,
                            planned_attendance_workers=repaired_attendance_workers,
                            dispatch_results=dispatch_results,
                        )
                        if corrected_state is not None:
                            hub_upload_json(
                                args.repo_id,
                                ROUND_STATE_PATH,
                                corrected_state,
                                args.hf_token,
                                message=(
                                    f"Force repair dispatch reconcile: stage={stage_k} "
                                    f"round={round_n} mode={corrected_state['mode']}"
                                ),
                            )
                    print(f"[coordinator] Done (force repair round {round_n}).")
                    return

                if triggered_at <= 0:
                    # triggered_at=0 means this round's dispatch was never confirmed —
                    # the Kaggle push may have failed silently, or the state was manually reset.
                    # Re-dispatch immediately instead of waiting up to 13h for the timeout.
                    print(
                        f"[coordinator] Round {round_n}: {missing_workers} marked triggered but "
                        f"triggered_at=0 (unconfirmed dispatch). Re-dispatching now."
                    )
                    new_state = {
                        **state,
                        "triggered_at": time.time(),
                        "last_updated": time.time(),
                        "dispatch_failures": [],
                    }
                    hub_upload_json(
                        args.repo_id,
                        ROUND_STATE_PATH,
                        new_state,
                        args.hf_token,
                        message=f"Re-dispatch unconfirmed: stage={stage_k} round={round_n}",
                    )
                    if not args.skip_trigger:
                        all_to_trigger = expected_workers + [
                            w for w in attendance_workers_prev if w not in expected_workers
                        ]
                        dispatch_results = trigger_kaggle_workers(
                            kaggle_creds,
                            active_workers=all_to_trigger,
                            notebook_path=Path(args.kaggle_notebook_path),
                            coordinator_args=args,
                        )
                        post_corrected = _reconcile_post_dispatch_state(
                            state=new_state,
                            planned_active_workers=expected_workers,
                            planned_attendance_workers=attendance_workers_prev,
                            dispatch_results=dispatch_results,
                        )
                        if post_corrected is not None:
                            hub_upload_json(
                                args.repo_id,
                                ROUND_STATE_PATH,
                                post_corrected,
                                args.hf_token,
                                message=f"Re-dispatch reconcile: stage={stage_k} round={round_n}",
                            )
                    print(f"[coordinator] Done (re-dispatch unconfirmed round {round_n}).")
                    return
                elif not is_round_timed_out:
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
            new_anchor = aggregate_worker_updates(
                anchor_weights,
                worker_weights_list,
                worker_samples_list,
                args.outer_lr,
                mode=mode_this_round,
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

        attendance_promoted = [w for w in attendance_workers_prev if w in attendance_ready_ids]
        if attendance_promoted:
            print(f"[coordinator] Attendance workers responded, promoting next round: {attendance_promoted}")

        newly_demoted = [
            w for w in (expected_workers or [])
            if w not in ready_ids and w not in attendance_workers_prev
        ] if is_round_timed_out else []

        still_attending = [w for w in attendance_workers_prev if w not in attendance_ready_ids]
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
            force_worker_ids=None,
        )
        eligible_for_training = [w for w in credentialed if w not in next_attendance_workers]
        planning_force_ids = force_ids if not expected_workers and not contributing_workers else None
        next_mode, next_active_workers = _determine_round_mode(
            projected_shards=projected_shards_next,
            credentialed_workers=eligible_for_training,
            min_shard_samples=args.min_shard_samples,
            force_worker_ids=planning_force_ids,
        )

        # ── Stage advance check ───────────────────────────────────────────────
        stage_complete = (
            final_stage_samples >= args.total_train_samples
            or completion_mode == "complete"
        )
        next_stage_k = stage_k
        next_round_n = round_n + 1
        if stage_complete:
            if stage_k >= DILOCO_TERMINAL_STAGE:
                print(
                    f"[coordinator] Stage {stage_k} COMPLETE "
                    f"({final_stage_samples}/{args.total_train_samples} samples). "
                    "Entering DGAC manual gate."
                )
                terminal_state = _terminal_stage_state(
                    state=state,
                    stage_k=DILOCO_TERMINAL_STAGE,
                    completed_stages=completed_stages,
                    total_samples_seen=total_samples_seen,
                    seed=seed,
                    last_round_workers=[w["worker_id"] for w in contributing_workers] if contributing_workers else [],
                    last_round_samples=sum(w.get("samples_seen", 0) for w in contributing_workers) if contributing_workers else 0,
                )
                hub_upload_json(
                    args.repo_id,
                    ROUND_STATE_PATH,
                    terminal_state,
                    args.hf_token,
                    message="DiLoCo terminal gate: stage 10 complete, DGAC manual",
                )
                _print_dgac_manual_gate_message(DILOCO_TERMINAL_STAGE)
                if coordinator_wandb_run is not None:
                    import wandb
                    wandb.log({"coordinator/stage_complete": 1}, step=round_n)
                    wandb.finish()
                    coordinator_wandb_run = None
                print("[coordinator] Done (DGAC manual gate).")
                return

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
            dispatch_results = trigger_kaggle_workers(
                kaggle_creds,
                active_workers=all_workers_to_trigger,
                notebook_path=Path(args.kaggle_notebook_path),
                coordinator_args=args,
            )
            corrected_state = _reconcile_post_dispatch_state(
                state=new_state,
                planned_active_workers=next_active_workers,
                planned_attendance_workers=next_attendance_workers,
                dispatch_results=dispatch_results,
            )
            if corrected_state is not None:
                hub_upload_json(
                    args.repo_id,
                    ROUND_STATE_PATH,
                    corrected_state,
                    args.hf_token,
                    message=(
                        f"Dispatch reconcile: stage={next_stage_k} round={next_round_n} "
                        f"mode={corrected_state['mode']}"
                    ),
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
