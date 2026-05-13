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

from ouroboros.coordinator_decision import (
    CoordinatorTransitionDecision,
    plan_dispatch_reconciliation,
    plan_missing_worker_transition,
    plan_post_aggregation_transition,
    plan_round_start,
    plan_waiting_mode_transition,
)
from ouroboros.diloco.aggregation import (
    ANCHOR_PREFIX,
    aggregate_worker_updates,
    load_adapter_weights_cpu,
    load_torch_state_cpu,
    save_and_upload_anchor,
    weighted_average_deltas,
    zero_like_state,
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
    _ordered_unique_worker_ids,
    _partition_ready_workers,
)
from ouroboros.mac_dgac_fallback import (
    MAC_DGAC_CLAIM_PATH,
    is_active_mac_claim,
    mac_claim_matches,
)
from ouroboros.runtime_env import resolve_wandb_key
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
DGAC_CANARY_RUN_MODE = "dgac-canary"
DGAC_DILOCO_RUN_MODE = "dgac-diloco"
DGAC_COMPLETE_MODE = "dgac-complete"


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
        choices=[DILOCO_RUN_MODE, DGAC_ANCHOR_EVAL_RUN_MODE, DGAC_TRAIN_RUN_MODE, DGAC_CANARY_RUN_MODE, DGAC_DILOCO_RUN_MODE],
        help=(
            "Kaggle notebook launch mode. Use 'dgac-anchor-eval' to push one "
            "GPU eval-only notebook for the terminal DiLoCo anchor; use "
            "'dgac-train' to launch Phase 3.4 DGAC from the terminal anchor, "
            "'dgac-canary' to launch a bounded short DGAC objective canary, "
            "or 'dgac-diloco' to initialize DGAC as a DiLoCo worker round. "
            "Anchor eval and dgac-train skip DiLoCo round_state mutations; dgac-diloco intentionally uses them."
        ),
    )
    parser.add_argument(
        "--mac_claim_id",
        default=os.environ.get("OUROBOROS_MAC_DGAC_CLAIM_ID"),
        help=(
            "Allow this local coordinator process to aggregate while the matching "
            "strict Mac DGAC fallback claim is active. GitHub Actions should leave "
            "this empty so a valid Mac claim blocks dispatch/aggregation races."
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
        default=resolve_wandb_key(),
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


def _print_dgac_manual_gate_message(stage_k: int) -> None:
    print(
        f"[coordinator] Stage {stage_k} is terminal for DiLoCo. "
        "DGAC is ready for manual quality review; no stage-11 DiLoCo dispatch will run."
    )
    print(
        "[coordinator] DGAC manual gate: review final stage-10 anchor, run CPU-smoke if needed, "
        "then launch DGAC explicitly."
    )


def _transition_dispatch_workers(decision: CoordinatorTransitionDecision) -> List[str]:
    return decision.workers_to_dispatch


def _format_transition_reconcile_message(
    template: str,
    *,
    corrected_state: Dict[str, Any],
) -> str:
    if not template:
        return "Dispatch reconcile"
    return template.format(
        mode=corrected_state.get("mode"),
        attendance_workers=corrected_state.get("attendance_workers", []),
        triggered_workers=corrected_state.get("triggered_workers", []),
    )


def _upload_transition_state(
    *,
    args: argparse.Namespace,
    decision: CoordinatorTransitionDecision,
) -> None:
    if decision.should_write_state and decision.state is not None:
        hub_upload_json(
            args.repo_id,
            ROUND_STATE_PATH,
            decision.state,
            args.hf_token,
            message=decision.hub_message,
        )


def _dispatch_transition(
    *,
    args: argparse.Namespace,
    kaggle_creds: Dict[str, Tuple[Optional[str], Optional[str]]],
    decision: CoordinatorTransitionDecision,
    require_active_workers: bool = False,
) -> None:
    workers_to_dispatch = _transition_dispatch_workers(decision)
    if args.skip_trigger:
        return
    if require_active_workers and not decision.dispatch_active_workers:
        return
    if not workers_to_dispatch:
        return
    dispatch_results = trigger_kaggle_workers(
        kaggle_creds,
        active_workers=workers_to_dispatch,
        notebook_path=Path(args.kaggle_notebook_path),
        coordinator_args=args,
    )
    if decision.state is None:
        return
    reconcile_plan = plan_dispatch_reconciliation(
        state=decision.state,
        planned_active_workers=decision.reconcile_active_workers,
        planned_attendance_workers=decision.reconcile_attendance_workers,
        dispatch_results=dispatch_results,
    )
    if reconcile_plan.corrected_state is not None:
        hub_upload_json(
            args.repo_id,
            ROUND_STATE_PATH,
            reconcile_plan.corrected_state,
            args.hf_token,
            message=_format_transition_reconcile_message(
                decision.dispatch_reconcile_message,
                corrected_state=reconcile_plan.corrected_state,
            ),
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


def _active_foreign_mac_claim(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    claim = hub_download_json(args.repo_id, MAC_DGAC_CLAIM_PATH, args.hf_token)
    if not is_active_mac_claim(claim, now=time.time()):
        return None
    if mac_claim_matches(claim, getattr(args, "mac_claim_id", None)):
        return None
    return claim


def _refuse_if_foreign_mac_claim_active(args: argparse.Namespace) -> bool:
    claim = _active_foreign_mac_claim(args)
    if claim is None:
        return False
    print(
        "[coordinator] Active strict Mac DGAC fallback claim detected; "
        "refusing GitHub/Kaggle coordinator work to avoid conflicting sessions. "
        f"claim_id={claim.get('claim_id')} expires_at={claim.get('expires_at')}"
    )
    return True


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


def _kaggle_dgac_diloco_worker_ids(args: argparse.Namespace) -> List[str]:
    if args.force_worker_ids:
        requested = _ordered_unique_worker_ids(
            [w.strip().upper() for w in str(args.force_worker_ids).split(",") if w.strip()]
        )
        if requested:
            return requested
    kaggle_creds = _build_kaggle_creds(args)
    credentialed = [worker_id for worker_id in WORKER_IDS if kaggle_creds[worker_id][0] and kaggle_creds[worker_id][1]]
    return credentialed or ["A"]


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


def _dispatch_post_dgac_completion_anchor_eval(args: argparse.Namespace) -> None:
    """Launch one eval-only notebook after DGAC DiLoCo reaches its sample target."""
    if args.skip_trigger:
        print("[kaggle-eval] --skip_trigger set. Skipping automatic DGAC anchor eval.")
        return
    original_mode = getattr(args, "kaggle_run_mode", DILOCO_RUN_MODE)
    try:
        args.kaggle_run_mode = DGAC_ANCHOR_EVAL_RUN_MODE
        run_kaggle_anchor_eval(args)
    finally:
        args.kaggle_run_mode = original_mode


def run_kaggle_dgac_train(args: argparse.Namespace) -> None:
    worker_ids = _kaggle_dgac_worker_ids(args)
    kaggle_creds = _build_kaggle_creds(args)
    run_mode = getattr(args, "kaggle_run_mode", DGAC_TRAIN_RUN_MODE)
    label = "DGAC canary" if run_mode == DGAC_CANARY_RUN_MODE else "Phase 3.4 DGAC training"
    print(
        f"[kaggle-dgac] Dispatching {label} notebook "
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
        "W&B train metrics, and any mode-specific output directory; "
        "this mode does not mutate DiLoCo round_state."
    )


def _initial_dgac_diloco_state(
    *,
    previous_state: Optional[Dict[str, Any]],
    worker_ids: List[str],
    projected_shards: Dict[str, int],
    seed: int,
    dgac_round_n: int = 0,
) -> Dict[str, Any]:
    previous_state = dict(previous_state or {})
    dedicated_round = max(int(dgac_round_n), 0)
    return {
        "stage_k": DILOCO_TERMINAL_STAGE,
        "round_n": dedicated_round,
        "dgac_round_n": dedicated_round,
        "dgac_round_label": f"DGAC dedicated round {dedicated_round:03d}",
        "mode": DGAC_DILOCO_RUN_MODE,
        "triggered_workers": worker_ids,
        "attendance_workers": [],
        "projected_shards": projected_shards,
        "anchor_path": ANCHOR_PREFIX,
        "total_samples_seen": {str(DILOCO_TERMINAL_STAGE): 0},
        "completed_stages": sorted({int(x) for x in previous_state.get("completed_stages", [])} | {DILOCO_TERMINAL_STAGE}),
        "dgac_manual_gate": False,
        "dgac_diloco": True,
        "pre_dgac_total_samples_seen": previous_state.get("total_samples_seen", {}),
        "last_updated": time.time(),
        "triggered_at": time.time(),
        "dispatch_failures": [],
        "last_round_workers": [],
        "last_round_samples": 0,
        "seed": seed,
    }


def _next_dgac_dedicated_round_n(previous_state: Optional[Dict[str, Any]]) -> int:
    """Return the next external DGAC dedicated round number.

    Normal DiLoCo terminal state is the pre-DGAC gate, so its first dedicated
    round is 000. Once any DGAC worker/complete state exists, a manual relaunch
    must advance the number to avoid reusing W&B ids such as dgac-a-r000.
    """
    state = dict(previous_state or {})
    if not state:
        return 0

    is_dgac_state = (
        state.get("mode") in {DGAC_DILOCO_RUN_MODE, DGAC_COMPLETE_MODE}
        or bool(state.get("dgac_diloco"))
        or bool(state.get("dgac_diloco_complete"))
        or "dgac_round_n" in state
        or "next_dgac_round_n" in state
    )
    if not is_dgac_state:
        return 0

    if state.get("next_dgac_round_n") is not None:
        return max(int(state["next_dgac_round_n"]), 0)

    candidates: List[int] = []
    for key in ("dgac_round_n", "round_n"):
        value = state.get(key)
        if value is None:
            continue
        try:
            candidates.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(candidates or [-1]) + 1


def run_kaggle_dgac_diloco(args: argparse.Namespace) -> None:
    worker_ids = _kaggle_dgac_diloco_worker_ids(args)
    kaggle_creds = _build_kaggle_creds(args)
    seed = int(getattr(args, "seed", 42))
    previous_state = None if args.dry_run else hub_download_json(args.repo_id, ROUND_STATE_PATH, args.hf_token)
    dgac_round_n = _next_dgac_dedicated_round_n(previous_state)
    projected_shards = _compute_projected_shards(
        total_samples=args.total_train_samples,
        stage_k=DILOCO_TERMINAL_STAGE,
        round_n=dgac_round_n,
        seed=seed,
        total_samples_seen=0,
    )
    print(
        "[kaggle-dgac-diloco] Initializing DGAC dedicated round "
        f"{dgac_round_n:03d} workers={worker_ids} repo={args.repo_id}"
    )
    if args.dry_run:
        print("[kaggle-dgac-diloco] DRY RUN — no round_state mutation or Kaggle dispatch.")
        print(f"  dgac_round_n={dgac_round_n}")
        print(f"  projected_shards={projected_shards}")
        return

    new_state = _initial_dgac_diloco_state(
        previous_state=previous_state,
        worker_ids=worker_ids,
        projected_shards=projected_shards,
        seed=seed,
        dgac_round_n=dgac_round_n,
    )
    hub_upload_json(
        args.repo_id,
        ROUND_STATE_PATH,
        new_state,
        args.hf_token,
        message=f"DGAC dedicated round {dgac_round_n:03d} start: workers={worker_ids}",
    )

    dispatch_results = trigger_kaggle_workers(
        kaggle_creds,
        active_workers=worker_ids,
        notebook_path=Path(args.kaggle_notebook_path),
        coordinator_args=args,
    )
    reconcile_plan = plan_dispatch_reconciliation(
        state=new_state,
        planned_active_workers=worker_ids,
        planned_attendance_workers=[],
        dispatch_results=dispatch_results,
    )
    if reconcile_plan.corrected_state is not None:
        hub_upload_json(
            args.repo_id,
            ROUND_STATE_PATH,
            reconcile_plan.corrected_state,
            args.hf_token,
            message=(
                f"DGAC dedicated round {dgac_round_n:03d} dispatch reconcile: "
                f"mode={reconcile_plan.corrected_state['mode']}"
            ),
        )
    if any(dispatch_results.get(worker_id) != "success" for worker_id in worker_ids):
        raise RuntimeError(
            f"DGAC dedicated round {dgac_round_n:03d} dispatch failed: "
            f"{dispatch_results}"
        )
    print(
        f"[kaggle-dgac-diloco] DGAC dedicated round {dgac_round_n:03d} dispatch accepted. "
        "Worker signals will resume aggregation for adapter + HaltGate anchors."
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

    if _refuse_if_foreign_mac_claim_active(args):
        return

    kaggle_run_mode = getattr(args, "kaggle_run_mode", DILOCO_RUN_MODE)
    if kaggle_run_mode == DGAC_ANCHOR_EVAL_RUN_MODE:
        run_kaggle_anchor_eval(args)
        return
    if kaggle_run_mode in {DGAC_TRAIN_RUN_MODE, DGAC_CANARY_RUN_MODE}:
        run_kaggle_dgac_train(args)
        return
    if kaggle_run_mode == DGAC_DILOCO_RUN_MODE:
        run_kaggle_dgac_diloco(args)
        return

    print("[coordinator] Reading round state...")
    state = hub_download_json(args.repo_id, ROUND_STATE_PATH, args.hf_token)
    if state is None:
        print("[coordinator] No round_state.json found. Nothing to aggregate.")
        return

    kaggle_creds = _build_kaggle_creds(args)
    credentialed = [w for w in WORKER_IDS if kaggle_creds[w][0] and kaggle_creds[w][1]]

    round_plan = plan_round_start(
        state=state,
        total_train_samples=args.total_train_samples,
        min_shard_samples=args.min_shard_samples,
        credentialed_workers=credentialed,
        force_worker_ids=args.force_worker_ids,
        worker_timeout_hours=args.worker_timeout_hours,
    )
    stage_k = round_plan.stage_k
    round_n = round_plan.round_n
    total_samples_seen = {str(k): int(v) for k, v in dict(state.get("total_samples_seen", {})).items()}
    completed_stages = [int(x) for x in state.get("completed_stages", [])]
    expected_workers = round_plan.expected_workers
    seed = int(state.get("seed", 42))
    current_mode = round_plan.current_mode
    if bool(state.get("dgac_diloco")) and current_mode != DGAC_COMPLETE_MODE:
        args.kaggle_run_mode = DGAC_DILOCO_RUN_MODE
    triggered_at = float(state.get("triggered_at", 0.0))

    if current_mode == DGAC_COMPLETE_MODE or bool(state.get("dgac_diloco_complete")):
        print(f"[coordinator] stage={stage_k} round={round_n} mode={current_mode}")
        next_round = state.get("next_dgac_round_n")
        suffix = f" Next manual DGAC dedicated round: {int(next_round):03d}." if next_round is not None else ""
        print(
            "[coordinator] DGAC dedicated round is complete. "
            "Review W&B/Hub final anchor before downstream packaging."
            f"{suffix}"
        )
        return

    if current_mode == "terminal" or bool(state.get("dgac_manual_gate")):
        print(f"[coordinator] stage={stage_k} round={round_n} mode={current_mode}")
        _print_dgac_manual_gate_message(min(stage_k, DILOCO_TERMINAL_STAGE))
        return

    attendance_workers_prev = round_plan.attendance_workers
    is_round_timed_out = round_plan.is_round_timed_out
    force_ids: Optional[List[str]] = round_plan.force_worker_ids or None
    projected_shards = round_plan.projected_shards
    remaining = round_plan.remaining_samples
    stage_samples_seen = int(total_samples_seen.get(str(stage_k), 0))
    next_mode = round_plan.next_mode
    next_active_workers = round_plan.next_active_workers
    next_attendance_workers = round_plan.next_attendance_workers

    print(f"[coordinator] stage={stage_k} round={round_n} mode={current_mode}")
    if attendance_workers_prev:
        print(f"[coordinator] Attendance workers: {attendance_workers_prev}")
    print(f"[coordinator] Remaining samples for stage {stage_k}: {remaining}")
    print(f"[coordinator] Projected shards: {projected_shards}")
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
        waiting_decision = plan_waiting_mode_transition(
            state=state,
            round_plan=round_plan,
            responded_worker_ids=[str(w.get("worker_id", "")).upper() for w in responded_in_waiting],
            credentialed_workers=credentialed,
            total_train_samples=args.total_train_samples,
            min_shard_samples=args.min_shard_samples,
            attendance_join_grace_minutes=args.attendance_join_grace_minutes,
            now=time.time(),
        )

        if waiting_decision.kind == "waiting_initial_dispatch":
            print(
                "[coordinator] Waiting mode: no confirmed dispatch timestamp yet; "
                "attempting attendance dispatch now."
            )
        elif waiting_decision.kind == "waiting_standby":
            print("[coordinator] Waiting mode: no responses yet, standing by.")
            return
        elif waiting_decision.kind == "waiting_redispatch":
            print(f"[coordinator] Waiting mode: re-dispatching attendance to {attendance_workers_prev}")
        elif waiting_decision.kind == "waiting_grace":
            print(
                "[coordinator] Waiting mode: attendance responders received "
                f"{waiting_decision.metadata['responded_workers']}, "
                f"still waiting for {waiting_decision.metadata['still_absent_workers']} "
                f"within {args.attendance_join_grace_minutes:g}m join grace."
            )
            return
        elif waiting_decision.kind == "waiting_promote":
            print(f"[coordinator] Waiting mode exit: promoting {waiting_decision.metadata['responded_workers']}")

        _upload_transition_state(args=args, decision=waiting_decision)
        if waiting_decision.state is not None:
            print(
                f"[coordinator] round_state updated: "
                f"stage={waiting_decision.state.get('stage_k')} "
                f"round={waiting_decision.state.get('round_n')} "
                f"mode={waiting_decision.state.get('mode')}"
            )
        _dispatch_transition(
            args=args,
            kaggle_creds=kaggle_creds,
            decision=waiting_decision,
            require_active_workers=waiting_decision.kind == "waiting_promote",
        )
        if waiting_decision.kind == "waiting_initial_dispatch":
            print("[coordinator] Done (waiting mode initial dispatch).")
        elif waiting_decision.kind == "waiting_redispatch":
            print("[coordinator] Done (waiting mode re-dispatch).")
        elif waiting_decision.kind == "waiting_promote":
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
                missing_decision = plan_missing_worker_transition(
                    state=state,
                    stage_k=stage_k,
                    round_n=round_n,
                    expected_workers=expected_workers,
                    attendance_workers=attendance_workers_prev,
                    missing_workers=missing_workers,
                    force_worker_ids=force_ids or [],
                    ready_worker_ids=ready_ids,
                    attendance_ready_ids=_positive_ready_worker_ids(attendance_ready_workers),
                    credentialed_workers=credentialed,
                    is_round_timed_out=is_round_timed_out,
                    now=time.time(),
                )
                repair_plan = missing_decision.metadata.get("repair_plan")
                if repair_plan is not None and repair_plan.unavailable_workers:
                    print(
                        "[coordinator] Force repair skipped unavailable workers: "
                        f"{repair_plan.unavailable_workers}"
                    )

                if missing_decision.kind == "force_repair":
                    assert repair_plan is not None
                    _upload_transition_state(args=args, decision=missing_decision)
                    if repair_plan.already_done_workers:
                        print(
                            "[coordinator] Force repair counted already-done workers without re-dispatch: "
                            f"{repair_plan.already_done_workers}"
                        )
                    if repair_plan.dispatch_workers:
                        print(
                            "[coordinator] Force repair dispatching missing workers: "
                            f"{repair_plan.dispatch_workers}; preserving active workers: {expected_workers}"
                        )
                    _dispatch_transition(
                        args=args,
                        kaggle_creds=kaggle_creds,
                        decision=missing_decision,
                    )
                    print(f"[coordinator] Done (force repair round {round_n}).")
                    return

                if missing_decision.kind == "force_repair_unavailable":
                    return

                if missing_decision.kind == "unconfirmed_redispatch":
                    print(
                        f"[coordinator] Round {round_n}: {missing_workers} marked triggered but "
                        f"triggered_at=0 (unconfirmed dispatch). Re-dispatching now."
                    )
                    _upload_transition_state(args=args, decision=missing_decision)
                    _dispatch_transition(
                        args=args,
                        kaggle_creds=kaggle_creds,
                        decision=missing_decision,
                    )
                    print(f"[coordinator] Done (re-dispatch unconfirmed round {round_n}).")
                    return

                if missing_decision.kind == "wait_for_missing_workers":
                    print(f"[coordinator] Waiting for workers to finish this round: {missing_workers}")
                    return

                if missing_decision.kind == "timeout_continue":
                    newly_demoted = missing_decision.metadata.get("newly_demoted_workers", [])
                    still_absent = missing_decision.metadata.get("still_absent_workers", [])
                    if newly_demoted:
                        print(
                            f"[coordinator] Timed out (>{args.worker_timeout_hours}h): "
                            f"{newly_demoted} — demoting to attendance"
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
            worker_halt_gate_list = []
            worker_samples_list = []
            requires_halt_gate = mode_this_round == DGAC_DILOCO_RUN_MODE or bool(state.get("dgac_diloco"))
            for status in contributing_workers:
                worker_weights_list.append(
                    load_adapter_weights_cpu(args.repo_id, status["weights_path"], args.hf_token)
                )
                if requires_halt_gate:
                    halt_gate_path = status.get("halt_gate_path")
                    if not halt_gate_path:
                        raise RuntimeError(
                            f"DGAC DiLoCo worker {status.get('worker_id')} did not upload halt_gate.pt"
                        )
                    gate_state = load_torch_state_cpu(args.repo_id, halt_gate_path, args.hf_token)
                    if gate_state is None:
                        raise RuntimeError(f"DGAC DiLoCo missing halt gate artifact: {halt_gate_path}")
                    worker_halt_gate_list.append(gate_state)
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
            new_halt_gate = None
            if requires_halt_gate:
                anchor_halt_gate = load_torch_state_cpu(
                    args.repo_id,
                    f"{ANCHOR_PREFIX}/halt_gate.pt",
                    args.hf_token,
                )
                if anchor_halt_gate is None:
                    anchor_halt_gate = zero_like_state(worker_halt_gate_list[0])
                new_halt_gate = aggregate_worker_updates(
                    anchor_halt_gate,
                    worker_halt_gate_list,
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
                halt_gate_state=new_halt_gate,
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

        post_decision = plan_post_aggregation_transition(
            state=state,
            stage_k=stage_k,
            round_n=round_n,
            current_mode=current_mode,
            total_train_samples=args.total_train_samples,
            min_shard_samples=args.min_shard_samples,
            credentialed_workers=credentialed,
            force_worker_ids=force_ids,
            expected_workers=expected_workers,
            attendance_workers=attendance_workers_prev,
            attendance_ready_ids=attendance_ready_ids,
            ready_worker_ids=ready_ids,
            is_round_timed_out=is_round_timed_out,
            total_samples_seen=total_samples_seen,
            stage_samples_seen=stage_samples_seen,
            completed_stages=completed_stages,
            seed=seed,
            contributing_workers=contributing_workers,
            anchor_path=ANCHOR_PREFIX,
            terminal_stage=DILOCO_TERMINAL_STAGE,
            dgac_complete_mode=DGAC_COMPLETE_MODE,
            now=time.time(),
        )
        final_stage_samples = int(post_decision.metadata.get("final_stage_samples", stage_samples_seen))
        stage_complete = bool(post_decision.metadata.get("stage_complete", False))

        if post_decision.kind == "dgac_diloco_complete":
            print(
                f"[coordinator] DGAC dedicated round COMPLETE "
                f"({final_stage_samples}/{args.total_train_samples} samples)."
            )
            _upload_transition_state(args=args, decision=post_decision)
            if coordinator_wandb_run is not None:
                import wandb
                wandb.log({"coordinator/dgac_diloco_complete": 1}, step=round_n)
                wandb.finish()
                coordinator_wandb_run = None
            _dispatch_post_dgac_completion_anchor_eval(args)
            print("[coordinator] Done (DGAC dedicated round complete; anchor eval dispatched).")
            return

        if post_decision.kind == "terminal_manual_gate":
            print(
                f"[coordinator] Stage {stage_k} COMPLETE "
                f"({final_stage_samples}/{args.total_train_samples} samples). "
                "Entering DGAC manual gate."
            )
            _upload_transition_state(args=args, decision=post_decision)
            _print_dgac_manual_gate_message(DILOCO_TERMINAL_STAGE)
            if coordinator_wandb_run is not None:
                import wandb
                wandb.log({"coordinator/stage_complete": 1}, step=round_n)
                wandb.finish()
                coordinator_wandb_run = None
            print("[coordinator] Done (DGAC manual gate).")
            return

        if post_decision.kind == "stage_advance":
            next_stage_k = post_decision.state["stage_k"] if post_decision.state else stage_k + 1
            print(
                f"[coordinator] Stage {stage_k} COMPLETE "
                f"({final_stage_samples}/{args.total_train_samples} samples). "
                f"Advancing to stage {next_stage_k}."
            )
            if coordinator_wandb_run is not None:
                import wandb
                wandb.log({"coordinator/stage_complete": 1}, step=round_n)

        if post_decision.kind == "all_absent_waiting":
            print("[coordinator] All workers absent — entering waiting mode. Coordinator idles until workers signal presence.")

        _upload_transition_state(args=args, decision=post_decision)
        assert post_decision.state is not None
        print(
            f"[coordinator] round_state.json updated: "
            f"stage={post_decision.state['stage_k']} "
            f"round={post_decision.state['round_n']} "
            f"mode={post_decision.state['mode']}"
        )

        # ── Trigger next workers ──────────────────────────────────────────────
        all_workers_to_trigger = _transition_dispatch_workers(post_decision)
        if args.skip_trigger:
            print("[coordinator] --skip_trigger set. Skipping worker trigger.")
        elif not all_workers_to_trigger:
            print("[coordinator] No workers to trigger (stage complete or waiting with no dispatch needed).")
        else:
            print(
                f"[coordinator] Triggering training: {post_decision.dispatch_active_workers}  "
                f"attendance: {post_decision.dispatch_attendance_workers}"
            )
            _dispatch_transition(
                args=args,
                kaggle_creds=kaggle_creds,
                decision=post_decision,
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
