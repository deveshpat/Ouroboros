#!/usr/bin/env python3
"""
DiLoCo Coordinator - CPU-only weight aggregation.
Runs in GitHub Actions after receiving worker signals.

Usage:
    python -m ouroboros.coordinator \
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

from ouroboros.coordinator.decision import (
    CoordinatorTransitionDecision,
    plan_dispatch_reconciliation,
    plan_missing_worker_transition,
    plan_post_aggregation_transition,
    plan_round_start,
    plan_waiting_mode_transition,
)
from ouroboros.coordinator.aggregation import (
    ANCHOR_PREFIX,
    aggregate_worker_updates,
    load_adapter_weights_cpu,
    load_torch_state_cpu,
    save_and_upload_anchor,
    weighted_average_deltas,
    zero_like_state,
)
from ouroboros.coordinator.dispatch import (
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
from ouroboros.coordinator.state import (
    WORKER_IDS,
    _compute_projected_shards,
    _determine_round_mode,
    _ordered_unique_worker_ids,
    _partition_ready_workers,
)
from ouroboros.utils.runtime_env import resolve_hf_token, resolve_wandb_key
from ouroboros.utils.wandb_runtime import wandb_init_kwargs


T = TypeVar("T")

ROUND_STATE_PATH = "diloco_state/round_state.json"
DEFAULT_KAGGLE_NOTEBOOK_PATH = Path(__file__).resolve().parents[2] / "kaggle-utils.ipynb"
DEFAULT_IO_RETRIES = 3
DEFAULT_IO_RETRY_BASE_DELAY_S = 1.5
DILOCO_TERMINAL_STAGE = 10
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
    parser.add_argument("--hf_token", default=resolve_hf_token())
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
        "--launch_worker_ids",
        default="",
        help=(
            "Comma-separated worker IDs to launch on Kaggle, e.g. 'A,B'. "
            "Empty means aggregate/check only and do not push notebooks."
        ),
    )
    parser.add_argument(
        "--force_worker_ids",
        default=None,
        help=argparse.SUPPRESS,
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
    parser.add_argument(
        "--attendance_join_grace_minutes",
        type=float,
        default=5.0,
        help=(
            "Minutes to wait in waiting mode after the first attendance response "
            "before promoting a partial attendance set. Default 5.0."
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
    parser.add_argument(
        "--eval_mode",
        choices=("none", "sample-25", "full"),
        default="none",
        help=(
            "Dispatch a generated-answer Coconut validation eval notebook instead of "
            "running the DiLoCo coordinator loop. sample-25 uses --limit_samples 25; "
            "full runs the whole validation split."
        ),
    )
    parser.add_argument(
        "--eval_worker_id",
        default="A",
        help="Kaggle worker account/notebook used for eval dispatch. Default: A.",
    )
    parser.add_argument("--eval_data_dir", default="data/coconut_v1")
    parser.add_argument("--eval_dataset_repo", default="WeirdRunner/Ouroboros")
    parser.add_argument("--eval_dataset_config", default="coconut-v1")
    parser.add_argument("--eval_dataset_split", default="validation")
    parser.add_argument(
        "--eval_dataset_revision",
        default="6a52cd0c47be1e7b85d9018225387950aefc4631",
    )
    parser.add_argument("--eval_baseline_model_id", default="ai21labs/AI21-Jamba-Reasoning-3B")
    parser.add_argument("--eval_candidate_repo_id", default="WeirdRunner/Ouroboros")
    parser.add_argument("--eval_candidate_subdir", default="diloco_state/anchor")
    parser.add_argument("--eval_stage_k", type=int, default=10)
    parser.add_argument("--eval_max_seq_len", type=int, default=512)
    parser.add_argument("--eval_gen_max_tokens", type=int, default=128)
    parser.add_argument("--eval_halt_threshold", type=float, default=0.5)
    parser.add_argument("--eval_device", default="auto")
    parser.add_argument("--eval_dtype", default="auto")
    parser.add_argument(
        "--eval_output_root",
        default="runs/eval",
        help="Root directory for generated eval artifacts inside the Kaggle notebook runtime.",
    )
    parser.add_argument(
        "--eval_output_dir",
        default="",
        help="Optional explicit output directory for generated eval artifacts.",
    )
    parser.add_argument(
        "--eval_disable_mamba_kernels",
        action="store_true",
        help="Forward --disable_mamba_kernels to compare-coconut-val.",
    )
    args = parser.parse_args()
    if args.force_worker_ids and not args.launch_worker_ids:
        args.launch_worker_ids = args.force_worker_ids
    args.force_worker_ids = args.launch_worker_ids or None
    return args



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



def _build_kaggle_creds(args: argparse.Namespace) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    return {
        "A": (args.kaggle_username_a, args.kaggle_key_a),
        "B": (args.kaggle_username_b, args.kaggle_key_b),
        "C": (args.kaggle_username_c, args.kaggle_key_c),
    }



def _requested_launch_worker_ids(args: argparse.Namespace) -> List[str]:
    raw = getattr(args, "launch_worker_ids", None) or getattr(args, "force_worker_ids", None)
    if raw is None:
        return []
    if isinstance(raw, list):
        values = raw
    else:
        values = [part.strip() for part in str(raw).split(",")]
    return _ordered_unique_worker_ids([str(value).upper() for value in values if str(value).strip()])


def _planning_credentialed_workers(
    *,
    credentialed_workers: List[str],
    launch_worker_ids: List[str],
) -> List[str]:
    # Non-empty launch_worker_ids is the explicit dispatch contract: the
    # coordinator may only push selected worker notebooks. Credentials are still
    # checked during dispatch; missing credentials become manual dispatches.
    if launch_worker_ids:
        return launch_worker_ids
    return credentialed_workers


def _eval_limit_samples(args: argparse.Namespace) -> Optional[int]:
    if args.eval_mode == "sample-25":
        return 25
    if args.eval_mode == "full":
        return None
    raise ValueError(f"Unsupported eval_mode: {args.eval_mode!r}")


def _eval_output_dir(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "eval_output_dir", "") or "").strip()
    if explicit:
        return explicit
    leaf = "coconut_val_compare_sample_25" if args.eval_mode == "sample-25" else "coconut_val_compare_full"
    return str(Path(args.eval_output_root) / leaf)


def _build_eval_runtime_env(args: argparse.Namespace, worker_id: str) -> Dict[str, str]:
    runtime_env = _build_worker_runtime_env(args, worker_id)
    runtime_env.update(
        {
            "OUROBOROS_KAGGLE_RUN_KIND": "eval",
            "OUROBOROS_EVAL_MODE": str(args.eval_mode),
            "OUROBOROS_EVAL_DATA_DIR": str(args.eval_data_dir),
            "OUROBOROS_EVAL_DATASET_REPO": str(args.eval_dataset_repo),
            "OUROBOROS_EVAL_DATASET_CONFIG": str(args.eval_dataset_config),
            "OUROBOROS_EVAL_DATASET_SPLIT": str(args.eval_dataset_split),
            "OUROBOROS_EVAL_DATASET_REVISION": str(args.eval_dataset_revision),
            "OUROBOROS_EVAL_BASELINE_MODEL_ID": str(args.eval_baseline_model_id),
            "OUROBOROS_EVAL_CANDIDATE_REPO_ID": str(args.eval_candidate_repo_id),
            "OUROBOROS_EVAL_CANDIDATE_SUBDIR": str(args.eval_candidate_subdir),
            "OUROBOROS_EVAL_STAGE_K": str(int(args.eval_stage_k)),
            "OUROBOROS_EVAL_MAX_SEQ_LEN": str(int(args.eval_max_seq_len)),
            "OUROBOROS_EVAL_GEN_MAX_TOKENS": str(int(args.eval_gen_max_tokens)),
            "OUROBOROS_EVAL_HALT_THRESHOLD": str(float(args.eval_halt_threshold)),
            "OUROBOROS_EVAL_DEVICE": str(args.eval_device),
            "OUROBOROS_EVAL_DTYPE": str(args.eval_dtype),
            "OUROBOROS_EVAL_OUTPUT_DIR": _eval_output_dir(args),
            "OUROBOROS_EVAL_CANDIDATE_REQUIRES_HALT_GATE": "1",
        }
    )
    limit_samples = _eval_limit_samples(args)
    if limit_samples is not None:
        runtime_env["OUROBOROS_EVAL_LIMIT_SAMPLES"] = str(limit_samples)
    if bool(getattr(args, "eval_disable_mamba_kernels", False)):
        runtime_env["OUROBOROS_EVAL_DISABLE_MAMBA_KERNELS"] = "1"
    return runtime_env


def _dispatch_eval_notebook(args: argparse.Namespace) -> None:
    worker_id = str(args.eval_worker_id or "").strip().upper()
    if worker_id not in WORKER_IDS:
        raise SystemExit(f"Invalid --eval_worker_id {args.eval_worker_id!r}; expected one of {WORKER_IDS}.")

    kaggle_creds = _build_kaggle_creds(args)
    username, key = kaggle_creds.get(worker_id, (None, None))
    if not username or not key:
        raise SystemExit(
            f"Eval dispatch requested on worker {worker_id}, but its Kaggle credentials are missing."
        )

    _, slug = WORKER_KAGGLE_SLUGS[worker_id]
    limit_samples = _eval_limit_samples(args)
    limit_label = "full validation split" if limit_samples is None else f"{limit_samples} samples"
    output_dir = _eval_output_dir(args)
    print(
        f"[coordinator] Dispatching generated-answer eval ({args.eval_mode}, {limit_label}) "
        f"to Kaggle worker {worker_id}: {slug}"
    )
    print(f"[coordinator] Eval artifacts will be written under: {output_dir}")
    ok = _trigger_single_worker(
        worker_id,
        username,
        key,
        slug,
        notebook_path=Path(args.kaggle_notebook_path),
        injected_env=_build_eval_runtime_env(args, worker_id),
    )
    if not ok:
        raise SystemExit(2)
    print("[coordinator] Eval notebook dispatched. Review Kaggle output/artifacts before launching the full run.")


def main() -> None:
    args = parse_args()
    if not args.hf_token:
        raise SystemExit("HF token required. Set HF_TOKEN or pass --hf_token.")

    if args.eval_mode != "none":
        _dispatch_eval_notebook(args)
        return

    launch_worker_ids = _requested_launch_worker_ids(args)
    if not launch_worker_ids:
        args.skip_trigger = True
        print("[coordinator] launch_worker_ids empty: aggregate/check only; no Kaggle notebooks will be pushed.")
    else:
        args.force_worker_ids = ",".join(launch_worker_ids)
        print(f"[coordinator] Explicit Kaggle launch workers: {launch_worker_ids}")

    print("[coordinator] Reading round state...")
    state = hub_download_json(args.repo_id, ROUND_STATE_PATH, args.hf_token)
    if state is None:
        print("[coordinator] No round_state.json found. Nothing to aggregate.")
        return

    kaggle_creds = _build_kaggle_creds(args)
    credentialed = [w for w in WORKER_IDS if kaggle_creds[w][0] and kaggle_creds[w][1]]
    planning_credentialed = _planning_credentialed_workers(
        credentialed_workers=credentialed,
        launch_worker_ids=launch_worker_ids,
    )

    round_plan = plan_round_start(
        state=state,
        total_train_samples=args.total_train_samples,
        min_shard_samples=args.min_shard_samples,
        credentialed_workers=planning_credentialed,
        force_worker_ids=launch_worker_ids or None,
        worker_timeout_hours=args.worker_timeout_hours,
    )
    stage_k = round_plan.stage_k
    round_n = round_plan.round_n
    total_samples_seen = {str(k): int(v) for k, v in dict(state.get("total_samples_seen", {})).items()}
    completed_stages = [int(x) for x in state.get("completed_stages", [])]
    expected_workers = round_plan.expected_workers
    seed = int(state.get("seed", 42))
    current_mode = round_plan.current_mode
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
            credentialed_workers=planning_credentialed,
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
                    **wandb_init_kwargs(wandb),
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
                    credentialed_workers=planning_credentialed,
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
            credentialed_workers=planning_credentialed,
            force_worker_ids=launch_worker_ids or None,
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
            print("[coordinator] Done (DGAC dedicated round complete; no automatic anchor eval dispatch).")
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
            print("[coordinator] No Kaggle notebook push requested; skipping worker trigger.")
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
