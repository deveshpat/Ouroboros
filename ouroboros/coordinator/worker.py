"""DiLoCo worker shard, state, upload, and signal operations."""

from __future__ import annotations

import base64
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ouroboros.bootstrap import _resolve_github_token_common
from ouroboros.coordinator.shared import RoundState, ordered_unique_workers, retry_io
from ouroboros.utils.wandb_runtime import wandb_init_kwargs
from ouroboros.models import (
    _is_main_process,
    _world_size,
    barrier,
    broadcast_parameters,
    get_trainable_parameters,
    _wandb_config,
)

_DILOCO_SHARD_STEP_FALLBACK = 385


def _diloco_wandb_identity(
    args: argparse.Namespace,
    *,
    stage_k: int,
    round_n: int,
    is_dgac_diloco: bool,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build W&B identity/config for normal DiLoCo or DGAC dedicated rounds."""
    worker_id = str(args.diloco_worker_id)
    worker_key = worker_id.lower()
    config: Dict[str, Any] = {
        **_wandb_config(args),
        "stage_k": int(stage_k),
        "round_n": int(round_n),
        "worker_id": worker_id,
    }
    if is_dgac_diloco:
        config.update(
            {
                "mode": "dgac-dedicated-round",
                "dgac_round_n": int(round_n),
                "dgac_round_label": f"DGAC dedicated round {int(round_n):03d}",
            }
        )
        run_id = f"dgac-{worker_key}-r{int(round_n):04d}"
        group_id = f"dgac-dedicated-r{int(round_n):04d}"
        name = f"DGAC Worker {worker_id} | Dedicated Round {int(round_n):03d}"
    else:
        config["mode"] = "diloco"
        run_id = f"diloco-{worker_key}-s{int(stage_k)}-r{int(round_n)}"
        group_id = f"diloco-{worker_key}-s{int(stage_k)}"
        name = f"Worker {worker_id} | Stage {int(stage_k)} | Round {int(round_n)}"

    if extra_config:
        config.update(extra_config)
    return {"id": run_id, "group": group_id, "name": name, "config": config}


def _partition_contiguous_range(n_items: int, n_parts: int, part_idx: int) -> Tuple[int, int]:
    if n_items <= 0:
        return 0, 0
    base = n_items // n_parts
    remainder = n_items % n_parts
    start = part_idx * base + min(part_idx, remainder)
    width = base + (1 if part_idx < remainder else 0)
    return start, start + width


def diloco_get_shard(
    train_samples: List[Dict[str, Any]],
    worker_id: str,
    stage_k: int,
    round_n: int,
    seed: int,
    samples_already_seen: int = 0,
) -> List[Dict[str, Any]]:
    """
    Deterministic shard assignment for the current DiLoCo round.

    The permutation is still stage/round dependent, but we trim the prefix that
    has already been counted in round_state.total_samples_seen for this stage.
    This keeps partially-complete stages from re-running a full 1/3 shard when
    only the stage remainder should be trained.
    """
    worker_idx = {"A": 0, "B": 1, "C": 2}[worker_id]
    n = len(train_samples)
    if n <= 0:
        return []

    rng = random.Random(seed + stage_k * 100_003 + round_n * 7)
    indices = list(range(n))
    rng.shuffle(indices)

    seen = max(0, min(int(samples_already_seen), n))
    remaining_indices = indices[seen:]
    if not remaining_indices:
        return []

    start, end = _partition_contiguous_range(len(remaining_indices), 3, worker_idx)
    shard_indices = remaining_indices[start:end]
    return [train_samples[i] for i in shard_indices]


def diloco_read_round_state(hf_token: str, repo_id: str) -> Dict[str, Any]:
    """
    Download and parse diloco_state/round_state.json from Hub.
    Returns default state if file doesn't exist (first run).
    """
    from huggingface_hub import hf_hub_download

    def _download() -> Dict[str, Any]:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="diloco_state/round_state.json",
            token=hf_token,
        )
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        return RoundState.from_dict(state).to_dict()

    state = retry_io(
        "  [diloco] Download round_state.json",
        _download,
        swallow=True,
        default=None,
        verbose=_is_main_process(),
    )
    if state is not None:
        return state
    return RoundState().to_dict()


def diloco_upload_worker_state(
    adapter_dir: Path,
    worker_id: str,
    stage_k: int,
    round_n: int,
    samples_seen: int,
    hf_token: str,
    repo_id: str,
    halt_gate=None,
) -> None:
    """
    Upload worker adapter weights and status to Hub.
    Paths:
      diloco_state/workers/{worker_id}/round_{round_n:04d}_stage_{stage_k}/adapter_model.safetensors
      diloco_state/workers/{worker_id}/status.json
    """
    import tempfile
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    remote_prefix = f"diloco_state/workers/{worker_id}/round_{round_n:04d}_stage_{stage_k}"

    if halt_gate is not None:
        torch.save(halt_gate.state_dict(), adapter_dir / "halt_gate.pt")

    uploaded_files = ["adapter_model.safetensors", "adapter_config.json"]
    if (adapter_dir / "halt_gate.pt").exists():
        uploaded_files.append("halt_gate.pt")

    for fname in uploaded_files:
        fpath = adapter_dir / fname
        if fpath.exists():
            retry_io(
                f"  [diloco] Upload worker artifact {worker_id}/{fname}",
                lambda fpath=fpath, fname=fname: api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=f"{remote_prefix}/{fname}",
                    repo_id=repo_id,
                    token=hf_token,
                    commit_message=f"Worker {worker_id} round {round_n} stage {stage_k}",
                ),
                verbose=_is_main_process(),
            )

    status = {
        "worker_id": worker_id,
        "stage_k": int(stage_k),
        "round_n": int(round_n),
        "samples_seen": int(samples_seen),
        "status": "done",
        "timestamp": time.time(),
        "weights_path": remote_prefix,
    }
    if (adapter_dir / "halt_gate.pt").exists():
        status["halt_gate_path"] = f"{remote_prefix}/halt_gate.pt"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(status, tf, indent=2)
        tmp_path = tf.name
    try:
        retry_io(
            f"  [diloco] Upload worker status {worker_id}",
            lambda: api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=f"diloco_state/workers/{worker_id}/status.json",
                repo_id=repo_id,
                token=hf_token,
                commit_message=f"Worker {worker_id} status update",
            ),
            verbose=_is_main_process(),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _set_peft_model_state_dict_compat(model, weights: Dict[str, torch.Tensor]) -> Any:
    """
    Load a saved PEFT adapter state dict while avoiding the Transformers-v5
    conversion path for adapters that are already in PEFT save_pretrained shape.

    PEFT 0.19 can otherwise route Jamba adapter keys through a Transformers
    WeightConverter path whose signature is version-sensitive. The DiLoCo
    anchor is already a PEFT adapter checkpoint, so the direct PEFT insertion
    path is the correct load behavior here.
    """
    from peft import set_peft_model_state_dict
    import peft.utils.save_and_load as peft_save_and_load

    old_transformers_v5 = getattr(peft_save_and_load, "is_transformers_ge_v5", None)
    if old_transformers_v5 is True:
        peft_save_and_load.is_transformers_ge_v5 = False
    try:
        return set_peft_model_state_dict(model, weights)
    finally:
        if old_transformers_v5 is not None:
            peft_save_and_load.is_transformers_ge_v5 = old_transformers_v5


def diloco_download_anchor(
    model,
    hf_token: str,
    repo_id: str,
    anchor_path: str,
    device: torch.device,
    halt_gate=None,
    required: bool = False,
) -> None:
    """
    Download anchor adapter weights from Hub and load them into the model in-place.
    If a DGAC halt gate is supplied, also attempts to load anchor_path/halt_gate.pt;
    absence is allowed so the first DGAC DiLoCo round can start from zero-init.
    Falls back silently if no anchor exists (first round uses random init).
    """
    try:
        def _download() -> None:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            dl_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{anchor_path}/adapter_model.safetensors",
                token=hf_token,
            )
            weights = load_file(dl_path, device=str(device))
            _set_peft_model_state_dict_compat(model, weights)

        retry_io(f"  [diloco] Download anchor {anchor_path}", _download, verbose=_is_main_process())
        if _is_main_process():
            print(f"  [diloco] Loaded anchor weights from {anchor_path}")

        if halt_gate is not None:
            def _download_halt_gate() -> bool:
                from huggingface_hub import hf_hub_download

                gate_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{anchor_path}/halt_gate.pt",
                    token=hf_token,
                )
                halt_gate.load_state_dict(torch.load(gate_path, map_location=device))
                return True

            gate_loaded = bool(
                retry_io(
                    f"  [diloco] Download halt gate {anchor_path}",
                    _download_halt_gate,
                    swallow=not required,
                    default=False,
                    verbose=_is_main_process(),
                )
            )
            if gate_loaded:
                if _is_main_process():
                    print(f"  [diloco] Loaded halt gate from {anchor_path}/halt_gate.pt")
            elif required:
                raise FileNotFoundError(f"Required DGAC halt gate missing at {anchor_path}/halt_gate.pt")
            elif _is_main_process():
                print("  [diloco] No halt gate anchor found; using zero-init HaltGate.")
    except Exception as exc:
        if required:
            raise RuntimeError(f"Required DiLoCo anchor load failed at {anchor_path}: {exc}") from exc
        if _is_main_process():
            print(f"  [diloco] No anchor found at {anchor_path} ({exc}); using current weights.")


def diloco_push_signal(
    worker_id: str,
    stage_k: int,
    round_n: int,
    github_token: str,
    github_repo: str,
) -> None:
    """
    Push a signal file to GitHub to trigger the coordinator GitHub Action.
    File: signals/worker_{id}_stage_{k}_round_{n}.json
    Uses GitHub API directly (no git clone needed).
    """
    import base64
    import requests

    signal_path = f"signals/worker_{worker_id}_stage_{stage_k}_round_{round_n}.json"
    content = json.dumps(
        {
            "worker_id": worker_id,
            "stage_k": int(stage_k),
            "round_n": int(round_n),
            "timestamp": time.time(),
        },
        indent=2,
    )
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")

    url = f"https://api.github.com/repos/{github_repo}/contents/{signal_path}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
    }

    existing = retry_io(
        f"  [diloco] Lookup existing signal {signal_path}",
        lambda: requests.get(url, headers=headers, timeout=30),
        swallow=True,
        default=None,
        verbose=_is_main_process(),
    )
    payload = {
        "message": f"Worker {worker_id} done: stage {stage_k} round {round_n}",
        "content": encoded,
    }
    if existing is not None and existing.status_code == 200:
        payload["sha"] = existing.json().get("sha")

    resp = retry_io(
        f"  [diloco] Push signal {signal_path}",
        lambda: requests.put(url, headers=headers, json=payload, timeout=30),
        swallow=True,
        default=None,
        verbose=_is_main_process(),
    )
    if resp is not None and resp.status_code in (200, 201):
        if _is_main_process():
            print(f"  [diloco] Signal pushed to GitHub: {signal_path}")
    else:
        if _is_main_process():
            if resp is None:
                print(f"  [diloco] WARNING: GitHub signal push failed: no response for {signal_path}")
            else:
                print(f"  [diloco] WARNING: GitHub signal push failed: {resp.status_code} {resp.text[:200]}")


def _diloco_reset_triggered_at(hf_token: str, repo_id: str) -> None:
    """
    Download diloco_state/round_state.json from Hub, set triggered_at=0,
    and re-upload. triggered_at=0 is the canonical signal for an unconfirmed
    dispatch; the coordinator immediately re-dispatches on its next run
    (verified working in Session 19/20).
    """
    _ROUND_STATE_REMOTE = "diloco_state/round_state.json"
    if not hf_token:
        print("  [diloco] WARNING: no HF token — cannot reset triggered_at.")
        return

    def _reset() -> None:
        from huggingface_hub import HfApi, hf_hub_download

        local = hf_hub_download(
            repo_id=repo_id,
            filename=_ROUND_STATE_REMOTE,
            token=hf_token,
        )
        with open(local, encoding="utf-8") as f:
            state = json.load(f)
        state["triggered_at"] = 0.0
        state["last_updated"] = time.time()
        api = HfApi(token=hf_token)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tf:
            json.dump(state, tf, indent=2)
            tmp_path = tf.name
        try:
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=_ROUND_STATE_REMOTE,
                repo_id=repo_id,
                token=hf_token,
                commit_message="GPU mismatch fast-fail: reset triggered_at=0",
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    retry_io(
        "  [diloco] Reset triggered_at in round_state.json",
        _reset,
        swallow=True,
        verbose=_is_main_process(),
    )
    if _is_main_process():
        print(
            "  [diloco] triggered_at reset to 0 in round_state.json ✓ "
            "— coordinator will re-dispatch on next run (≤30 min)."
        )


def run_diloco_worker(
    *,
    model,
    tokenizer,
    halt_gate: Optional[HaltGate],
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    curriculum_max_stage: int,
    lat_token_id: int,
    pad_id: int,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    session_start: float,
    wandb_run,
    hf_token: str,
) -> Dict[str, Any]:
    if args.use_halt_gate and halt_gate is None:
        raise ValueError("--use_halt_gate in DiLoCo mode requires a HaltGate instance")
    if args.use_halt_gate and not getattr(args, "resume_from_diloco_anchor", False):
        raise ValueError("DGAC DiLoCo requires --resume_from_diloco_anchor so workers share the terminal anchor")
    if args.diloco_worker_id is None:
        raise ValueError("--diloco_worker_id required with --diloco_mode")
    if not hf_token:
        raise ValueError("HF token required for DiLoCo mode")
    if args.resume_from and _is_main_process():
        print("  [diloco] Ignoring --resume_from; the shared anchor defines worker startup state.")

    round_state = diloco_read_round_state(hf_token, args.diloco_state_repo)
    stage_k = int(round_state.get("stage_k", 0))
    round_n = int(round_state.get("round_n", 0))
    anchor_path = round_state.get("anchor_path", "diloco_state/anchor")

    if stage_k > curriculum_max_stage:
        if _is_main_process():
            print(f"  [diloco] stage={stage_k} exceeds max configured stage={curriculum_max_stage}. Nothing to do.")
        return {"stage_k": stage_k, "round_n": round_n, "samples_seen": 0}

    triggered_workers_raw = round_state.get("triggered_workers")
    triggered_workers = (
        ordered_unique_workers(triggered_workers_raw)
        if triggered_workers_raw is not None
        else None
    )
    attendance_workers = ordered_unique_workers(round_state.get("attendance_workers"))
    if triggered_workers:
        attendance_workers = [w for w in attendance_workers if w not in set(triggered_workers)]

    is_selected_for_training = (
        triggered_workers is None or args.diloco_worker_id in triggered_workers
    )
    is_attendance_only = (
        args.diloco_worker_id in attendance_workers and not is_selected_for_training
    )

    if not is_selected_for_training and not is_attendance_only:
        if _is_main_process():
            print(
                f"  [diloco] Worker {args.diloco_worker_id} not scheduled for "
                f"stage={stage_k} round={round_n}; exiting without training."
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
            "skipped_not_scheduled": True,
        }

    if is_attendance_only:
        if _is_main_process():
            print(
                f"  [diloco] Worker {args.diloco_worker_id} in attendance mode — "
                f"signaling presence (no training this round)."
            )
            diloco_download_anchor(
                model,
                hf_token,
                args.diloco_state_repo,
                anchor_path,
                device,
                halt_gate=halt_gate,
                required=bool(getattr(args, "resume_from_diloco_anchor", False)),
            )

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
                halt_gate=halt_gate if args.use_halt_gate else None,
            )
            github_token = _resolve_github_token_common()
            if github_token and args.diloco_signal_repo:
                diloco_push_signal(
                    args.diloco_worker_id,
                    stage_k,
                    round_n,
                    github_token,
                    args.diloco_signal_repo,
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

    if _is_main_process():
        print(f"  [diloco] Worker {args.diloco_worker_id} | stage={stage_k} round={round_n}")
        if args.push_to_hub:
            print("  [diloco] Regular stage checkpoint Hub sync is disabled in DiLoCo mode; worker uploads go to diloco_state/ only.")
        diloco_download_anchor(
            model,
            hf_token,
            args.diloco_state_repo,
            anchor_path,
            device,
            halt_gate=halt_gate,
            required=bool(getattr(args, "resume_from_diloco_anchor", False)),
        )
    barrier()

    if _world_size() > 1:
        broadcast_parameters(get_trainable_parameters(model, halt_gate if args.use_halt_gate else None), src=0)
        barrier()

    stage_samples_seen = int(round_state.get("total_samples_seen", {}).get(str(stage_k), 0))
    stage_samples_seen = max(0, min(stage_samples_seen, len(train_samples)))
    remaining_stage_samples = max(len(train_samples) - stage_samples_seen, 0)
    is_new_stage = stage_samples_seen == 0

    train_shard = diloco_get_shard(
        train_samples,
        args.diloco_worker_id,
        stage_k,
        round_n,
        args.seed,
        samples_already_seen=stage_samples_seen,
    )
    if _is_main_process():
        print(
            f"  [diloco] Stage progress before round: "
            f"{stage_samples_seen}/{len(train_samples)} samples"
        )
        print(f"  [diloco] Remaining global samples: {remaining_stage_samples}")
        print(f"  [diloco] Shard size: {len(train_shard)} samples")

    if len(train_shard) == 0:
        if _is_main_process():
            print("  [diloco] Empty shard — uploading passthrough status and signal.")
            # Save current adapter (anchor weights, unchanged) for status upload
            _passthrough_dir = output_dir / "diloco_worker_upload" / f"worker_{args.diloco_worker_id}_stage_{stage_k}_round_{round_n}_passthrough"
            _passthrough_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(_passthrough_dir))
            diloco_upload_worker_state(
                adapter_dir=_passthrough_dir,
                worker_id=args.diloco_worker_id,
                stage_k=stage_k,
                round_n=round_n,
                samples_seen=0,
                hf_token=hf_token,
                repo_id=args.diloco_state_repo,
                halt_gate=halt_gate if args.use_halt_gate else None,
            )
            github_token = _resolve_github_token_common()
            if github_token and args.diloco_signal_repo:
                diloco_push_signal(
                    args.diloco_worker_id, stage_k, round_n,
                    github_token, args.diloco_signal_repo,
                )
            else:
                print("  [diloco] No GITHUB_TOKEN — coordinator must be triggered manually.")
            print(f"  [diloco] Worker {args.diloco_worker_id} passthrough done. stage={stage_k} round={round_n}")
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

    # ── Step-offset (always computed, regardless of W&B mode) ─────────────────
    # Keep the step budget based on a full nominal 1/3 worker shard so W&B step
    # offsets remain monotonic across rounds even when the final round is a
    # trimmed remainder. Reserve one extra marker step between rounds so the
    # round-start log for round N is strictly greater than the round-complete
    # log from round N-1.
    shard_step_estimate = math.ceil(
        len(train_samples) / 3 / max(args.batch_size * args.grad_accum, 1)
    )
    if shard_step_estimate <= 0:
        shard_step_estimate = _DILOCO_SHARD_STEP_FALLBACK
    round_step_span = shard_step_estimate + 1
    global_step_offset = round_n * round_step_span
    is_dgac_diloco = bool(args.use_halt_gate and getattr(args, "resume_from_diloco_anchor", False))

    if is_dgac_diloco:
        args.dgac_round_n = int(round_n)
        args.dgac_round_label = f"DGAC dedicated round {int(round_n):03d}"
        output_dir = output_dir / f"round_{int(round_n):04d}"

    # ── W&B init (DiLoCo path only) ──────────────────────────────────────────
    # Normal DiLoCo keeps stage/round names. DGAC uses dedicated-round names so
    # manual relaunches never collide with prior W&B run ids such as r0.
    diloco_wandb_run = None
    if _is_main_process() and args.wandb_mode != "disabled":
        try:
            import wandb as _wandb
            identity = _diloco_wandb_identity(
                args,
                stage_k=stage_k,
                round_n=round_n,
                is_dgac_diloco=is_dgac_diloco,
                extra_config={
                    "shard_step_estimate": shard_step_estimate,
                    "wandb_round_step_span": round_step_span,
                    "remaining_stage_samples": remaining_stage_samples,
                    "planned_shard_samples": len(train_shard),
                },
            )
            diloco_wandb_run = _wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=identity["id"],
                group=identity["group"],
                name=identity["name"],
                # No resume= needed: each round is a guaranteed-fresh run
                config=identity["config"],
                mode=args.wandb_mode,
                **wandb_init_kwargs(_wandb),
            )
            if is_dgac_diloco:
                _wandb.log(
                    {"dgac/round": round_n, "diloco/stage": stage_k},
                    step=global_step_offset,
                )
            else:
                _wandb.log(
                    {"diloco/round": round_n, "diloco/stage": stage_k},
                    step=global_step_offset,
                )
        except Exception as _we:
            print(f"  [diloco] W&B init failed: {_we}")

    active_workers_for_pre_val = triggered_workers or [args.diloco_worker_id]
    pre_val_leader = active_workers_for_pre_val[0] if active_workers_for_pre_val else args.diloco_worker_id
    forced_pre_val = bool(args.diloco_run_val and args.diloco_worker_id == pre_val_leader)
    default_stage_pre_val = bool(
        round_n == 0
        and is_new_stage
        and args.diloco_worker_id == pre_val_leader
    )
    should_run_pre_val = bool(forced_pre_val or default_stage_pre_val)
    if is_dgac_diloco and _is_main_process():
        if should_run_pre_val:
            print(
                "  [dgac-diloco] Running leader pre-val before DGAC shard training "
                f"(worker={args.diloco_worker_id}, round={round_n})."
            )
        else:
            print(
                "  [dgac-diloco] Skipping duplicate worker pre-val "
                f"(leader={pre_val_leader}, worker={args.diloco_worker_id})."
            )
    if should_run_pre_val and val_samples:
        from ouroboros.coconut.evaluation import evaluate_stage
        val_ce, val_acc = evaluate_stage(
            model=model,
            val_samples=val_samples,
            tokenizer=tokenizer,
            lat_token_id=lat_token_id,
            stage_k=stage_k,
            device=device,
            args=args,
            halt_gate=halt_gate if args.use_halt_gate else None,
        )
        if _is_main_process():
            print(f"  [diloco] Pre-training val: stage={stage_k} ce={val_ce:.4f} acc={val_acc:.4f}")
            if diloco_wandb_run is not None:
                import wandb
                wandb.log(
                    {
                        "diloco/pre_val_ce": val_ce,
                        "diloco/pre_val_acc": val_acc,
                        "diloco/stage": stage_k,
                        "diloco/round": round_n,
                    },
                    step=global_step_offset,
                )
    barrier()

    original_push_to_hub = args.push_to_hub
    original_stage0_epochs = args.stage_0_epochs
    original_epochs_per_stage = args.epochs_per_stage
    args.push_to_hub = False
    args.epochs_per_stage = 1
    if stage_k == 0:
        args.stage_0_epochs = 1
    try:
        from ouroboros.coconut.stage_runner import run_training_stages

        result = run_training_stages(
            model=model,
            tokenizer=tokenizer,
            halt_gate=halt_gate if args.use_halt_gate else None,
            train_samples=train_shard,
            val_samples=val_samples,
            lat_token_id=lat_token_id,
            pad_id=pad_id,
            args=args,
            device=device,
            output_dir=output_dir,
            session_start=session_start,
            wandb_run=diloco_wandb_run,
            stages=[stage_k],
            curriculum_max_stage=curriculum_max_stage,
            resume_path=None,
            resume_same_stage=False,
            resume_stage=stage_k,
            resume_epoch=0,
            resume_step_in_epoch=-1,
            global_step=global_step_offset,
            step_in_phase=0,
            load_best_between_stages=False,
            run_epoch_end_val=False,
        )
    finally:
        args.push_to_hub = original_push_to_hub
        args.stage_0_epochs = original_stage0_epochs
        args.epochs_per_stage = original_epochs_per_stage

    samples_seen_this_round = int(min(result["samples_seen"], len(train_shard)))

    barrier()
    if _is_main_process():
        upload_dir = output_dir / "diloco_worker_upload" / f"worker_{args.diloco_worker_id}_stage_{stage_k}_round_{round_n}"
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
        upload_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(upload_dir))

        diloco_upload_worker_state(
            adapter_dir=upload_dir,
            worker_id=args.diloco_worker_id,
            stage_k=stage_k,
            round_n=round_n,
            samples_seen=samples_seen_this_round,
            hf_token=hf_token,
            repo_id=args.diloco_state_repo,
            halt_gate=halt_gate if args.use_halt_gate else None,
        )

        github_token = _resolve_github_token_common()
        if github_token and args.diloco_signal_repo:
            diloco_push_signal(
                args.diloco_worker_id,
                stage_k,
                round_n,
                github_token,
                args.diloco_signal_repo,
            )
        else:
            print("  [diloco] No GITHUB_TOKEN - coordinator must be triggered manually.")

        print(
            f"  [diloco] Worker {args.diloco_worker_id} done. "
            f"stage={stage_k} round={round_n} samples_seen={samples_seen_this_round}"
        )
    barrier()

    if diloco_wandb_run is not None:
        import wandb
        wandb.log(
            {
                "diloco/round": round_n,
                "diloco/stage": stage_k,
                "diloco/samples_seen_this_round": samples_seen_this_round,
                "diloco/round_complete": 1,
            },
            step=global_step_offset + shard_step_estimate,
        )
        wandb.finish()

    return {
        "stage_k": stage_k,
        "round_n": round_n,
        "samples_seen": samples_seen_this_round,
        **result,
    }
