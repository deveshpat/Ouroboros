"""Strict local Mac fallback orchestration for DGAC DiLoCo rounds.

This module keeps the Mac path deliberately separate from the GitHub/Kaggle
dispatcher. It owns only the guardrails and command construction needed for a
single local Apple Silicon fallback worker run:

1. prove the live Hub round has not drifted from the expected DGAC state;
2. prove the local runtime is an MPS + Jamba-compatible path;
3. write a short-lived Hub claim so GitHub Actions will not race the Mac run;
4. reactivate the exact waiting DGAC round for one local Mac worker; and
5. run local aggregation with ``--skip_trigger`` under the matching claim.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from ouroboros.runtime_env import parse_worker_id_list, resolve_hf_token


MAC_DGAC_CLAIM_PATH = "diloco_state/locks/mac_dgac_fallback.json"
ROUND_STATE_PATH = "diloco_state/round_state.json"
ANCHOR_PREFIX = "diloco_state/anchor"
DEFAULT_DGAC_TOTAL_TRAIN_SAMPLES = 36906
DEFAULT_MAC_DGAC_WORKERS = ("A",)
CANARY_LATEST_DIRNAME = "canary_latest"
CANARY_ACTIVE_PREFIX = "canary_active_"
ACTIVE_MAC_CLAIM_STATUSES = {"claimed", "running", "preflight-passed"}


def _env_text(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class ExpectedRoundState:
    stage_k: int = 10
    round_n: int = 3
    mode: str = "waiting"
    total_samples_seen: int = 23481
    projected_shards: Mapping[str, int] = field(
        default_factory=lambda: {"A": 4475, "B": 4475, "C": 4475}
    )
    require_dgac_diloco: bool = True
    max_total_samples_seen_drift: int = 0


@dataclass(frozen=True)
class MacRuntimeProbe:
    platform_system: str
    platform_machine: str
    mps_available: bool
    cuda_available: bool
    mamba_ssm_macos_available: bool
    mamba_probe_passed: bool
    jamba_forward_backward_passed: bool
    anchor_adapter_loaded: bool
    halt_gate_loaded: bool
    use_4bit_requested: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MacPreflightReport:
    canonical_writes_allowed: bool
    quantization: str
    parity_classification: str
    reasons: list[str]
    expected: ExpectedRoundState
    runtime_probe: MacRuntimeProbe


def _stage_total_samples_seen(state: Mapping[str, Any], stage_k: int) -> int:
    totals = dict(state.get("total_samples_seen", {}) or {})
    return int(totals.get(str(stage_k), totals.get(stage_k, 0)))


def _round_state_drift_reasons(
    state: Optional[Mapping[str, Any]],
    *,
    expected: ExpectedRoundState,
) -> list[str]:
    if not isinstance(state, Mapping):
        return ["round_state is missing or not a JSON object"]

    reasons: list[str] = []
    stage_k = int(state.get("stage_k", -1))
    round_n = int(state.get("round_n", -1))
    mode = str(state.get("mode", ""))
    if stage_k != int(expected.stage_k):
        reasons.append(f"stage_k drifted: expected {expected.stage_k}, live {stage_k}")
    if round_n != int(expected.round_n):
        reasons.append(f"round_n drifted: expected {expected.round_n}, live {round_n}")
    if mode != expected.mode:
        reasons.append(f"mode drifted: expected {expected.mode!r}, live {mode!r}")

    actual_seen = _stage_total_samples_seen(state, expected.stage_k)
    allowed = max(int(expected.max_total_samples_seen_drift), 0)
    if abs(actual_seen - int(expected.total_samples_seen)) > allowed:
        reasons.append(
            f"total_samples_seen[{expected.stage_k}] drifted: "
            f"expected {expected.total_samples_seen}, live {actual_seen}"
        )

    projected = dict(state.get("projected_shards", {}) or {})
    for worker_id, expected_size in expected.projected_shards.items():
        actual_size = int(projected.get(worker_id, -1))
        if actual_size != int(expected_size):
            reasons.append(
                f"projected_shards[{worker_id}] drifted: "
                f"expected {expected_size}, live {actual_size}"
            )

    if expected.require_dgac_diloco and not bool(state.get("dgac_diloco")):
        reasons.append("round_state is not marked dgac_diloco=true")
    return reasons


def _runtime_probe_reasons(probe: MacRuntimeProbe) -> list[str]:
    reasons: list[str] = []
    if probe.platform_system != "Darwin":
        reasons.append(f"platform must be Darwin/macOS, got {probe.platform_system!r}")
    if probe.platform_machine not in {"arm64", "aarch64"}:
        reasons.append(f"platform machine must be Apple Silicon arm64, got {probe.platform_machine!r}")
    if not probe.mps_available:
        reasons.append("torch.backends.mps.is_available() is false")
    if probe.cuda_available:
        reasons.append("CUDA is available; strict Mac fallback requires CUDA-unavailable MPS execution")
    if probe.use_4bit_requested:
        reasons.append("--use_4bit is forbidden on strict Mac fallback because bitsandbytes requires CUDA")
    if not probe.mamba_ssm_macos_available:
        reasons.append("mamba-ssm-macos distribution is not installed")
    if not probe.mamba_probe_passed:
        reasons.append("mamba-ssm-macos probe did not pass")
    if not probe.jamba_forward_backward_passed:
        reasons.append("Jamba forward/backward MPS probe did not pass")
    if not probe.anchor_adapter_loaded:
        reasons.append("DiLoCo anchor adapter could not be loaded from Hub")
    if not probe.halt_gate_loaded:
        reasons.append("DiLoCo anchor halt_gate.pt could not be loaded from Hub")
    return reasons


def _quantization_classification(probe: MacRuntimeProbe) -> str:
    if probe.use_4bit_requested:
        return "unsupported-4bit-on-mps"
    if probe.mps_available and not probe.cuda_available:
        return "mps-fp16-lora-no-4bit"
    return "unknown-or-unsupported"


def evaluate_mac_preflight(
    *,
    state: Optional[Mapping[str, Any]],
    expected: ExpectedRoundState,
    runtime_probe: MacRuntimeProbe,
) -> MacPreflightReport:
    reasons = [
        *_round_state_drift_reasons(state, expected=expected),
        *_runtime_probe_reasons(runtime_probe),
    ]
    allowed = not reasons
    return MacPreflightReport(
        canonical_writes_allowed=allowed,
        quantization=_quantization_classification(runtime_probe),
        parity_classification="canonical-write-eligible" if allowed else "refuse-canonical-write",
        reasons=reasons,
        expected=expected,
        runtime_probe=runtime_probe,
    )


def is_active_mac_claim(claim: Optional[Mapping[str, Any]], *, now: Optional[float] = None) -> bool:
    if not isinstance(claim, Mapping):
        return False
    status = str(claim.get("status", "")).lower()
    if status not in ACTIVE_MAC_CLAIM_STATUSES:
        return False
    expires_at = float(claim.get("expires_at", 0.0) or 0.0)
    current = time.time() if now is None else float(now)
    return expires_at > current


def mac_claim_matches(claim: Optional[Mapping[str, Any]], claim_id: Optional[str]) -> bool:
    if not isinstance(claim, Mapping) or not claim_id:
        return False
    return str(claim.get("claim_id", "")) == str(claim_id)


def create_mac_claim(
    *,
    claim_id: Optional[str] = None,
    now: Optional[float] = None,
    ttl_hours: float = 16.0,
    worker_ids: Sequence[str] = DEFAULT_MAC_DGAC_WORKERS,
    expected: ExpectedRoundState = ExpectedRoundState(),
) -> dict[str, Any]:
    current = time.time() if now is None else float(now)
    ttl_s = max(float(ttl_hours), 0.1) * 3600.0
    normalized_workers = parse_worker_id_list(list(worker_ids)) or list(DEFAULT_MAC_DGAC_WORKERS)
    return {
        "claim_id": claim_id or f"mac-dgac-{uuid.uuid4().hex[:12]}",
        "status": "running",
        "created_at": current,
        "updated_at": current,
        "expires_at": current + ttl_s,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "worker_ids": normalized_workers,
        "expected_stage_k": int(expected.stage_k),
        "expected_round_n": int(expected.round_n),
        "expected_mode": expected.mode,
        "expected_total_samples_seen": int(expected.total_samples_seen),
        "expected_projected_shards": dict(expected.projected_shards),
    }


def build_mac_controlled_round_state(
    *,
    state: Mapping[str, Any],
    worker_ids: Sequence[str],
    claim_id: str,
    now: Optional[float] = None,
) -> dict[str, Any]:
    current = time.time() if now is None else float(now)
    normalized_workers = parse_worker_id_list(list(worker_ids)) or list(DEFAULT_MAC_DGAC_WORKERS)
    return {
        **dict(state),
        "mode": "dgac-diloco",
        "triggered_workers": normalized_workers,
        "attendance_workers": [],
        "anchor_path": state.get("anchor_path", ANCHOR_PREFIX),
        "projected_shards": dict(state.get("projected_shards", {}) or {}),
        "total_samples_seen": dict(state.get("total_samples_seen", {}) or {}),
        "dgac_diloco": True,
        "dgac_manual_gate": False,
        "mac_dgac_claim_id": claim_id,
        "mac_dgac_claimed_at": current,
        "last_updated": current,
        "triggered_at": current,
        "dispatch_failures": [],
    }


def build_mac_failed_claim(
    *,
    claim: Mapping[str, Any],
    error: str,
    now: Optional[float] = None,
) -> dict[str, Any]:
    current = time.time() if now is None else float(now)
    return {
        **dict(claim),
        "status": "failed",
        "updated_at": current,
        "failure": str(error),
    }


def build_mac_failure_round_state(
    *,
    state: Mapping[str, Any],
    claim_id: str,
    error: str,
    now: Optional[float] = None,
) -> dict[str, Any]:
    current = time.time() if now is None else float(now)
    return {
        **dict(state),
        "mac_dgac_claim_id": claim_id,
        "mac_dgac_failed_at": current,
        "mac_dgac_failure": str(error),
        "last_updated": current,
    }


def build_local_dgac_worker_command(
    *,
    worker_id: str,
    repo_id: str,
    output_root: str = "runs/mac_dgac_fallback",
    outer_lr: float = 0.7,
    max_seq_len: int = 384,
    grad_accum: int = 8,
    dgac_halt_probe_steps: str = "stage_k",
    python_executable: str = sys.executable,
    script: str = "jamba_coconut_finetune.py",
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_mode: str = "online",
) -> list[str]:
    worker = parse_worker_id_list([worker_id])[0]
    command = [
        python_executable,
        script,
        "--data_dir",
        "data/coconut_v1",
        "--use_halt_gate",
        "--resume_from_diloco_anchor",
        "--mac_mps_mamba_kernels",
        "--stage_0_epochs",
        "1",
        "--epochs_per_stage",
        "1",
        "--max_stage",
        "10",
        "--max_seq_len",
        str(int(max_seq_len)),
        "--max_grad_norm",
        "0.3",
        "--dgac_halt_probe_steps",
        dgac_halt_probe_steps,
        "--batch_size",
        "1",
        "--grad_accum",
        str(int(grad_accum)),
        "--val_batch_size",
        "1",
        "--val_skip_buffer_minutes",
        "720",
        "--session_timeout_hours",
        "24.0",
        "--graceful_exit_buffer_minutes",
        "20",
        "--diloco_mode",
        "--diloco_worker_id",
        worker,
        "--diloco_outer_lr",
        str(float(outer_lr)),
        "--diloco_state_repo",
        repo_id,
        "--diloco_signal_repo",
        "",
        "--push_to_hub",
        "--output_dir",
        f"{output_root.rstrip('/')}/worker_{worker}",
        "--wandb_mode",
        wandb_mode,
        "--wandb_project",
        wandb_project or _env_text("OUROBOROS_WANDB_PROJECT") or "ouroboros-stage3-jamba",
        "--log_every",
        "1",
        "--profile_training_timing",
    ]
    resolved_entity = wandb_entity or _env_text("OUROBOROS_WANDB_ENTITY")
    if resolved_entity:
        command.extend(["--wandb_entity", resolved_entity])
    return command


def build_local_dgac_canary_command(
    *,
    repo_id: str,
    output_root: str = "runs/mac_dgac_fallback",
    output_dir: Optional[str] = None,
    python_executable: str = sys.executable,
    script: str = "jamba_coconut_finetune.py",
    max_train_steps: int = 1,
    grad_accum: int = 8,
    max_samples: int = 512,
    max_seq_len: int = 384,
    dgac_halt_probe_steps: str = "stage_k",
    disable_grad_checkpoint: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_mode: str = "online",
) -> list[str]:
    command = [
        python_executable,
        script,
        "--data_dir",
        "data/coconut_v1",
        "--use_halt_gate",
        "--resume_from_diloco_anchor",
        "--mac_mps_mamba_kernels",
        "--stage_0_epochs",
        "1",
        "--epochs_per_stage",
        "1",
        "--max_stage",
        "10",
        "--max_seq_len",
        str(int(max_seq_len)),
        "--max_grad_norm",
        "0.3",
        "--dgac_halt_probe_steps",
        dgac_halt_probe_steps,
        "--batch_size",
        "1",
        "--grad_accum",
        str(int(grad_accum)),
        "--max_train_steps",
        str(int(max_train_steps)),
        "--max_samples",
        str(int(max_samples)),
        "--val_batch_size",
        "1",
        "--val_skip_buffer_minutes",
        "720",
        "--session_timeout_hours",
        "2.0",
        "--graceful_exit_buffer_minutes",
        "5",
        "--diloco_state_repo",
        repo_id,
        "--output_dir",
        output_dir or f"{output_root.rstrip('/')}/{CANARY_LATEST_DIRNAME}",
        "--wandb_mode",
        wandb_mode,
        "--wandb_project",
        wandb_project or _env_text("OUROBOROS_WANDB_PROJECT") or "ouroboros-stage3-jamba",
        "--wandb_run_name",
        "Mac DGAC profiler canary",
        "--log_every",
        "1",
        "--no-gen_every_stage",
        "--profile_training_timing",
    ]
    if disable_grad_checkpoint:
        command.append("--no-grad_checkpoint")
    resolved_entity = wandb_entity or _env_text("OUROBOROS_WANDB_ENTITY")
    if resolved_entity:
        command.extend(["--wandb_entity", resolved_entity])
    return command


def _canary_path_key(path: Path) -> tuple[float, str]:
    try:
        return (path.stat().st_mtime, path.name)
    except FileNotFoundError:
        return (0.0, path.name)


def promote_canary_output(active_dir: Path, latest_dir: Path) -> None:
    if not active_dir.exists():
        return
    backup_dir: Optional[Path] = None
    if latest_dir.exists():
        backup_dir = latest_dir.with_name(f"{latest_dir.name}.previous")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        latest_dir.rename(backup_dir)
    active_dir.rename(latest_dir)
    if backup_dir is not None and backup_dir.exists():
        shutil.rmtree(backup_dir)


def prune_canary_outputs(output_root: Path, *, keep_latest: bool = True) -> None:
    if not output_root.exists():
        return
    latest = output_root / CANARY_LATEST_DIRNAME
    keep_names = {latest.name} if keep_latest and latest.exists() else set()
    for path in sorted(output_root.iterdir(), key=_canary_path_key):
        if not path.is_dir():
            continue
        if path.name in keep_names:
            continue
        if path.name == "canary" or path.name.startswith("canary_") or path.name.startswith(CANARY_ACTIVE_PREFIX):
            shutil.rmtree(path)


def build_local_dgac_aggregation_command(
    *,
    repo_id: str,
    claim_id: str,
    worker_ids: Sequence[str] = DEFAULT_MAC_DGAC_WORKERS,
    total_train_samples: int = DEFAULT_DGAC_TOTAL_TRAIN_SAMPLES,
    outer_lr: float = 0.7,
    min_shard_samples: int = 32,
    python_executable: str = sys.executable,
    script: str = "diloco_coordinator.py",
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
) -> list[str]:
    normalized_workers = parse_worker_id_list(list(worker_ids)) or list(DEFAULT_MAC_DGAC_WORKERS)
    command = [
        python_executable,
        script,
        "--repo_id",
        repo_id,
        "--outer_lr",
        str(float(outer_lr)),
        "--worker_timeout_hours",
        "13.0",
        "--total_train_samples",
        str(int(total_train_samples)),
        "--min_shard_samples",
        str(int(min_shard_samples)),
        "--force_worker_ids",
        ",".join(normalized_workers),
        "--skip_trigger",
        "--mac_claim_id",
        claim_id,
        "--wandb_project",
        wandb_project or _env_text("OUROBOROS_WANDB_PROJECT") or "ouroboros-stage3-jamba",
    ]
    resolved_entity = wandb_entity or _env_text("OUROBOROS_WANDB_ENTITY")
    if resolved_entity:
        command.extend(["--wandb_entity", resolved_entity])
    return command


def _run_mamba_probe() -> tuple[bool, bool, dict[str, Any]]:
    details: dict[str, Any] = {}
    try:
        from importlib import metadata

        details["mamba_ssm_macos_version"] = metadata.version("mamba-ssm-macos")
        distribution_available = True
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic probe
        details["mamba_ssm_macos_error"] = f"{type(exc).__name__}: {exc}"
        distribution_available = False

    try:
        importlib.invalidate_caches()
        mamba = importlib.import_module("mamba_ssm")
        details["mamba_ssm_module"] = getattr(mamba, "__file__", "")
        importlib.import_module("mamba_ssm.ops.selective_scan_interface")
        return distribution_available, True, details
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic probe
        details["mamba_probe_error"] = f"{type(exc).__name__}: {exc}"
        return distribution_available, False, details


def _run_anchor_probe(*, repo_id: str, token: str) -> tuple[bool, bool, dict[str, Any]]:
    details: dict[str, Any] = {}
    try:
        from huggingface_hub import hf_hub_download

        adapter = hf_hub_download(repo_id=repo_id, filename=f"{ANCHOR_PREFIX}/adapter_model.safetensors", token=token)
        config = hf_hub_download(repo_id=repo_id, filename=f"{ANCHOR_PREFIX}/adapter_config.json", token=token)
        gate = hf_hub_download(repo_id=repo_id, filename=f"{ANCHOR_PREFIX}/halt_gate.pt", token=token)
        details["anchor_adapter_path"] = adapter
        details["anchor_config_path"] = config
        details["halt_gate_path"] = gate
        return Path(adapter).exists() and Path(config).exists(), Path(gate).exists(), details
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic probe
        details["anchor_probe_error"] = f"{type(exc).__name__}: {exc}"
        return False, False, details


def _run_jamba_forward_backward_probe(*, model_id: str, device: str = "mps") -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {}
    model = None
    outputs = None
    loss = None
    batch = None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_device = torch.device(device)
        dtype = torch.float16 if torch_device.type == "mps" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            use_mamba_kernels=False,
            attn_implementation="eager",
        ).to(torch_device)
        model.train()
        batch = tokenizer("Ouroboros Mac DGAC probe.", return_tensors="pt")
        batch = {key: value.to(torch_device) for key, value in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss.backward()
        details["loss"] = float(loss.detach().cpu())
        details["dtype"] = str(dtype).replace("torch.", "")
        details["use_mamba_kernels"] = False
        return True, details
    except Exception as exc:  # noqa: BLE001 - strict preflight reports the failure
        details["jamba_probe_error"] = f"{type(exc).__name__}: {exc}"
        return False, details
    finally:
        del outputs, loss, batch, model
        try:
            import torch

            if device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        except Exception:
            pass


def run_strict_mac_runtime_probe(
    *,
    repo_id: str,
    token: str,
    model_id: str = "ai21labs/AI21-Jamba-Reasoning-3B",
    use_4bit_requested: bool = False,
    jamba_probe: Optional[Callable[[], tuple[bool, Mapping[str, Any]]]] = None,
) -> MacRuntimeProbe:
    import torch

    mamba_available, mamba_passed, mamba_details = _run_mamba_probe()
    anchor_loaded, halt_gate_loaded, anchor_details = _run_anchor_probe(repo_id=repo_id, token=token)
    if jamba_probe is None:
        jamba_passed, jamba_details = _run_jamba_forward_backward_probe(model_id=model_id)
    else:
        jamba_passed, jamba_details = jamba_probe()

    details = {
        **mamba_details,
        **anchor_details,
        **dict(jamba_details),
        "torch_version": getattr(torch, "__version__", ""),
    }
    mps_backend = getattr(torch.backends, "mps", None)
    return MacRuntimeProbe(
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        mps_available=bool(mps_backend is not None and mps_backend.is_available()),
        cuda_available=bool(torch.cuda.is_available()),
        mamba_ssm_macos_available=mamba_available,
        mamba_probe_passed=mamba_passed,
        jamba_forward_backward_passed=bool(jamba_passed),
        anchor_adapter_loaded=anchor_loaded,
        halt_gate_loaded=halt_gate_loaded,
        use_4bit_requested=use_4bit_requested,
        details=details,
    )


def _worker_env(base_env: Mapping[str, str], *, claim_id: str) -> dict[str, str]:
    env = dict(base_env)
    env["OUROBOROS_MAC_DGAC_CLAIM_ID"] = claim_id
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OUROBOROS_WANDB_INIT_TIMEOUT", "300")
    env.pop("GITHUB_TOKEN", None)
    env.pop("GH_TOKEN", None)
    return env


def _print_report(report: MacPreflightReport) -> None:
    print(f"[mac-dgac] preflight parity={report.parity_classification} quantization={report.quantization}")
    if report.reasons:
        print("[mac-dgac] refusing canonical Hub writes:")
        for reason in report.reasons:
            print(f"  - {reason}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict local Mac DGAC fallback runner")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--repo_id", default="WeirdRunner/Ouroboros")
    parser.add_argument("--model_id", default="ai21labs/AI21-Jamba-Reasoning-3B")
    parser.add_argument(
        "--workers",
        default=",".join(DEFAULT_MAC_DGAC_WORKERS),
        help=(
            "Local Mac fallback worker IDs. Default is one local worker only; "
            "A/B/C multi-worker orchestration belongs to Kaggle accounts."
        ),
    )
    parser.add_argument("--output_root", default="runs/mac_dgac_fallback")
    parser.add_argument("--outer_lr", type=float, default=0.7)
    parser.add_argument(
        "--local_grad_accum",
        type=int,
        default=8,
        help="Local Mac worker gradient accumulation. Lower than CUDA so W&B emits usable live metrics.",
    )
    parser.add_argument("--total_train_samples", type=int, default=DEFAULT_DGAC_TOTAL_TRAIN_SAMPLES)
    parser.add_argument("--min_shard_samples", type=int, default=32)
    parser.add_argument("--claim_ttl_hours", type=float, default=16.0)
    parser.add_argument("--claim_id", default=None)
    parser.add_argument("--expected_stage_k", type=int, default=10)
    parser.add_argument("--expected_round_n", type=int, default=3)
    parser.add_argument("--expected_mode", default="waiting")
    parser.add_argument("--expected_total_samples_seen", type=int, default=23481)
    parser.add_argument("--force_claim_takeover", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--canary",
        action="store_true",
        help="Run a standalone one-step Mac profiler canary without Hub claims, worker uploads, or aggregation.",
    )
    parser.add_argument("--canary_max_train_steps", type=int, default=1)
    parser.add_argument("--canary_grad_accum", type=int, default=8)
    parser.add_argument("--canary_max_samples", type=int, default=512)
    parser.add_argument("--canary_max_seq_len", type=int, default=384)
    parser.add_argument("--canary_dgac_halt_probe_steps", default="stage_k")
    parser.add_argument(
        "--canary_no_grad_checkpoint",
        action="store_true",
        help="Experimental: disable gradient checkpointing for the Mac canary. May OOM on MPS.",
    )
    parser.add_argument("--use_4bit", action="store_true", help="Rejected by strict Mac preflight; present only as an explicit guard.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    hf_token = resolve_hf_token(args.hf_token)
    if not hf_token:
        print("[mac-dgac] FATAL: HF token required. Set HF_TOKEN or pass --hf_token.")
        return 2
    os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    from ouroboros.diloco.coordinator import hub_download_json, hub_upload_json

    worker_ids = parse_worker_id_list(args.workers) or list(DEFAULT_MAC_DGAC_WORKERS)
    expected = ExpectedRoundState(
        stage_k=args.expected_stage_k,
        round_n=args.expected_round_n,
        mode=args.expected_mode,
        total_samples_seen=args.expected_total_samples_seen,
    )
    state = hub_download_json(args.repo_id, ROUND_STATE_PATH, hf_token)
    runtime_probe = run_strict_mac_runtime_probe(
        repo_id=args.repo_id,
        token=hf_token,
        model_id=args.model_id,
        use_4bit_requested=bool(args.use_4bit),
    )
    report = evaluate_mac_preflight(state=state, expected=expected, runtime_probe=runtime_probe)
    _print_report(report)
    if not report.canonical_writes_allowed:
        return 2

    if args.canary:
        canary_claim_id = args.claim_id or f"mac-dgac-canary-{uuid.uuid4().hex[:12]}"
        env = _worker_env(os.environ, claim_id=canary_claim_id)
        canary_output_root = Path(args.output_root)
        active_output_dir = canary_output_root / f"{CANARY_ACTIVE_PREFIX}{uuid.uuid4().hex[:12]}"
        latest_output_dir = canary_output_root / CANARY_LATEST_DIRNAME
        prune_canary_outputs(canary_output_root, keep_latest=True)
        command = build_local_dgac_canary_command(
            repo_id=args.repo_id,
            output_root=args.output_root,
            output_dir=str(active_output_dir),
            max_train_steps=args.canary_max_train_steps,
            grad_accum=args.canary_grad_accum,
            max_samples=args.canary_max_samples,
            max_seq_len=args.canary_max_seq_len,
            dgac_halt_probe_steps=args.canary_dgac_halt_probe_steps,
            disable_grad_checkpoint=bool(args.canary_no_grad_checkpoint),
        )
        print("[mac-dgac] CANARY: no Hub claim, no worker upload, no aggregation.")
        print("[mac-dgac] running canary:", " ".join(command))
        if args.dry_run:
            return 0
        try:
            subprocess.run(command, check=True, env=env)
            promote_canary_output(active_output_dir, latest_output_dir)
            prune_canary_outputs(canary_output_root, keep_latest=True)
            return 0
        except subprocess.CalledProcessError as exc:
            if active_output_dir.exists():
                shutil.rmtree(active_output_dir)
            command_text = " ".join(str(part) for part in (exc.cmd or []))
            print(f"[mac-dgac] FATAL: canary subprocess exited {exc.returncode}: {command_text}")
            return int(exc.returncode or 1)
        except KeyboardInterrupt:
            if active_output_dir.exists():
                shutil.rmtree(active_output_dir)
            print("[mac-dgac] FATAL: canary interrupted by user")
            return 130

    existing_claim = hub_download_json(args.repo_id, MAC_DGAC_CLAIM_PATH, hf_token)
    if is_active_mac_claim(existing_claim) and not args.force_claim_takeover:
        print(f"[mac-dgac] FATAL: active Mac claim already exists: {existing_claim}")
        return 2

    claim = create_mac_claim(
        claim_id=args.claim_id,
        ttl_hours=args.claim_ttl_hours,
        worker_ids=worker_ids,
        expected=expected,
    )
    controlled_state = build_mac_controlled_round_state(
        state=state or {},
        worker_ids=worker_ids,
        claim_id=claim["claim_id"],
    )
    if args.dry_run:
        print("[mac-dgac] DRY RUN: preflight passed; no Hub writes or local workers started.")
        print(json.dumps({"claim": claim, "round_state": controlled_state}, indent=2))
        return 0

    hub_upload_json(args.repo_id, MAC_DGAC_CLAIM_PATH, claim, hf_token, "Mac DGAC fallback claim")
    hub_upload_json(
        args.repo_id,
        ROUND_STATE_PATH,
        controlled_state,
        hf_token,
        f"Mac DGAC fallback: claim {claim['claim_id']} controls round {expected.round_n}",
    )

    env = _worker_env(os.environ, claim_id=claim["claim_id"])

    def _record_failed_run(error: str, exit_code: int) -> int:
        failed_claim = build_mac_failed_claim(claim=claim, error=error)
        failure_state = build_mac_failure_round_state(
            state=state or {},
            claim_id=claim["claim_id"],
            error=error,
        )
        hub_upload_json(
            args.repo_id,
            MAC_DGAC_CLAIM_PATH,
            failed_claim,
            hf_token,
            "Mac DGAC fallback claim failed",
        )
        hub_upload_json(
            args.repo_id,
            ROUND_STATE_PATH,
            failure_state,
            hf_token,
            f"Mac DGAC fallback failed: restore waiting round {expected.round_n}",
        )
        return int(exit_code or 1)

    try:
        for worker_id in worker_ids:
            command = build_local_dgac_worker_command(
                worker_id=worker_id,
                repo_id=args.repo_id,
                output_root=args.output_root,
                outer_lr=args.outer_lr,
                grad_accum=args.local_grad_accum,
            )
            print("[mac-dgac] running worker:", " ".join(command))
            subprocess.run(command, check=True, env=env)

        aggregation = build_local_dgac_aggregation_command(
            repo_id=args.repo_id,
            claim_id=claim["claim_id"],
            worker_ids=worker_ids,
            total_train_samples=args.total_train_samples,
            outer_lr=args.outer_lr,
            min_shard_samples=args.min_shard_samples,
        )
        print("[mac-dgac] running local aggregation:", " ".join(aggregation))
        subprocess.run(aggregation, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        command_text = " ".join(str(part) for part in (exc.cmd or []))
        error = f"subprocess exited {exc.returncode}: {command_text}".strip()
        print(f"[mac-dgac] FATAL: {error}")
        return _record_failed_run(error, int(exc.returncode or 1))
    except KeyboardInterrupt:
        error = "interrupted by user"
        print(f"[mac-dgac] FATAL: {error}")
        return _record_failed_run(error, 130)

    completed_claim = {**claim, "status": "complete", "updated_at": time.time()}
    hub_upload_json(
        args.repo_id,
        MAC_DGAC_CLAIM_PATH,
        completed_claim,
        hf_token,
        "Mac DGAC fallback claim complete",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
