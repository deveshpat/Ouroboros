"""CPU-safe Kaggle workflow validation helpers.

The validation path proves notebook/runtime plumbing without importing torch,
starting torchrun, touching CUDA, pushing to Hub, or consuming Kaggle GPU quota.
It is intentionally stdlib-only so the Kaggle notebook can call it before the
training bootstrap.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ouroboros.kaggle import kaggle_secret_presence, resolve_diloco_worker_id
from ouroboros.kaggle_runtime import KaggleRepoSpec, resolve_kaggle_repo_spec

CPU_SMOKE_MODE = "cpu-smoke"
DEFAULT_VALIDATION_DIR = Path("runs/workflow_validation")
Emitter = Callable[[str], None]


@dataclass(frozen=True)
class CpuSmokeValidationReport:
    """Serializable report emitted by the CPU-safe workflow validation path."""

    mode: str
    worker_id: str
    repo_url: str
    repo_ref: str
    repo_commit: str
    command: list[str]
    status_path: str
    secret_presence: dict[str, bool]
    gpu_requested: bool
    torchrun_requested: bool
    push_to_hub_requested: bool
    timestamp: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def workflow_validation_mode(env: Mapping[str, str] | None = None) -> str | None:
    """Return the requested validation mode, if any."""
    env = os.environ if env is None else env
    value = str(env.get("OUROBOROS_WORKFLOW_VALIDATE", "")).strip().lower()
    return value or None


def is_cpu_smoke_validation_enabled(env: Mapping[str, str] | None = None) -> bool:
    return workflow_validation_mode(env) == CPU_SMOKE_MODE


def _int_from_env(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(str(env.get(name, "")).strip() or default)
    except ValueError:
        return default


def build_cpu_smoke_validation_command(
    *,
    worker_id: str,
    status_path: Path | str,
    stage_k: int = 0,
    round_n: int = 0,
) -> list[str]:
    """Build a CPU-only fake worker command for documentation/logging contracts."""
    return [
        sys.executable,
        "-m",
        "ouroboros.workflow_validation_worker",
        "--worker_id",
        worker_id,
        "--stage_k",
        str(int(stage_k)),
        "--round_n",
        str(int(round_n)),
        "--status_path",
        str(status_path),
    ]


def write_cpu_smoke_worker_status(
    *,
    worker_id: str,
    stage_k: int,
    round_n: int,
    status_path: Path | str,
    timestamp: float | None = None,
) -> dict[str, object]:
    """Write a local fake worker status matching the coordinator's status schema."""
    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "worker_id": worker_id,
        "stage_k": int(stage_k),
        "round_n": int(round_n),
        "samples_seen": 0,
        "status": "done",
        "timestamp": time.time() if timestamp is None else float(timestamp),
        "weights_path": f"local_validation/workers/{worker_id}/round_{int(round_n):04d}_stage_{int(stage_k)}",
        "validation_mode": CPU_SMOKE_MODE,
    }
    path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def _build_report(
    *,
    env: Mapping[str, str],
    repo_spec: KaggleRepoSpec,
    worker_id: str,
    command: Sequence[str],
    status_path: Path,
) -> CpuSmokeValidationReport:
    command_list = list(command)
    return CpuSmokeValidationReport(
        mode=CPU_SMOKE_MODE,
        worker_id=worker_id,
        repo_url=repo_spec.repo_url,
        repo_ref=repo_spec.repo_ref,
        repo_commit=repo_spec.repo_commit,
        command=command_list,
        status_path=str(status_path),
        secret_presence=kaggle_secret_presence(env),
        gpu_requested=any(part in {"--use_4bit", "--diloco_mode"} for part in command_list),
        torchrun_requested=bool(command_list and Path(command_list[0]).name == "torchrun"),
        push_to_hub_requested="--push_to_hub" in command_list,
        timestamp=time.time(),
    )


def run_cpu_smoke_validation(
    env: Mapping[str, str] | None = None,
    *,
    output_dir: Path | str = DEFAULT_VALIDATION_DIR,
    emit: Emitter = print,
) -> CpuSmokeValidationReport:
    """Run the CPU-safe validation branch used by staged Kaggle notebooks.

    This writes a local coordinator-compatible status JSON and prints a compact
    report. It does not run torchrun, import torch, request a GPU, or push to Hub.
    """
    env = os.environ if env is None else env
    worker_id = resolve_diloco_worker_id(env)
    repo_spec = resolve_kaggle_repo_spec(env)
    stage_k = _int_from_env(env, "OUROBOROS_VALIDATION_STAGE_K", 0)
    round_n = _int_from_env(env, "OUROBOROS_VALIDATION_ROUND_N", 0)
    output_path = Path(output_dir)
    status_path = output_path / f"worker_{worker_id}_status.json"
    command = build_cpu_smoke_validation_command(
        worker_id=worker_id,
        stage_k=stage_k,
        round_n=round_n,
        status_path=status_path,
    )
    write_cpu_smoke_worker_status(
        worker_id=worker_id,
        stage_k=stage_k,
        round_n=round_n,
        status_path=status_path,
    )
    report = _build_report(
        env=env,
        repo_spec=repo_spec,
        worker_id=worker_id,
        command=command,
        status_path=status_path,
    )
    if report.gpu_requested or report.torchrun_requested or report.push_to_hub_requested:
        raise RuntimeError(f"CPU smoke validation built an unsafe command: {command!r}")
    report_path = output_path / "report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    emit(f"[workflow-validate] CPU smoke validation complete: {report_path}")
    emit(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return report


__all__ = [
    "CPU_SMOKE_MODE",
    "CpuSmokeValidationReport",
    "build_cpu_smoke_validation_command",
    "is_cpu_smoke_validation_enabled",
    "run_cpu_smoke_validation",
    "workflow_validation_mode",
    "write_cpu_smoke_worker_status",
]
