"""CPU-safe Kaggle workflow validation helpers.

The validation path proves notebook/runtime plumbing without importing torch,
starting torchrun, touching CUDA, or consuming Kaggle GPU quota. It is safe to
import before the training bootstrap. Publishing the validation result to Hub is
explicitly opt-in so ordinary local smoke tests remain network-free.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ouroboros.coordinator.kaggle_commands import kaggle_secret_presence, resolve_diloco_worker_id
from ouroboros.utils.runtime_env import env_bool, env_int, normalize_text, resolve_env_alias, resolve_hf_token
from ouroboros.utils.kaggle_runtime import KaggleRepoSpec, resolve_kaggle_repo_spec

CPU_SMOKE_MODE = "cpu-smoke"
DEFAULT_VALIDATION_DIR = Path("runs/workflow_validation")
DEFAULT_VALIDATION_PREFIX = "diloco_state/workflow_validation"
DEFAULT_STATE_REPO = "WeirdRunner/Ouroboros"
Emitter = Callable[[str], None]
JsonUploader = Callable[[str, str, dict[str, object], str, str], None]


@dataclass(frozen=True)
class CpuSmokeValidationReport:
    """Serializable report emitted by the CPU-safe workflow validation path."""

    mode: str
    worker_id: str
    validation_run_id: str
    repo_url: str
    repo_ref: str
    repo_commit: str
    state_repo: str
    command: list[str]
    status_path: str
    remote_status_path: str
    remote_report_path: str
    secret_presence: dict[str, bool]
    gpu_requested: bool
    torchrun_requested: bool
    push_to_hub_requested: bool
    publish_requested: bool
    published: bool
    timestamp: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def workflow_validation_mode(env: Mapping[str, str] | None = None) -> str | None:
    """Return the requested validation mode, if any."""
    env = os.environ if env is None else env
    value = normalize_text(env.get("OUROBOROS_WORKFLOW_VALIDATE"))
    return value.lower() if value else None


def is_cpu_smoke_validation_enabled(env: Mapping[str, str] | None = None) -> bool:
    return workflow_validation_mode(env) == CPU_SMOKE_MODE


def is_validation_publish_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return True when CPU smoke should publish remote validation artifacts."""
    env = os.environ if env is None else env
    return env_bool(env, "OUROBOROS_WORKFLOW_VALIDATION_PUBLISH")


def resolve_workflow_validation_run_id(env: Mapping[str, str] | None = None) -> str:
    """Resolve a stable run id for grouping remote workflow-validation reports."""
    env = os.environ if env is None else env
    for key in ("OUROBOROS_WORKFLOW_VALIDATION_RUN_ID", "GITHUB_RUN_ID"):
        value = str(env.get(key, "")).strip()
        if value:
            attempt = str(env.get("GITHUB_RUN_ATTEMPT", "")).strip()
            if key == "GITHUB_RUN_ID" and attempt:
                return f"{value}-{attempt}"
            return value
    return f"local-{int(time.time())}"


def resolve_workflow_validation_state_repo(env: Mapping[str, str] | None = None) -> str:
    """Resolve the Hub repo that stores validation artifacts."""
    env = os.environ if env is None else env
    return resolve_env_alias(
        env,
        ("OUROBOROS_WORKFLOW_VALIDATION_STATE_REPO", "OUROBOROS_DILOCO_STATE_REPO"),
    ) or DEFAULT_STATE_REPO


def workflow_validation_remote_paths(
    *,
    run_id: str,
    worker_id: str,
    prefix: str = DEFAULT_VALIDATION_PREFIX,
) -> tuple[str, str]:
    """Return remote Hub paths for a worker's validation status and report."""
    normalized_worker = worker_id.strip().upper()
    clean_prefix = prefix.strip("/") or DEFAULT_VALIDATION_PREFIX
    clean_run = str(run_id).strip().strip("/")
    if not clean_run:
        raise ValueError("workflow validation run id is required")
    return (
        f"{clean_prefix}/{clean_run}/worker_{normalized_worker}_status.json",
        f"{clean_prefix}/{clean_run}/worker_{normalized_worker}_report.json",
    )


def _int_from_env(env: Mapping[str, str], name: str, default: int) -> int:
    return env_int(env, name, default=default)


def _token_from_env(env: Mapping[str, str]) -> str:
    return resolve_hf_token(env=env) or ""


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
        "ouroboros.eval.smoke_worker",
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
    validation_run_id: str | None = None,
    remote_status_path: str | None = None,
    timestamp: float | None = None,
) -> dict[str, object]:
    """Write a local fake worker status matching the coordinator's status schema."""
    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {
        "worker_id": worker_id,
        "stage_k": int(stage_k),
        "round_n": int(round_n),
        "samples_seen": 0,
        "status": "done",
        "timestamp": time.time() if timestamp is None else float(timestamp),
        "weights_path": f"local_validation/workers/{worker_id}/round_{int(round_n):04d}_stage_{int(stage_k)}",
        "validation_mode": CPU_SMOKE_MODE,
    }
    if validation_run_id is not None:
        status["validation_run_id"] = validation_run_id
    if remote_status_path is not None:
        status["remote_status_path"] = remote_status_path
    path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def upload_json_to_hub(
    repo_id: str,
    token: str,
    path: str,
    data: dict[str, object],
    message: str,
) -> None:
    """Upload JSON to Hugging Face Hub.

    The import is intentionally local so importing this module remains cheap and
    torch-free in the Kaggle notebook bootstrap path.
    """
    from huggingface_hub import HfApi

    if not token:
        raise ValueError("HF token is required to publish workflow validation artifacts")
    api = HfApi(token=token)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(data, tf, indent=2)
        tf.write("\n")
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


def _build_report(
    *,
    env: Mapping[str, str],
    repo_spec: KaggleRepoSpec,
    worker_id: str,
    validation_run_id: str,
    state_repo: str,
    command: Sequence[str],
    status_path: Path,
    remote_status_path: str,
    remote_report_path: str,
    publish_requested: bool,
    published: bool,
) -> CpuSmokeValidationReport:
    command_list = list(command)
    return CpuSmokeValidationReport(
        mode=CPU_SMOKE_MODE,
        worker_id=worker_id,
        validation_run_id=validation_run_id,
        repo_url=repo_spec.repo_url,
        repo_ref=repo_spec.repo_ref,
        repo_commit=repo_spec.repo_commit,
        state_repo=state_repo,
        command=command_list,
        status_path=str(status_path),
        remote_status_path=remote_status_path,
        remote_report_path=remote_report_path,
        secret_presence=kaggle_secret_presence(env),
        gpu_requested=any(part in {"--use_4bit", "--diloco_mode"} for part in command_list),
        torchrun_requested=bool(command_list and Path(command_list[0]).name == "torchrun"),
        push_to_hub_requested="--push_to_hub" in command_list,
        publish_requested=publish_requested,
        published=published,
        timestamp=time.time(),
    )


def run_cpu_smoke_validation(
    env: Mapping[str, str] | None = None,
    *,
    output_dir: Path | str = DEFAULT_VALIDATION_DIR,
    emit: Emitter = print,
    upload_json: JsonUploader | None = None,
) -> CpuSmokeValidationReport:
    """Run the CPU-safe validation branch used by staged Kaggle notebooks.

    The local path writes a coordinator-compatible status JSON and a report. When
    `OUROBOROS_WORKFLOW_VALIDATION_PUBLISH=1`, the same status/report are also
    uploaded under `diloco_state/workflow_validation/<run_id>/` so the GitHub
    coordinator can verify the Kaggle notebook completed without consuming GPU.
    """
    env = os.environ if env is None else env
    worker_id = resolve_diloco_worker_id(env)
    repo_spec = resolve_kaggle_repo_spec(env)
    stage_k = _int_from_env(env, "OUROBOROS_VALIDATION_STAGE_K", 0)
    round_n = _int_from_env(env, "OUROBOROS_VALIDATION_ROUND_N", 0)
    validation_run_id = resolve_workflow_validation_run_id(env)
    state_repo = resolve_workflow_validation_state_repo(env)
    remote_status_path, remote_report_path = workflow_validation_remote_paths(
        run_id=validation_run_id,
        worker_id=worker_id,
    )
    output_path = Path(output_dir)
    status_path = output_path / f"worker_{worker_id}_status.json"
    command = build_cpu_smoke_validation_command(
        worker_id=worker_id,
        stage_k=stage_k,
        round_n=round_n,
        status_path=status_path,
    )
    status = write_cpu_smoke_worker_status(
        worker_id=worker_id,
        stage_k=stage_k,
        round_n=round_n,
        status_path=status_path,
        validation_run_id=validation_run_id,
        remote_status_path=remote_status_path,
    )
    publish_requested = is_validation_publish_enabled(env)
    report = _build_report(
        env=env,
        repo_spec=repo_spec,
        worker_id=worker_id,
        validation_run_id=validation_run_id,
        state_repo=state_repo,
        command=command,
        status_path=status_path,
        remote_status_path=remote_status_path,
        remote_report_path=remote_report_path,
        publish_requested=publish_requested,
        published=False,
    )
    if report.gpu_requested or report.torchrun_requested or report.push_to_hub_requested:
        raise RuntimeError(f"CPU smoke validation built an unsafe command: {command!r}")

    uploader = upload_json or upload_json_to_hub
    if publish_requested:
        token = _token_from_env(env)
        uploader(
            state_repo,
            token,
            remote_status_path,
            status,
            f"Workflow validation status: {validation_run_id} worker {worker_id}",
        )
        report = _build_report(
            env=env,
            repo_spec=repo_spec,
            worker_id=worker_id,
            validation_run_id=validation_run_id,
            state_repo=state_repo,
            command=command,
            status_path=status_path,
            remote_status_path=remote_status_path,
            remote_report_path=remote_report_path,
            publish_requested=True,
            published=True,
        )
        uploader(
            state_repo,
            token,
            remote_report_path,
            report.to_dict(),
            f"Workflow validation report: {validation_run_id} worker {worker_id}",
        )

    report_path = output_path / "report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    emit(f"[workflow-validate] CPU smoke validation complete: {report_path}")
    if report.published:
        emit(
            "[workflow-validate] Published CPU smoke validation artifacts: "
            f"{report.remote_status_path}, {report.remote_report_path}"
        )
    emit(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return report


__all__ = [
    "CPU_SMOKE_MODE",
    "DEFAULT_VALIDATION_PREFIX",
    "CpuSmokeValidationReport",
    "build_cpu_smoke_validation_command",
    "is_cpu_smoke_validation_enabled",
    "is_validation_publish_enabled",
    "resolve_workflow_validation_run_id",
    "resolve_workflow_validation_state_repo",
    "run_cpu_smoke_validation",
    "upload_json_to_hub",
    "workflow_validation_mode",
    "workflow_validation_remote_paths",
    "write_cpu_smoke_worker_status",
]
