"""Stdlib-only Kaggle launch helpers for Ouroboros notebooks.

The Kaggle notebook should stay a thin orchestration adapter. This module owns
reusable launch contracts that can be tested locally without Kaggle, CUDA, Hub,
or GitHub access.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping
from typing import Optional

_VALID_DILOCO_WORKER_IDS = {"A", "B", "C"}
DILOCO_RUN_MODE = "diloco"
DGAC_ANCHOR_EVAL_RUN_MODE = "dgac-anchor-eval"
DGAC_TRAIN_RUN_MODE = "dgac-train"
_VALID_KAGGLE_RUN_MODES = {DILOCO_RUN_MODE, DGAC_ANCHOR_EVAL_RUN_MODE, DGAC_TRAIN_RUN_MODE}


def _normalize_text(value: object | None, *, uppercase: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.upper() if uppercase else text


def _env_has_any(env: Mapping[str, str], names: tuple[str, ...]) -> bool:
    return any(_normalize_text(env.get(name)) is not None for name in names)


def resolve_diloco_worker_id(env: Mapping[str, str] | None = None) -> str:
    """Resolve and validate the DiLoCo worker id from notebook/runtime env aliases."""
    env = os.environ if env is None else env
    for env_name in ("DILOCO_WORKER_ID", "OUROBOROS_DILOCO_WORKER_ID", "WORKER_ID"):
        worker_id = _normalize_text(env.get(env_name), uppercase=True)
        if worker_id is None:
            continue
        if worker_id not in _VALID_DILOCO_WORKER_IDS:
            raise ValueError(
                f"Invalid DiLoCo worker id {worker_id!r}. Expected one of A, B, or C."
            )
        return worker_id
    raise ValueError(
        "DiLoCo worker id is required. Set DILOCO_WORKER_ID, "
        "OUROBOROS_DILOCO_WORKER_ID, or WORKER_ID."
    )


def kaggle_secret_presence(env: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Return secret availability booleans without exposing secret values."""
    env = os.environ if env is None else env
    return {
        "HF_TOKEN": _env_has_any(env, ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")),
        "WANDB_KEY": _env_has_any(env, ("WANDB_API_KEY", "WANDB_KEY")),
        "GITHUB_TOKEN": _env_has_any(env, ("GITHUB_TOKEN", "GH_TOKEN")),
        "DILOCO_WORKER_ID": _env_has_any(
            env,
            ("DILOCO_WORKER_ID", "OUROBOROS_DILOCO_WORKER_ID", "WORKER_ID"),
        ),
    }


def resolve_kaggle_run_mode(env: Mapping[str, str] | None = None) -> str:
    """Resolve the notebook launch mode from runtime env aliases."""
    env = os.environ if env is None else env
    mode = _normalize_text(
        env.get("OUROBOROS_KAGGLE_RUN_MODE") or env.get("OUROBOROS_RUN_MODE")
    )
    if mode is None:
        return DILOCO_RUN_MODE
    mode = mode.lower()
    if mode not in _VALID_KAGGLE_RUN_MODES:
        raise ValueError(
            f"Invalid Kaggle run mode {mode!r}. Expected one of: "
            f"{', '.join(sorted(_VALID_KAGGLE_RUN_MODES))}."
        )
    return mode


def build_diloco_training_command(
    *,
    worker_id: str,
    script: str = "jamba_coconut_finetune.py",
    nproc_per_node: int = 2,
    data_dir: str = "data/coconut_v1",
    use_4bit: bool = True,
    stage_0_epochs: int = 1,
    epochs_per_stage: int = 1,
    max_stage: int = 10,
    batch_size: int = 4,
    grad_accum: int = 8,
    val_batch_size: int = 2,
    val_skip_buffer_minutes: int = 60,
    session_timeout_hours: float = 12.0,
    graceful_exit_buffer_minutes: int = 20,
    diloco_outer_lr: float = 0.7,
    diloco_state_repo: str = "WeirdRunner/Ouroboros",
    diloco_signal_repo: str = "deveshpat/Ouroboros",
    output_dir: str = "runs/diloco",
    push_to_hub: bool = True,
    wandb_mode: str | None = None,
) -> list[str]:
    """Build the tested Kaggle Dual-GPU DiLoCo training command."""
    normalized_worker_id = _normalize_text(worker_id, uppercase=True)
    if normalized_worker_id not in _VALID_DILOCO_WORKER_IDS:
        raise ValueError(
            f"Invalid DiLoCo worker id {normalized_worker_id!r}. Expected one of A, B, or C."
        )

    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={int(nproc_per_node)}",
        script,
        "--data_dir",
        data_dir,
    ]
    if use_4bit:
        command.append("--use_4bit")
    command.extend(
        [
            "--stage_0_epochs",
            str(int(stage_0_epochs)),
            "--epochs_per_stage",
            str(int(epochs_per_stage)),
            "--max_stage",
            str(int(max_stage)),
            "--batch_size",
            str(int(batch_size)),
            "--grad_accum",
            str(int(grad_accum)),
            "--val_batch_size",
            str(int(val_batch_size)),
            "--val_skip_buffer_minutes",
            str(int(val_skip_buffer_minutes)),
            "--session_timeout_hours",
            str(float(session_timeout_hours)),
            "--graceful_exit_buffer_minutes",
            str(int(graceful_exit_buffer_minutes)),
            "--diloco_mode",
            "--diloco_worker_id",
            normalized_worker_id,
            "--diloco_outer_lr",
            str(float(diloco_outer_lr)),
            "--diloco_state_repo",
            diloco_state_repo,
            "--diloco_signal_repo",
            diloco_signal_repo,
        ]
    )
    if push_to_hub:
        command.append("--push_to_hub")
    command.extend(["--output_dir", output_dir])
    if wandb_mode is not None:
        command.extend(["--wandb_mode", wandb_mode])
    return command



def build_dgac_training_command(
    *,
    script: str = "jamba_coconut_finetune.py",
    nproc_per_node: int = 2,
    data_dir: str = "data/coconut_v1",
    use_4bit: bool = True,
    epochs_per_stage: int = 3,
    max_stage: int = 10,
    max_grad_norm: float = 0.3,
    batch_size: int = 4,
    grad_accum: int = 8,
    val_batch_size: int = 2,
    val_skip_buffer_minutes: int = 60,
    session_timeout_hours: float = 12.0,
    graceful_exit_buffer_minutes: int = 20,
    diloco_state_repo: str = "WeirdRunner/Ouroboros",
    output_dir: str = "runs/stage3_dgac",
    hf_stage_subdir: str | None = None,
    push_to_hub: bool = True,
    wandb_project: str | None = "ouroboros-stage3-jamba",
    wandb_entity: str | None = None,
    wandb_mode: str | None = None,
) -> list[str]:
    """Build the tested Kaggle command for Phase 3.4 DGAC training."""
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={int(nproc_per_node)}",
        script,
        "--use_halt_gate",
        "--resume_from_diloco_anchor",
        "--diloco_state_repo",
        diloco_state_repo,
        "--data_dir",
        data_dir,
    ]
    if use_4bit:
        command.append("--use_4bit")
    command.extend(
        [
            "--epochs_per_stage",
            str(int(epochs_per_stage)),
            "--max_stage",
            str(int(max_stage)),
            "--max_grad_norm",
            str(float(max_grad_norm)),
            "--batch_size",
            str(int(batch_size)),
            "--grad_accum",
            str(int(grad_accum)),
            "--val_batch_size",
            str(int(val_batch_size)),
            "--val_skip_buffer_minutes",
            str(int(val_skip_buffer_minutes)),
            "--session_timeout_hours",
            str(float(session_timeout_hours)),
            "--graceful_exit_buffer_minutes",
            str(int(graceful_exit_buffer_minutes)),
        ]
    )
    if push_to_hub:
        command.append("--push_to_hub")
    command.extend(["--output_dir", output_dir])
    command.extend(["--hf_stage_subdir", hf_stage_subdir or output_dir])
    if wandb_project is not None:
        command.extend(["--wandb_project", wandb_project])
    if wandb_entity is not None:
        command.extend(["--wandb_entity", wandb_entity])
    if wandb_mode is not None:
        command.extend(["--wandb_mode", wandb_mode])
    return command


def build_dgac_anchor_eval_command(
    *,
    script: str = "jamba_coconut_finetune.py",
    nproc_per_node: int = 2,
    data_dir: str = "data/coconut_v1",
    use_4bit: bool = True,
    max_stage: int = 10,
    max_grad_norm: float = 0.3,
    batch_size: int = 4,
    grad_accum: int = 8,
    val_batch_size: int = 2,
    val_skip_buffer_minutes: int = 60,
    session_timeout_hours: float = 12.0,
    graceful_exit_buffer_minutes: int = 20,
    diloco_state_repo: str = "WeirdRunner/Ouroboros",
    output_dir: str = "runs/stage10_anchor_eval",
    wandb_project: str | None = "ouroboros-stage3-jamba",
    wandb_entity: str | None = None,
    wandb_mode: str | None = None,
) -> list[str]:
    """Build the tested Kaggle command for terminal Stage-10 anchor eval-only."""
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={int(nproc_per_node)}",
        script,
        "--use_halt_gate",
        "--resume_from_diloco_anchor",
        "--eval_only",
        "--diloco_state_repo",
        diloco_state_repo,
        "--data_dir",
        data_dir,
    ]
    if use_4bit:
        command.append("--use_4bit")
    command.extend(
        [
            "--max_stage",
            str(int(max_stage)),
            "--max_grad_norm",
            str(float(max_grad_norm)),
            "--batch_size",
            str(int(batch_size)),
            "--grad_accum",
            str(int(grad_accum)),
            "--val_batch_size",
            str(int(val_batch_size)),
            "--val_skip_buffer_minutes",
            str(int(val_skip_buffer_minutes)),
            "--session_timeout_hours",
            str(float(session_timeout_hours)),
            "--graceful_exit_buffer_minutes",
            str(int(graceful_exit_buffer_minutes)),
            "--output_dir",
            output_dir,
        ]
    )
    if wandb_project is not None:
        command.extend(["--wandb_project", wandb_project])
    if wandb_entity is not None:
        command.extend(["--wandb_entity", wandb_entity])
    if wandb_mode is not None:
        command.extend(["--wandb_mode", wandb_mode])
    return command


def format_shell_command(command: list[str]) -> str:
    """Render a command list as a pasteable shell command for notebook logs."""
    return " ".join(shlex.quote(part) for part in command)


__all__ = [
    "DGAC_ANCHOR_EVAL_RUN_MODE",
    "DGAC_TRAIN_RUN_MODE",
    "DILOCO_RUN_MODE",
    "build_dgac_anchor_eval_command",
    "build_dgac_training_command",
    "build_diloco_training_command",
    "format_shell_command",
    "kaggle_secret_presence",
    "resolve_diloco_worker_id",
    "resolve_kaggle_run_mode",
]
