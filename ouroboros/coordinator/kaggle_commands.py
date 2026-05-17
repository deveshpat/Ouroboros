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

from ouroboros.coordinator.kaggle_contract import (
    BENCHMARK_RUN_MODE,
    DGAC_ANCHOR_EVAL_RUN_MODE,
    DGAC_CANARY_RUN_MODE,
    DGAC_DILOCO_RUN_MODE,
    DGAC_TRAIN_RUN_MODE,
    DILOCO_RUN_MODE,
    known_kaggle_launch_modes,
    resolve_kaggle_launch_contract,
)
from ouroboros.eval.benchmark_suites import (
    DEFAULT_BENCHMARK_SUITE,
    resolve_benchmark_tasks,
)
from ouroboros.utils.runtime_env import (
    GITHUB_TOKEN_ALIASES,
    HF_TOKEN_ALIASES,
    WANDB_KEY_ALIASES,
    WORKER_ID_ALIASES,
    WORKER_IDS,
    normalize_benchmark_limit,
    normalize_text,
    require_known_worker_id,
    require_worker_id,
    resolve_env_alias,
)

_VALID_DILOCO_WORKER_IDS = set(WORKER_IDS)
_VALID_KAGGLE_RUN_MODES = set(known_kaggle_launch_modes())


def _normalize_text(value: object | None, *, uppercase: bool = False) -> Optional[str]:
    return normalize_text(value, uppercase=uppercase)


def _env_has_any(env: Mapping[str, str], names: tuple[str, ...]) -> bool:
    return resolve_env_alias(env, names) is not None


def resolve_diloco_worker_id(env: Mapping[str, str] | None = None) -> str:
    """Resolve and validate the DiLoCo worker id from notebook/runtime env aliases."""
    return require_worker_id(os.environ if env is None else env)


def kaggle_secret_presence(env: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Return secret availability booleans without exposing secret values."""
    env = os.environ if env is None else env
    return {
        "HF_TOKEN": _env_has_any(env, HF_TOKEN_ALIASES),
        "WANDB_KEY": _env_has_any(env, WANDB_KEY_ALIASES),
        "GITHUB_TOKEN": _env_has_any(env, GITHUB_TOKEN_ALIASES),
        "DILOCO_WORKER_ID": _env_has_any(env, WORKER_ID_ALIASES),
    }


def resolve_kaggle_run_mode(env: Mapping[str, str] | None = None) -> str:
    """Resolve the notebook launch mode from runtime env aliases."""
    try:
        return resolve_kaggle_launch_contract(os.environ if env is None else env).mode
    except ValueError as exc:
        raise ValueError(str(exc).replace("Kaggle launch mode", "Kaggle run mode")) from exc


def _entrypoint_argv(script: str) -> list[str]:
    """Return argv fragments for a script path or ``-m package`` module entrypoint."""
    parts = script.split()
    if len(parts) == 2 and parts[0] == "-m":
        return ["-m", parts[1]]
    return [script]


def build_diloco_training_command(
    *,
    worker_id: str,
    script: str = "-m ouroboros.coconut",
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
    use_halt_gate: bool = False,
    resume_from_diloco_anchor: bool = False,
    latent_cache: bool = True,
    max_grad_norm: float | None = None,
    diloco_run_val: bool = False,
    gen_every_stage: bool | None = None,
) -> list[str]:
    """Build the tested Kaggle Dual-GPU DiLoCo training command.

    When use_halt_gate/resume_from_diloco_anchor are enabled, the same
    DiLoCo worker contract is used for DGAC shard training: workers load the
    terminal anchor, train HaltGate + adapters on their shard, then upload
    worker artifacts for coordinator aggregation.
    """
    normalized_worker_id = require_known_worker_id(worker_id)

    is_dgac_diloco = bool(use_halt_gate and resume_from_diloco_anchor)
    if is_dgac_diloco and int(epochs_per_stage) != 1:
        raise ValueError(
            "DGAC DiLoCo runs exactly one local epoch per worker round. "
            "Launch another dgac-diloco pass from the aggregated anchor if more "
            "HaltGate training is needed."
        )

    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={int(nproc_per_node)}",
        *_entrypoint_argv(script),
        "--data_dir",
        data_dir,
    ]
    if use_4bit:
        command.append("--use_4bit")
    if use_halt_gate:
        command.append("--use_halt_gate")
    if resume_from_diloco_anchor:
        command.append("--resume_from_diloco_anchor")
    if latent_cache:
        command.append("--latent_cache")
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
    if max_grad_norm is not None:
        command.extend(["--max_grad_norm", str(float(max_grad_norm))])
    if diloco_run_val:
        command.append("--diloco_run_val")
    if gen_every_stage is not None:
        command.append("--gen_every_stage" if gen_every_stage else "--no-gen_every_stage")
    if push_to_hub:
        command.append("--push_to_hub")
    command.extend(["--output_dir", output_dir])
    if wandb_mode is not None:
        command.extend(["--wandb_mode", wandb_mode])
    return command



def build_dgac_training_command(
    *,
    script: str = "-m ouroboros.coconut",
    nproc_per_node: int = 2,
    data_dir: str = "data/coconut_v1",
    use_4bit: bool = True,
    epochs_per_stage: int = 3,
    max_stage: int = 10,
    max_samples: int | None = None,
    max_train_steps: int | None = None,
    log_every: int | None = None,
    gen_every_stage: bool | None = None,
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
    latent_cache: bool = True,
) -> list[str]:
    """Build the tested Kaggle command for Phase 3.4 DGAC training."""
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={int(nproc_per_node)}",
        *_entrypoint_argv(script),
        "--use_halt_gate",
        "--resume_from_diloco_anchor",
        "--diloco_state_repo",
        diloco_state_repo,
        "--data_dir",
        data_dir,
    ]
    if use_4bit:
        command.append("--use_4bit")
    if latent_cache:
        command.append("--latent_cache")
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
    if max_samples is not None:
        command.extend(["--max_samples", str(int(max_samples))])
    if max_train_steps is not None:
        command.extend(["--max_train_steps", str(int(max_train_steps))])
    if log_every is not None:
        command.extend(["--log_every", str(int(log_every))])
    if gen_every_stage is not None:
        command.append("--gen_every_stage" if gen_every_stage else "--no-gen_every_stage")
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



def build_dgac_canary_command(
    *,
    script: str = "-m ouroboros.coconut",
    nproc_per_node: int = 2,
    data_dir: str = "data/coconut_v1",
    use_4bit: bool = True,
    max_stage: int = 10,
    max_samples: int = 512,
    max_train_steps: int = 20,
    max_grad_norm: float = 0.3,
    batch_size: int = 4,
    grad_accum: int = 8,
    val_batch_size: int = 2,
    val_skip_buffer_minutes: int = 720,
    session_timeout_hours: float = 12.0,
    graceful_exit_buffer_minutes: int = 20,
    diloco_state_repo: str = "WeirdRunner/Ouroboros",
    output_dir: str = "runs/stage3_dgac_canary",
    hf_stage_subdir: str | None = None,
    push_to_hub: bool = False,
    wandb_project: str | None = "ouroboros-stage3-jamba",
    wandb_entity: str | None = None,
    wandb_mode: str | None = None,
    latent_cache: bool = True,
) -> list[str]:
    """Build a bounded DGAC canary command that exits before full-epoch training."""
    return build_dgac_training_command(
        script=script,
        nproc_per_node=nproc_per_node,
        data_dir=data_dir,
        use_4bit=use_4bit,
        epochs_per_stage=1,
        max_stage=max_stage,
        max_samples=max_samples,
        max_train_steps=max_train_steps,
        log_every=1,
        gen_every_stage=False,
        max_grad_norm=max_grad_norm,
        batch_size=batch_size,
        grad_accum=grad_accum,
        val_batch_size=val_batch_size,
        val_skip_buffer_minutes=val_skip_buffer_minutes,
        session_timeout_hours=session_timeout_hours,
        graceful_exit_buffer_minutes=graceful_exit_buffer_minutes,
        diloco_state_repo=diloco_state_repo,
        output_dir=output_dir,
        hf_stage_subdir=hf_stage_subdir or output_dir,
        push_to_hub=push_to_hub,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        latent_cache=latent_cache,
    )

def build_dgac_anchor_eval_command(
    *,
    script: str = "-m ouroboros.coconut",
    nproc_per_node: int = 2,
    data_dir: str = "data/coconut_v1",
    use_4bit: bool = True,
    max_stage: int = 10,
    max_grad_norm: float = 0.3,
    batch_size: int = 4,
    grad_accum: int = 8,
    val_batch_size: int = 1,
    dgac_diagnostics_batch_size: int | None = 1,
    dgac_diagnostics_only: bool = False,
    dgac_diagnostics_forced_kmax_ce: float | None = None,
    val_skip_buffer_minutes: int = 60,
    session_timeout_hours: float = 12.0,
    graceful_exit_buffer_minutes: int = 20,
    diloco_state_repo: str = "WeirdRunner/Ouroboros",
    output_dir: str = "runs/dgac_anchor_eval",
    wandb_project: str | None = "ouroboros-stage3-jamba",
    wandb_entity: str | None = None,
    wandb_mode: str | None = None,
    latent_cache: bool = True,
) -> list[str]:
    """Build the tested Kaggle command for current DGAC/DiLoCo anchor eval-only."""
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={int(nproc_per_node)}",
        *_entrypoint_argv(script),
        "--use_halt_gate",
        "--resume_from_diloco_anchor",
        "--eval_only",
        "--dgac_diagnostics",
        "--diloco_state_repo",
        diloco_state_repo,
        "--data_dir",
        data_dir,
    ]
    if use_4bit:
        command.append("--use_4bit")
    if latent_cache:
        command.append("--latent_cache")
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
    if dgac_diagnostics_only:
        command.append("--dgac_diagnostics_only")
    if dgac_diagnostics_batch_size is not None:
        command.extend(["--dgac_diagnostics_batch_size", str(int(dgac_diagnostics_batch_size))])
    if dgac_diagnostics_forced_kmax_ce is not None:
        command.extend(["--dgac_diagnostics_forced_kmax_ce", str(float(dgac_diagnostics_forced_kmax_ce))])
    if wandb_project is not None:
        command.extend(["--wandb_project", wandb_project])
    if wandb_entity is not None:
        command.extend(["--wandb_entity", wandb_entity])
    if wandb_mode is not None:
        command.extend(["--wandb_mode", wandb_mode])
    return command


def build_lm_eval_benchmark_command(
    *,
    tasks: str | None = None,
    suite: str = DEFAULT_BENCHMARK_SUITE,
    limit: str | int | None = None,
    output_dir: str = "runs/lm_eval_benchmark",
    base_model: str = "ai21labs/AI21-Jamba-Reasoning-3B",
    adapter_repo: str = "WeirdRunner/Ouroboros",
    adapter_subfolder: str = "diloco_state/anchor",
    batch_size: str = "1",
    device: str = "cuda:0",
    dtype: str = "float16",
    model_args: str | None = None,
    publish_to_hub: bool = True,
    adapter_cache_dir: str | None = None,
) -> list[str]:
    """Build the Kaggle command for standardized lm-evaluation-harness benchmarks."""
    tasks = resolve_benchmark_tasks(suite=suite, tasks=tasks)
    command = [
        "python",
        "-m",
        "ouroboros.eval.benchmark_harness",
        "--tasks",
        str(tasks),
        "--output_dir",
        output_dir,
        "--base_model",
        base_model,
        "--adapter_repo",
        adapter_repo,
        "--adapter_subfolder",
        adapter_subfolder,
        "--batch_size",
        str(batch_size),
        "--device",
        device,
        "--dtype",
        dtype,
    ]
    limit = normalize_benchmark_limit(limit)
    if limit is not None:
        command.extend(["--limit", limit])
    if model_args is not None and str(model_args).strip():
        command.extend(["--model_args", str(model_args).strip()])
    if adapter_cache_dir is not None and str(adapter_cache_dir).strip():
        command.extend(["--adapter_cache_dir", str(adapter_cache_dir).strip()])
    if publish_to_hub:
        command.append("--publish_to_hub")
    return command


def build_lm_eval_benchmark_multi_gpu_command(
    *,
    tasks: str | None = None,
    suite: str = DEFAULT_BENCHMARK_SUITE,
    devices: str | None = None,
    limit: str | int | None = None,
    output_dir: str = "runs/lm_eval_benchmark",
    base_model: str = "ai21labs/AI21-Jamba-Reasoning-3B",
    adapter_repo: str = "WeirdRunner/Ouroboros",
    adapter_subfolder: str = "diloco_state/anchor",
    batch_size: str = "1",
    dtype: str = "float16",
    model_args: str | None = None,
    publish_to_hub: bool = True,
    parallelism: str | None = None,
) -> list[str]:
    """Build a Kaggle command that shards lm-eval tasks across multiple GPUs."""
    tasks = resolve_benchmark_tasks(suite=suite, tasks=tasks)
    command = [
        "python",
        "-m",
        "ouroboros.eval.benchmark_multi_gpu",
        "--tasks",
        str(tasks),
        "--output_dir",
        output_dir,
        "--base_model",
        base_model,
        "--adapter_repo",
        adapter_repo,
        "--adapter_subfolder",
        adapter_subfolder,
        "--batch_size",
        str(batch_size),
        "--dtype",
        dtype,
    ]
    if devices is not None and str(devices).strip():
        command.extend(["--devices", str(devices).strip()])
    if parallelism is not None and str(parallelism).strip():
        command.extend(["--parallelism", str(parallelism).strip()])
    limit = normalize_benchmark_limit(limit)
    if limit is not None:
        command.extend(["--limit", limit])
    if model_args is not None and str(model_args).strip():
        command.extend(["--model_args", str(model_args).strip()])
    if publish_to_hub:
        command.append("--publish_to_hub")
    return command


def format_shell_command(command: list[str]) -> str:
    """Render a command list as a pasteable shell command for notebook logs."""
    return " ".join(shlex.quote(part) for part in command)


__all__ = [
    "BENCHMARK_RUN_MODE",
    "DGAC_ANCHOR_EVAL_RUN_MODE",
    "DGAC_TRAIN_RUN_MODE",
    "DGAC_CANARY_RUN_MODE",
    "DGAC_DILOCO_RUN_MODE",
    "DILOCO_RUN_MODE",
    "build_dgac_anchor_eval_command",
    "build_dgac_training_command",
    "build_dgac_canary_command",
    "build_diloco_training_command",
    "build_lm_eval_benchmark_command",
    "build_lm_eval_benchmark_multi_gpu_command",
    "format_shell_command",
    "kaggle_secret_presence",
    "resolve_diloco_worker_id",
    "resolve_kaggle_run_mode",
]
