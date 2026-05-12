"""Deep Kaggle launch-mode matrix for Ouroboros.

``ouroboros.kaggle_contract`` remains the stdlib-safe policy layer. This
module binds each launch mode to the command builders, notebook shell-magic
expectations, output env keys, and workflow-facing labels used by adapters.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ouroboros.kaggle import (
    build_dgac_anchor_eval_command,
    build_dgac_canary_command,
    build_dgac_training_command,
    build_diloco_training_command,
)
from ouroboros.kaggle_contract import (
    CPU_SMOKE_MODE,
    DGAC_ANCHOR_EVAL_RUN_MODE,
    DGAC_CANARY_RUN_MODE,
    DGAC_DILOCO_RUN_MODE,
    DGAC_TRAIN_RUN_MODE,
    DILOCO_RUN_MODE,
    KaggleLaunchContract,
    get_kaggle_launch_contract,
    known_kaggle_launch_modes,
)
from ouroboros.runtime_env import normalize_text, require_known_worker_id, require_worker_id


@dataclass(frozen=True)
class KaggleLaunchModeSpec:
    mode: str
    contract: KaggleLaunchContract
    env_defaults: Mapping[str, str]
    command_builder: Callable[..., list[str]]
    notebook_shell_template: str
    output_env_key: str | None
    requires_worker_id: bool
    workflow_label: str


def _readonly(defaults: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(defaults))


def _with_defaults(defaults: Mapping[str, str], env: Mapping[str, str] | None) -> dict[str, str]:
    merged = dict(defaults)
    source = os.environ if env is None else env
    for key, value in source.items():
        if value is not None:
            merged[str(key)] = str(value)
    return merged


def _value(env: Mapping[str, str], key: str) -> str:
    value = normalize_text(env.get(key))
    if value is None:
        raise ValueError(f"Missing required Kaggle launch env key: {key}")
    return value


def _float_value(env: Mapping[str, str], key: str) -> float:
    return float(_value(env, key))


def _resolve_worker(env: Mapping[str, str], worker_id: str | None) -> str:
    if worker_id is not None:
        return require_known_worker_id(worker_id)
    return require_worker_id(env)


def _build_diloco(env: Mapping[str, str], *, worker_id: str | None = None) -> list[str]:
    return build_diloco_training_command(
        worker_id=_resolve_worker(env, worker_id),
        diloco_state_repo=_value(env, "OUROBOROS_DILOCO_STATE_REPO"),
        diloco_signal_repo=_value(env, "OUROBOROS_DILOCO_SIGNAL_REPO"),
        diloco_outer_lr=_float_value(env, "OUROBOROS_DILOCO_OUTER_LR"),
        output_dir=_value(env, "OUROBOROS_DILOCO_OUTPUT_DIR"),
    )


def _build_dgac_diloco(env: Mapping[str, str], *, worker_id: str | None = None) -> list[str]:
    return build_diloco_training_command(
        worker_id=_resolve_worker(env, worker_id),
        diloco_state_repo=_value(env, "OUROBOROS_DILOCO_STATE_REPO"),
        diloco_signal_repo=_value(env, "OUROBOROS_DILOCO_SIGNAL_REPO"),
        diloco_outer_lr=_float_value(env, "OUROBOROS_DILOCO_OUTER_LR"),
        output_dir=_value(env, "OUROBOROS_DGAC_DILOCO_OUTPUT_DIR"),
        epochs_per_stage=1,
        use_halt_gate=True,
        resume_from_diloco_anchor=True,
        max_grad_norm=0.3,
    )


def _build_dgac_canary(env: Mapping[str, str], *, worker_id: str | None = None) -> list[str]:
    del worker_id
    output_dir = _value(env, "OUROBOROS_DGAC_CANARY_OUTPUT_DIR")
    return build_dgac_canary_command(
        diloco_state_repo=_value(env, "OUROBOROS_DILOCO_STATE_REPO"),
        output_dir=output_dir,
        hf_stage_subdir=output_dir,
        wandb_project=_value(env, "OUROBOROS_WANDB_PROJECT"),
    )


def _build_dgac_train(env: Mapping[str, str], *, worker_id: str | None = None) -> list[str]:
    del worker_id
    output_dir = _value(env, "OUROBOROS_DGAC_OUTPUT_DIR")
    return build_dgac_training_command(
        diloco_state_repo=_value(env, "OUROBOROS_DILOCO_STATE_REPO"),
        output_dir=output_dir,
        hf_stage_subdir=output_dir,
        wandb_project=_value(env, "OUROBOROS_WANDB_PROJECT"),
    )


def _build_dgac_anchor_eval(env: Mapping[str, str], *, worker_id: str | None = None) -> list[str]:
    del worker_id
    return build_dgac_anchor_eval_command(
        diloco_state_repo=_value(env, "OUROBOROS_DILOCO_STATE_REPO"),
        output_dir=_value(env, "OUROBOROS_DGAC_ANCHOR_EVAL_OUTPUT_DIR"),
        wandb_project=_value(env, "OUROBOROS_WANDB_PROJECT"),
    )


def _build_cpu_smoke(env: Mapping[str, str], *, worker_id: str | None = None) -> list[str]:
    del env, worker_id
    return ["python", "-m", "ouroboros.workflow_validation_worker"]


_COMMON_DEFAULTS = {
    "OUROBOROS_DILOCO_STATE_REPO": "WeirdRunner/Ouroboros",
    "OUROBOROS_DILOCO_SIGNAL_REPO": "deveshpat/Ouroboros",
    "OUROBOROS_DILOCO_OUTER_LR": "0.7",
    "OUROBOROS_WANDB_PROJECT": "ouroboros-stage3-jamba",
}

_DILOCO_SHELL = (
    'torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py '
    '--data_dir data/coconut_v1 --use_4bit --stage_0_epochs 1 --epochs_per_stage 1 '
    '--max_stage 10 --batch_size 4 --grad_accum 8 --val_batch_size 2 '
    '--val_skip_buffer_minutes 60 --session_timeout_hours 12.0 '
    '--graceful_exit_buffer_minutes 20 --diloco_mode --diloco_worker_id "$DILOCO_WORKER_ID" '
    '--diloco_outer_lr "$OUROBOROS_DILOCO_OUTER_LR" '
    '--diloco_state_repo "$OUROBOROS_DILOCO_STATE_REPO" '
    '--diloco_signal_repo "$OUROBOROS_DILOCO_SIGNAL_REPO" --push_to_hub '
    '--output_dir "$OUROBOROS_DILOCO_OUTPUT_DIR"'
)
_DGAC_DILOCO_SHELL = (
    'torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py '
    '--data_dir data/coconut_v1 --use_4bit --use_halt_gate --resume_from_diloco_anchor '
    '--stage_0_epochs 1 --epochs_per_stage 1 --max_stage 10 --max_grad_norm 0.3 '
    '--batch_size 4 --grad_accum 8 --val_batch_size 2 --val_skip_buffer_minutes 60 '
    '--session_timeout_hours 12.0 --graceful_exit_buffer_minutes 20 --diloco_mode '
    '--diloco_worker_id "$DILOCO_WORKER_ID" --diloco_outer_lr "$OUROBOROS_DILOCO_OUTER_LR" '
    '--diloco_state_repo "$OUROBOROS_DILOCO_STATE_REPO" '
    '--diloco_signal_repo "$OUROBOROS_DILOCO_SIGNAL_REPO" --push_to_hub '
    '--output_dir "$OUROBOROS_DGAC_DILOCO_OUTPUT_DIR"'
)
_DGAC_CANARY_SHELL = (
    'torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py '
    '--use_halt_gate --resume_from_diloco_anchor --diloco_state_repo "$OUROBOROS_DILOCO_STATE_REPO" '
    '--data_dir data/coconut_v1 --use_4bit --epochs_per_stage 1 --max_stage 10 '
    '--max_grad_norm 0.3 --batch_size 4 --grad_accum 8 --val_batch_size 2 '
    '--val_skip_buffer_minutes 720 --session_timeout_hours 12.0 '
    '--graceful_exit_buffer_minutes 20 --max_samples 512 --max_train_steps 20 '
    '--log_every 1 --no-gen_every_stage --output_dir "$OUROBOROS_DGAC_CANARY_OUTPUT_DIR" '
    '--hf_stage_subdir "$OUROBOROS_DGAC_CANARY_OUTPUT_DIR" --wandb_project "$OUROBOROS_WANDB_PROJECT"'
)
_DGAC_TRAIN_SHELL = (
    'torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py '
    '--use_halt_gate --resume_from_diloco_anchor --diloco_state_repo "$OUROBOROS_DILOCO_STATE_REPO" '
    '--data_dir data/coconut_v1 --use_4bit --epochs_per_stage 3 --max_stage 10 '
    '--max_grad_norm 0.3 --batch_size 4 --grad_accum 8 --val_batch_size 2 '
    '--val_skip_buffer_minutes 60 --session_timeout_hours 12.0 '
    '--graceful_exit_buffer_minutes 20 --push_to_hub --output_dir "$OUROBOROS_DGAC_OUTPUT_DIR" '
    '--hf_stage_subdir "$OUROBOROS_DGAC_OUTPUT_DIR" --wandb_project "$OUROBOROS_WANDB_PROJECT"'
)
_DGAC_ANCHOR_EVAL_SHELL = (
    'torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py '
    '--use_halt_gate --resume_from_diloco_anchor --eval_only '
    '--diloco_state_repo "$OUROBOROS_DILOCO_STATE_REPO" --data_dir data/coconut_v1 '
    '--use_4bit --max_stage 10 --max_grad_norm 0.3 --batch_size 4 --grad_accum 8 '
    '--val_batch_size 2 --val_skip_buffer_minutes 60 --session_timeout_hours 12.0 '
    '--graceful_exit_buffer_minutes 20 --output_dir "$OUROBOROS_DGAC_ANCHOR_EVAL_OUTPUT_DIR" '
    '--wandb_project "$OUROBOROS_WANDB_PROJECT"'
)

_SPECS: dict[str, KaggleLaunchModeSpec] = {
    DILOCO_RUN_MODE: KaggleLaunchModeSpec(
        mode=DILOCO_RUN_MODE,
        contract=get_kaggle_launch_contract(DILOCO_RUN_MODE),
        env_defaults=_readonly({**_COMMON_DEFAULTS, "OUROBOROS_DILOCO_OUTPUT_DIR": "runs/diloco"}),
        command_builder=_build_diloco,
        notebook_shell_template=_DILOCO_SHELL,
        output_env_key="OUROBOROS_DILOCO_OUTPUT_DIR",
        requires_worker_id=True,
        workflow_label="normal coordinator DiLoCo dispatch",
    ),
    DGAC_ANCHOR_EVAL_RUN_MODE: KaggleLaunchModeSpec(
        mode=DGAC_ANCHOR_EVAL_RUN_MODE,
        contract=get_kaggle_launch_contract(DGAC_ANCHOR_EVAL_RUN_MODE),
        env_defaults=_readonly({**_COMMON_DEFAULTS, "OUROBOROS_DGAC_ANCHOR_EVAL_OUTPUT_DIR": "runs/dgac_anchor_eval"}),
        command_builder=_build_dgac_anchor_eval,
        notebook_shell_template=_DGAC_ANCHOR_EVAL_SHELL,
        output_env_key="OUROBOROS_DGAC_ANCHOR_EVAL_OUTPUT_DIR",
        requires_worker_id=False,
        workflow_label="current anchor eval-only",
    ),
    DGAC_TRAIN_RUN_MODE: KaggleLaunchModeSpec(
        mode=DGAC_TRAIN_RUN_MODE,
        contract=get_kaggle_launch_contract(DGAC_TRAIN_RUN_MODE),
        env_defaults=_readonly({**_COMMON_DEFAULTS, "OUROBOROS_DGAC_OUTPUT_DIR": "runs/stage3_dgac"}),
        command_builder=_build_dgac_train,
        notebook_shell_template=_DGAC_TRAIN_SHELL,
        output_env_key="OUROBOROS_DGAC_OUTPUT_DIR",
        requires_worker_id=False,
        workflow_label="sequential DGAC fallback",
    ),
    DGAC_CANARY_RUN_MODE: KaggleLaunchModeSpec(
        mode=DGAC_CANARY_RUN_MODE,
        contract=get_kaggle_launch_contract(DGAC_CANARY_RUN_MODE),
        env_defaults=_readonly({**_COMMON_DEFAULTS, "OUROBOROS_DGAC_CANARY_OUTPUT_DIR": "runs/stage3_dgac_canary"}),
        command_builder=_build_dgac_canary,
        notebook_shell_template=_DGAC_CANARY_SHELL,
        output_env_key="OUROBOROS_DGAC_CANARY_OUTPUT_DIR",
        requires_worker_id=False,
        workflow_label="bounded DGAC objective canary",
    ),
    DGAC_DILOCO_RUN_MODE: KaggleLaunchModeSpec(
        mode=DGAC_DILOCO_RUN_MODE,
        contract=get_kaggle_launch_contract(DGAC_DILOCO_RUN_MODE),
        env_defaults=_readonly({**_COMMON_DEFAULTS, "OUROBOROS_DGAC_DILOCO_OUTPUT_DIR": "runs/dgac_dedicated"}),
        command_builder=_build_dgac_diloco,
        notebook_shell_template=_DGAC_DILOCO_SHELL,
        output_env_key="OUROBOROS_DGAC_DILOCO_OUTPUT_DIR",
        requires_worker_id=True,
        workflow_label="DGAC dedicated worker rounds",
    ),
    CPU_SMOKE_MODE: KaggleLaunchModeSpec(
        mode=CPU_SMOKE_MODE,
        contract=get_kaggle_launch_contract(CPU_SMOKE_MODE),
        env_defaults=_readonly({"OUROBOROS_WORKFLOW_VALIDATE": CPU_SMOKE_MODE}),
        command_builder=_build_cpu_smoke,
        notebook_shell_template="",
        output_env_key=None,
        requires_worker_id=True,
        workflow_label="read-only CPU workflow validation",
    ),
}


def known_launch_specs(*, include_cpu_smoke: bool = False) -> tuple[KaggleLaunchModeSpec, ...]:
    """Return launch specs for public Kaggle launch modes."""
    return tuple(_SPECS[mode] for mode in known_kaggle_launch_modes(include_cpu_smoke=include_cpu_smoke))


def get_launch_spec(mode: str) -> KaggleLaunchModeSpec:
    """Resolve one launch-mode spec using the same normalization as the policy contract."""
    contract = get_kaggle_launch_contract(mode)
    return _SPECS[contract.mode]


def build_launch_command(
    mode: str,
    env: Mapping[str, str] | None = None,
    *,
    worker_id: str | None = None,
) -> list[str]:
    """Build the command argv for a Kaggle launch mode from matrix defaults plus env."""
    spec = get_launch_spec(mode)
    merged_env = _with_defaults(spec.env_defaults, env)
    return spec.command_builder(merged_env, worker_id=worker_id)


def expected_notebook_shell_tokens(mode: str) -> tuple[str, ...]:
    """Return the literal IPython shell-magic argv expected in kaggle-utils.ipynb."""
    template = get_launch_spec(mode).notebook_shell_template
    if not template:
        return ()
    return tuple(shlex.split(template))


def requires_kaggle_gpu(mode: str) -> bool:
    """Return whether Kaggle metadata/CLI should request GPU for this launch mode."""
    return get_launch_spec(mode).contract.requires_gpu


__all__ = [
    "KaggleLaunchModeSpec",
    "build_launch_command",
    "expected_notebook_shell_tokens",
    "get_launch_spec",
    "known_launch_specs",
    "requires_kaggle_gpu",
]
