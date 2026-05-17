"""Data contract for Kaggle launch modes.

Coordinator and notebook adapters use this module as the stdlib-only source
of truth for launch-mode policies. Command builders remain in
``ouroboros.coordinator.kaggle_commands`` so this contract stays importable in bootstrap contexts.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

from ouroboros.utils.runtime_env import normalize_text

DILOCO_RUN_MODE = "diloco"
DGAC_ANCHOR_EVAL_RUN_MODE = "dgac-anchor-eval"
DGAC_TRAIN_RUN_MODE = "dgac-train"
DGAC_CANARY_RUN_MODE = "dgac-canary"
DGAC_DILOCO_RUN_MODE = "dgac-diloco"
BENCHMARK_RUN_MODE = "benchmark"


@dataclass(frozen=True)
class KaggleLaunchContract:
    mode: str
    requires_gpu: bool
    worker_mode: bool
    trains: bool
    validates: bool
    mutates_round_state: bool
    env_keys: tuple[str, ...]
    required_cli_args: tuple[str, ...]
    notebook_path: str
    allowed_from_github_actions: bool
    allowed_manually: bool


_COMMON_ENV = (
    "OUROBOROS_KAGGLE_RUN_MODE",
    "OUROBOROS_REPO_URL",
    "OUROBOROS_REPO_REF",
    "OUROBOROS_REPO_COMMIT",
    "OUROBOROS_DILOCO_STATE_REPO",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "WANDB_API_KEY",
    "WANDB_KEY",
)
_WORKER_ENV = (
    "DILOCO_WORKER_ID",
    "OUROBOROS_DILOCO_WORKER_ID",
    "WORKER_ID",
    "OUROBOROS_AUTO_TRIGGERED",
)

_CONTRACTS: dict[str, KaggleLaunchContract] = {
    DILOCO_RUN_MODE: KaggleLaunchContract(
        mode=DILOCO_RUN_MODE,
        requires_gpu=True,
        worker_mode=True,
        trains=True,
        validates=False,
        mutates_round_state=True,
        env_keys=_COMMON_ENV + _WORKER_ENV + ("OUROBOROS_DILOCO_SIGNAL_REPO", "OUROBOROS_DILOCO_OUTER_LR"),
        required_cli_args=("--diloco_mode", "--diloco_worker_id", "--diloco_state_repo"),
        notebook_path="kaggle-utils.ipynb",
        allowed_from_github_actions=True,
        allowed_manually=True,
    ),
    DGAC_ANCHOR_EVAL_RUN_MODE: KaggleLaunchContract(
        mode=DGAC_ANCHOR_EVAL_RUN_MODE,
        requires_gpu=True,
        worker_mode=True,
        trains=False,
        validates=True,
        mutates_round_state=False,
        env_keys=_COMMON_ENV + _WORKER_ENV,
        required_cli_args=("--resume_from_diloco_anchor", "--eval_only", "--dgac_diagnostics"),
        notebook_path="kaggle-utils.ipynb",
        allowed_from_github_actions=True,
        allowed_manually=True,
    ),
    DGAC_TRAIN_RUN_MODE: KaggleLaunchContract(
        mode=DGAC_TRAIN_RUN_MODE,
        requires_gpu=True,
        worker_mode=True,
        trains=True,
        validates=True,
        mutates_round_state=False,
        env_keys=_COMMON_ENV + _WORKER_ENV,
        required_cli_args=("--use_halt_gate", "--resume_from_diloco_anchor"),
        notebook_path="kaggle-utils.ipynb",
        allowed_from_github_actions=True,
        allowed_manually=True,
    ),
    DGAC_CANARY_RUN_MODE: KaggleLaunchContract(
        mode=DGAC_CANARY_RUN_MODE,
        requires_gpu=True,
        worker_mode=True,
        trains=True,
        validates=False,
        mutates_round_state=False,
        env_keys=_COMMON_ENV + _WORKER_ENV,
        required_cli_args=("--use_halt_gate", "--resume_from_diloco_anchor", "--max_train_steps"),
        notebook_path="kaggle-utils.ipynb",
        allowed_from_github_actions=True,
        allowed_manually=True,
    ),
    DGAC_DILOCO_RUN_MODE: KaggleLaunchContract(
        mode=DGAC_DILOCO_RUN_MODE,
        requires_gpu=True,
        worker_mode=True,
        trains=True,
        validates=False,
        mutates_round_state=True,
        env_keys=_COMMON_ENV + _WORKER_ENV + ("OUROBOROS_DILOCO_SIGNAL_REPO", "OUROBOROS_DILOCO_OUTER_LR"),
        required_cli_args=("--diloco_mode", "--use_halt_gate", "--resume_from_diloco_anchor"),
        notebook_path="kaggle-utils.ipynb",
        allowed_from_github_actions=True,
        allowed_manually=True,
    ),
    BENCHMARK_RUN_MODE: KaggleLaunchContract(
        mode=BENCHMARK_RUN_MODE,
        requires_gpu=True,
        worker_mode=True,
        trains=False,
        validates=True,
        mutates_round_state=False,
        env_keys=_COMMON_ENV + _WORKER_ENV + (
            "OUROBOROS_BENCHMARK_SUITE",
            "OUROBOROS_BENCHMARK_TASKS",
            "OUROBOROS_BENCHMARK_LIMIT",
            "OUROBOROS_BENCHMARK_OUTPUT_DIR",
            "OUROBOROS_BENCHMARK_BASE_MODEL",
            "OUROBOROS_BENCHMARK_ADAPTER_REPO",
            "OUROBOROS_BENCHMARK_ADAPTER_SUBFOLDER",
            "OUROBOROS_BENCHMARK_BATCH_SIZE",
            "OUROBOROS_BENCHMARK_DEVICE",
            "OUROBOROS_BENCHMARK_DEVICES",
            "OUROBOROS_BENCHMARK_DTYPE",
            "OUROBOROS_BENCHMARK_MODEL_ARGS",
            "OUROBOROS_BENCHMARK_PUBLISH_TO_HUB",
            "OUROBOROS_BENCHMARK_HUB_PREFIX",
            "OUROBOROS_BENCHMARK_SKIP_INSTALL",
            "OUROBOROS_BENCHMARK_PARALLELISM",
        ),
        required_cli_args=("-m", "ouroboros.eval.benchmark_multi_gpu"),
        notebook_path="kaggle-utils.ipynb",
        allowed_from_github_actions=True,
        allowed_manually=True,
    ),
}


def known_kaggle_launch_modes() -> tuple[str, ...]:
    return tuple(_CONTRACTS)


def get_kaggle_launch_contract(mode: str) -> KaggleLaunchContract:
    normalized = normalize_text(mode)
    if normalized is None:
        normalized = DILOCO_RUN_MODE
    normalized = normalized.lower()
    if normalized not in _CONTRACTS:
        raise ValueError(
            f"Invalid Kaggle launch mode {normalized!r}. Expected one of: "
            f"{', '.join(known_kaggle_launch_modes())}."
        )
    return _CONTRACTS[normalized]


def resolve_kaggle_launch_contract(env: Mapping[str, str] | None = None) -> KaggleLaunchContract:
    env = os.environ if env is None else env
    mode = normalize_text(env.get("OUROBOROS_KAGGLE_RUN_MODE") or env.get("OUROBOROS_RUN_MODE"))
    return get_kaggle_launch_contract(mode or DILOCO_RUN_MODE)
