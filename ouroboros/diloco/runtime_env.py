"""Runtime environment construction for DiLoCo worker dispatch.

This module is a deep module for coordinator-side runtime-env injection. Callers only
provide coordinator args + worker id and receive a normalized env payload.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional, Sequence

WORKER_IDS: Sequence[str] = ("A", "B", "C")


def normalize_optional_text(value: Optional[Any], *, uppercase: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.upper() if uppercase else text


def first_nonempty_text(*values: Optional[Any], uppercase: bool = False) -> Optional[str]:
    for value in values:
        text = normalize_optional_text(value, uppercase=uppercase)
        if text is not None:
            return text
    return None


def set_env_if_present(
    target: Dict[str, str],
    key: str,
    value: Optional[Any],
    *,
    uppercase: bool = False,
) -> None:
    text = normalize_optional_text(value, uppercase=uppercase)
    if text is not None:
        target[key] = text


def infer_runtime_repo_url(default: str = "https://github.com/deveshpat/Ouroboros.git") -> str:
    explicit = first_nonempty_text(os.environ.get("OUROBOROS_REPO_URL"))
    if explicit:
        return explicit
    repo = first_nonempty_text(os.environ.get("GITHUB_REPOSITORY"))
    server = first_nonempty_text(os.environ.get("GITHUB_SERVER_URL"), "https://github.com")
    if repo:
        return f"{server.rstrip('/')}/{repo}.git"
    return default


def infer_runtime_repo_ref(default: str = "main") -> str:
    return (
        first_nonempty_text(
            os.environ.get("OUROBOROS_REPO_REF"),
            os.environ.get("GITHUB_REF_NAME"),
            default,
        )
        or default
    )


def infer_runtime_repo_commit() -> Optional[str]:
    return first_nonempty_text(
        os.environ.get("OUROBOROS_REPO_COMMIT"),
        os.environ.get("GITHUB_SHA"),
    )


def build_worker_runtime_env(args: argparse.Namespace, worker_id: str) -> Dict[str, str]:
    worker = normalize_optional_text(worker_id, uppercase=True)
    if worker not in WORKER_IDS:
        raise ValueError(f"Invalid worker id for runtime env injection: {worker_id!r}")

    runtime_env: Dict[str, str] = {}

    for name, value in os.environ.items():
        if name.startswith("OUROBOROS_"):
            set_env_if_present(runtime_env, name, value)

    set_env_if_present(runtime_env, "DILOCO_WORKER_ID", worker, uppercase=True)
    set_env_if_present(runtime_env, "OUROBOROS_DILOCO_WORKER_ID", worker, uppercase=True)
    set_env_if_present(runtime_env, "WORKER_ID", worker, uppercase=True)
    runtime_env["OUROBOROS_AUTO_TRIGGERED"] = "1"

    hf_token = first_nonempty_text(
        getattr(args, "hf_token", None),
        os.environ.get("HF_TOKEN"),
        os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    if hf_token:
        runtime_env["HF_TOKEN"] = hf_token
        runtime_env["HUGGINGFACE_HUB_TOKEN"] = hf_token

    wandb_key = first_nonempty_text(
        getattr(args, "wandb_key", None),
        os.environ.get("WANDB_API_KEY"),
        os.environ.get("WANDB_KEY"),
    )
    if wandb_key:
        runtime_env["WANDB_API_KEY"] = wandb_key
        runtime_env["WANDB_KEY"] = wandb_key

    github_token = first_nonempty_text(
        os.environ.get("GITHUB_TOKEN"),
        os.environ.get("GH_TOKEN"),
    )
    if github_token:
        runtime_env["GITHUB_TOKEN"] = github_token
        runtime_env["GH_TOKEN"] = github_token

    set_env_if_present(runtime_env, "OUROBOROS_REPO_URL", infer_runtime_repo_url())
    set_env_if_present(runtime_env, "OUROBOROS_REPO_REF", infer_runtime_repo_ref())
    set_env_if_present(runtime_env, "OUROBOROS_REPO_COMMIT", infer_runtime_repo_commit())
    set_env_if_present(
        runtime_env,
        "OUROBOROS_DILOCO_STATE_REPO",
        first_nonempty_text(os.environ.get("OUROBOROS_DILOCO_STATE_REPO"), getattr(args, "repo_id", None)),
    )
    set_env_if_present(
        runtime_env,
        "OUROBOROS_DILOCO_SIGNAL_REPO",
        first_nonempty_text(
            os.environ.get("OUROBOROS_DILOCO_SIGNAL_REPO"),
            os.environ.get("GITHUB_REPOSITORY"),
        ),
    )
    set_env_if_present(
        runtime_env,
        "OUROBOROS_DILOCO_OUTER_LR",
        first_nonempty_text(
            os.environ.get("OUROBOROS_DILOCO_OUTER_LR"),
            f"{float(getattr(args, 'outer_lr', 0.7)):g}",
        ),
    )
    set_env_if_present(
        runtime_env,
        "OUROBOROS_WANDB_PROJECT",
        first_nonempty_text(os.environ.get("OUROBOROS_WANDB_PROJECT"), getattr(args, "wandb_project", None)),
    )
    set_env_if_present(
        runtime_env,
        "OUROBOROS_WANDB_ENTITY",
        first_nonempty_text(os.environ.get("OUROBOROS_WANDB_ENTITY"), getattr(args, "wandb_entity", None)),
    )
    set_env_if_present(
        runtime_env,
        "OUROBOROS_DILOCO_OUTPUT_DIR",
        first_nonempty_text(os.environ.get("OUROBOROS_DILOCO_OUTPUT_DIR"), "runs/diloco"),
    )
    return runtime_env
