"""Environment-payload helpers for DiLoCo Kaggle workers."""

from __future__ import annotations

import base64
import json
import os
import zlib
from argparse import Namespace
from typing import Any, MutableMapping

from ouroboros.diloco.protocol import WORKER_IDS


def normalize_optional_text(value: Any, *, uppercase: bool = False) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "undefined"}:
        return None
    return text.upper() if uppercase else text


def first_nonempty_text(*values: Any, uppercase: bool = False) -> str | None:
    for value in values:
        text = normalize_optional_text(value, uppercase=uppercase)
        if text:
            return text
    return None


def set_env_if_present(target: MutableMapping[str, str], name: str, value: Any) -> None:
    text = normalize_optional_text(value)
    if text is not None:
        target[name] = text


def _github_repo_url_from_env() -> str | None:
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not server or not repo:
        return None
    return f"{server.rstrip('/')}/{repo}.git"


def infer_runtime_repo_url(args: Namespace) -> str | None:
    return first_nonempty_text(
        getattr(args, "runtime_repo_url", None),
        os.environ.get("OUROBOROS_REPO_URL"),
        os.environ.get("OUROBOROS_RUNTIME_REPO_URL"),
        _github_repo_url_from_env(),
    )


def infer_runtime_repo_ref(args: Namespace) -> str | None:
    return first_nonempty_text(
        getattr(args, "runtime_repo_ref", None),
        os.environ.get("OUROBOROS_REPO_REF"),
        os.environ.get("OUROBOROS_RUNTIME_REPO_REF"),
        os.environ.get("GITHUB_REF_NAME"),
        os.environ.get("GITHUB_REF"),
    )


def infer_runtime_repo_commit(args: Namespace) -> str | None:
    return first_nonempty_text(
        getattr(args, "runtime_repo_commit", None),
        os.environ.get("OUROBOROS_REPO_COMMIT"),
        os.environ.get("OUROBOROS_RUNTIME_REPO_COMMIT"),
        os.environ.get("GITHUB_SHA"),
    )


def _arg_or_env(args: Namespace, arg_name: str, *env_names: str) -> str | None:
    values = [getattr(args, arg_name, None)]
    values.extend(os.environ.get(name) for name in env_names)
    return first_nonempty_text(*values)


def build_worker_runtime_env(args: Namespace, worker_id: str) -> dict[str, str]:
    wid = normalize_optional_text(worker_id, uppercase=True)
    if wid not in WORKER_IDS:
        raise ValueError(f"invalid worker_id: {worker_id!r}")

    env: dict[str, str] = {}
    for name, value in os.environ.items():
        if name.startswith("OUROBOROS_"):
            set_env_if_present(env, name, value)

    repo_id = first_nonempty_text(getattr(args, "hub_repo_id", None), getattr(args, "repo_id", None))
    hf_token = _arg_or_env(args, "hf_token", "HF_TOKEN")
    github_token = _arg_or_env(args, "github_token", "GITHUB_TOKEN", "GH_TOKEN")
    wandb_key = _arg_or_env(args, "wandb_key", "WANDB_API_KEY", "WANDB_KEY")
    repo_url = infer_runtime_repo_url(args)
    repo_ref = infer_runtime_repo_ref(args)
    repo_commit = infer_runtime_repo_commit(args)

    # Worker identity: keep both historical notebook names and namespaced names.
    env["DILOCO_WORKER_ID"] = wid
    env["WORKER_ID"] = wid
    env["OUROBOROS_AUTO_TRIGGERED"] = "1"
    env["OUROBOROS_DILOCO_WORKER_ID"] = wid
    env["OUROBOROS_DILOCO_WORKER_IDS"] = ",".join(WORKER_IDS)

    set_env_if_present(env, "HF_TOKEN", hf_token)
    set_env_if_present(env, "OUROBOROS_HF_TOKEN", hf_token)
    set_env_if_present(env, "GITHUB_TOKEN", github_token)
    set_env_if_present(env, "GH_TOKEN", github_token)
    set_env_if_present(env, "OUROBOROS_GITHUB_TOKEN", github_token)
    set_env_if_present(env, "WANDB_API_KEY", wandb_key)
    set_env_if_present(env, "WANDB_KEY", wandb_key)
    set_env_if_present(env, "WANDB_PROJECT", getattr(args, "wandb_project", None))
    set_env_if_present(env, "WANDB_ENTITY", getattr(args, "wandb_entity", None))
    set_env_if_present(env, "WANDB_RUN_NAME", getattr(args, "wandb_run_name", None))

    set_env_if_present(env, "OUROBOROS_REPO_URL", repo_url)
    set_env_if_present(env, "OUROBOROS_RUNTIME_REPO_URL", repo_url)
    set_env_if_present(env, "OUROBOROS_REPO_REF", repo_ref)
    set_env_if_present(env, "OUROBOROS_RUNTIME_REPO_REF", repo_ref)
    set_env_if_present(env, "OUROBOROS_REPO_COMMIT", repo_commit)
    set_env_if_present(env, "OUROBOROS_RUNTIME_REPO_COMMIT", repo_commit)

    for name in ("OUROBOROS_HF_REPO_ID", "OUROBOROS_HUB_REPO_ID", "OUROBOROS_DILOCO_REPO_ID"):
        set_env_if_present(env, name, repo_id)

    # Mirror common DiLoCo arguments under env names consumed by worker notebooks.
    for arg_name, env_name in {
        "stage": "OUROBOROS_DILOCO_STAGE",
        "stage_k": "OUROBOROS_DILOCO_STAGE_K",
        "round": "OUROBOROS_DILOCO_ROUND",
        "round_n": "OUROBOROS_DILOCO_ROUND_N",
        "total_samples": "OUROBOROS_DILOCO_TOTAL_SAMPLES",
        "total_train_samples": "OUROBOROS_DILOCO_TOTAL_TRAIN_SAMPLES",
        "total_samples_seen": "OUROBOROS_DILOCO_TOTAL_SAMPLES_SEEN",
        "outer_lr": "OUROBOROS_DILOCO_OUTER_LR",
        "min_shard_samples": "OUROBOROS_DILOCO_MIN_SHARD_SAMPLES",
    }.items():
        set_env_if_present(env, env_name, getattr(args, arg_name, None))
    return env


def encode_runtime_env_payload(runtime_env: MutableMapping[str, str]) -> str:
    raw = json.dumps(dict(runtime_env), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(zlib.compress(raw)).decode("ascii")


# Backward-compatible private names used by the legacy coordinator module/tests.
_normalize_optional_text = normalize_optional_text
_first_nonempty_text = first_nonempty_text
_set_env_if_present = set_env_if_present
_infer_runtime_repo_url = infer_runtime_repo_url
_infer_runtime_repo_ref = infer_runtime_repo_ref
_infer_runtime_repo_commit = infer_runtime_repo_commit
_build_worker_runtime_env = build_worker_runtime_env
_encode_runtime_env_payload = encode_runtime_env_payload


__all__ = [
    "build_worker_runtime_env",
    "encode_runtime_env_payload",
    "first_nonempty_text",
    "infer_runtime_repo_commit",
    "infer_runtime_repo_ref",
    "infer_runtime_repo_url",
    "normalize_optional_text",
    "set_env_if_present",
]
