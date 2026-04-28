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


def infer_runtime_repo_url(args: Namespace) -> str | None:
    return first_nonempty_text(
        getattr(args, "runtime_repo_url", None),
        os.environ.get("OUROBOROS_RUNTIME_REPO_URL"),
        os.environ.get("GITHUB_SERVER_URL") and os.environ.get("GITHUB_REPOSITORY")
        and f"{os.environ['GITHUB_SERVER_URL'].rstrip('/')}/{os.environ['GITHUB_REPOSITORY']}",
    )


def infer_runtime_repo_ref(args: Namespace) -> str | None:
    return first_nonempty_text(
        getattr(args, "runtime_repo_ref", None),
        os.environ.get("OUROBOROS_RUNTIME_REPO_REF"),
        os.environ.get("GITHUB_REF_NAME"),
        os.environ.get("GITHUB_REF"),
    )


def infer_runtime_repo_commit(args: Namespace) -> str | None:
    return first_nonempty_text(
        getattr(args, "runtime_repo_commit", None),
        os.environ.get("OUROBOROS_RUNTIME_REPO_COMMIT"),
        os.environ.get("GITHUB_SHA"),
    )


def build_worker_runtime_env(args: Namespace, worker_id: str) -> dict[str, str]:
    wid = normalize_optional_text(worker_id, uppercase=True)
    if wid not in WORKER_IDS:
        raise ValueError(f"invalid worker_id: {worker_id!r}")

    env: dict[str, str] = {}
    for name, value in os.environ.items():
        if name.startswith("OUROBOROS_"):
            set_env_if_present(env, name, value)

    env["OUROBOROS_DILOCO_WORKER_ID"] = wid
    env["OUROBOROS_DILOCO_WORKER_IDS"] = ",".join(WORKER_IDS)
    set_env_if_present(env, "OUROBOROS_HF_REPO_ID", getattr(args, "hub_repo_id", None))
    set_env_if_present(env, "OUROBOROS_HUB_REPO_ID", getattr(args, "hub_repo_id", None))
    set_env_if_present(env, "OUROBOROS_HF_TOKEN", getattr(args, "hf_token", None) or os.environ.get("HF_TOKEN"))
    set_env_if_present(env, "HF_TOKEN", getattr(args, "hf_token", None) or os.environ.get("HF_TOKEN"))
    set_env_if_present(env, "OUROBOROS_GITHUB_TOKEN", getattr(args, "github_token", None) or os.environ.get("GITHUB_TOKEN"))
    set_env_if_present(env, "GITHUB_TOKEN", getattr(args, "github_token", None) or os.environ.get("GITHUB_TOKEN"))
    set_env_if_present(env, "OUROBOROS_RUNTIME_REPO_URL", infer_runtime_repo_url(args))
    set_env_if_present(env, "OUROBOROS_RUNTIME_REPO_REF", infer_runtime_repo_ref(args))
    set_env_if_present(env, "OUROBOROS_RUNTIME_REPO_COMMIT", infer_runtime_repo_commit(args))

    # Mirror common DiLoCo arguments under env names consumed by worker notebooks.
    for arg_name, env_name in {
        "stage": "OUROBOROS_DILOCO_STAGE",
        "round": "OUROBOROS_DILOCO_ROUND",
        "total_samples": "OUROBOROS_DILOCO_TOTAL_SAMPLES",
        "total_samples_seen": "OUROBOROS_DILOCO_TOTAL_SAMPLES_SEEN",
        "outer_lr": "OUROBOROS_DILOCO_OUTER_LR",
        "hub_repo_id": "OUROBOROS_DILOCO_REPO_ID",
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
