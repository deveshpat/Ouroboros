"""Stdlib-safe runtime environment alias resolution for Ouroboros.

This module is intentionally importable before torch, transformers, Kaggle,
Hugging Face, or W&B. Bootstrap, notebook launch, workflow validation,
coordinator, and worker code should use these helpers instead of hand-rolled
env parsing so alias behavior cannot drift.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Optional, Sequence

WORKER_IDS: tuple[str, ...] = ("A", "B", "C")
WORKER_ID_ALIASES: tuple[str, ...] = (
    "DILOCO_WORKER_ID",
    "OUROBOROS_DILOCO_WORKER_ID",
    "WORKER_ID",
)
HF_TOKEN_ALIASES: tuple[str, ...] = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
GITHUB_TOKEN_ALIASES: tuple[str, ...] = ("GITHUB_TOKEN", "GH_TOKEN")
WANDB_KEY_ALIASES: tuple[str, ...] = ("WANDB_API_KEY", "WANDB_KEY")
KAGGLE_USERNAME_TEMPLATE = "KAGGLE_USERNAME_{worker_id}"
KAGGLE_KEY_TEMPLATE = "KAGGLE_KEY_{worker_id}"

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def normalize_text(value: Any | None, *, uppercase: bool = False) -> Optional[str]:
    """Trim text values and normalize empty/missing values to None."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.upper() if uppercase else text


def normalize_worker_id(value: Any | None) -> Optional[str]:
    """Return a canonical worker id or None for missing/invalid values."""
    worker_id = normalize_text(value, uppercase=True)
    if worker_id in WORKER_IDS:
        return worker_id
    return None


def parse_worker_id_list(value: Any | None) -> list[str]:
    """Parse a comma/list/tuple worker selection into ordered unique A/B/C ids."""
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, Sequence):
        raw_items = list(value)
    else:
        raw_items = [value]

    parsed: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        worker_id = normalize_worker_id(item)
        if worker_id is None or worker_id in seen:
            continue
        parsed.append(worker_id)
        seen.add(worker_id)
    return parsed


def resolve_env_alias(env: Mapping[str, Any] | None, names: Sequence[str]) -> Optional[str]:
    """Resolve the first non-empty value among aliases from env."""
    env = os.environ if env is None else env
    for name in names:
        value = normalize_text(env.get(name))
        if value is not None:
            return value
    return None


def resolve_worker_id(env: Mapping[str, Any] | None = None, *, cli_value: Any | None = None) -> Optional[str]:
    """Resolve a worker id from CLI override first, then runtime env aliases."""
    worker_id = normalize_worker_id(cli_value)
    if worker_id is not None:
        return worker_id
    env = os.environ if env is None else env
    for name in WORKER_ID_ALIASES:
        worker_id = normalize_worker_id(env.get(name))
        if worker_id is not None:
            return worker_id
    return None


def require_worker_id(env: Mapping[str, Any] | None = None, *, cli_value: Any | None = None) -> str:
    """Resolve a worker id or fail with a user-actionable error."""
    worker_id = resolve_worker_id(env, cli_value=cli_value)
    if worker_id is None:
        raise ValueError(
            "DiLoCo worker id is required. Set DILOCO_WORKER_ID, "
            "OUROBOROS_DILOCO_WORKER_ID, WORKER_ID, or pass --diloco_worker_id."
        )
    return worker_id


def resolve_hf_token(cli_value: Any | None = None, env: Mapping[str, Any] | None = None) -> Optional[str]:
    token = normalize_text(cli_value)
    if token is not None:
        return token
    return resolve_env_alias(env, HF_TOKEN_ALIASES)


def resolve_github_token(cli_value: Any | None = None, env: Mapping[str, Any] | None = None) -> Optional[str]:
    token = normalize_text(cli_value)
    if token is not None:
        return token
    return resolve_env_alias(env, GITHUB_TOKEN_ALIASES)


def resolve_wandb_key(cli_value: Any | None = None, env: Mapping[str, Any] | None = None) -> Optional[str]:
    token = normalize_text(cli_value)
    if token is not None:
        return token
    return resolve_env_alias(env, WANDB_KEY_ALIASES)


def resolve_kaggle_credentials(
    env: Mapping[str, Any] | None,
    worker_id: str,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve per-worker Kaggle credentials without exposing values."""
    env = os.environ if env is None else env
    worker = require_known_worker_id(worker_id)
    username = normalize_text(env.get(KAGGLE_USERNAME_TEMPLATE.format(worker_id=worker)))
    key = normalize_text(env.get(KAGGLE_KEY_TEMPLATE.format(worker_id=worker)))
    return username, key


def require_known_worker_id(worker_id: Any) -> str:
    normalized = normalize_worker_id(worker_id)
    if normalized is None:
        raise ValueError(f"Invalid DiLoCo worker id {worker_id!r}. Expected one of A, B, or C.")
    return normalized


def env_bool(env: Mapping[str, Any] | None, name: str, *, default: bool = False) -> bool:
    env = os.environ if env is None else env
    value = normalize_text(env.get(name))
    if value is None:
        return bool(default)
    lowered = value.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    return bool(default)


def env_int(env: Mapping[str, Any] | None, name: str, *, default: int = 0) -> int:
    env = os.environ if env is None else env
    value = normalize_text(env.get(name))
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)
