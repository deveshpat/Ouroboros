"""Stdlib-safe runtime environment alias resolution for Ouroboros."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Optional, Sequence

HF_TOKEN_ALIASES: tuple[str, ...] = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
WANDB_KEY_ALIASES: tuple[str, ...] = ("WANDB_API_KEY", "WANDB_KEY")

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


def resolve_env_alias(env: Mapping[str, Any] | None, names: Sequence[str]) -> Optional[str]:
    """Resolve the first non-empty value among aliases from env."""
    env = os.environ if env is None else env
    for name in names:
        value = normalize_text(env.get(name))
        if value is not None:
            return value
    return None


def resolve_hf_token(cli_value: Any | None = None, env: Mapping[str, Any] | None = None) -> Optional[str]:
    token = normalize_text(cli_value)
    if token is not None:
        return token
    return resolve_env_alias(env, HF_TOKEN_ALIASES)


def resolve_wandb_key(cli_value: Any | None = None, env: Mapping[str, Any] | None = None) -> Optional[str]:
    token = normalize_text(cli_value)
    if token is not None:
        return token
    return resolve_env_alias(env, WANDB_KEY_ALIASES)


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


def process_rank(env: Mapping[str, Any] | None = None) -> int:
    env = os.environ if env is None else env
    return env_int(env, "RANK", default=0)


def local_process_rank(env: Mapping[str, Any] | None = None) -> int:
    env = os.environ if env is None else env
    value = normalize_text(env.get("LOCAL_RANK"))
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass
    return process_rank(env)


def world_size(env: Mapping[str, Any] | None = None) -> int:
    return env_int(env, "WORLD_SIZE", default=1)


def is_main_process(env: Mapping[str, Any] | None = None) -> bool:
    return process_rank(env) == 0
