"""Utils public interface: provider/env/Hub/W&B/Kaggle/Azure helpers.

Importing this package is intentionally lightweight. Expensive or cyclic helpers
remain available from their owning submodules, e.g. ``ouroboros.utils.hub``.
"""

from __future__ import annotations

from .runtime_env import (
    env_bool,
    env_int,
    normalize_text,
    parse_worker_id_list,
    require_known_worker_id,
    resolve_env_alias,
    resolve_github_token,
    resolve_hf_token,
    resolve_kaggle_credentials,
    resolve_wandb_key,
    resolve_worker_id,
)
from .wandb_runtime import wandb_init_kwargs
from .kaggle_runtime import KaggleRepoSpec, resolve_kaggle_repo_spec

__all__ = (
    "env_bool",
    "env_int",
    "normalize_text",
    "parse_worker_id_list",
    "require_known_worker_id",
    "resolve_env_alias",
    "resolve_github_token",
    "resolve_hf_token",
    "resolve_kaggle_credentials",
    "resolve_wandb_key",
    "resolve_worker_id",
    "wandb_init_kwargs",
    "KaggleRepoSpec",
    "resolve_kaggle_repo_spec",
)
