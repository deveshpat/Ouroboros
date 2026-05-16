"""Small W&B runtime helpers shared by local and worker entrypoints."""

from __future__ import annotations

import os
from typing import Any, Mapping


def _wandb_init_timeout_seconds(env: Mapping[str, Any] | None = None) -> float:
    values = os.environ if env is None else env
    raw = values.get("OUROBOROS_WANDB_INIT_TIMEOUT") or values.get("WANDB_INIT_TIMEOUT") or 300
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        timeout = 300.0
    return max(timeout, 90.0)


def wandb_init_kwargs(wandb_module: Any, env: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return robust kwargs for wandb.init without hard-requiring newer W&B APIs."""
    try:
        settings = wandb_module.Settings(init_timeout=_wandb_init_timeout_seconds(env))
    except Exception:
        return {}
    return {"settings": settings}
