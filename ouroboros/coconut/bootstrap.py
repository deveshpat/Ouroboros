"""Runtime bootstrap seam for Kaggle/CUDA/Mamba execution.

This module is intentionally side-effect-light at import time.  Heavy dependency
installation is triggered only from :func:`ensure_runtime_ready`, which keeps tests
and simple imports fast while preserving the old runtime's execution boundary.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class RuntimeInfo:
    python: str
    cuda_available: bool
    device_name: Optional[str]
    device_capability: Optional[str]
    hf_home: Optional[str]
    rank: int = 0
    world_size: int = 1


def _int_env(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def runtime_rank() -> int:
    return _int_env("RANK", _int_env("LOCAL_RANK", 0))


def runtime_world_size() -> int:
    return max(_int_env("WORLD_SIZE", 1), 1)


def inspect_runtime() -> RuntimeInfo:
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            capability = f"sm{major}{minor}"
        else:
            name = None
            capability = None
    except Exception:
        cuda_available = False
        name = None
        capability = None

    return RuntimeInfo(
        python=platform.python_version(),
        cuda_available=cuda_available,
        device_name=name,
        device_capability=capability,
        hf_home=os.environ.get("HF_HOME"),
        rank=runtime_rank(),
        world_size=runtime_world_size(),
    )


def _run(cmd: Iterable[str]) -> None:
    proc = subprocess.run(list(cmd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def ensure_runtime_ready(*, install_requirements: bool | None = None) -> RuntimeInfo:
    """Return runtime metadata and optionally install a requirements file.

    By default this does not install anything. Set
    ``OUROBOROS_INSTALL_REQUIREMENTS=1`` or pass ``install_requirements=True`` to
    opt in from Kaggle/notebook runs.
    """

    should_install = (
        install_requirements
        if install_requirements is not None
        else os.environ.get("OUROBOROS_INSTALL_REQUIREMENTS") == "1"
    )
    if should_install:
        req = Path(os.environ.get("OUROBOROS_REQUIREMENTS", "requirements.txt"))
        if req.exists():
            _run([sys.executable, "-m", "pip", "install", "-r", str(req)])
    return inspect_runtime()


__all__ = [
    "RuntimeInfo",
    "ensure_runtime_ready",
    "inspect_runtime",
    "runtime_rank",
    "runtime_world_size",
]
