"""Runtime bootstrap seam for the Kaggle/CUDA/Mamba environment.

The original script still owns the exact install sequence. This module provides a
stable interface for the next extraction pass and lightweight runtime metadata
used by tests and orchestration code.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RuntimeInfo:
    python: str
    cuda_available: bool
    device_name: Optional[str]
    device_capability: Optional[str]
    hf_home: Optional[str]


def inspect_runtime() -> RuntimeInfo:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            capability = f"sm{cc[0]}{cc[1]}"
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
    )


def ensure_runtime_ready() -> RuntimeInfo:
    """Return runtime metadata; heavy install remains in the legacy entrypoint."""

    return inspect_runtime()
