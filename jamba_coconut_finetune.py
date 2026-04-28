#!/usr/bin/env python3
"""Thin adapter for the Coconut/Jamba fine-tuning runtime.

The heavy Kaggle bootstrap and training implementation lives in
:mod:`ouroboros.coconut.finetune_runtime`. This file remains as the stable CLI
path used by notebooks, torchrun, and existing docs.
"""

from __future__ import annotations

import runpy


RUNTIME_MODULE = "ouroboros.coconut.finetune_runtime"


def main() -> None:
    # Avoid importing the runtime at module import time: the runtime intentionally
    # performs Kaggle dependency bootstrap at top level when executed.
    runpy.run_module(RUNTIME_MODULE, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
