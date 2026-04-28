#!/usr/bin/env python3
"""Thin CLI adapter for Coconut/Jamba fine-tuning.

The former runtime file duplicated bootstrap, CLI parsing, dataset preparation,
latent-forward logic, training orchestration, and DiLoCo worker publication in one
multi-thousand-line module.  Keep this file import-safe and delegate to seams that
can be tested independently.
"""

from __future__ import annotations

from ouroboros.coconut.training_runtime import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
