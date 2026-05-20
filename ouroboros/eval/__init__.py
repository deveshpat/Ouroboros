"""Evaluation artifact and comparison commands for Ouroboros public alpha."""

from __future__ import annotations

__all__ = ("main",)


def main(argv=None) -> None:
    from ouroboros.eval.cli import main as _main

    _main(argv)
