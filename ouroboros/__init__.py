"""Project Ouroboros runtime package.

This package contains behavior-preserving seams extracted from the original
single-file coordinator/training scripts. The scripts remain the CLI entrypoints;
these modules make the protocol/runtime behavior testable through small
interfaces.
"""

__all__ = ["diloco", "coconut"]
