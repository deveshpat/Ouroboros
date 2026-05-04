#!/usr/bin/env python3
"""Thin compatibility adapter for the Ouroboros training entrypoint.

This root script intentionally owns only process startup concerns:
  * critical environment variables that must be set before any torch import;
  * bootstrap-free help rendering;
  * runtime dependency bootstrap; and
  * delegation into the packaged training CLI.

All reusable training, checkpoint, DiLoCo worker, DGAC, and data behavior lives in
``ouroboros/*``. Keep this file small so Kaggle/notebook launch behavior remains
observable while the old training monolith is retired.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Sequence

# Set critical env vars BEFORE any torch/NCCL import. The packaged training
# module is loaded only after dependency bootstrap.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", str(4 * 3600))
os.environ.setdefault("NCCL_TIMEOUT", str(4 * 3600))

# Capture startup before bootstrap so dependency install and model load time are
# counted in Kaggle timeout accounting.
_SCRIPT_START = time.perf_counter()


def _print_bootstrap_free_help_and_exit() -> None:
    """Render CLI help without dependency installs, CUDA probes, or network calls."""
    from ouroboros.cli import print_bootstrap_free_help_and_exit

    print_bootstrap_free_help_and_exit()


# Keep ``--help`` bootstrap-free. This must stay before ``ensure_environment()``
# and before importing ``ouroboros.train``.
if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    _print_bootstrap_free_help_and_exit()


def main(argv: Sequence[str] | None = None) -> None:
    """Bootstrap runtime dependencies, parse CLI args, and run training."""
    from ouroboros.bootstrap import ensure_environment
    from ouroboros.cli import parse_args

    ensure_environment()

    from ouroboros.train import run_cli

    args = parse_args(argv)
    run_cli(args, script_start=_SCRIPT_START)


if __name__ == "__main__":
    main()
