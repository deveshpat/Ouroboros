"""Run Coconut training via ``python -m ouroboros.coconut``."""

from __future__ import annotations

import os
import sys
import time
from typing import Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", str(4 * 3600))
os.environ.setdefault("NCCL_TIMEOUT", str(4 * 3600))

_SCRIPT_START = time.perf_counter()


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if any(arg in {"-h", "--help"} for arg in argv):
        from ouroboros.coconut.cli import print_bootstrap_free_help_and_exit

        print_bootstrap_free_help_and_exit()
    from ouroboros.bootstrap import ensure_environment
    from ouroboros.coconut.cli import parse_args

    ensure_environment()

    from ouroboros.coconut import run_cli

    run_cli(parse_args(argv), script_start=_SCRIPT_START)


if __name__ == "__main__":
    main()
