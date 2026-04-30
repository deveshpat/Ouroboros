#!/usr/bin/env python3
"""Thin compatibility shim — implementation lives in coordinator_runner."""

from __future__ import annotations

from ouroboros.diloco.coordinator_runner import *  # noqa: F401, F403
from ouroboros.diloco.coordinator_runner import (
    ROUND_STATE_PATH,
    collect_ready_workers,
    hub_download_json,
    main,
)
from ouroboros.diloco.protocol import (
    WORKER_IDS as _WORKER_IDS,
    ordered_unique_worker_ids as _ordered_unique_worker_ids,
)

# Legacy tests assert list equality, not tuple.
WORKER_IDS: list[str] = list(_WORKER_IDS)

if __name__ == "__main__":
    main()
