#!/usr/bin/env python3
"""Thin compatibility adapter for the DiLoCo coordinator runtime."""

from __future__ import annotations

from typing import Any

from ouroboros.diloco import coordinator_runner as _runner
from ouroboros.diloco.coordinator_cli import parse_args
from ouroboros.diloco.coordinator_dispatch import (
    mode_from_active_workers as _mode_from_active_workers,
    reconcile_after_dispatch as _reconcile_post_dispatch_state,
    trigger_kaggle_workers,
)
from ouroboros.diloco.coordinator_io import (
    hub_download_file,
    hub_download_json,
    hub_upload_json,
    load_adapter_weights_cpu,
    retry_io as _retry_io,
    save_and_upload_anchor,
    weighted_average_deltas,
)
from ouroboros.diloco.coordinator_runner import (
    ROUND_STATE_PATH,
    main,
    _compute_projected_shards,
    _determine_round_mode,
    _ordered_unique_worker_ids,
    _partition_ready_workers,
)
from ouroboros.diloco.runtime_env import (
    build_worker_runtime_env as _build_worker_runtime_env,
    encode_runtime_env_payload as _encode_runtime_env_payload,
    first_nonempty_text as _first_nonempty_text,
    normalize_optional_text as _normalize_optional_text,
    set_env_if_present as _set_env_if_present,
)

# Legacy repo-root adapter/tests expect list equality, not the protocol tuple.
WORKER_IDS = ["A", "B", "C"]


def collect_ready_workers(*args: Any, **kwargs: Any) -> Any:
    """Delegate to coordinator_runner while honoring monkeypatches on this module."""

    old_download_json = _runner.hub_download_json
    old_download_file = _runner.hub_download_file
    _runner.hub_download_json = hub_download_json
    _runner.hub_download_file = hub_download_file
    try:
        return _runner.collect_ready_workers(*args, **kwargs)
    finally:
        _runner.hub_download_json = old_download_json
        _runner.hub_download_file = old_download_file


__all__ = [name for name in globals() if not name.startswith("__")]


if __name__ == "__main__":
    main()
