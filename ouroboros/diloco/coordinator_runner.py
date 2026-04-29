"""Compatibility wrapper for DiLoCo coordinator orchestration."""

from __future__ import annotations

from typing import Any

from ouroboros.diloco import coordinator_runner_logic as _logic

# Preserve the historical import and monkeypatch surface while the expanded
# single-pass decision tree lives in the dedicated logic seam.
for _name in _logic.__all__:
    globals()[_name] = getattr(_logic, _name)


def collect_ready_workers(*args: Any, **kwargs: Any) -> Any:
    _sync_logic_globals()
    return _logic.collect_ready_workers(*args, **kwargs)


def main(argv: Any = None) -> None:
    _sync_logic_globals()
    _logic.main(argv)


def _sync_logic_globals() -> None:
    for name in (
        "hub_download_file",
        "hub_download_json",
        "hub_upload_json",
        "trigger_kaggle_workers",
        "_aggregate_ready_workers",
        "_next_round_state",
    ):
        setattr(_logic, name, globals()[name])


__all__ = [name for name in globals() if not name.startswith("__")]
