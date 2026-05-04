from __future__ import annotations

import argparse
from pathlib import Path

from ouroboros.diloco import coordinator


def _args(**overrides):
    values = dict(
        repo_id="fake/repo",
        hf_token="hf_fake",
        min_shard_samples=1,
        outer_lr=0.7,
        worker_timeout_hours=13.0,
        force_worker_ids=None,
        kaggle_username_a="user-a",
        kaggle_key_a="key-a",
        kaggle_username_b="user-b",
        kaggle_key_b="key-b",
        kaggle_username_c="user-c",
        kaggle_key_c="key-c",
        kaggle_notebook_path=str(Path("kaggle-utils.ipynb")),
        skip_trigger=True,
        dry_run=False,
        wandb_key=None,
        wandb_project="project",
        wandb_entity=None,
        total_train_samples=9,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_packaged_coordinator_exits_cleanly_when_round_state_is_missing(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args())
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: None)
    uploads = []
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append(args))

    coordinator.main()

    assert uploads == []


def test_packaged_coordinator_waits_without_mutating_state_before_timeout(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args())
    monkeypatch.setattr(coordinator.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        coordinator,
        "hub_download_json",
        lambda *args, **kwargs: {
            "stage_k": 0,
            "round_n": 2,
            "mode": "diloco",
            "triggered_workers": ["A", "B"],
            "attendance_workers": [],
            "triggered_at": 99.0,
            "total_samples_seen": {"0": 0},
            "completed_stages": [],
            "seed": 42,
        },
    )
    monkeypatch.setattr(coordinator, "collect_ready_workers", lambda *args, **kwargs: [])
    uploads = []
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append(args))
    monkeypatch.setattr(coordinator, "load_adapter_weights_cpu", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("aggregate must not run")))

    coordinator.main()

    assert uploads == []


def test_packaged_coordinator_advances_stage_with_fake_services(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args(total_train_samples=2))
    times = iter([1000.0, 1001.0, 1002.0])
    monkeypatch.setattr(coordinator.time, "time", lambda: next(times, 1003.0))
    state = {
        "stage_k": 0,
        "round_n": 0,
        "mode": "solo",
        "triggered_workers": ["A"],
        "attendance_workers": [],
        "triggered_at": 900.0,
        "total_samples_seen": {"0": 0},
        "completed_stages": [],
        "seed": 42,
    }
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        coordinator,
        "collect_ready_workers",
        lambda *args, **kwargs: [
            {"worker_id": "A", "stage_k": 0, "round_n": 0, "status": "done", "samples_seen": 2, "weights_path": "workers/A"}
        ],
    )
    monkeypatch.setattr(coordinator, "load_adapter_weights_cpu", lambda repo_id, weights_path, token: {"w": weights_path})
    monkeypatch.setattr(coordinator, "hub_download_text", lambda *args, **kwargs: "{}")
    saved = []
    monkeypatch.setattr(coordinator, "save_and_upload_anchor", lambda *args, **kwargs: saved.append((args, kwargs)))
    uploads = []
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append((args, kwargs)))

    coordinator.main()

    assert saved, "anchor upload seam should receive the promoted solo worker weights"
    assert saved[0][0][0] == {"w": "workers/A"}
    new_state = uploads[-1][0][2]
    assert new_state["stage_k"] == 1
    assert new_state["round_n"] == 0
    assert new_state["completed_stages"] == [0]
    assert new_state["last_round_workers"] == ["A"]
    assert new_state["last_round_samples"] == 2


def test_packaged_coordinator_reconciles_fake_kaggle_failure_after_fake_worker_status(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args(skip_trigger=False, total_train_samples=9))
    times = iter([1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0])
    monkeypatch.setattr(coordinator.time, "time", lambda: next(times, 1006.0))
    state = {
        "stage_k": 0,
        "round_n": 0,
        "mode": "solo",
        "triggered_workers": ["A"],
        "attendance_workers": [],
        "triggered_at": 900.0,
        "total_samples_seen": {"0": 0},
        "completed_stages": [],
        "seed": 42,
    }
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        coordinator,
        "collect_ready_workers",
        lambda *args, **kwargs: [
            {
                "worker_id": "A",
                "stage_k": 0,
                "round_n": 0,
                "status": "done",
                "samples_seen": 1,
                "weights_path": "workers/A",
            }
        ],
    )
    monkeypatch.setattr(coordinator, "load_adapter_weights_cpu", lambda repo_id, weights_path, token: {"w": weights_path})
    monkeypatch.setattr(coordinator, "hub_download_text", lambda *args, **kwargs: "{}")
    monkeypatch.setattr(coordinator, "save_and_upload_anchor", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        coordinator,
        "trigger_kaggle_workers",
        lambda *args, **kwargs: {"A": "success", "B": "failed", "C": "manual"},
    )
    uploads = []
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append((args, kwargs)))

    coordinator.main()

    assert len(uploads) == 2
    planned_state = uploads[0][0][2]
    corrected_state = uploads[1][0][2]
    assert planned_state["triggered_workers"] == ["A", "B", "C"]
    assert corrected_state["triggered_workers"] == ["A", "C"]
    assert corrected_state["attendance_workers"] == ["B"]
    assert corrected_state["dispatch_failures"] == ["B"]
    assert corrected_state["mode"] == "diloco"
    assert corrected_state["last_round_workers"] == ["A"]
    assert corrected_state["last_round_samples"] == 1
