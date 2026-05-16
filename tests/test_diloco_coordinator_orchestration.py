from __future__ import annotations

import argparse
from pathlib import Path

from ouroboros.coordinator import coordinator


def _args(**overrides):
    values = dict(
        repo_id="fake/repo",
        hf_token="hf_fake",
        min_shard_samples=1,
        outer_lr=0.7,
        worker_timeout_hours=13.0,
        attendance_join_grace_minutes=5.0,
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
        kaggle_run_mode="diloco",
    )
    values.update(overrides)
    return argparse.Namespace(**values)



def test_packaged_coordinator_dgac_anchor_eval_dispatches_one_gpu_notebook_without_round_state(monkeypatch):
    monkeypatch.setattr(
        coordinator,
        "parse_args",
        lambda: _args(
            kaggle_run_mode="dgac-anchor-eval",
            force_worker_ids=None,
            skip_trigger=False,
        ),
    )
    triggered = []

    def fake_trigger(kaggle_creds, *, active_workers, notebook_path, coordinator_args):
        triggered.append((active_workers, notebook_path, coordinator_args.kaggle_run_mode))
        return {worker_id: "success" for worker_id in active_workers}

    monkeypatch.setattr(coordinator, "trigger_kaggle_workers", fake_trigger)
    monkeypatch.setattr(
        coordinator,
        "hub_download_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("anchor eval dispatch must not read Hub state")
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "hub_upload_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("anchor eval dispatch must not mutate Hub state")),
    )

    coordinator.main()

    assert triggered == [(["A"], Path("kaggle-utils.ipynb"), "dgac-anchor-eval")]


def test_packaged_coordinator_benchmark_dispatches_one_gpu_notebook_without_round_state(monkeypatch):
    monkeypatch.setattr(
        coordinator,
        "parse_args",
        lambda: _args(
            kaggle_run_mode="benchmark",
            force_worker_ids=None,
            skip_trigger=False,
        ),
    )
    triggered = []

    def fake_trigger(kaggle_creds, *, active_workers, notebook_path, coordinator_args):
        triggered.append((active_workers, notebook_path, coordinator_args.kaggle_run_mode))
        return {worker_id: "success" for worker_id in active_workers}

    monkeypatch.setattr(coordinator, "trigger_kaggle_workers", fake_trigger)

    monkeypatch.setattr(
        coordinator,
        "hub_download_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("benchmark dispatch must not read Hub state")
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "hub_upload_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("benchmark dispatch must not mutate Hub state")),
    )

    coordinator.main()

    assert triggered == [(["A"], Path("kaggle-utils.ipynb"), "benchmark")]


def test_packaged_coordinator_dgac_train_dispatches_one_gpu_notebook_without_round_state(monkeypatch):
    monkeypatch.setattr(
        coordinator,
        "parse_args",
        lambda: _args(
            kaggle_run_mode="dgac-train",
            force_worker_ids=None,
            skip_trigger=False,
        ),
    )
    triggered = []

    def fake_trigger(kaggle_creds, *, active_workers, notebook_path, coordinator_args):
        triggered.append((active_workers, notebook_path, coordinator_args.kaggle_run_mode))
        return {worker_id: "success" for worker_id in active_workers}

    monkeypatch.setattr(coordinator, "trigger_kaggle_workers", fake_trigger)
    monkeypatch.setattr(
        coordinator,
        "hub_download_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("DGAC train dispatch must not read Hub state")
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "hub_upload_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("DGAC train dispatch must not mutate Hub state")),
    )

    coordinator.main()

    assert triggered == [(["A"], Path("kaggle-utils.ipynb"), "dgac-train")]


def test_packaged_coordinator_dgac_train_uses_one_forced_worker_only(monkeypatch):
    monkeypatch.setattr(
        coordinator,
        "parse_args",
        lambda: _args(
            kaggle_run_mode="dgac-train",
            force_worker_ids="B,C",
            skip_trigger=False,
        ),
    )
    triggered = []

    def fake_trigger(kaggle_creds, *, active_workers, notebook_path, coordinator_args):
        triggered.append(active_workers)
        return {worker_id: "success" for worker_id in active_workers}

    monkeypatch.setattr(coordinator, "trigger_kaggle_workers", fake_trigger)
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: None)

    coordinator.main()

    assert triggered == [["B"]]


def test_packaged_coordinator_dgac_anchor_eval_respects_force_worker_ids(monkeypatch):
    monkeypatch.setattr(
        coordinator,
        "parse_args",
        lambda: _args(
            kaggle_run_mode="dgac-anchor-eval",
            force_worker_ids="B",
            skip_trigger=False,
        ),
    )
    triggered = []

    def fake_trigger(kaggle_creds, *, active_workers, notebook_path, coordinator_args):
        triggered.append(active_workers)
        return {worker_id: "success" for worker_id in active_workers}

    monkeypatch.setattr(coordinator, "trigger_kaggle_workers", fake_trigger)
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: None)

    coordinator.main()

    assert triggered == [["B"]]



def test_packaged_coordinator_dgac_diloco_initializes_round_state_and_dispatches_all_forced_workers(monkeypatch):
    monkeypatch.setattr(
        coordinator,
        "parse_args",
        lambda: _args(
            kaggle_run_mode="dgac-diloco",
            force_worker_ids="A,C",
            skip_trigger=False,
            total_train_samples=9,
        ),
    )
    triggered = []
    uploads = []

    def fake_trigger(kaggle_creds, *, active_workers, notebook_path, coordinator_args):
        triggered.append((active_workers, notebook_path, coordinator_args.kaggle_run_mode))
        return {worker_id: "success" for worker_id in active_workers}

    monkeypatch.setattr(coordinator, "trigger_kaggle_workers", fake_trigger)
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: {
        "stage_k": 10,
        "round_n": 0,
        "mode": "terminal",
        "total_samples_seen": {"10": 36906},
        "completed_stages": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append((args, kwargs)))

    coordinator.main()

    assert triggered == [(["A", "C"], Path("kaggle-utils.ipynb"), "dgac-diloco")]
    state = uploads[0][0][2]
    assert state["mode"] == "dgac-diloco"
    assert state["triggered_workers"] == ["A", "C"]
    assert state["dgac_diloco"] is True
    assert state["total_samples_seen"] == {"10": 0}
    assert state["pre_dgac_total_samples_seen"] == {"10": 36906}


def test_initial_dgac_diloco_state_preserves_previous_totals_and_resets_stage_counter():
    state = coordinator._initial_dgac_diloco_state(
        previous_state={
            "total_samples_seen": {"10": 36906},
            "completed_stages": [0, 1, 10],
        },
        worker_ids=["A", "B", "C"],
        projected_shards={"A": 3, "B": 3, "C": 3},
        seed=123,
    )

    assert state["mode"] == "dgac-diloco"
    assert state["total_samples_seen"] == {"10": 0}
    assert state["pre_dgac_total_samples_seen"] == {"10": 36906}
    assert state["completed_stages"] == [0, 1, 10]
    assert state["triggered_workers"] == ["A", "B", "C"]
    assert state["seed"] == 123


def test_initial_dgac_diloco_state_uses_successive_dedicated_rounds_after_cancelled_run():
    next_round = coordinator._next_dgac_dedicated_round_n(
        {
            "stage_k": 10,
            "round_n": 0,
            "dgac_round_n": 0,
            "mode": "dgac-diloco",
            "dgac_diloco": True,
            "total_samples_seen": {"10": 100},
        }
    )
    state = coordinator._initial_dgac_diloco_state(
        previous_state={
            "stage_k": 10,
            "round_n": 0,
            "dgac_round_n": 0,
            "mode": "dgac-diloco",
            "dgac_diloco": True,
            "total_samples_seen": {"10": 100},
            "completed_stages": [10],
        },
        worker_ids=["A", "C"],
        projected_shards={"A": 3, "C": 3},
        seed=123,
        dgac_round_n=next_round,
    )

    assert next_round == 1
    assert state["round_n"] == 1
    assert state["dgac_round_n"] == 1
    assert state["dgac_round_label"] == "DGAC dedicated round 001"
    assert state["total_samples_seen"] == {"10": 0}
    assert state["triggered_workers"] == ["A", "C"]


def test_initial_dgac_diloco_state_starts_first_dedicated_round_from_terminal_gate():
    next_round = coordinator._next_dgac_dedicated_round_n(
        {
            "stage_k": 10,
            "round_n": 0,
            "mode": "terminal",
            "dgac_manual_gate": True,
            "total_samples_seen": {"10": 36906},
        }
    )

    assert next_round == 0



def test_packaged_coordinator_auto_dispatches_anchor_eval_after_dgac_completion(monkeypatch):
    monkeypatch.setattr(
        coordinator,
        "parse_args",
        lambda: _args(
            kaggle_run_mode="diloco",
            skip_trigger=False,
            total_train_samples=9,
        ),
    )
    state = {
        "stage_k": 10,
        "round_n": 3,
        "dgac_round_n": 3,
        "mode": "dgac-diloco",
        "triggered_workers": ["B"],
        "attendance_workers": [],
        "triggered_at": 900.0,
        "anchor_path": "diloco_state/anchor",
        "total_samples_seen": {"10": 8},
        "completed_stages": [10],
        "dgac_diloco": True,
        "seed": 42,
    }
    triggered = []
    uploads = []
    saved = []

    monkeypatch.setattr(coordinator.time, "time", lambda: 1000.0)
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        coordinator,
        "collect_ready_workers",
        lambda *args, **kwargs: [
            {
                "worker_id": "B",
                "stage_k": 10,
                "round_n": 3,
                "status": "done",
                "samples_seen": 1,
                "weights_path": "workers/B",
                "halt_gate_path": "workers/B/halt_gate.pt",
            }
        ],
    )
    monkeypatch.setattr(coordinator, "load_adapter_weights_cpu", lambda repo_id, weights_path, token: {"w": weights_path})
    monkeypatch.setattr(coordinator, "load_torch_state_cpu", lambda repo_id, path, token: {"gate": path})
    monkeypatch.setattr(coordinator, "hub_download_text", lambda *args, **kwargs: "{}")
    monkeypatch.setattr(coordinator, "save_and_upload_anchor", lambda *args, **kwargs: saved.append((args, kwargs)))
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append((args, kwargs)))

    def fake_trigger(kaggle_creds, *, active_workers, notebook_path, coordinator_args):
        triggered.append((active_workers, notebook_path, coordinator_args.kaggle_run_mode))
        return {worker_id: "success" for worker_id in active_workers}

    monkeypatch.setattr(coordinator, "trigger_kaggle_workers", fake_trigger)

    coordinator.main()

    assert saved
    assert uploads[-1][0][2]["mode"] == "dgac-complete"
    assert uploads[-1][0][2]["dgac_diloco_complete"] is True
    assert triggered == [(["A"], Path("kaggle-utils.ipynb"), "dgac-anchor-eval")]
