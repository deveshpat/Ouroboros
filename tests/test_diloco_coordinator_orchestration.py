from __future__ import annotations

import argparse
from pathlib import Path

from ouroboros.diloco import coordinator
from ouroboros.mac_dgac_fallback import MAC_DGAC_CLAIM_PATH


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
        workflow_validate=None,
        workflow_validation_run_id=None,
        workflow_validation_timeout_s=900.0,
        workflow_validation_poll_s=10.0,
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
    def fake_download(repo_id, path, token):
        if path == MAC_DGAC_CLAIM_PATH:
            return None
        raise AssertionError("anchor eval dispatch must not read round_state")

    monkeypatch.setattr(coordinator, "hub_download_json", fake_download)
    monkeypatch.setattr(
        coordinator,
        "hub_upload_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("anchor eval dispatch must not mutate Hub state")),
    )

    coordinator.main()

    assert triggered == [(["A"], Path("kaggle-utils.ipynb"), "dgac-anchor-eval")]


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
    def fake_download(repo_id, path, token):
        if path == MAC_DGAC_CLAIM_PATH:
            return None
        raise AssertionError("DGAC train dispatch must not read round_state")

    monkeypatch.setattr(coordinator, "hub_download_json", fake_download)
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

def test_packaged_coordinator_cpu_smoke_validation_defaults_to_worker_a_and_polls_remote_status(monkeypatch):
    monkeypatch.setattr(
        coordinator,
        "parse_args",
        lambda: _args(
            workflow_validate="cpu-smoke",
            workflow_validation_run_id="run-999-1",
            workflow_validation_timeout_s=1.0,
            workflow_validation_poll_s=0.1,
            force_worker_ids=None,
            skip_trigger=False,
        ),
    )
    triggered = []

    def fake_trigger(kaggle_creds, *, active_workers, notebook_path, coordinator_args):
        triggered.append((active_workers, notebook_path, coordinator_args.workflow_validate))
        return {worker_id: "success" for worker_id in active_workers}

    def fake_download(repo_id, path, token):
        assert path != coordinator.ROUND_STATE_PATH, "workflow validation must not read or mutate live round_state"
        assert path == "diloco_state/workflow_validation/run-999-1/worker_A_status.json"
        return {
            "worker_id": "A",
            "stage_k": 0,
            "round_n": 0,
            "status": "done",
            "samples_seen": 0,
            "validation_mode": "cpu-smoke",
            "validation_run_id": "run-999-1",
        }

    monkeypatch.setattr(coordinator, "trigger_kaggle_workers", fake_trigger)
    monkeypatch.setattr(coordinator, "hub_download_json", fake_download)
    monkeypatch.setattr(
        coordinator,
        "hub_upload_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("workflow validation is read-only")),
    )

    coordinator.main()

    assert triggered == [(["A"], Path("kaggle-utils.ipynb"), "cpu-smoke")]


def test_packaged_coordinator_cpu_smoke_validation_respects_force_worker_ids(monkeypatch):
    monkeypatch.setattr(
        coordinator,
        "parse_args",
        lambda: _args(
            workflow_validate="cpu-smoke",
            workflow_validation_run_id="run-force",
            workflow_validation_timeout_s=1.0,
            workflow_validation_poll_s=0.1,
            force_worker_ids="B,C",
            skip_trigger=False,
        ),
    )
    triggered = []

    def fake_trigger(kaggle_creds, *, active_workers, notebook_path, coordinator_args):
        triggered.append(active_workers)
        return {worker_id: "success" for worker_id in active_workers}

    def fake_download(repo_id, path, token):
        worker = "B" if "worker_B" in path else "C"
        return {
            "worker_id": worker,
            "stage_k": 0,
            "round_n": 0,
            "status": "done",
            "samples_seen": 0,
            "validation_mode": "cpu-smoke",
            "validation_run_id": "run-force",
        }

    monkeypatch.setattr(coordinator, "trigger_kaggle_workers", fake_trigger)
    monkeypatch.setattr(coordinator, "hub_download_json", fake_download)

    coordinator.main()

    assert triggered == [["B", "C"]]

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


def test_force_repair_adds_missing_workers_without_dropping_active_worker(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args(skip_trigger=False, force_worker_ids="A,C"))
    monkeypatch.setattr(coordinator.time, "time", lambda: 1000.0)
    state = {
        "stage_k": 10,
        "round_n": 2,
        "mode": "solo",
        "triggered_workers": ["B"],
        "attendance_workers": ["A", "C"],
        "triggered_at": 999.0,
        "total_samples_seen": {"10": 0},
        "completed_stages": list(range(1, 10)),
        "seed": 42,
        "dispatch_failures": ["A", "B", "C"],
    }
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        coordinator,
        "collect_ready_workers",
        lambda *args, **kwargs: [
            {"worker_id": "A", "stage_k": 10, "round_n": 2, "status": "done", "samples_seen": 0},
            {"worker_id": "C", "stage_k": 10, "round_n": 2, "status": "done", "samples_seen": 0},
        ],
    )
    triggered = []
    monkeypatch.setattr(
        coordinator,
        "trigger_kaggle_workers",
        lambda *args, **kwargs: triggered.append(kwargs["active_workers"]) or {w: "success" for w in kwargs["active_workers"]},
    )
    uploads = []
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append((args, kwargs)))
    monkeypatch.setattr(
        coordinator,
        "load_adapter_weights_cpu",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("force repair must not aggregate while B is pending")),
    )

    coordinator.main()

    assert triggered == [["A", "C"]]
    repaired_state = uploads[-1][0][2]
    assert repaired_state["triggered_workers"] == ["B", "A", "C"]
    assert repaired_state["attendance_workers"] == []
    assert repaired_state["mode"] == "diloco"
    assert repaired_state["stage_k"] == 10
    assert repaired_state["round_n"] == 2
    assert repaired_state["dispatch_failures"] == []


def test_force_repair_abc_only_triggers_workers_missing_from_active_set(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args(skip_trigger=False, force_worker_ids="A,B,C"))
    monkeypatch.setattr(coordinator.time, "time", lambda: 1000.0)
    state = {
        "stage_k": 10,
        "round_n": 2,
        "mode": "solo",
        "triggered_workers": ["B"],
        "attendance_workers": ["A", "C"],
        "triggered_at": 999.0,
        "total_samples_seen": {"10": 0},
        "completed_stages": list(range(1, 10)),
        "seed": 42,
    }
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: state)
    monkeypatch.setattr(coordinator, "collect_ready_workers", lambda *args, **kwargs: [])
    triggered = []
    monkeypatch.setattr(
        coordinator,
        "trigger_kaggle_workers",
        lambda *args, **kwargs: triggered.append(kwargs["active_workers"]) or {w: "success" for w in kwargs["active_workers"]},
    )
    uploads = []
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append((args, kwargs)))

    coordinator.main()

    assert triggered == [["A", "C"]]
    repaired_state = uploads[-1][0][2]
    assert repaired_state["triggered_workers"] == ["B", "A", "C"]
    assert repaired_state["attendance_workers"] == []


def test_force_does_not_discard_completed_non_forced_work(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args(skip_trigger=True, force_worker_ids="A,C", total_train_samples=9))
    monkeypatch.setattr(coordinator.time, "time", lambda: 1000.0)
    state = {
        "stage_k": 9,
        "round_n": 2,
        "mode": "diloco",
        "triggered_workers": ["A", "B", "C"],
        "attendance_workers": [],
        "triggered_at": 900.0,
        "total_samples_seen": {"9": 0},
        "completed_stages": list(range(1, 9)),
        "seed": 42,
    }
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        coordinator,
        "collect_ready_workers",
        lambda *args, **kwargs: [
            {"worker_id": "A", "stage_k": 9, "round_n": 2, "status": "done", "samples_seen": 1, "weights_path": "workers/A"},
            {"worker_id": "B", "stage_k": 9, "round_n": 2, "status": "done", "samples_seen": 1, "weights_path": "workers/B"},
            {"worker_id": "C", "stage_k": 9, "round_n": 2, "status": "done", "samples_seen": 1, "weights_path": "workers/C"},
        ],
    )
    loaded = []
    monkeypatch.setattr(coordinator, "load_adapter_weights_cpu", lambda repo_id, weights_path, token: loaded.append(weights_path) or {"w": weights_path})
    monkeypatch.setattr(coordinator, "hub_download_text", lambda *args, **kwargs: "{}")
    saved = []
    monkeypatch.setattr(coordinator, "aggregate_worker_updates", lambda *args, **kwargs: {"w": "new-anchor"})
    monkeypatch.setattr(coordinator, "save_and_upload_anchor", lambda *args, **kwargs: saved.append((args, kwargs)))
    uploads = []
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append((args, kwargs)))

    coordinator.main()

    assert loaded == ["diloco_state/anchor", "workers/A", "workers/B", "workers/C"]
    assert saved, "all completed workers should be aggregated even when B was not forced"
    new_state = uploads[-1][0][2]
    assert new_state["last_round_workers"] == ["A", "B", "C"]
    assert new_state["last_round_samples"] == 3


def test_force_repair_ignores_already_done_and_active_workers_for_dispatch(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args(skip_trigger=False, force_worker_ids="A,B,C"))
    monkeypatch.setattr(coordinator.time, "time", lambda: 1000.0)
    state = {
        "stage_k": 10,
        "round_n": 2,
        "mode": "solo",
        "triggered_workers": ["B"],
        "attendance_workers": ["A"],
        "triggered_at": 999.0,
        "total_samples_seen": {"10": 0},
        "completed_stages": list(range(1, 10)),
        "seed": 42,
    }
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        coordinator,
        "collect_ready_workers",
        lambda *args, **kwargs: [
            {"worker_id": "A", "stage_k": 10, "round_n": 2, "status": "done", "samples_seen": 4, "weights_path": "workers/A"},
        ],
    )
    triggered = []
    monkeypatch.setattr(
        coordinator,
        "trigger_kaggle_workers",
        lambda *args, **kwargs: triggered.append(kwargs["active_workers"]) or {w: "success" for w in kwargs["active_workers"]},
    )
    uploads = []
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append((args, kwargs)))

    coordinator.main()

    assert triggered == [["C"]]
    repaired_state = uploads[-1][0][2]
    assert repaired_state["triggered_workers"] == ["B", "A", "C"]
    assert repaired_state["attendance_workers"] == []


def test_force_repair_failed_push_does_not_falsely_mark_worker_active(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args(skip_trigger=False, force_worker_ids="A,C"))
    times = iter([1000.0, 1001.0, 1002.0, 1003.0])
    monkeypatch.setattr(coordinator.time, "time", lambda: next(times, 1004.0))
    state = {
        "stage_k": 10,
        "round_n": 2,
        "mode": "solo",
        "triggered_workers": ["B"],
        "attendance_workers": ["A", "C"],
        "triggered_at": 999.0,
        "total_samples_seen": {"10": 0},
        "completed_stages": list(range(1, 10)),
        "seed": 42,
    }
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: state)
    monkeypatch.setattr(coordinator, "collect_ready_workers", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        coordinator,
        "trigger_kaggle_workers",
        lambda *args, **kwargs: {"A": "success", "C": "failed"},
    )
    uploads = []
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append((args, kwargs)))

    coordinator.main()

    planned_state = uploads[0][0][2]
    corrected_state = uploads[1][0][2]
    assert planned_state["triggered_workers"] == ["B", "A", "C"]
    assert corrected_state["triggered_workers"] == ["B", "A"]
    assert corrected_state["attendance_workers"] == ["C"]
    assert corrected_state["dispatch_failures"] == ["C"]


def test_stage_10_completion_enters_terminal_manual_gate_without_dispatch(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args(skip_trigger=False, total_train_samples=2))
    monkeypatch.setattr(coordinator.time, "time", lambda: 1000.0)
    state = {
        "stage_k": 10,
        "round_n": 4,
        "mode": "solo",
        "triggered_workers": ["A"],
        "attendance_workers": [],
        "triggered_at": 900.0,
        "total_samples_seen": {"10": 0},
        "completed_stages": list(range(1, 10)),
        "seed": 42,
        "dispatch_failures": ["A", "B", "C"],
    }
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        coordinator,
        "collect_ready_workers",
        lambda *args, **kwargs: [
            {"worker_id": "A", "stage_k": 10, "round_n": 4, "status": "done", "samples_seen": 2, "weights_path": "workers/A"},
        ],
    )
    monkeypatch.setattr(coordinator, "load_adapter_weights_cpu", lambda repo_id, weights_path, token: {"w": weights_path})
    monkeypatch.setattr(coordinator, "hub_download_text", lambda *args, **kwargs: "{}")
    monkeypatch.setattr(coordinator, "save_and_upload_anchor", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        coordinator,
        "trigger_kaggle_workers",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stage 10 terminal must not dispatch stage 11")),
    )
    uploads = []
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append((args, kwargs)))

    coordinator.main()

    terminal_state = uploads[-1][0][2]
    assert terminal_state["stage_k"] == 10
    assert terminal_state["round_n"] == 0
    assert terminal_state["mode"] == "terminal"
    assert terminal_state["triggered_workers"] == []
    assert terminal_state["attendance_workers"] == []
    assert terminal_state["dgac_manual_gate"] is True
    assert terminal_state["triggered_at"] == 0.0
    assert terminal_state["dispatch_failures"] == []
    assert 10 in terminal_state["completed_stages"]


def test_terminal_manual_gate_state_is_idempotent_and_never_dispatches(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args(skip_trigger=False))
    state = {
        "stage_k": 10,
        "round_n": 0,
        "mode": "terminal",
        "dgac_manual_gate": True,
        "triggered_workers": [],
        "attendance_workers": [],
        "triggered_at": 0.0,
        "total_samples_seen": {"10": 9},
        "completed_stages": list(range(1, 11)),
        "seed": 42,
    }
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        coordinator,
        "trigger_kaggle_workers",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("terminal state must not dispatch")),
    )
    uploads = []
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append((args, kwargs)))

    coordinator.main()

    assert uploads == []


def test_stage_9_completion_still_advances_to_stage_10(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args(skip_trigger=True, total_train_samples=2))
    monkeypatch.setattr(coordinator.time, "time", lambda: 1000.0)
    state = {
        "stage_k": 9,
        "round_n": 4,
        "mode": "solo",
        "triggered_workers": ["A"],
        "attendance_workers": [],
        "triggered_at": 900.0,
        "total_samples_seen": {"9": 0},
        "completed_stages": list(range(1, 9)),
        "seed": 42,
    }
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        coordinator,
        "collect_ready_workers",
        lambda *args, **kwargs: [
            {"worker_id": "A", "stage_k": 9, "round_n": 4, "status": "done", "samples_seen": 2, "weights_path": "workers/A"},
        ],
    )
    monkeypatch.setattr(coordinator, "load_adapter_weights_cpu", lambda repo_id, weights_path, token: {"w": weights_path})
    monkeypatch.setattr(coordinator, "hub_download_text", lambda *args, **kwargs: "{}")
    monkeypatch.setattr(coordinator, "save_and_upload_anchor", lambda *args, **kwargs: None)
    uploads = []
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append((args, kwargs)))

    coordinator.main()

    new_state = uploads[-1][0][2]
    assert new_state["stage_k"] == 10
    assert new_state["round_n"] == 0
    assert new_state["mode"] != "terminal"
    assert 9 in new_state["completed_stages"]


def test_waiting_mode_does_not_promote_first_attendance_responder_before_grace(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args(skip_trigger=False, attendance_join_grace_minutes=5.0))
    monkeypatch.setattr(coordinator.time, "time", lambda: 1100.0)
    state = {
        "stage_k": 10,
        "round_n": 2,
        "mode": "waiting",
        "triggered_workers": [],
        "attendance_workers": ["A", "B", "C"],
        "triggered_at": 1000.0,
        "total_samples_seen": {"10": 0},
        "completed_stages": list(range(1, 10)),
        "seed": 42,
    }
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        coordinator,
        "collect_ready_workers",
        lambda *args, **kwargs: [
            {"worker_id": "A", "stage_k": 10, "round_n": 2, "status": "done", "samples_seen": 0},
        ],
    )
    uploads = []
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append((args, kwargs)))
    monkeypatch.setattr(
        coordinator,
        "trigger_kaggle_workers",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("first responder must not dispatch alone before grace")),
    )

    coordinator.main()

    assert uploads == []


def test_waiting_mode_promotes_multiple_attendance_responders_together_after_grace(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args(skip_trigger=True, attendance_join_grace_minutes=5.0))
    monkeypatch.setattr(coordinator.time, "time", lambda: 1400.0)
    state = {
        "stage_k": 10,
        "round_n": 2,
        "mode": "waiting",
        "triggered_workers": [],
        "attendance_workers": ["A", "B", "C"],
        "triggered_at": 1000.0,
        "total_samples_seen": {"10": 0},
        "completed_stages": list(range(1, 10)),
        "seed": 42,
    }
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        coordinator,
        "collect_ready_workers",
        lambda *args, **kwargs: [
            {"worker_id": "A", "stage_k": 10, "round_n": 2, "status": "done", "samples_seen": 0},
            {"worker_id": "C", "stage_k": 10, "round_n": 2, "status": "done", "samples_seen": 0},
        ],
    )
    uploads = []
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append((args, kwargs)))

    coordinator.main()

    new_state = uploads[-1][0][2]
    assert new_state["triggered_workers"] == ["A", "C"]
    assert new_state["attendance_workers"] == ["B"]
    assert new_state["mode"] == "diloco"


def test_waiting_mode_dispatch_reconciliation_demotes_failed_promoted_worker(monkeypatch):
    monkeypatch.setattr(coordinator, "parse_args", lambda: _args(skip_trigger=False, attendance_join_grace_minutes=5.0))
    times = iter([1400.0, 1401.0, 1402.0, 1403.0])
    monkeypatch.setattr(coordinator.time, "time", lambda: next(times, 1404.0))
    state = {
        "stage_k": 10,
        "round_n": 2,
        "mode": "waiting",
        "triggered_workers": [],
        "attendance_workers": ["A", "C"],
        "triggered_at": 1000.0,
        "total_samples_seen": {"10": 0},
        "completed_stages": list(range(1, 10)),
        "seed": 42,
    }
    monkeypatch.setattr(coordinator, "hub_download_json", lambda *args, **kwargs: state)
    monkeypatch.setattr(
        coordinator,
        "collect_ready_workers",
        lambda *args, **kwargs: [
            {"worker_id": "A", "stage_k": 10, "round_n": 2, "status": "done", "samples_seen": 0},
            {"worker_id": "C", "stage_k": 10, "round_n": 2, "status": "done", "samples_seen": 0},
        ],
    )
    monkeypatch.setattr(coordinator, "trigger_kaggle_workers", lambda *args, **kwargs: {"A": "success", "C": "failed"})
    uploads = []
    monkeypatch.setattr(coordinator, "hub_upload_json", lambda *args, **kwargs: uploads.append((args, kwargs)))

    coordinator.main()

    planned_state = uploads[0][0][2]
    corrected_state = uploads[1][0][2]
    assert planned_state["triggered_workers"] == ["A", "C"]
    assert corrected_state["triggered_workers"] == ["A"]
    assert corrected_state["attendance_workers"] == ["C"]
    assert corrected_state["dispatch_failures"] == ["C"]
