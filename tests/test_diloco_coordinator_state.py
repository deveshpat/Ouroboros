from __future__ import annotations

from ouroboros.coordinator import state


def test_worker_ordering_mode_selection_and_projected_shards_match_coordinator_contract():
    assert state._ordered_unique_worker_ids(["b", "A", "b", "z"], ["C", "a"]) == ["B", "A", "C"]
    assert state._compute_projected_shards(10, stage_k=0, round_n=0, seed=42, total_samples_seen=1) == {
        "A": 3,
        "B": 3,
        "C": 3,
    }
    assert state._determine_round_mode({"A": 4, "B": 1, "C": 4}, ["A", "B", "C"], 3) == (
        "diloco",
        ["A", "C"],
    )
    assert state._determine_round_mode({"A": 4, "B": 1, "C": 1}, ["A", "B", "C"], 3) == (
        "solo",
        ["A"],
    )
    assert state._determine_round_mode({"A": 1, "B": 1, "C": 1}, ["A", "B", "C"], 3) == (
        "complete",
        [],
    )
    assert state._determine_round_mode({"A": 0, "B": 0, "C": 0}, ["A", "B"], 3, force_worker_ids=["B", "C"]) == (
        "solo",
        ["B"],
    )
    assert state._determine_round_mode({"A": 4, "B": 4, "C": 1}, ["A", "B", "C"], 3, force_worker_ids=["C"]) == (
        "diloco",
        ["A", "B", "C"],
    )


def test_partition_ready_workers_keeps_active_precedence_over_attendance():
    ready = [
        {"worker_id": "A", "samples_seen": 3},
        {"worker_id": "B", "samples_seen": 0},
        {"worker_id": "C", "samples_seen": 0},
    ]

    active, attendance = state._partition_ready_workers(
        ready,
        expected_workers=["A", "B"],
        attendance_workers=["B", "C"],
    )

    assert [w["worker_id"] for w in active] == ["A", "B"]
    assert [w["worker_id"] for w in attendance] == ["C"]


def test_reconcile_post_dispatch_state_demotes_failed_workers_and_preserves_unknown_fields(monkeypatch):
    monkeypatch.setattr(state.time, "time", lambda: 123.0)
    original = {"mode": "diloco", "triggered_at": 88.0, "unrelated": "preserved"}

    corrected = state._reconcile_post_dispatch_state(
        state=original,
        planned_active_workers=["A", "B"],
        planned_attendance_workers=["C"],
        dispatch_results={"A": "success", "B": "failed", "C": "failed"},
    )

    assert corrected is not None
    assert corrected["mode"] == "solo"
    assert corrected["triggered_workers"] == ["A"]
    assert corrected["attendance_workers"] == ["B", "C"]
    assert corrected["triggered_at"] == 123.0
    assert corrected["last_updated"] == 123.0
    assert corrected["dispatch_failures"] == ["B", "C"]
    assert corrected["unrelated"] == "preserved"


def test_reconcile_post_dispatch_state_enters_waiting_when_only_attendance_remains(monkeypatch):
    monkeypatch.setattr(state.time, "time", lambda: 456.0)

    corrected = state._reconcile_post_dispatch_state(
        state={"mode": "diloco", "triggered_at": 10.0},
        planned_active_workers=["A"],
        planned_attendance_workers=[],
        dispatch_results={"A": "failed"},
    )

    assert corrected is not None
    assert corrected["mode"] == "waiting"
    assert corrected["triggered_workers"] == []
    assert corrected["attendance_workers"] == ["A"]
    assert corrected["triggered_at"] == 0.0


def test_reconcile_post_dispatch_state_is_noop_without_failed_dispatches():
    assert state._reconcile_post_dispatch_state(
        state={"mode": "diloco"},
        planned_active_workers=["A"],
        planned_attendance_workers=[],
        dispatch_results={"A": "manual"},
    ) is None


def test_reconcile_post_dispatch_state_preserves_already_active_workers_without_retrigger_result(monkeypatch):
    monkeypatch.setattr(state.time, "time", lambda: 789.0)

    corrected = state._reconcile_post_dispatch_state(
        state={"mode": "diloco", "triggered_workers": ["B"], "attendance_workers": []},
        planned_active_workers=["B", "A", "C"],
        planned_attendance_workers=[],
        dispatch_results={"A": "success", "C": "failed"},
    )

    assert corrected is not None
    assert corrected["triggered_workers"] == ["B", "A"]
    assert corrected["attendance_workers"] == ["C"]
    assert corrected["dispatch_failures"] == ["C"]
