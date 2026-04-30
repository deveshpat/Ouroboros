from __future__ import annotations

from typing import Any

from ouroboros.diloco.hub_state import InMemoryHubStateStore, ROUND_STATE_PATH, worker_status_path


def _argv(*extra: str) -> list[str]:
    return ["--hub_repo_id", "owner/model", "--total_samples", "100", *extra]


def _round_state(**overrides: Any) -> dict[str, Any]:
    state = {
        "stage": 7,
        "stage_k": 7,
        "round": 0,
        "round_n": 0,
        "mode": "diloco",
        "active_workers": ["A", "B"],
        "triggered_workers": ["A", "B"],
        "attendance_workers": [],
        "triggered_at": 1000.0,
        "projected_shards": {"A": 34, "B": 33, "C": 33},
        "total_samples": 100,
        "total_train_samples": 100,
        "total_samples_seen": 0,
    }
    state.update(overrides)
    return state


def _ready_status(worker_id: str, *, weights_path: str | None = None) -> dict[str, Any]:
    status = {"worker_id": worker_id, "stage_k": 7, "round_n": 0, "status": "done", "samples_seen": 10}
    if weights_path:
        status["weights_path"] = weights_path
    return status


def _make_store_and_dispatch(
    initial_files: dict[str, dict[str, Any]],
    dispatches: list[tuple[str, ...]],
) -> tuple[InMemoryHubStateStore, Any]:
    store = InMemoryHubStateStore(initial_files)

    def fake_dispatch(args: Any, workers: list[str] | tuple[str, ...], round_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        normalized = tuple(str(worker).strip().upper() for worker in workers)
        dispatches.append(normalized)
        return {worker: {"worker_id": worker, "outcome": "success", "ok": True} for worker in normalized}

    return store, fake_dispatch


def test_main_retries_dispatch_when_triggered_at_is_zero() -> None:
    import ouroboros.diloco.coordinator_runner as runner

    dispatches: list[tuple[str, ...]] = []
    store, fake_dispatch = _make_store_and_dispatch(
        {ROUND_STATE_PATH: _round_state(triggered_at=0.0)},
        dispatches,
    )

    runner.main(_argv(), store=store, dispatch=fake_dispatch)

    assert dispatches == [("A", "B")]
    saved = store.load_round_state_raw()
    assert saved is not None
    assert saved["triggered_workers"] == ["A", "B"]
    assert saved["triggered_at"] > 0.0


def test_main_aggregates_ready_workers_and_dispatches_next_round(monkeypatch: Any) -> None:
    import ouroboros.diloco.coordinator_runner as runner

    dispatches: list[tuple[str, ...]] = []
    store, fake_dispatch = _make_store_and_dispatch(
        {
            ROUND_STATE_PATH: _round_state(),
            worker_status_path("A"): _ready_status("A", weights_path="a.safetensors"),
            worker_status_path("B"): _ready_status("B", weights_path="b.safetensors"),
        },
        dispatches,
    )

    monkeypatch.setattr(
        runner,
        "_aggregate_ready_workers",
        lambda args, round_state, statuses, ready: {
            "anchor_path": "diloco_state/anchor/stage_7/round_0/anchor.safetensors",
            "samples_by_worker": {worker: 10 for worker in ready},
        },
    )
    monkeypatch.setattr(
        runner,
        "_next_round_state",
        lambda args, current, aggregate_result: _round_state(
            round=1,
            round_n=1,
            active_workers=["C"],
            triggered_workers=[],
            triggered_at=0.0,
            anchor_path=aggregate_result["anchor_path"],
            previous_anchor_path=aggregate_result["anchor_path"],
            total_samples_seen=20,
        ),
    )

    runner.main(_argv(), store=store, dispatch=fake_dispatch)

    assert dispatches == [("C",)]
    saved = store.load_round_state_raw()
    assert saved is not None
    assert saved["round_n"] == 1
    assert saved["triggered_workers"] == ["C"]
    assert saved["anchor_path"].endswith("anchor.safetensors")


def test_main_demotes_timed_out_missing_worker_to_attendance(monkeypatch: Any) -> None:
    import ouroboros.diloco.coordinator_runner as runner

    dispatches: list[tuple[str, ...]] = []
    store, fake_dispatch = _make_store_and_dispatch(
        {
            ROUND_STATE_PATH: _round_state(triggered_at=1000.0),
            worker_status_path("A"): _ready_status("A", weights_path="a.safetensors"),
        },
        dispatches,
    )
    monkeypatch.setattr(runner.time, "time", lambda: 1000.0 + 14 * 3600)
    monkeypatch.setattr(
        runner,
        "_aggregate_ready_workers",
        lambda args, round_state, statuses, ready: {
            "anchor_path": "diloco_state/anchor/stage_7/round_0/anchor.safetensors",
            "samples_by_worker": {worker: 10 for worker in ready},
        },
    )
    monkeypatch.setattr(
        runner,
        "_next_round_state",
        lambda args, current, aggregate_result: _round_state(
            round=1,
            round_n=1,
            active_workers=["A"],
            triggered_workers=[],
            attendance_workers=["B"],
            timed_out_workers=["B"],
            triggered_at=0.0,
            anchor_path=aggregate_result["anchor_path"],
            previous_anchor_path=aggregate_result["anchor_path"],
            total_samples_seen=10,
        ),
    )

    runner.main(_argv("--worker_timeout_hours", "13"), store=store, dispatch=fake_dispatch)

    assert dispatches == [("B", "A")]
    saved = store.load_round_state_raw()
    assert saved is not None
    assert saved["triggered_workers"] == ["A"]
    assert saved["attendance_workers"] == ["B"]
    assert saved["timed_out_workers"] == ["B"]


def test_main_waits_without_error_when_workers_not_ready_or_timed_out(monkeypatch: Any, capsys: Any) -> None:
    import ouroboros.diloco.coordinator_runner as runner

    dispatches: list[tuple[str, ...]] = []
    store, fake_dispatch = _make_store_and_dispatch({ROUND_STATE_PATH: _round_state(triggered_at=1000.0)}, dispatches)
    monkeypatch.setattr(runner.time, "time", lambda: 1300.0)

    runner.main(_argv("--worker_timeout_hours", "13"), store=store, dispatch=fake_dispatch)

    assert dispatches == []
    assert "[coordinator] Waiting for workers: ['A', 'B'] (triggered 300s ago)" in capsys.readouterr().out
    saved = store.load_round_state_raw()
    assert saved is not None
    assert saved["triggered_at"] == 1000.0


def test_main_contains_no_polling_loop_or_sleep() -> None:
    import inspect
    import ouroboros.diloco.coordinator_runner as runner

    source = inspect.getsource(runner.main)
    assert "while True" not in source
    assert "time.sleep" not in source
