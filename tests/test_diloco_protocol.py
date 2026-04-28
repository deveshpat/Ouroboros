from ouroboros.diloco.protocol import (
    ProtocolConfig,
    RoundState,
    WorkerStatus,
    compute_projected_shards,
    determine_round_mode,
    ordered_unique_worker_ids,
    partition_ready_workers,
    plan_next_round,
    reconcile_post_dispatch_state,
)


def test_projected_shards_keep_three_way_determinism():
    assert compute_projected_shards(total_samples=10, total_samples_seen=0) == {"A": 4, "B": 3, "C": 3}
    assert compute_projected_shards(total_samples=10, total_samples_seen=4) == {"A": 2, "B": 2, "C": 2}


def test_round_mode_complete_solo_diloco_and_force():
    projected = {"A": 32, "B": 31, "C": 40}
    assert determine_round_mode(projected_shards=projected, credentialed_workers=["A", "B", "C"], min_shard_samples=32) == ("diloco", ["A", "C"])
    assert determine_round_mode(projected_shards=projected, credentialed_workers=["B"], min_shard_samples=32) == ("complete", [])
    assert determine_round_mode(projected_shards=projected, credentialed_workers=["A"], min_shard_samples=32) == ("solo", ["A"])
    assert determine_round_mode(projected_shards=projected, credentialed_workers=["A", "B"], min_shard_samples=100, force_worker_ids=["B"]) == ("solo", ["B"])


def test_partition_ready_workers_keeps_attendance_non_blocking():
    ready = [{"worker_id": "A"}, {"worker_id": "C"}]
    active, attendance = partition_ready_workers(ready, expected_workers=["A"], attendance_workers=["C"])
    assert [x["worker_id"] for x in active] == ["A"]
    assert [x["worker_id"] for x in attendance] == ["C"]


def test_triggered_at_zero_means_retry_dispatch_not_timeout():
    state = RoundState.from_mapping({
        "stage_k": 3,
        "round_n": 0,
        "mode": "diloco",
        "triggered_workers": ["A", "B"],
        "attendance_workers": [],
        "triggered_at": 0,
        "total_samples_seen": {"3": 0},
    })
    plan = plan_next_round(
        state=state,
        statuses=[],
        credentialed_workers=["A", "B", "C"],
        config=ProtocolConfig(total_train_samples=36906, min_shard_samples=32),
        now=999999,
    )
    assert plan.should_dispatch is True
    assert plan.should_aggregate is False
    assert plan.active_workers == ("A", "B")
    assert "sentinel" in plan.reason


def test_missing_worker_after_timeout_is_demoted_to_attendance():
    state = RoundState.from_mapping({
        "stage_k": 3,
        "round_n": 0,
        "mode": "diloco",
        "triggered_workers": ["A", "B"],
        "attendance_workers": [],
        "triggered_at": 100,
        "total_samples_seen": {"3": 0},
    })
    statuses = [WorkerStatus.from_mapping({"worker_id": "A", "stage_k": 3, "round_n": 0, "status": "done", "samples_seen": 123})]
    plan = plan_next_round(
        state=state,
        statuses=statuses,
        credentialed_workers=["A", "B", "C"],
        config=ProtocolConfig(total_train_samples=36906, min_shard_samples=32, worker_timeout_hours=13),
        now=100 + 14 * 3600,
    )
    assert plan.should_aggregate is True
    assert plan.active_workers == ("A",)
    assert plan.attendance_workers == ("B",)
    assert plan.timed_out_workers == ("B",)


def test_remaining_below_min_shard_advances_stage():
    state = RoundState.from_mapping({
        "stage_k": 3,
        "round_n": 9,
        "mode": "diloco",
        "triggered_workers": [],
        "attendance_workers": [],
        "total_samples_seen": {"3": 99},
    })
    plan = plan_next_round(
        state=state,
        statuses=[],
        credentialed_workers=["A", "B", "C"],
        config=ProtocolConfig(total_train_samples=100, min_shard_samples=32),
        now=123,
    )
    assert plan.mode == "complete"
    assert plan.should_advance_stage is True
    assert plan.should_dispatch is False


def test_reconcile_failed_dispatch_demotes_failed_active_and_resets_timestamp_when_none_sent():
    corrected = reconcile_post_dispatch_state(
        state={"mode": "diloco", "triggered_workers": ["A", "B"], "attendance_workers": []},
        planned_active_workers=["A", "B"],
        planned_attendance_workers=[],
        dispatch_results={"A": "failed", "B": "failed"},
        now=123.0,
    )
    assert corrected is not None
    assert corrected["mode"] == "waiting"
    assert corrected["triggered_workers"] == []
    assert corrected["attendance_workers"] == ["A", "B"]
    assert corrected["triggered_at"] == 0.0
