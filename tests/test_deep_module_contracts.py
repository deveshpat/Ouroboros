from __future__ import annotations

import argparse


def test_kaggle_launch_contracts_cover_runtime_modes_and_policies():
    from ouroboros.kaggle_contract import (
        CPU_SMOKE_MODE,
        DGAC_CANARY_RUN_MODE,
        DGAC_DILOCO_RUN_MODE,
        DILOCO_RUN_MODE,
        get_kaggle_launch_contract,
        known_kaggle_launch_modes,
        resolve_kaggle_launch_contract,
    )

    modes = set(known_kaggle_launch_modes(include_cpu_smoke=True))
    assert {"diloco", "dgac-anchor-eval", "dgac-train", "dgac-canary", "dgac-diloco", "cpu-smoke"} <= modes

    diloco = get_kaggle_launch_contract(DILOCO_RUN_MODE)
    assert diloco.requires_gpu is True
    assert diloco.worker_mode is True
    assert diloco.trains is True
    assert diloco.mutates_round_state is True
    assert "OUROBOROS_KAGGLE_RUN_MODE" in diloco.env_keys
    assert "DILOCO_WORKER_ID" in diloco.env_keys

    canary = get_kaggle_launch_contract(DGAC_CANARY_RUN_MODE)
    assert canary.requires_gpu is True
    assert canary.trains is True
    assert canary.mutates_round_state is False

    dgac_diloco = get_kaggle_launch_contract(DGAC_DILOCO_RUN_MODE)
    assert dgac_diloco.worker_mode is True
    assert dgac_diloco.mutates_round_state is True

    cpu = get_kaggle_launch_contract(CPU_SMOKE_MODE)
    assert cpu.requires_gpu is False
    assert cpu.cpu_smoke_safe is True
    assert cpu.trains is False
    assert cpu.validates is True

    assert resolve_kaggle_launch_contract({"OUROBOROS_KAGGLE_RUN_MODE": "DGAC-CANARY"}) == canary


def test_training_session_planner_classifies_branches_without_heavy_imports():
    from ouroboros.training_plan import TrainingPlanKind, plan_training_session

    base = argparse.Namespace(
        diloco_mode=False,
        eval_only=False,
        resume_from=None,
        resume_from_diloco_anchor=False,
        use_halt_gate=False,
        max_train_steps=None,
        workflow_validate=None,
        gen_every_stage=True,
        diloco_run_val=False,
    )

    assert plan_training_session(base).kind == TrainingPlanKind.STANDARD_TRAIN

    eval_args = argparse.Namespace(**{**vars(base), "eval_only": True})
    eval_plan = plan_training_session(eval_args)
    assert eval_plan.kind == TrainingPlanKind.EVAL_ONLY
    assert eval_plan.should_train is False
    assert eval_plan.should_validate is True

    resume_args = argparse.Namespace(**{**vars(base), "resume_from": "runs/checkpoint"})
    resume_plan = plan_training_session(resume_args)
    assert resume_plan.kind == TrainingPlanKind.RESUME_TRAIN
    assert resume_plan.resume_source == "runs/checkpoint"

    dgac_args = argparse.Namespace(**{**vars(base), "use_halt_gate": True, "resume_from_diloco_anchor": True})
    dgac_plan = plan_training_session(dgac_args)
    assert dgac_plan.kind == TrainingPlanKind.DGAC_TRAIN
    assert dgac_plan.should_train is True

    worker_args = argparse.Namespace(**{**vars(dgac_args), "diloco_mode": True, "diloco_worker_id": "b"})
    worker_plan = plan_training_session(worker_args)
    assert worker_plan.kind == TrainingPlanKind.DGAC_DILOCO_WORKER
    assert worker_plan.delegates_to_diloco is True
    assert worker_plan.skip_worker_pre_validation is False

    bad_args = argparse.Namespace(**{**vars(base), "diloco_mode": True, "use_halt_gate": True})
    try:
        plan_training_session(bad_args)
    except ValueError as exc:
        assert "resume_from_diloco_anchor" in str(exc)
    else:
        raise AssertionError("expected unsupported DGAC DiLoCo combination to fail")


def test_worker_lifecycle_classifier_expresses_training_attendance_and_passthrough():
    from ouroboros.worker_lifecycle import WorkerLifecycleKind, classify_worker_lifecycle

    state = {"triggered_workers": ["A"], "attendance_workers": ["B"], "stage_k": 1, "round_n": 2}

    active = classify_worker_lifecycle(worker_id="a", round_state=state, shard_samples=3)
    assert active.kind == WorkerLifecycleKind.NORMAL_DILOCO_WORKER
    assert active.should_train is True
    assert active.contributes_to_aggregation is True

    attendance = classify_worker_lifecycle(worker_id="B", round_state=state, shard_samples=3)
    assert attendance.kind == WorkerLifecycleKind.ATTENDANCE_ONLY
    assert attendance.should_train is False
    assert attendance.should_upload_status is True
    assert attendance.contributes_to_aggregation is False

    passthrough = classify_worker_lifecycle(worker_id="A", round_state=state, shard_samples=0)
    assert passthrough.kind == WorkerLifecycleKind.EMPTY_SHARD_PASSTHROUGH
    assert passthrough.should_train is False
    assert passthrough.should_push_signal is True

    dgac = classify_worker_lifecycle(
        worker_id="A",
        round_state=state,
        shard_samples=3,
        use_halt_gate=True,
        resume_from_diloco_anchor=True,
    )
    assert dgac.kind == WorkerLifecycleKind.DGAC_DILOCO_WORKER
    assert dgac.skip_pre_validation is False

    skipped = classify_worker_lifecycle(worker_id="C", round_state=state, shard_samples=3)
    assert skipped.kind == WorkerLifecycleKind.NOOP
    assert skipped.should_train is False

    try:
        classify_worker_lifecycle(worker_id="A", round_state=state, shard_samples=3, use_halt_gate=True)
    except ValueError as exc:
        assert "resume_from_diloco_anchor" in str(exc)
    else:
        raise AssertionError("expected unsupported halt-gate combination")


def test_coordinator_decision_module_plans_force_repair_and_dispatch_reconcile():
    from ouroboros.coordinator_decision import plan_force_repair, plan_round_start
    from ouroboros.diloco.state import _reconcile_post_dispatch_state

    state = {
        "stage_k": 2,
        "round_n": 5,
        "mode": "diloco",
        "triggered_workers": ["A"],
        "attendance_workers": ["C"],
        "total_samples_seen": {"2": 0},
        "seed": 42,
        "triggered_at": 0.0,
    }
    plan = plan_round_start(
        state=state,
        total_train_samples=96,
        min_shard_samples=16,
        credentialed_workers=["A", "B", "C"],
        force_worker_ids="B,C",
        worker_timeout_hours=13.0,
        now=100.0,
    )

    assert plan.expected_workers == ["A"]
    assert plan.attendance_workers == ["C"]
    assert plan.force_worker_ids == ["B", "C"]
    assert plan.next_active_workers == ["A", "B"]
    assert plan.next_attendance_workers == ["C"]
    assert plan.unconfirmed_dispatch is True

    repair = plan_force_repair(
        expected_workers=["A"],
        attendance_workers=["C"],
        force_worker_ids=["A", "B", "C"],
        ready_worker_ids={"A"},
        attendance_ready_ids=set(),
        credentialed_workers=["A", "B", "C"],
    )
    assert repair.already_done_workers == ["A"]
    assert repair.dispatch_workers == ["B", "C"]
    assert repair.active_workers == ["A", "B", "C"]
    assert repair.attendance_workers == []

    corrected = _reconcile_post_dispatch_state(
        state={**state, "triggered_workers": ["A", "B"], "attendance_workers": ["C"]},
        planned_active_workers=["A", "B"],
        planned_attendance_workers=["C"],
        dispatch_results={"B": "failed", "C": "manual"},
    )
    assert corrected is not None
    assert corrected["triggered_workers"] == ["A"]
    assert corrected["attendance_workers"] == ["C", "B"]
    assert "B" in corrected["dispatch_failures"]


def test_coordinator_transition_planner_handles_waiting_mode_actions():
    from ouroboros.coordinator_decision import plan_round_start, plan_waiting_mode_transition

    state = {
        "stage_k": 10,
        "round_n": 2,
        "mode": "waiting",
        "triggered_workers": [],
        "attendance_workers": ["A", "B", "C"],
        "triggered_at": 0.0,
        "total_samples_seen": {"10": 0},
        "completed_stages": list(range(1, 10)),
        "seed": 42,
    }
    round_plan = plan_round_start(
        state=state,
        total_train_samples=9,
        min_shard_samples=1,
        credentialed_workers=["A", "B", "C"],
        now=1000.0,
    )

    initial = plan_waiting_mode_transition(
        state=state,
        round_plan=round_plan,
        responded_worker_ids=[],
        credentialed_workers=["A", "B", "C"],
        total_train_samples=9,
        min_shard_samples=1,
        attendance_join_grace_minutes=5.0,
        now=1234.0,
    )
    assert initial.kind == "waiting_initial_dispatch"
    assert initial.should_write_state is True
    assert initial.dispatch_attendance_workers == ["A", "B", "C"]
    assert initial.state["triggered_at"] == 1234.0

    waiting_state = {**state, "triggered_at": 1000.0}
    waiting_plan = plan_round_start(
        state=waiting_state,
        total_train_samples=9,
        min_shard_samples=1,
        credentialed_workers=["A", "B", "C"],
        now=1100.0,
    )
    grace = plan_waiting_mode_transition(
        state=waiting_state,
        round_plan=waiting_plan,
        responded_worker_ids=["A"],
        credentialed_workers=["A", "B", "C"],
        total_train_samples=9,
        min_shard_samples=1,
        attendance_join_grace_minutes=5.0,
        now=1100.0,
    )
    assert grace.kind == "waiting_grace"
    assert grace.should_write_state is False
    assert grace.metadata["still_absent_workers"] == ["B", "C"]

    promoted = plan_waiting_mode_transition(
        state=waiting_state,
        round_plan=waiting_plan,
        responded_worker_ids=["A", "C"],
        credentialed_workers=["A", "B", "C"],
        total_train_samples=9,
        min_shard_samples=1,
        attendance_join_grace_minutes=5.0,
        now=1400.0,
    )
    assert promoted.kind == "waiting_promote"
    assert promoted.state["round_n"] == 3
    assert promoted.state["triggered_workers"] == ["A", "C"]
    assert promoted.state["attendance_workers"] == ["B"]


def test_coordinator_transition_planner_handles_missing_worker_repair_and_unconfirmed_dispatch():
    from ouroboros.coordinator_decision import plan_missing_worker_transition

    state = {
        "stage_k": 10,
        "round_n": 2,
        "mode": "solo",
        "triggered_workers": ["B"],
        "attendance_workers": ["A", "C"],
        "triggered_at": 999.0,
    }
    repair = plan_missing_worker_transition(
        state=state,
        stage_k=10,
        round_n=2,
        expected_workers=["B"],
        attendance_workers=["A", "C"],
        missing_workers=["B"],
        force_worker_ids=["A", "C"],
        ready_worker_ids=set(),
        attendance_ready_ids=set(),
        credentialed_workers=["A", "B", "C"],
        is_round_timed_out=False,
        now=1000.0,
    )
    assert repair.kind == "force_repair"
    assert repair.dispatch_active_workers == ["A", "C"]
    assert repair.state["triggered_workers"] == ["B", "A", "C"]
    assert repair.state["attendance_workers"] == []

    unconfirmed = plan_missing_worker_transition(
        state={**state, "triggered_workers": ["A", "B"], "attendance_workers": ["C"], "triggered_at": 0.0},
        stage_k=10,
        round_n=2,
        expected_workers=["A", "B"],
        attendance_workers=["C"],
        missing_workers=["B"],
        force_worker_ids=[],
        ready_worker_ids={"A"},
        attendance_ready_ids=set(),
        credentialed_workers=["A", "B", "C"],
        is_round_timed_out=False,
        now=1000.0,
    )
    assert unconfirmed.kind == "unconfirmed_redispatch"
    assert unconfirmed.workers_to_dispatch == ["A", "B", "C"]
    assert unconfirmed.reconcile_active_workers == ["A", "B"]
    assert unconfirmed.reconcile_attendance_workers == ["C"]

    timed_out = plan_missing_worker_transition(
        state=state,
        stage_k=10,
        round_n=2,
        expected_workers=["A", "B"],
        attendance_workers=["C"],
        missing_workers=["B"],
        force_worker_ids=[],
        ready_worker_ids={"A"},
        attendance_ready_ids=set(),
        credentialed_workers=["A", "B", "C"],
        is_round_timed_out=True,
        now=1000.0,
    )
    assert timed_out.kind == "timeout_continue"
    assert timed_out.should_run_aggregation is True
    assert timed_out.metadata["newly_demoted_workers"] == ["B"]


def test_coordinator_post_aggregation_planner_handles_terminal_and_absent_transitions():
    from ouroboros.coordinator_decision import plan_post_aggregation_transition

    base_state = {
        "stage_k": 10,
        "round_n": 4,
        "mode": "solo",
        "triggered_workers": ["A"],
        "attendance_workers": [],
        "total_samples_seen": {"10": 2},
        "completed_stages": list(range(1, 10)),
        "seed": 42,
    }
    terminal = plan_post_aggregation_transition(
        state=base_state,
        stage_k=10,
        round_n=4,
        current_mode="solo",
        total_train_samples=2,
        min_shard_samples=1,
        credentialed_workers=["A", "B", "C"],
        force_worker_ids=None,
        expected_workers=["A"],
        attendance_workers=[],
        attendance_ready_ids=set(),
        ready_worker_ids={"A"},
        is_round_timed_out=False,
        total_samples_seen={"10": 2},
        stage_samples_seen=0,
        completed_stages=list(range(1, 10)),
        seed=42,
        contributing_workers=[{"worker_id": "A", "samples_seen": 2}],
        anchor_path="anchor",
        terminal_stage=10,
        dgac_complete_mode="dgac-complete",
        now=1000.0,
    )
    assert terminal.kind == "terminal_manual_gate"
    assert terminal.state["mode"] == "terminal"
    assert terminal.state["dgac_manual_gate"] is True
    assert terminal.workers_to_dispatch == []

    dgac = plan_post_aggregation_transition(
        state={**base_state, "mode": "dgac-diloco", "dgac_diloco": True},
        stage_k=10,
        round_n=4,
        current_mode="dgac-diloco",
        total_train_samples=2,
        min_shard_samples=1,
        credentialed_workers=["A", "B", "C"],
        force_worker_ids=None,
        expected_workers=["A"],
        attendance_workers=[],
        attendance_ready_ids=set(),
        ready_worker_ids={"A"},
        is_round_timed_out=False,
        total_samples_seen={"10": 2},
        stage_samples_seen=0,
        completed_stages=list(range(1, 10)),
        seed=42,
        contributing_workers=[{"worker_id": "A", "samples_seen": 2}],
        anchor_path="anchor",
        terminal_stage=10,
        dgac_complete_mode="dgac-complete",
        now=1000.0,
    )
    assert dgac.kind == "dgac_diloco_complete"
    assert dgac.state["mode"] == "dgac-complete"
    assert dgac.state["dgac_diloco_complete"] is True
    assert dgac.state["round_n"] == 4
    assert dgac.state["dgac_round_n"] == 4
    assert dgac.state["next_dgac_round_n"] == 5

    absent = plan_post_aggregation_transition(
        state={"stage_k": 0, "round_n": 0, "mode": "diloco", "seed": 42},
        stage_k=0,
        round_n=0,
        current_mode="diloco",
        total_train_samples=9,
        min_shard_samples=1,
        credentialed_workers=["A", "B", "C"],
        force_worker_ids=None,
        expected_workers=["A", "B", "C"],
        attendance_workers=[],
        attendance_ready_ids=set(),
        ready_worker_ids=set(),
        is_round_timed_out=True,
        total_samples_seen={"0": 0},
        stage_samples_seen=0,
        completed_stages=[],
        seed=42,
        contributing_workers=[],
        anchor_path="anchor",
        terminal_stage=10,
        dgac_complete_mode="dgac-complete",
        now=1000.0,
    )
    assert absent.kind == "all_absent_waiting"
    assert absent.state["round_n"] == 0
    assert absent.state["mode"] == "waiting"
    assert absent.state["attendance_workers"] == ["A", "B", "C"]


def test_dispatch_reconciliation_is_owned_by_decision_layer():
    from ouroboros.coordinator_decision import plan_dispatch_reconciliation

    plan = plan_dispatch_reconciliation(
        state={"mode": "diloco", "triggered_workers": ["A", "B"], "attendance_workers": []},
        planned_active_workers=["A", "B"],
        planned_attendance_workers=[],
        dispatch_results={"A": "success", "B": "failed"},
    )
    assert plan.should_write_state is True
    assert plan.corrected_state["triggered_workers"] == ["A"]
    assert plan.corrected_state["attendance_workers"] == ["B"]
    assert plan.corrected_state["dispatch_failures"] == ["B"]
