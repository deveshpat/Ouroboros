"""Pure coordinator decision seams for DiLoCo/DGAC round orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from ouroboros.coordinator.state import (
    WORKER_IDS,
    _compute_projected_shards,
    _determine_round_mode,
    _mode_from_active_workers,
    _reconcile_post_dispatch_state,
)
from ouroboros.utils.runtime_env import parse_worker_id_list


@dataclass(frozen=True)
class CoordinatorRoundStartPlan:
    stage_k: int
    round_n: int
    current_mode: str
    expected_workers: list[str]
    attendance_workers: list[str]
    force_worker_ids: list[str]
    projected_shards: dict[str, int]
    remaining_samples: int
    next_mode: str
    next_active_workers: list[str]
    next_attendance_workers: list[str]
    is_round_timed_out: bool
    unconfirmed_dispatch: bool


@dataclass(frozen=True)
class ForceRepairPlan:
    active_workers: list[str]
    attendance_workers: list[str]
    already_done_workers: list[str]
    dispatch_workers: list[str]
    unavailable_workers: list[str]

    @property
    def has_work(self) -> bool:
        return bool(self.already_done_workers or self.dispatch_workers)


@dataclass(frozen=True)
class CoordinatorTransitionDecision:
    """Declarative coordinator state-machine outcome for the side-effect runner."""

    kind: str
    reason: str
    should_write_state: bool = False
    state: Optional[dict[str, Any]] = None
    hub_message: str = ""
    dispatch_active_workers: list[str] = field(default_factory=list)
    dispatch_attendance_workers: list[str] = field(default_factory=list)
    reconcile_active_workers: list[str] = field(default_factory=list)
    reconcile_attendance_workers: list[str] = field(default_factory=list)
    dispatch_reconcile_message: str = ""
    should_run_aggregation: bool = False
    should_stop: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def workers_to_dispatch(self) -> list[str]:
        return _ordered_workers(self.dispatch_active_workers, self.dispatch_attendance_workers)


@dataclass(frozen=True)
class DispatchReconciliationPlan:
    corrected_state: Optional[dict[str, Any]]

    @property
    def should_write_state(self) -> bool:
        return self.corrected_state is not None


def _ordered_workers(*groups: Iterable[str] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for worker in parse_worker_id_list(list(group or [])):
            if worker not in seen:
                result.append(worker)
                seen.add(worker)
    return result


def _ordered_known_workers(*groups: Iterable[str] | None) -> list[str]:
    wanted = set(_ordered_workers(*groups))
    return [worker for worker in WORKER_IDS if worker in wanted]


def _last_round_worker_ids(contributing_workers: list[Mapping[str, Any]]) -> list[str]:
    return [str(worker.get("worker_id", "")).upper() for worker in contributing_workers]


def _last_round_samples(contributing_workers: list[Mapping[str, Any]]) -> int:
    return sum(int(worker.get("samples_seen", 0)) for worker in contributing_workers)


def plan_round_start(
    *,
    state: Mapping[str, Any],
    total_train_samples: int,
    min_shard_samples: int,
    credentialed_workers: list[str],
    force_worker_ids: str | list[str] | None = None,
    worker_timeout_hours: float = 13.0,
    now: Optional[float] = None,
) -> CoordinatorRoundStartPlan:
    """Plan the deterministic start-of-run coordinator state."""
    now = time.time() if now is None else float(now)
    stage_k = int(state.get("stage_k", 0))
    round_n = int(state.get("round_n", 0))
    current_mode = str(state.get("mode", "diloco"))
    total_samples_seen = {str(k): int(v) for k, v in dict(state.get("total_samples_seen", {})).items()}
    stage_samples_seen = int(total_samples_seen.get(str(stage_k), 0))
    seed = int(state.get("seed", 42))
    expected_workers = parse_worker_id_list(state.get("triggered_workers"))
    attendance_workers = [
        worker for worker in parse_worker_id_list(state.get("attendance_workers"))
        if worker not in set(expected_workers)
    ]
    force_ids = parse_worker_id_list(force_worker_ids)
    triggered_at = float(state.get("triggered_at", 0.0))
    worker_timeout_s = max(float(worker_timeout_hours), 0.0) * 3600.0
    is_round_timed_out = triggered_at > 0 and (now - triggered_at) > worker_timeout_s

    projected_shards = _compute_projected_shards(
        total_samples=int(total_train_samples),
        stage_k=stage_k,
        round_n=round_n,
        seed=seed,
        total_samples_seen=stage_samples_seen,
    )
    eligible_for_training = [worker for worker in parse_worker_id_list(credentialed_workers) if worker not in attendance_workers]
    next_mode, next_active_workers = _determine_round_mode(
        projected_shards=projected_shards,
        credentialed_workers=eligible_for_training,
        min_shard_samples=int(min_shard_samples),
        force_worker_ids=force_ids or None,
    )
    return CoordinatorRoundStartPlan(
        stage_k=stage_k,
        round_n=round_n,
        current_mode=current_mode,
        expected_workers=expected_workers,
        attendance_workers=attendance_workers,
        force_worker_ids=force_ids,
        projected_shards=projected_shards,
        remaining_samples=max(int(total_train_samples) - stage_samples_seen, 0),
        next_mode=next_mode,
        next_active_workers=next_active_workers,
        next_attendance_workers=list(attendance_workers),
        is_round_timed_out=is_round_timed_out,
        unconfirmed_dispatch=bool(expected_workers) and triggered_at <= 0,
    )


def plan_force_repair(
    *,
    expected_workers: list[str],
    attendance_workers: list[str],
    force_worker_ids: list[str],
    ready_worker_ids: set[str],
    attendance_ready_ids: set[str],
    credentialed_workers: list[str],
) -> ForceRepairPlan:
    """Plan additive manual repair without discarding valid worker work."""
    expected = _ordered_workers(expected_workers)
    attendance = _ordered_workers(attendance_workers)
    credentialed = set(_ordered_workers(credentialed_workers))
    ready_or_attendance = set(_ordered_workers(list(ready_worker_ids), list(attendance_ready_ids)))

    already_done: list[str] = []
    dispatch: list[str] = []
    unavailable: list[str] = []
    for worker_id in _ordered_workers(force_worker_ids):
        if worker_id in expected:
            if worker_id in ready_or_attendance and worker_id not in already_done:
                already_done.append(worker_id)
            continue
        if worker_id in ready_or_attendance:
            already_done.append(worker_id)
            continue
        if worker_id not in credentialed:
            unavailable.append(worker_id)
            continue
        dispatch.append(worker_id)

    active_workers = _ordered_workers(expected, already_done, dispatch)
    attendance_after = [worker for worker in attendance if worker not in set(active_workers)]
    return ForceRepairPlan(
        active_workers=active_workers,
        attendance_workers=attendance_after,
        already_done_workers=already_done,
        dispatch_workers=dispatch,
        unavailable_workers=unavailable,
    )


def plan_waiting_mode_transition(
    *,
    state: Mapping[str, Any],
    round_plan: CoordinatorRoundStartPlan,
    responded_worker_ids: Iterable[str],
    credentialed_workers: list[str],
    total_train_samples: int,
    min_shard_samples: int,
    attendance_join_grace_minutes: float,
    now: float,
) -> CoordinatorTransitionDecision:
    """Plan waiting-mode attendance dispatch, idle wait, re-dispatch, or promotion."""
    attendance_workers = list(round_plan.attendance_workers)
    responded_ids = set(_ordered_workers(responded_worker_ids))
    still_absent = [worker for worker in attendance_workers if worker not in responded_ids]
    triggered_at = float(state.get("triggered_at", 0.0))

    if not responded_ids:
        if triggered_at <= 0:
            new_state = {
                **dict(state),
                "triggered_at": now,
                "last_updated": now,
                "dispatch_failures": [],
            }
            return CoordinatorTransitionDecision(
                kind="waiting_initial_dispatch",
                reason="Waiting mode has no confirmed dispatch timestamp; dispatch attendance workers.",
                should_write_state=True,
                state=new_state,
                hub_message=f"Waiting mode: initial attendance dispatch round={round_plan.round_n}",
                dispatch_attendance_workers=attendance_workers,
                reconcile_attendance_workers=attendance_workers,
                dispatch_reconcile_message=(
                    f"Waiting mode dispatch reconcile: stage={round_plan.stage_k} "
                    f"round={round_plan.round_n} attendance={{attendance_workers}}"
                ),
            )
        if not round_plan.is_round_timed_out:
            return CoordinatorTransitionDecision(
                kind="waiting_standby",
                reason="Waiting mode has no attendance responses and remains inside timeout.",
            )
        new_state = {
            **dict(state),
            "triggered_at": now,
            "last_updated": now,
            "dispatch_failures": [],
        }
        return CoordinatorTransitionDecision(
            kind="waiting_redispatch",
            reason="Waiting mode timed out without responses; re-dispatch attendance workers.",
            should_write_state=True,
            state=new_state,
            hub_message=f"Waiting mode: re-dispatch attendance round={round_plan.round_n}",
            dispatch_attendance_workers=attendance_workers,
            reconcile_attendance_workers=attendance_workers,
            dispatch_reconcile_message=(
                f"Waiting mode re-dispatch reconcile: round={round_plan.round_n} attendance={{attendance_workers}}"
            ),
        )

    attendance_grace_s = max(float(attendance_join_grace_minutes), 0.0) * 60.0
    waiting_elapsed_s = (now - triggered_at) if triggered_at > 0 else 0.0
    all_attendance_responded = not still_absent
    grace_expired = triggered_at > 0 and waiting_elapsed_s >= attendance_grace_s
    if not all_attendance_responded and not grace_expired:
        return CoordinatorTransitionDecision(
            kind="waiting_grace",
            reason="Some attendance workers responded, but the join grace window is still open.",
            metadata={
                "responded_workers": sorted(responded_ids),
                "still_absent_workers": still_absent,
                "waiting_elapsed_s": waiting_elapsed_s,
            },
        )

    responded_list = sorted(responded_ids)
    total_samples_seen = {str(k): int(v) for k, v in dict(state.get("total_samples_seen", {})).items()}
    stage_samples_seen = int(total_samples_seen.get(str(round_plan.stage_k), 0))
    total_samples_seen[str(round_plan.stage_k)] = stage_samples_seen
    next_round_n = round_plan.round_n + 1
    projected_shards_next = _compute_projected_shards(
        total_samples=int(total_train_samples),
        stage_k=round_plan.stage_k,
        round_n=next_round_n,
        seed=int(state.get("seed", 42)),
        total_samples_seen=stage_samples_seen,
    )
    eligible_for_training = [worker for worker in _ordered_workers(credentialed_workers) if worker in responded_ids]
    next_mode, next_active_workers = _determine_round_mode(
        projected_shards=projected_shards_next,
        credentialed_workers=eligible_for_training,
        min_shard_samples=int(min_shard_samples),
        force_worker_ids=round_plan.force_worker_ids or None,
    )
    next_attendance_workers = still_absent
    if not next_active_workers:
        next_mode = "waiting"
        next_attendance_workers = attendance_workers

    new_state = {
        **dict(state),
        "stage_k": round_plan.stage_k,
        "round_n": next_round_n,
        "mode": next_mode,
        "triggered_workers": next_active_workers,
        "attendance_workers": next_attendance_workers,
        "projected_shards": projected_shards_next,
        "total_samples_seen": total_samples_seen,
        "last_updated": now,
        "triggered_at": now,
        "dispatch_failures": [],
        "last_round_workers": responded_list,
        "last_round_samples": 0,
        "seed": int(state.get("seed", 42)),
    }
    return CoordinatorTransitionDecision(
        kind="waiting_promote",
        reason="Waiting mode responders are ready to be promoted into a training round.",
        should_write_state=True,
        state=new_state,
        hub_message=f"Waiting mode resolved: stage={round_plan.stage_k} round={next_round_n} mode={next_mode}",
        dispatch_active_workers=next_active_workers,
        dispatch_attendance_workers=next_attendance_workers,
        reconcile_active_workers=next_active_workers,
        reconcile_attendance_workers=next_attendance_workers,
        dispatch_reconcile_message=(
            f"Waiting mode resolved dispatch reconcile: stage={round_plan.stage_k} "
            f"round={next_round_n} mode={{mode}}"
        ),
        metadata={"responded_workers": responded_list, "still_absent_workers": still_absent},
    )


def plan_missing_worker_transition(
    *,
    state: Mapping[str, Any],
    stage_k: int,
    round_n: int,
    expected_workers: list[str],
    attendance_workers: list[str],
    missing_workers: list[str],
    force_worker_ids: list[str],
    ready_worker_ids: set[str],
    attendance_ready_ids: set[str],
    credentialed_workers: list[str],
    is_round_timed_out: bool,
    now: float,
) -> CoordinatorTransitionDecision:
    """Plan active-round missing worker wait, force repair, or unconfirmed re-dispatch."""
    repair_plan = plan_force_repair(
        expected_workers=expected_workers,
        attendance_workers=attendance_workers,
        force_worker_ids=force_worker_ids,
        ready_worker_ids=ready_worker_ids,
        attendance_ready_ids=attendance_ready_ids,
        credentialed_workers=credentialed_workers,
    )
    if repair_plan.has_work:
        repaired_state = {
            **dict(state),
            "mode": _mode_from_active_workers(repair_plan.active_workers),
            "triggered_workers": repair_plan.active_workers,
            "attendance_workers": repair_plan.attendance_workers,
            "last_updated": now,
            "triggered_at": now,
            "dispatch_failures": [],
        }
        return CoordinatorTransitionDecision(
            kind="force_repair",
            reason="Forced workers can be added without discarding already-planned active work.",
            should_write_state=True,
            state=repaired_state,
            hub_message=f"Force repair dispatch: stage={stage_k} round={round_n} active={repair_plan.active_workers}",
            dispatch_active_workers=repair_plan.dispatch_workers,
            reconcile_active_workers=repair_plan.active_workers,
            reconcile_attendance_workers=repair_plan.attendance_workers,
            dispatch_reconcile_message=f"Force repair dispatch reconcile: stage={stage_k} round={round_n} mode={{mode}}",
            metadata={"repair_plan": repair_plan},
        )

    if repair_plan.unavailable_workers:
        return CoordinatorTransitionDecision(
            kind="force_repair_unavailable",
            reason="Forced workers were requested but have no usable credentials and cannot be dispatched.",
            metadata={"repair_plan": repair_plan},
        )

    if float(state.get("triggered_at", 0.0)) <= 0:
        new_state = {
            **dict(state),
            "triggered_at": now,
            "last_updated": now,
            "dispatch_failures": [],
        }
        all_to_trigger = _ordered_workers(expected_workers, [w for w in attendance_workers if w not in set(expected_workers)])
        return CoordinatorTransitionDecision(
            kind="unconfirmed_redispatch",
            reason="Round was marked triggered but has no confirmed dispatch timestamp; re-dispatch immediately.",
            should_write_state=True,
            state=new_state,
            hub_message=f"Re-dispatch unconfirmed: stage={stage_k} round={round_n}",
            dispatch_active_workers=all_to_trigger,
            reconcile_active_workers=expected_workers,
            reconcile_attendance_workers=attendance_workers,
            dispatch_reconcile_message=f"Re-dispatch reconcile: stage={stage_k} round={round_n}",
            metadata={"missing_workers": missing_workers, "repair_plan": repair_plan},
        )

    if not is_round_timed_out:
        return CoordinatorTransitionDecision(
            kind="wait_for_missing_workers",
            reason="Active workers are missing but the round has not timed out.",
            metadata={"missing_workers": missing_workers, "repair_plan": repair_plan},
        )

    newly_demoted = [worker for worker in missing_workers if worker not in attendance_workers]
    still_absent = [worker for worker in missing_workers if worker in attendance_workers]
    return CoordinatorTransitionDecision(
        kind="timeout_continue",
        reason="Missing workers timed out; continue toward aggregation/next planning with attendance demotion.",
        should_stop=False,
        should_run_aggregation=True,
        metadata={
            "missing_workers": missing_workers,
            "newly_demoted_workers": newly_demoted,
            "still_absent_workers": still_absent,
            "repair_plan": repair_plan,
        },
    )


def plan_post_aggregation_transition(
    *,
    state: Mapping[str, Any],
    stage_k: int,
    round_n: int,
    current_mode: str,
    total_train_samples: int,
    min_shard_samples: int,
    credentialed_workers: list[str],
    force_worker_ids: list[str] | None,
    expected_workers: list[str],
    attendance_workers: list[str],
    attendance_ready_ids: set[str],
    ready_worker_ids: set[str],
    is_round_timed_out: bool,
    total_samples_seen: Mapping[str, int],
    stage_samples_seen: int,
    completed_stages: list[int],
    seed: int,
    contributing_workers: list[Mapping[str, Any]],
    anchor_path: str,
    terminal_stage: int,
    dgac_complete_mode: str,
    now: float,
) -> CoordinatorTransitionDecision:
    """Plan the next state after any ready worker aggregation has completed."""
    totals = {str(k): int(v) for k, v in dict(total_samples_seen).items()}
    final_stage_samples = int(totals.get(str(stage_k), stage_samples_seen))
    newly_demoted = [
        worker for worker in (expected_workers or [])
        if worker not in ready_worker_ids and worker not in attendance_workers
    ] if is_round_timed_out else []
    still_attending = [worker for worker in attendance_workers if worker not in attendance_ready_ids]
    next_attendance_workers = _ordered_known_workers(newly_demoted, still_attending)

    projected_shards_next = _compute_projected_shards(
        total_samples=int(total_train_samples),
        stage_k=stage_k,
        round_n=round_n + 1,
        seed=seed,
        total_samples_seen=final_stage_samples,
    )
    completion_mode, _ = _determine_round_mode(
        projected_shards=projected_shards_next,
        credentialed_workers=credentialed_workers,
        min_shard_samples=int(min_shard_samples),
        force_worker_ids=None,
    )
    eligible_for_training = [worker for worker in credentialed_workers if worker not in next_attendance_workers]
    planning_force_ids = force_worker_ids if not expected_workers and not contributing_workers else None
    next_mode, next_active_workers = _determine_round_mode(
        projected_shards=projected_shards_next,
        credentialed_workers=eligible_for_training,
        min_shard_samples=int(min_shard_samples),
        force_worker_ids=planning_force_ids,
    )

    last_workers = _last_round_worker_ids(contributing_workers)
    last_samples = _last_round_samples(contributing_workers)
    completed = sorted(set(int(stage) for stage in completed_stages))
    stage_complete = final_stage_samples >= int(total_train_samples) or completion_mode == "complete"
    next_stage_k = stage_k
    next_round_n = round_n + 1

    if stage_complete:
        completed = sorted(set(completed + [stage_k]))
        if stage_k >= terminal_stage:
            if current_mode == "dgac-diloco" or bool(state.get("dgac_diloco")):
                dgac_round_n = int(state.get("dgac_round_n", round_n))
                terminal_state = {
                    **dict(state),
                    "stage_k": terminal_stage,
                    "round_n": dgac_round_n,
                    "dgac_round_n": dgac_round_n,
                    "dgac_round_label": f"DGAC dedicated round {dgac_round_n:03d}",
                    "next_dgac_round_n": dgac_round_n + 1,
                    "mode": dgac_complete_mode,
                    "triggered_workers": [],
                    "attendance_workers": [],
                    "projected_shards": {},
                    "anchor_path": anchor_path,
                    "total_samples_seen": totals,
                    "completed_stages": completed,
                    "dgac_manual_gate": True,
                    "dgac_diloco": True,
                    "dgac_diloco_complete": True,
                    "last_updated": now,
                    "triggered_at": 0.0,
                    "dispatch_failures": [],
                    "last_round_workers": last_workers,
                    "last_round_samples": last_samples,
                    "seed": seed,
                }
                return CoordinatorTransitionDecision(
                    kind="dgac_diloco_complete",
                    reason="Terminal DGAC DiLoCo stage is complete.",
                    should_write_state=True,
                    state=terminal_state,
                    hub_message="DGAC DiLoCo complete: final anchor uploaded",
                    metadata={"stage_complete": True, "final_stage_samples": final_stage_samples},
                )

            terminal_state = {
                **dict(state),
                "stage_k": terminal_stage,
                "round_n": 0,
                "mode": "terminal",
                "triggered_workers": [],
                "attendance_workers": [],
                "projected_shards": {},
                "anchor_path": anchor_path,
                "total_samples_seen": totals,
                "completed_stages": completed,
                "dgac_manual_gate": True,
                "last_updated": now,
                "triggered_at": 0.0,
                "dispatch_failures": [],
                "last_round_workers": last_workers,
                "last_round_samples": last_samples,
                "seed": seed,
            }
            return CoordinatorTransitionDecision(
                kind="terminal_manual_gate",
                reason="Terminal DiLoCo stage completed; enter DGAC manual gate.",
                should_write_state=True,
                state=terminal_state,
                hub_message="DiLoCo terminal gate: stage 10 complete, DGAC manual",
                metadata={"stage_complete": True, "final_stage_samples": final_stage_samples},
            )

        next_stage_k = stage_k + 1
        next_round_n = 0
        projected_shards_next = _compute_projected_shards(
            total_samples=int(total_train_samples),
            stage_k=next_stage_k,
            round_n=0,
            seed=seed,
            total_samples_seen=0,
        )
        eligible_for_training = [worker for worker in credentialed_workers if worker not in next_attendance_workers]
        next_mode, next_active_workers = _determine_round_mode(
            projected_shards=projected_shards_next,
            credentialed_workers=eligible_for_training,
            min_shard_samples=int(min_shard_samples),
            force_worker_ids=force_worker_ids,
        )

    if not stage_complete and not next_active_workers and next_attendance_workers:
        next_mode = "waiting"
        next_round_n = round_n
        next_stage_k = stage_k

    new_state = {
        "stage_k": next_stage_k,
        "round_n": next_round_n,
        "mode": next_mode,
        "triggered_workers": next_active_workers,
        "attendance_workers": next_attendance_workers,
        "projected_shards": projected_shards_next,
        "anchor_path": anchor_path,
        "total_samples_seen": totals,
        "completed_stages": completed,
        "dgac_manual_gate": False if bool(state.get("dgac_diloco")) else bool(state.get("dgac_manual_gate", False)),
        "dgac_diloco": bool(state.get("dgac_diloco", False)),
        "pre_dgac_total_samples_seen": state.get("pre_dgac_total_samples_seen", {}),
        "last_updated": now,
        "triggered_at": now if (next_active_workers or next_attendance_workers) else 0.0,
        "last_round_workers": last_workers,
        "last_round_samples": last_samples,
        "seed": seed,
    }
    kind = "stage_advance" if stage_complete else "next_round"
    if not stage_complete and next_mode == "waiting":
        kind = "all_absent_waiting"
    return CoordinatorTransitionDecision(
        kind=kind,
        reason="Post-aggregation next coordinator state planned.",
        should_write_state=True,
        state=new_state,
        hub_message=f"Round state: stage={next_stage_k} round={next_round_n} mode={next_mode}",
        dispatch_active_workers=next_active_workers,
        dispatch_attendance_workers=next_attendance_workers,
        reconcile_active_workers=next_active_workers,
        reconcile_attendance_workers=next_attendance_workers,
        dispatch_reconcile_message=f"Dispatch reconcile: stage={next_stage_k} round={next_round_n} mode={{mode}}",
        should_stop=False,
        metadata={
            "stage_complete": stage_complete,
            "final_stage_samples": final_stage_samples,
            "newly_demoted_workers": newly_demoted,
            "still_attending_workers": still_attending,
        },
    )


def plan_dispatch_reconciliation(
    *,
    state: dict[str, Any],
    planned_active_workers: list[str],
    planned_attendance_workers: list[str],
    dispatch_results: dict[str, str],
) -> DispatchReconciliationPlan:
    """Own the dispatch-failure reconciliation seam from the coordinator decision layer."""
    return DispatchReconciliationPlan(
        corrected_state=_reconcile_post_dispatch_state(
            state=state,
            planned_active_workers=planned_active_workers,
            planned_attendance_workers=planned_attendance_workers,
            dispatch_results=dispatch_results,
        )
    )


def mode_for_workers(workers: list[str], fallback: str = "complete") -> str:
    return _mode_from_active_workers(_ordered_workers(workers), fallback=fallback)
