"""Pure training session planning for Ouroboros CLI runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ouroboros.utils.runtime_env import normalize_text


class TrainingPlanKind(str, Enum):
    STANDARD_TRAIN = "standard-train"
    EVAL_ONLY = "eval-only"
    RESUME_TRAIN = "resume-train"
    DGAC_TRAIN = "dgac-train"
    DGAC_EVAL_ONLY = "dgac-eval-only"
    DGAC_CANARY = "dgac-canary"
    DILOCO_WORKER = "diloco-worker"
    DGAC_DILOCO_WORKER = "dgac-diloco-worker"


@dataclass(frozen=True)
class TrainingSessionPlan:
    kind: TrainingPlanKind
    should_train: bool
    should_validate: bool
    should_generate: bool
    should_resume_checkpoint: bool = False
    resume_source: Optional[str] = None
    delegates_to_diloco: bool = False
    skip_worker_pre_validation: bool = False
    reason: str = ""


def _truthy_attr(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def plan_training_session(args: Any) -> TrainingSessionPlan:
    """Classify the requested run before heavy model/dataset execution."""

    diloco_mode = _truthy_attr(args, "diloco_mode")
    use_halt_gate = _truthy_attr(args, "use_halt_gate")
    resume_from_anchor = _truthy_attr(args, "resume_from_diloco_anchor")
    eval_only = _truthy_attr(args, "eval_only")
    resume_from = normalize_text(getattr(args, "resume_from", None))
    max_train_steps = getattr(args, "max_train_steps", None)
    gen_every_stage = _truthy_attr(args, "gen_every_stage", True)

    if resume_from_anchor and not use_halt_gate:
        raise ValueError("resume_from_diloco_anchor requires use_halt_gate")
    if diloco_mode and use_halt_gate and not resume_from_anchor:
        raise ValueError("DGAC DiLoCo worker mode requires resume_from_diloco_anchor")

    if diloco_mode:
        is_dgac = use_halt_gate and resume_from_anchor
        return TrainingSessionPlan(
            kind=TrainingPlanKind.DGAC_DILOCO_WORKER if is_dgac else TrainingPlanKind.DILOCO_WORKER,
            should_train=True,
            should_validate=bool(getattr(args, "diloco_run_val", False)),
            should_generate=gen_every_stage if is_dgac else False,
            delegates_to_diloco=True,
            skip_worker_pre_validation=False,
            reason="worker execution delegated to DiLoCo runtime",
        )

    if eval_only:
        is_dgac = use_halt_gate and resume_from_anchor
        return TrainingSessionPlan(
            kind=TrainingPlanKind.DGAC_EVAL_ONLY if is_dgac else TrainingPlanKind.EVAL_ONLY,
            should_train=False,
            should_validate=True,
            should_generate=True,
            reason="eval-only CLI branch",
        )

    if use_halt_gate and resume_from_anchor:
        is_canary = max_train_steps is not None and int(max_train_steps) > 0
        return TrainingSessionPlan(
            kind=TrainingPlanKind.DGAC_CANARY if is_canary else TrainingPlanKind.DGAC_TRAIN,
            should_train=True,
            should_validate=True,
            should_generate=gen_every_stage,
            reason="DGAC training from terminal DiLoCo anchor",
        )

    if resume_from:
        return TrainingSessionPlan(
            kind=TrainingPlanKind.RESUME_TRAIN,
            should_train=True,
            should_validate=True,
            should_generate=gen_every_stage,
            should_resume_checkpoint=True,
            resume_source=resume_from,
            reason="checkpoint resume",
        )

    return TrainingSessionPlan(
        kind=TrainingPlanKind.STANDARD_TRAIN,
        should_train=True,
        should_validate=True,
        should_generate=gen_every_stage,
        reason="standard sequential curriculum training",
    )
