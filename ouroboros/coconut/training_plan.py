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


@dataclass(frozen=True)
class TrainingSessionPlan:
    kind: TrainingPlanKind
    should_train: bool
    should_validate: bool
    should_resume_checkpoint: bool = False
    resume_source: Optional[str] = None
    reason: str = ""


def _truthy_attr(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def plan_training_session(args: Any) -> TrainingSessionPlan:
    """Classify the requested run before heavy model/dataset execution."""

    use_halt_gate = _truthy_attr(args, "use_halt_gate")
    resume_from_anchor = _truthy_attr(args, "resume_from_anchor")
    eval_only = _truthy_attr(args, "eval_only")
    resume_from = normalize_text(getattr(args, "resume_from", None))
    max_train_steps = getattr(args, "max_train_steps", None)

    if resume_from_anchor and not use_halt_gate:
        raise ValueError("resume_from_anchor requires use_halt_gate")

    if eval_only:
        is_dgac = use_halt_gate and resume_from_anchor
        return TrainingSessionPlan(
            kind=TrainingPlanKind.DGAC_EVAL_ONLY if is_dgac else TrainingPlanKind.EVAL_ONLY,
            should_train=False,
            should_validate=True,
            reason="eval-only CLI branch",
        )

    if use_halt_gate and resume_from_anchor:
        is_canary = max_train_steps is not None and int(max_train_steps) > 0
        return TrainingSessionPlan(
            kind=TrainingPlanKind.DGAC_CANARY if is_canary else TrainingPlanKind.DGAC_TRAIN,
            should_train=True,
            should_validate=True,
            reason="DGAC training from Hub anchor",
        )

    if resume_from:
        return TrainingSessionPlan(
            kind=TrainingPlanKind.RESUME_TRAIN,
            should_train=True,
            should_validate=True,
            should_resume_checkpoint=True,
            resume_source=resume_from,
            reason="checkpoint resume",
        )

    return TrainingSessionPlan(
        kind=TrainingPlanKind.STANDARD_TRAIN,
        should_train=True,
        should_validate=True,
        reason="standard sequential curriculum training",
    )
