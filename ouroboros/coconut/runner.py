"""Compatibility facade for training execution.

Implementation ownership lives under ``ouroboros.coconut``. Keep importing
from this module for existing scripts; prefer ``ouroboros.coconut.*`` for new
internal code.
"""

from __future__ import annotations

import argparse

from ouroboros.coconut.checkpointing import (
    _cleanup_distributed_resume_artifacts,
    _distributed_resume_marker,
    _resolve_resume_checkpoint_for_all_ranks,
    find_latest_resume_checkpoint,
    load_checkpoint,
    prune_epoch_checkpoints,
    save_checkpoint,
    startup_hub_sync_and_prune,
)
from ouroboros.coconut.evaluation import (
    GEN_PROMPTS,
    _collect_local_halt_gate_stage_plan,
    _evaluate_ce_for_sample_stage_pairs,
    _percentile_from_histogram,
    _summarize_halt_histogram,
    evaluate_stage,
    run_dgac_diagnostics,
    run_eval_only,
    run_generation_callback,
)
from ouroboros.coconut.stage_runner import (
    _best_state_for_stage,
    _optimizer_step_sample_count,
    _stage_grad_clip_norm,
    build_optimizer_and_scheduler,
    make_timeout_checker,
    run_training_stages,
)


def run_cli(args: argparse.Namespace, *, script_start: float) -> None:
    from ouroboros.coconut.session import run_training_session

    return run_training_session(args, script_start=script_start)


__all__ = [
    "GEN_PROMPTS",
    "build_optimizer_and_scheduler",
    "evaluate_stage",
    "find_latest_resume_checkpoint",
    "load_checkpoint",
    "make_timeout_checker",
    "prune_epoch_checkpoints",
    "run_cli",
    "run_dgac_diagnostics",
    "run_eval_only",
    "run_generation_callback",
    "run_training_stages",
    "save_checkpoint",
    "startup_hub_sync_and_prune",
]
