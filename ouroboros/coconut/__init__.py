"""Coconut public interface: training, latent execution, DGAC, and checkpoints.

The package root is lazy so ``ouroboros.coconut.cli`` remains bootstrap-safe and
can be imported before torch/transformers/runtime setup.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "run_cli": ("session", "run_training_session"),
    "plan_training_session": ("training_plan", "plan_training_session"),
    "TrainingSessionPlan": ("training_plan", "TrainingSessionPlan"),
    "TrainingPlanKind": ("training_plan", "TrainingPlanKind"),
    "build_sample_at_stage": ("data", "build_sample_at_stage"),
    "collate_stage_k": ("data", "collate_stage_k"),
    "get_max_stage": ("data", "get_max_stage"),
    "load_canonical_dataset": ("data", "load_canonical_dataset"),
    "HaltGate": ("dgac", "HaltGate"),
    "coconut_forward": ("dgac", "coconut_forward"),
    "compute_dgac_lambda1": ("dgac", "compute_dgac_lambda1"),
    "normalize_pred": ("dgac", "normalize_pred"),
    "build_dgac_halt_probe_depths": ("dgac", "build_dgac_halt_probe_depths"),
    "construct_dgac_halt_targets": ("dgac", "construct_dgac_halt_targets"),
    "build_halt_supervision_labels": ("dgac", "build_halt_supervision_labels"),
    "prepare_latent_runtime": ("latent", "prepare_latent_runtime"),
    "run_latent_passes": ("latent", "run_latent_passes"),
    "forward_latent_batch": ("latent", "forward_latent_batch"),
    "decode_from_latent_context": ("latent", "decode_from_latent_context"),
    "compute_ce_from_hidden": ("latent", "compute_ce_from_hidden"),
    "compute_ce_mean_by_row": ("latent", "compute_ce_mean_by_row"),
    "compute_ce_sum_and_count": ("latent", "compute_ce_sum_and_count"),
    "build_question_context": ("latent", "build_question_context"),
    "collect_latent_hidden_sequences": ("latent", "collect_latent_hidden_sequences"),
    "save_checkpoint": ("checkpointing", "save_checkpoint"),
    "load_checkpoint": ("checkpointing", "load_checkpoint"),
    "prune_epoch_checkpoints": ("checkpointing", "prune_epoch_checkpoints"),
    "find_latest_resume_checkpoint": ("checkpointing", "find_latest_resume_checkpoint"),
    "startup_hub_sync_and_prune": ("checkpointing", "startup_hub_sync_and_prune"),
    "evaluate_stage": ("evaluation", "evaluate_stage"),
    "run_eval_only": ("evaluation", "run_eval_only"),
    "run_training_stages": ("stage_runner", "run_training_stages"),
    "make_timeout_checker": ("stage_runner", "make_timeout_checker"),
    "build_optimizer_and_scheduler": ("stage_runner", "build_optimizer_and_scheduler"),
}

_PUBLIC_EXPORTS = (
    "run_cli",
    "plan_training_session",
    "TrainingSessionPlan",
    "TrainingPlanKind",
    "HaltGate",
    "run_dgac_diagnostics",
    "run_eval_only",
    "run_training_stages",
    "save_checkpoint",
    "load_checkpoint",
)

__all__ = _PUBLIC_EXPORTS


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attr_name)
    globals()[name] = value
    return value
