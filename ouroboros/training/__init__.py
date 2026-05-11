"""Deep training-session modules.

Prefer importing training execution, checkpointing, and evaluation seams from
this package. ``ouroboros.train`` remains a compatibility facade.
"""

from ouroboros.training.checkpointing import (
    _cleanup_distributed_resume_artifacts,
    _distributed_resume_marker,
    _resolve_resume_checkpoint_for_all_ranks,
    find_latest_resume_checkpoint,
    load_checkpoint,
    prune_epoch_checkpoints,
    save_checkpoint,
    startup_hub_sync_and_prune,
)
from ouroboros.training.evaluation import (
    GEN_PROMPTS,
    evaluate_stage,
    run_dgac_diagnostics,
    run_eval_only,
    run_generation_callback,
)
from ouroboros.training.stage_runner import (
    build_optimizer_and_scheduler,
    make_timeout_checker,
    run_training_stages,
)
from ouroboros.training.session import run_training_session

__all__ = [
    "GEN_PROMPTS",
    "build_optimizer_and_scheduler",
    "evaluate_stage",
    "find_latest_resume_checkpoint",
    "load_checkpoint",
    "make_timeout_checker",
    "prune_epoch_checkpoints",
    "run_dgac_diagnostics",
    "run_eval_only",
    "run_generation_callback",
    "run_training_session",
    "run_training_stages",
    "save_checkpoint",
    "startup_hub_sync_and_prune",
]
