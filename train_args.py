"""
train_args.py
============
OuroborosTrainingArguments: TrainingArguments + the few project-specific fields
the callbacks and trainer read (Kaggle session budget, current stage, halt-gate
flag). Kept in its own file because TrainingArguments subclassing is a distinct,
small concern and pulling transformers into train.py's top would break the
torch-free --help path.
"""

from __future__ import annotations

import time
from typing import Optional

from transformers import TrainingArguments


class OuroborosTrainingArguments(TrainingArguments):
    """
    Standard TrainingArguments plus:

    - session_timeout_hours / graceful_exit_buffer_minutes: the SessionTimeoutCallback
      trips (save + stop) when elapsed + buffer >= timeout — load-bearing on Kaggle,
      which kills the process at 12h.
    - val_skip_buffer_minutes: the ValBudgetGuardCallback skips an eval that can't
      finish before the kill (val takes ~37min on Dual T4).
    - session_start: perf_counter baseline the callbacks share across stages.
    - stage_k / use_halt_gate: per-stage context for the sidecar + DGAC ramp.
    """

    # Kaggle session budget (hours/minutes).
    session_timeout_hours: float = 11.0
    graceful_exit_buffer_minutes: float = 20.0
    val_skip_buffer_minutes: float = 60.0
    # Shared wall-clock baseline; set by the session driver per stage.
    session_start: float = 0.0
    # Per-stage context.
    stage_k: int = 0
    use_halt_gate: bool = False

    def __init__(self, *args, **kwargs):
        # Pop our fields before forwarding to TrainingArguments (it rejects unknown
        # kwargs in strict versions), then set them as attributes.
        own = {}
        for name in ("session_timeout_hours", "graceful_exit_buffer_minutes",
                     "val_skip_buffer_minutes", "session_start", "stage_k", "use_halt_gate"):
            if name in kwargs:
                own[name] = kwargs.pop(name)
        super().__init__(*args, **kwargs)
        for name, value in own.items():
            setattr(self, name, value)
        if not self.session_start:
            self.session_start = time.perf_counter()
