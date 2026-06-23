"""
callbacks.py
============
The Kaggle-hardening callbacks for CurriculumTrainer. Three concerns, one file
because each is small and they share the same TrainerCallback base + the same
project-specific TrainingArguments fields.

These encode the runtime constraints that made the old hand-rolled loop
load-bearing on Kaggle — session timeout + emergency checkpoint, val-budget
guard (val takes ~37min on Dual T4 and can't finish before the 12h kill), and
the per-checkpoint sidecar that records the fields Trainer's trainer_state.json
doesn't (stage_k, DGAC phase step, val metrics) so cross-session resume can find
the right stage.

Every TrainerControl field is treated as version-sensitive: the callbacks set
should_save / should_training_stop / should_evaluate defensively and rely only
on the long-stable subset of the TrainerCallback contract (on_step_end /
on_epoch_end / on_save receive (args, state, control) and return control).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback


class SessionTimeoutCallback(TrainerCallback):
    """
    Trip (save + stop) when the session budget is exhausted, so a Kaggle run
    always lands a resumable checkpoint before the 12h kill. After on_step_end
    sets should_save + should_training_stop, Trainer saves (via its post-step
    save hook) then unwinds out of train() — the saved checkpoint is a normal
    Trainer checkpoint, fully resumable next session.

    The check is pure wall-clock against a shared session_start, so all DDP
    ranks trip within ~1s of each other; the buffer (default 20min) is orders
    of magnitude larger than inter-rank step skew. A defensive broadcast of the
    trip flag keeps ranks from diverging if clocks ever drift.
    """

    def on_step_end(self, args, state, control, **kwargs):
        timeout = getattr(args, "session_timeout_hours", 0)
        if not timeout:
            return control
        buffer_s = getattr(args, "graceful_exit_buffer_minutes", 0) * 60.0
        elapsed = time.perf_counter() - getattr(args, "session_start", time.perf_counter())
        if elapsed + buffer_s >= timeout * 3600.0:
            control.should_save = True
            control.should_training_stop = True
            self._broadcast_trip()
            try:
                print(f"\n  [timeout] {elapsed/3600:.2f}h elapsed; saving emergency checkpoint.")
            except Exception:
                pass
        return control

    @staticmethod
    def _broadcast_trip() -> None:
        """Defensive: keep all ranks agreeing on the trip even if wall-clocks skew."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
        if torch.distributed.get_world_size() <= 1:
            return
        flag = torch.tensor([1], device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.int32)
        torch.distributed.broadcast(flag, src=0)


class ValBudgetGuardCallback(TrainerCallback):
    """
    Skip an epoch-end eval that can't finish before the kill, saving a checkpoint
    instead. Mirrors the old val_skip_buffer guard: val on Dual T4 takes ~37min,
    so if fewer than val_skip_buffer_minutes remain, run eval = wasted work that
    gets cut off mid-way. Setting should_evaluate=False skips it;
    should_save=True + should_training_stop=True lands a clean checkpoint.
    """

    def on_epoch_end(self, args, state, control, **kwargs):
        skip_buffer = getattr(args, "val_skip_buffer_minutes", 0)
        timeout = getattr(args, "session_timeout_hours", 0)
        if not (skip_buffer and timeout):
            return control
        elapsed = time.perf_counter() - getattr(args, "session_start", time.perf_counter())
        remaining_min = (timeout * 3600.0 - elapsed) / 60.0
        if remaining_min < skip_buffer:
            control.should_evaluate = False
            control.should_save = True
            control.should_training_stop = True
            try:
                print(f"  [val-guard] {remaining_min:.0f}min remaining < {skip_buffer:.0f}min; "
                      "skipping val, saving checkpoint.")
            except Exception:
                pass
        return control


class CheckpointSidecarCallback(TrainerCallback):
    """
    Write ouroboros_stage.json into every checkpoint dir Trainer saves, recording
    the fields trainer_state.json doesn't: stage_k, DGAC phase step, use_halt_gate,
    model_id, and the latest val metrics. JSON (not torch) so cross-session resume
    can read it at startup without importing torch — faster Kaggle boot, no pickle
    risk. on_save fires after Trainer writes its own checkpoint files.
    """

    def __init__(self, args: Any) -> None:
        self.args = args

    def on_save(self, args, state, control, **kwargs):
        output_dir = Path(getattr(args, "output_dir", ""))
        # Trainer saves into checkpoint-{global_step}/ under output_dir.
        ckpt = output_dir / f"checkpoint-{state.global_step}"
        if not ckpt.exists():
            # Some versions pass the just-saved path differently; scan for the newest.
            ckpts = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")] \
                if output_dir.exists() else []
            ckpt = max(ckpts, key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1) \
                if ckpts else None
        if ckpt is None:
            return control

        best = state.best_metric
        sidecar = {
            "stage_k": int(getattr(args, "stage_k", 0)),
            "global_step": int(state.global_step),
            "use_halt_gate": bool(getattr(args, "use_halt_gate", False)),
            "model_id": getattr(self.args, "model_id", ""),
            "val_ce": None,
            "val_token_acc": float(best) if best is not None else None,
        }
        try:
            (ckpt / "ouroboros_stage.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        except Exception:
            pass
        return control
