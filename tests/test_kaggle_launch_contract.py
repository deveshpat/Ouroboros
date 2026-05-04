from __future__ import annotations

import os

from ouroboros.kaggle import (
    build_diloco_training_command,
    format_shell_command,
    kaggle_secret_presence,
    resolve_diloco_worker_id,
)


def test_resolve_diloco_worker_id_accepts_env_aliases_and_normalizes():
    assert resolve_diloco_worker_id({"WORKER_ID": " b "}) == "B"
    assert resolve_diloco_worker_id({"OUROBOROS_DILOCO_WORKER_ID": "c"}) == "C"
    assert resolve_diloco_worker_id({"DILOCO_WORKER_ID": "A"}) == "A"


def test_resolve_diloco_worker_id_rejects_missing_or_invalid_values():
    for env in ({}, {"WORKER_ID": ""}, {"WORKER_ID": "D"}):
        try:
            resolve_diloco_worker_id(env)
        except ValueError as exc:
            assert "DiLoCo worker id" in str(exc)
        else:  # pragma: no cover - branch exists to make failures explicit
            raise AssertionError(f"expected ValueError for {env!r}")


def test_build_diloco_training_command_preserves_notebook_launch_contract():
    command = build_diloco_training_command(worker_id="b")

    assert command[:4] == ["torchrun", "--standalone", "--nproc_per_node=2", "jamba_coconut_finetune.py"]
    assert "--data_dir" in command and "data/coconut_v1" in command
    assert "--use_4bit" in command
    assert "--stage_0_epochs" in command and "1" in command
    assert "--epochs_per_stage" in command and "1" in command
    assert "--max_stage" in command and "10" in command
    assert "--batch_size" in command and "4" in command
    assert "--grad_accum" in command and "8" in command
    assert "--val_batch_size" in command and "2" in command
    assert "--val_skip_buffer_minutes" in command and "60" in command
    assert "--session_timeout_hours" in command and "12.0" in command
    assert "--graceful_exit_buffer_minutes" in command and "20" in command
    assert "--diloco_mode" in command
    assert "--diloco_worker_id" in command and "B" in command
    assert "--diloco_outer_lr" in command and "0.7" in command
    assert "--diloco_state_repo" in command and "WeirdRunner/Ouroboros" in command
    assert "--diloco_signal_repo" in command and "deveshpat/Ouroboros" in command
    assert "--push_to_hub" in command
    assert "--output_dir" in command and "runs/diloco" in command


def test_build_diloco_training_command_allows_safe_overrides():
    command = build_diloco_training_command(
        worker_id="A",
        nproc_per_node=1,
        max_stage=2,
        batch_size=1,
        grad_accum=2,
        output_dir="runs/smoke",
        wandb_mode="offline",
    )

    assert "--nproc_per_node=1" in command
    assert command[command.index("--max_stage") + 1] == "2"
    assert command[command.index("--batch_size") + 1] == "1"
    assert command[command.index("--grad_accum") + 1] == "2"
    assert command[command.index("--output_dir") + 1] == "runs/smoke"
    assert command[command.index("--wandb_mode") + 1] == "offline"


def test_format_shell_command_quotes_arguments_for_notebook_logging():
    command = ["python", "jamba_coconut_finetune.py", "--output_dir", "runs/with spaces"]
    assert format_shell_command(command) == "python jamba_coconut_finetune.py --output_dir 'runs/with spaces'"


def test_kaggle_secret_presence_reports_booleans_without_values(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_123")
    monkeypatch.setenv("WANDB_KEY", "wandb_123")
    monkeypatch.setenv("GH_TOKEN", "gh_123")
    monkeypatch.setenv("WORKER_ID", "A")

    assert kaggle_secret_presence(os.environ) == {
        "HF_TOKEN": True,
        "WANDB_KEY": True,
        "GITHUB_TOKEN": True,
        "DILOCO_WORKER_ID": True,
    }
