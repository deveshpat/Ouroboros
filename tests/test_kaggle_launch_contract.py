from __future__ import annotations

import os

from ouroboros.kaggle import (
    DGAC_ANCHOR_EVAL_RUN_MODE,
    DGAC_CANARY_RUN_MODE,
    DGAC_DILOCO_RUN_MODE,
    DGAC_TRAIN_RUN_MODE,
    DILOCO_RUN_MODE,
    build_dgac_anchor_eval_command,
    build_dgac_canary_command,
    build_dgac_training_command,
    build_diloco_training_command,
    format_shell_command,
    kaggle_secret_presence,
    resolve_diloco_worker_id,
    resolve_kaggle_run_mode,
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
    assert "--latent_cache" in command
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
        latent_cache=False,
    )

    assert "--nproc_per_node=1" in command
    assert command[command.index("--max_stage") + 1] == "2"
    assert command[command.index("--batch_size") + 1] == "1"
    assert command[command.index("--grad_accum") + 1] == "2"
    assert command[command.index("--output_dir") + 1] == "runs/smoke"
    assert command[command.index("--wandb_mode") + 1] == "offline"
    assert "--latent_cache" not in command




def test_build_diloco_training_command_can_launch_dgac_diloco_worker():
    command = build_diloco_training_command(
        worker_id="c",
        epochs_per_stage=1,
        output_dir="runs/dgac_dedicated",
        use_halt_gate=True,
        resume_from_diloco_anchor=True,
        max_grad_norm=0.3,
    )

    assert "--diloco_mode" in command
    assert "--use_halt_gate" in command
    assert "--resume_from_diloco_anchor" in command
    assert "--latent_cache" in command
    assert command[command.index("--diloco_worker_id") + 1] == "C"
    assert command[command.index("--epochs_per_stage") + 1] == "1"
    assert command[command.index("--max_grad_norm") + 1] == "0.3"
    assert "--diloco_run_val" not in command
    assert "--gen_every_stage" not in command
    assert command[command.index("--output_dir") + 1] == "runs/dgac_dedicated"


def test_build_diloco_training_command_can_enable_dgac_diloco_eval_hooks():
    command = build_diloco_training_command(
        worker_id="a",
        epochs_per_stage=1,
        output_dir="runs/dgac_dedicated",
        use_halt_gate=True,
        resume_from_diloco_anchor=True,
        diloco_run_val=True,
        gen_every_stage=True,
    )

    assert "--diloco_run_val" in command
    assert "--gen_every_stage" in command
    assert "--no-gen_every_stage" not in command


def test_build_diloco_training_command_rejects_multi_epoch_dgac_diloco_worker():
    try:
        build_diloco_training_command(
            worker_id="A",
            epochs_per_stage=3,
            output_dir="runs/dgac_dedicated",
            use_halt_gate=True,
            resume_from_diloco_anchor=True,
        )
    except ValueError as exc:
        assert "DGAC DiLoCo runs exactly one local epoch" in str(exc)
    else:  # pragma: no cover - failure branch kept explicit
        raise AssertionError("expected ValueError for multi-epoch DGAC DiLoCo")

def test_resolve_kaggle_run_mode_defaults_to_diloco_and_accepts_dgac_modes():
    assert resolve_kaggle_run_mode({}) == DILOCO_RUN_MODE
    assert resolve_kaggle_run_mode({"OUROBOROS_KAGGLE_RUN_MODE": "dgac-anchor-eval"}) == DGAC_ANCHOR_EVAL_RUN_MODE
    assert resolve_kaggle_run_mode({"OUROBOROS_KAGGLE_RUN_MODE": "dgac-train"}) == DGAC_TRAIN_RUN_MODE
    assert resolve_kaggle_run_mode({"OUROBOROS_KAGGLE_RUN_MODE": "dgac-canary"}) == DGAC_CANARY_RUN_MODE
    assert resolve_kaggle_run_mode({"OUROBOROS_KAGGLE_RUN_MODE": "dgac-diloco"}) == DGAC_DILOCO_RUN_MODE
    assert resolve_kaggle_run_mode({"OUROBOROS_RUN_MODE": "DILOCO"}) == DILOCO_RUN_MODE

    try:
        resolve_kaggle_run_mode({"OUROBOROS_KAGGLE_RUN_MODE": "unknown"})
    except ValueError as exc:
        assert "Invalid Kaggle run mode" in str(exc)
    else:  # pragma: no cover - branch exists to make failures explicit
        raise AssertionError("expected ValueError for invalid run mode")


def test_build_dgac_training_command_loads_anchor_trains_halt_gate_and_pushes_checkpoints():
    command = build_dgac_training_command()

    assert command[:4] == ["torchrun", "--standalone", "--nproc_per_node=2", "jamba_coconut_finetune.py"]
    assert "--use_halt_gate" in command
    assert "--resume_from_diloco_anchor" in command
    assert "--eval_only" not in command
    assert "--diloco_mode" not in command
    assert "--diloco_state_repo" in command and "WeirdRunner/Ouroboros" in command
    assert "--data_dir" in command and "data/coconut_v1" in command
    assert "--use_4bit" in command
    assert "--latent_cache" in command
    assert "--epochs_per_stage" in command and command[command.index("--epochs_per_stage") + 1] == "3"
    assert "--max_stage" in command and command[command.index("--max_stage") + 1] == "10"
    assert "--max_grad_norm" in command and command[command.index("--max_grad_norm") + 1] == "0.3"
    assert "--push_to_hub" in command
    assert "--output_dir" in command and command[command.index("--output_dir") + 1] == "runs/stage3_dgac"
    assert "--hf_stage_subdir" in command and command[command.index("--hf_stage_subdir") + 1] == "runs/stage3_dgac"
    assert "--wandb_project" in command and "ouroboros-stage3-jamba" in command


def test_build_dgac_training_command_allows_safe_overrides():
    command = build_dgac_training_command(
        nproc_per_node=1,
        use_4bit=False,
        diloco_state_repo="state/repo",
        output_dir="runs/dgac-smoke",
        hf_stage_subdir="runs/custom-dgac",
        push_to_hub=False,
        wandb_mode="offline",
        latent_cache=False,
    )

    assert "--nproc_per_node=1" in command
    assert "--use_4bit" not in command
    assert command[command.index("--diloco_state_repo") + 1] == "state/repo"
    assert command[command.index("--output_dir") + 1] == "runs/dgac-smoke"
    assert command[command.index("--hf_stage_subdir") + 1] == "runs/custom-dgac"
    assert "--push_to_hub" not in command
    assert command[command.index("--wandb_mode") + 1] == "offline"
    assert "--latent_cache" not in command



def test_build_dgac_canary_command_is_bounded_and_does_not_push_checkpoints():
    command = build_dgac_canary_command()

    assert command[:4] == ["torchrun", "--standalone", "--nproc_per_node=2", "jamba_coconut_finetune.py"]
    assert "--use_halt_gate" in command
    assert "--resume_from_diloco_anchor" in command
    assert "--latent_cache" in command
    assert "--eval_only" not in command
    assert "--diloco_mode" not in command
    assert command[command.index("--epochs_per_stage") + 1] == "1"
    assert command[command.index("--max_samples") + 1] == "512"
    assert command[command.index("--max_train_steps") + 1] == "20"
    assert command[command.index("--log_every") + 1] == "1"
    assert "--no-gen_every_stage" in command
    assert "--push_to_hub" not in command
    assert command[command.index("--output_dir") + 1] == "runs/stage3_dgac_canary"


def test_build_dgac_anchor_eval_command_loads_anchor_and_skips_training():
    command = build_dgac_anchor_eval_command()

    assert command[:4] == ["torchrun", "--standalone", "--nproc_per_node=2", "jamba_coconut_finetune.py"]
    assert "--use_halt_gate" in command
    assert "--resume_from_diloco_anchor" in command
    assert "--eval_only" in command
    assert "--dgac_diagnostics" in command
    assert "--diloco_state_repo" in command and "WeirdRunner/Ouroboros" in command
    assert "--data_dir" in command and "data/coconut_v1" in command
    assert "--use_4bit" in command
    assert "--max_stage" in command and command[command.index("--max_stage") + 1] == "10"
    assert "--max_grad_norm" in command and command[command.index("--max_grad_norm") + 1] == "0.3"
    assert "--val_batch_size" in command and command[command.index("--val_batch_size") + 1] == "1"
    assert "--dgac_diagnostics_batch_size" in command
    assert command[command.index("--dgac_diagnostics_batch_size") + 1] == "1"
    assert "--output_dir" in command and "runs/dgac_anchor_eval" in command
    assert "--diloco_mode" not in command
    assert "--push_to_hub" not in command


def test_build_dgac_anchor_eval_command_allows_safe_overrides():
    command = build_dgac_anchor_eval_command(
        nproc_per_node=1,
        use_4bit=False,
        diloco_state_repo="state/repo",
        output_dir="runs/eval",
        dgac_diagnostics_batch_size=3,
        dgac_diagnostics_only=True,
        dgac_diagnostics_forced_kmax_ce=0.4112,
        wandb_mode="offline",
        latent_cache=False,
    )

    assert "--nproc_per_node=1" in command
    assert "--use_4bit" not in command
    assert command[command.index("--diloco_state_repo") + 1] == "state/repo"
    assert command[command.index("--output_dir") + 1] == "runs/eval"
    assert command[command.index("--dgac_diagnostics_batch_size") + 1] == "3"
    assert "--dgac_diagnostics_only" in command
    assert command[command.index("--dgac_diagnostics_forced_kmax_ce") + 1] == "0.4112"
    assert command[command.index("--wandb_mode") + 1] == "offline"
    assert "--latent_cache" not in command

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
