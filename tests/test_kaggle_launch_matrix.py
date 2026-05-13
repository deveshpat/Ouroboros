from __future__ import annotations

import json
import shlex
from pathlib import Path

from ouroboros.kaggle import (
    build_dgac_anchor_eval_command,
    build_dgac_canary_command,
    build_dgac_training_command,
    build_diloco_training_command,
)
from ouroboros.kaggle_contract import (
    CPU_SMOKE_MODE,
    DGAC_ANCHOR_EVAL_RUN_MODE,
    DGAC_CANARY_RUN_MODE,
    DGAC_DILOCO_RUN_MODE,
    DGAC_TRAIN_RUN_MODE,
    DILOCO_RUN_MODE,
    get_kaggle_launch_contract,
    known_kaggle_launch_modes,
)
from ouroboros.kaggle_launch_matrix import (
    build_launch_command,
    expected_notebook_shell_tokens,
    get_launch_spec,
    known_launch_specs,
    requires_kaggle_gpu,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "kaggle-utils.ipynb"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "diloco_coordinator.yml"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def _shell_tokens_after(marker: str, stop_marker: str | None = None) -> tuple[str, ...]:
    source = _notebook_source()
    branch = source.split(marker, 1)[1]
    if stop_marker is not None:
        branch = branch.split(stop_marker, 1)[0]
    for line in branch.splitlines():
        stripped = line.strip()
        if stripped.startswith("!torchrun "):
            return tuple(shlex.split(stripped.removeprefix("!")))
    raise AssertionError(f"no !torchrun command found after {marker!r}")


def test_launch_matrix_covers_every_declared_mode_and_contract():
    specs = {spec.mode: spec for spec in known_launch_specs(include_cpu_smoke=True)}

    assert set(specs) == set(known_kaggle_launch_modes(include_cpu_smoke=True))
    for mode, spec in specs.items():
        assert spec.contract == get_kaggle_launch_contract(mode)
        assert spec.workflow_label
        assert spec.mode == mode


def test_launch_matrix_keeps_cpu_smoke_out_of_public_gpu_modes():
    public_modes = tuple(spec.mode for spec in known_launch_specs())

    assert CPU_SMOKE_MODE not in public_modes
    assert CPU_SMOKE_MODE in tuple(spec.mode for spec in known_launch_specs(include_cpu_smoke=True))
    assert requires_kaggle_gpu(CPU_SMOKE_MODE) is False
    for mode in public_modes:
        assert requires_kaggle_gpu(mode) is True


def test_launch_matrix_builds_same_commands_as_compatibility_builders():
    env = {
        "DILOCO_WORKER_ID": "B",
        "OUROBOROS_DILOCO_STATE_REPO": "WeirdRunner/Ouroboros",
        "OUROBOROS_DILOCO_SIGNAL_REPO": "deveshpat/Ouroboros",
        "OUROBOROS_DILOCO_OUTER_LR": "0.7",
        "OUROBOROS_DILOCO_OUTPUT_DIR": "runs/diloco",
        "OUROBOROS_DGAC_ANCHOR_EVAL_OUTPUT_DIR": "runs/dgac_anchor_eval",
        "OUROBOROS_DGAC_OUTPUT_DIR": "runs/stage3_dgac",
        "OUROBOROS_DGAC_CANARY_OUTPUT_DIR": "runs/stage3_dgac_canary",
        "OUROBOROS_DGAC_DILOCO_OUTPUT_DIR": "runs/dgac_dedicated",
        "OUROBOROS_WANDB_PROJECT": "ouroboros-stage3-jamba",
    }

    assert build_launch_command(DILOCO_RUN_MODE, env) == build_diloco_training_command(
        worker_id="B",
        diloco_state_repo="WeirdRunner/Ouroboros",
        diloco_signal_repo="deveshpat/Ouroboros",
        diloco_outer_lr=0.7,
        output_dir="runs/diloco",
    )
    assert build_launch_command(DGAC_DILOCO_RUN_MODE, env) == build_diloco_training_command(
        worker_id="B",
        diloco_state_repo="WeirdRunner/Ouroboros",
        diloco_signal_repo="deveshpat/Ouroboros",
        diloco_outer_lr=0.7,
        output_dir="runs/dgac_dedicated",
        epochs_per_stage=1,
        use_halt_gate=True,
        resume_from_diloco_anchor=True,
        max_grad_norm=0.3,
        diloco_run_val=True,
        gen_every_stage=True,
    )
    assert build_launch_command(DGAC_CANARY_RUN_MODE, env) == build_dgac_canary_command(
        diloco_state_repo="WeirdRunner/Ouroboros",
        output_dir="runs/stage3_dgac_canary",
        hf_stage_subdir="runs/stage3_dgac_canary",
        wandb_project="ouroboros-stage3-jamba",
    )
    assert build_launch_command(DGAC_TRAIN_RUN_MODE, env) == build_dgac_training_command(
        diloco_state_repo="WeirdRunner/Ouroboros",
        output_dir="runs/stage3_dgac",
        hf_stage_subdir="runs/stage3_dgac",
        wandb_project="ouroboros-stage3-jamba",
    )
    assert build_launch_command(DGAC_ANCHOR_EVAL_RUN_MODE, env) == build_dgac_anchor_eval_command(
        diloco_state_repo="WeirdRunner/Ouroboros",
        output_dir="runs/dgac_anchor_eval",
        wandb_project="ouroboros-stage3-jamba",
    )


def test_notebook_literal_torchrun_commands_match_launch_matrix():
    branch_markers = {
        DGAC_DILOCO_RUN_MODE: ("if run_mode == DGAC_DILOCO_RUN_MODE:", "elif run_mode == DGAC_CANARY_RUN_MODE:"),
        DGAC_CANARY_RUN_MODE: ("elif run_mode == DGAC_CANARY_RUN_MODE:", "elif run_mode == DGAC_TRAIN_RUN_MODE:"),
        DGAC_TRAIN_RUN_MODE: ("elif run_mode == DGAC_TRAIN_RUN_MODE:", "elif run_mode == DGAC_ANCHOR_EVAL_RUN_MODE:"),
        DGAC_ANCHOR_EVAL_RUN_MODE: ("elif run_mode == DGAC_ANCHOR_EVAL_RUN_MODE:", "elif run_mode == DILOCO_RUN_MODE:"),
        DILOCO_RUN_MODE: ("elif run_mode == DILOCO_RUN_MODE:", "else:"),
    }

    for mode, (start, stop) in branch_markers.items():
        assert _shell_tokens_after(start, stop) == expected_notebook_shell_tokens(mode)


def test_workflow_dispatch_exposes_only_valid_matrix_modes():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    public_modes = tuple(spec.mode for spec in known_launch_specs())

    assert "workflow_dispatch:" in workflow
    assert "push:" in workflow
    assert "signals/*.json" in workflow
    assert "schedule:" not in workflow
    assert "cron:" not in workflow
    for mode in public_modes:
        assert mode in workflow
    assert CPU_SMOKE_MODE in workflow.split("workflow_validate", 1)[1].split("kaggle_run_mode", 1)[0]
    assert CPU_SMOKE_MODE not in workflow.split("kaggle_run_mode", 1)[1].split("attendance_join_grace_minutes", 1)[0]
    assert "default: 'dgac-diloco'" in workflow
    assert get_launch_spec("dgac-diloco").mode == DGAC_DILOCO_RUN_MODE
