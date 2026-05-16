from __future__ import annotations

import json
from pathlib import Path

from ouroboros.coordinator.kaggle_contract import (
    BENCHMARK_RUN_MODE,
    DGAC_ANCHOR_EVAL_RUN_MODE,
    DGAC_CANARY_RUN_MODE,
    DGAC_DILOCO_RUN_MODE,
    DGAC_TRAIN_RUN_MODE,
    DILOCO_RUN_MODE,
)
from ouroboros.coordinator.kaggle_launch_matrix import build_launch_command

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "kaggle-utils.ipynb"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def test_kaggle_notebook_keeps_shell_magic_not_python_subprocess():
    source = _notebook_source()

    assert "from ouroboros.coordinator.kaggle_commands import" in source
    assert "resolve_diloco_worker_id" in source
    assert "resolve_kaggle_run_mode" in source
    assert "!{shell_command}" in source
    assert "subprocess.run(command, check=True)" not in source
    assert "import subprocess" not in source.split("from ouroboros.coordinator.kaggle_commands import", 1)[-1]


def test_kaggle_notebook_describes_itself_as_thin_adapter():
    source = _notebook_source()

    assert "thin adapter" in source
    assert "Reusable training, checkpoint, DGAC, DiLoCo worker" in source


def test_kaggle_notebook_delegates_all_gpu_launch_argv_to_matrix():
    source = _notebook_source()

    assert "build_launch_command(run_mode, os.environ, worker_id=worker_id)" in source
    assert "format_shell_command(command)" in source
    assert "The command argv comes from ouroboros.coordinator.kaggle_launch_matrix" in source
    assert "!torchrun --standalone" not in source

    forbidden_parallel_builders = (
        "build_dgac_anchor_eval_command",
        "build_dgac_canary_command",
        "build_dgac_training_command",
        "build_diloco_training_command",
    )
    for name in forbidden_parallel_builders:
        assert name not in source


def test_matrix_supports_expected_kaggle_gpu_modes_without_notebook_branches():
    env = {
        "DILOCO_WORKER_ID": "A",
        "OUROBOROS_DILOCO_STATE_REPO": "WeirdRunner/Ouroboros",
        "OUROBOROS_DILOCO_SIGNAL_REPO": "deveshpat/Ouroboros",
        "OUROBOROS_DILOCO_OUTER_LR": "0.7",
        "OUROBOROS_DILOCO_OUTPUT_DIR": "runs/diloco",
        "OUROBOROS_DGAC_ANCHOR_EVAL_OUTPUT_DIR": "runs/dgac_anchor_eval",
        "OUROBOROS_DGAC_OUTPUT_DIR": "runs/stage3_dgac",
        "OUROBOROS_DGAC_CANARY_OUTPUT_DIR": "runs/stage3_dgac_canary",
        "OUROBOROS_DGAC_DILOCO_OUTPUT_DIR": "runs/dgac_dedicated",
        "OUROBOROS_BENCHMARK_TASKS": "arc_easy",
        "OUROBOROS_BENCHMARK_LIMIT": "10",
        "OUROBOROS_BENCHMARK_OUTPUT_DIR": "runs/lm_eval_benchmark",
        "OUROBOROS_BENCHMARK_BASE_MODEL": "ai21labs/AI21-Jamba-Reasoning-3B",
        "OUROBOROS_BENCHMARK_ADAPTER_REPO": "WeirdRunner/Ouroboros",
        "OUROBOROS_BENCHMARK_ADAPTER_SUBFOLDER": "diloco_state/anchor",
        "OUROBOROS_BENCHMARK_BATCH_SIZE": "1",
        "OUROBOROS_BENCHMARK_DEVICE": "cuda:0",
        "OUROBOROS_BENCHMARK_DTYPE": "float16",
        "OUROBOROS_WANDB_PROJECT": "ouroboros-stage3-jamba",
    }

    commands = {
        mode: build_launch_command(mode, env)
        for mode in (
            DILOCO_RUN_MODE,
            DGAC_DILOCO_RUN_MODE,
            DGAC_CANARY_RUN_MODE,
            DGAC_TRAIN_RUN_MODE,
            DGAC_ANCHOR_EVAL_RUN_MODE,
            BENCHMARK_RUN_MODE,
        )
    }

    assert "--diloco_mode" in commands[DILOCO_RUN_MODE]
    assert "--diloco_mode" in commands[DGAC_DILOCO_RUN_MODE]
    assert "--use_halt_gate" in commands[DGAC_DILOCO_RUN_MODE]
    assert "--max_train_steps" in commands[DGAC_CANARY_RUN_MODE]
    assert "--push_to_hub" in commands[DGAC_TRAIN_RUN_MODE]
    assert "--eval_only" in commands[DGAC_ANCHOR_EVAL_RUN_MODE]
    assert "--dgac_diagnostics" in commands[DGAC_ANCHOR_EVAL_RUN_MODE]
    assert commands[BENCHMARK_RUN_MODE][:3] == ["python", "-m", "ouroboros.eval.benchmark_multi_gpu"]
    assert "--devices" not in commands[BENCHMARK_RUN_MODE]
    assert "--tasks" in commands[BENCHMARK_RUN_MODE]
    assert "--publish_to_hub" in commands[BENCHMARK_RUN_MODE]
    assert "--diloco_mode" not in commands[DGAC_ANCHOR_EVAL_RUN_MODE]
    assert "--push_to_hub" not in commands[DGAC_ANCHOR_EVAL_RUN_MODE]
