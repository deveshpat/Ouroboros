from __future__ import annotations

import json
from pathlib import Path

from ouroboros.kaggle import (
    build_lm_eval_benchmark_command,
    build_dgac_anchor_eval_command,
    build_dgac_canary_command,
    build_dgac_training_command,
    build_diloco_training_command,
)
from ouroboros.kaggle_contract import (
    BENCHMARK_RUN_MODE,
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
    apply_launch_environment_defaults,
    build_launch_command,
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


def _arg_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


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
        "OUROBOROS_BENCHMARK_TASKS": "arc_easy,hellaswag,winogrande",
        "OUROBOROS_BENCHMARK_LIMIT": "100",
        "OUROBOROS_BENCHMARK_OUTPUT_DIR": "runs/lm_eval_benchmark",
        "OUROBOROS_BENCHMARK_BASE_MODEL": "ai21labs/AI21-Jamba-Reasoning-3B",
        "OUROBOROS_BENCHMARK_ADAPTER_REPO": "WeirdRunner/Ouroboros",
        "OUROBOROS_BENCHMARK_ADAPTER_SUBFOLDER": "diloco_state/anchor",
        "OUROBOROS_BENCHMARK_BATCH_SIZE": "1",
        "OUROBOROS_BENCHMARK_DEVICE": "cuda:0",
        "OUROBOROS_BENCHMARK_DTYPE": "float16",
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
    assert build_launch_command(BENCHMARK_RUN_MODE, env) == build_lm_eval_benchmark_command(
        tasks="arc_easy,hellaswag,winogrande",
        limit="100",
        output_dir="runs/lm_eval_benchmark",
        base_model="ai21labs/AI21-Jamba-Reasoning-3B",
        adapter_repo="WeirdRunner/Ouroboros",
        adapter_subfolder="diloco_state/anchor",
        batch_size="1",
        device="cuda:0",
        dtype="float16",
    )


def test_dgac_anchor_eval_uses_t4_safe_eval_and_diagnostic_microbatches():
    command = build_launch_command(DGAC_ANCHOR_EVAL_RUN_MODE, {})

    assert _arg_value(command, "--val_batch_size") == "1"
    assert _arg_value(command, "--dgac_diagnostics_batch_size") == "1"
    assert "--eval_only" in command
    assert "--dgac_diagnostics" in command


def test_dgac_anchor_eval_can_resume_at_diagnostics_only_from_env():
    command = build_launch_command(
        DGAC_ANCHOR_EVAL_RUN_MODE,
        {
            "OUROBOROS_DGAC_DIAGNOSTICS_ONLY": "1",
            "OUROBOROS_DGAC_DIAGNOSTICS_FORCED_KMAX_CE": "0.4112",
        },
    )

    assert "--dgac_diagnostics_only" in command
    assert _arg_value(command, "--dgac_diagnostics_forced_kmax_ce") == "0.4112"
    assert _arg_value(command, "--val_batch_size") == "1"
    assert _arg_value(command, "--dgac_diagnostics_batch_size") == "1"


def test_benchmark_mode_builds_harness_command_from_env():
    command = build_launch_command(
        BENCHMARK_RUN_MODE,
        {
            "OUROBOROS_BENCHMARK_TASKS": "mmlu,arc_challenge",
            "OUROBOROS_BENCHMARK_LIMIT": "50",
            "OUROBOROS_BENCHMARK_OUTPUT_DIR": "runs/custom_benchmark",
            "OUROBOROS_BENCHMARK_BASE_MODEL": "base/model",
            "OUROBOROS_BENCHMARK_ADAPTER_REPO": "adapter/repo",
            "OUROBOROS_BENCHMARK_ADAPTER_SUBFOLDER": "anchor",
            "OUROBOROS_BENCHMARK_BATCH_SIZE": "auto",
            "OUROBOROS_BENCHMARK_DEVICE": "cuda:0",
            "OUROBOROS_BENCHMARK_DTYPE": "bfloat16",
            "OUROBOROS_BENCHMARK_MODEL_ARGS": "pretrained=merged/model,trust_remote_code=True",
        },
    )

    assert command[:3] == ["python", "-m", "ouroboros.benchmark_harness"]
    assert _arg_value(command, "--tasks") == "mmlu,arc_challenge"
    assert _arg_value(command, "--limit") == "50"
    assert _arg_value(command, "--output_dir") == "runs/custom_benchmark"
    assert _arg_value(command, "--base_model") == "base/model"
    assert _arg_value(command, "--adapter_repo") == "adapter/repo"
    assert _arg_value(command, "--adapter_subfolder") == "anchor"
    assert _arg_value(command, "--batch_size") == "auto"
    assert _arg_value(command, "--dtype") == "bfloat16"
    assert _arg_value(command, "--model_args") == "pretrained=merged/model,trust_remote_code=True"
    assert "--publish_to_hub" in command


def test_benchmark_mode_treats_empty_and_full_limit_as_full_run():
    empty_limit = build_launch_command(
        BENCHMARK_RUN_MODE,
        {
            "OUROBOROS_BENCHMARK_LIMIT": "",
        },
    )
    explicit_full = build_launch_command(
        BENCHMARK_RUN_MODE,
        {
            "OUROBOROS_BENCHMARK_LIMIT": "full",
        },
    )

    assert "--limit" not in empty_limit
    assert "--limit" not in explicit_full


def test_launch_environment_defaults_are_centralized_and_non_destructive():
    env = {"OUROBOROS_WANDB_PROJECT": "custom-project"}

    apply_launch_environment_defaults(DGAC_ANCHOR_EVAL_RUN_MODE, env)

    assert env["OUROBOROS_WANDB_PROJECT"] == "custom-project"
    assert env["OUROBOROS_DILOCO_STATE_REPO"] == "WeirdRunner/Ouroboros"
    assert env["OUROBOROS_DGAC_ANCHOR_EVAL_OUTPUT_DIR"] == "runs/dgac_anchor_eval"
    apply_launch_environment_defaults(BENCHMARK_RUN_MODE, env)
    assert env["OUROBOROS_BENCHMARK_OUTPUT_DIR"] == "runs/lm_eval_benchmark"
    assert env["OUROBOROS_BENCHMARK_TASKS"] == "arc_easy,hellaswag,winogrande"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_notebook_executes_single_matrix_built_shell_command():
    source = _notebook_source()

    assert "build_launch_command(run_mode, os.environ, worker_id=worker_id)" in source
    assert "apply_launch_environment_defaults(run_mode, os.environ)" in source
    assert "!{shell_command}" in source
    assert "!torchrun --standalone" not in source
    assert "build_dgac_anchor_eval_command" not in source
    assert "build_dgac_training_command" not in source
    assert "build_diloco_training_command" not in source


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
    assert "benchmark" in workflow
    assert "benchmark_tasks" in workflow
    assert "default: ''" in workflow
    assert "Empty/full/none/0 = full benchmark" in workflow
    assert "OUROBOROS_BENCHMARK_TASKS" in workflow
    assert "dgac_anchor_eval_resume_mode" in workflow
    assert "diagnostics-only" in workflow
    assert "dgac_diagnostics_forced_kmax_ce" in workflow
    assert "OUROBOROS_DGAC_DIAGNOSTICS_ONLY" in workflow
    assert "OUROBOROS_DGAC_DIAGNOSTICS_FORCED_KMAX_CE" in workflow
    assert get_launch_spec("dgac-diloco").mode == DGAC_DILOCO_RUN_MODE
