from __future__ import annotations

import json
from pathlib import Path

from ouroboros.coordinator.kaggle_commands import (
    build_lm_eval_benchmark_command,
    build_lm_eval_benchmark_multi_gpu_command,
    build_dgac_anchor_eval_command,
    build_dgac_canary_command,
    build_dgac_training_command,
    build_diloco_training_command,
)
from ouroboros.coordinator.kaggle_contract import (
    BENCHMARK_RUN_MODE,
    DGAC_ANCHOR_EVAL_RUN_MODE,
    DGAC_CANARY_RUN_MODE,
    DGAC_DILOCO_RUN_MODE,
    DGAC_TRAIN_RUN_MODE,
    DILOCO_RUN_MODE,
    get_kaggle_launch_contract,
    known_kaggle_launch_modes,
)
from ouroboros.coordinator.kaggle_launch_matrix import (
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
    specs = {spec.mode: spec for spec in known_launch_specs()}

    assert set(specs) == set(known_kaggle_launch_modes())
    for mode, spec in specs.items():
        assert spec.contract == get_kaggle_launch_contract(mode)
        assert spec.workflow_label
        assert spec.mode == mode
