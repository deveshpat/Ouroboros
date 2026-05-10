from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "kaggle-utils.ipynb"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def test_kaggle_notebook_keeps_torchrun_shell_magic_not_python_subprocess():
    source = _notebook_source()

    assert "from ouroboros.kaggle import" in source
    assert "resolve_diloco_worker_id" in source
    assert "resolve_kaggle_run_mode" in source
    assert "!torchrun --standalone" in source
    assert "subprocess.run(command, check=True)" not in source
    assert "import subprocess" not in source.split("!torchrun --standalone", 1)[0].split("from ouroboros.kaggle import", 1)[-1]


def test_kaggle_notebook_describes_itself_as_thin_adapter():
    source = _notebook_source()

    assert "thin adapter" in source
    assert "Reusable training, checkpoint, DGAC, DiLoCo worker" in source


def test_kaggle_notebook_supports_dgac_anchor_eval_mode_without_diloco_training():
    source = _notebook_source()

    assert "OUROBOROS_KAGGLE_RUN_MODE" in source
    assert "DGAC_ANCHOR_EVAL_RUN_MODE" in source
    assert "build_dgac_anchor_eval_command" in source
    assert "--resume_from_diloco_anchor" in source
    assert "--eval_only" in source
    assert "--use_halt_gate" in source
    assert "runs/dgac_anchor_eval" in source
    eval_branch = source.split("elif run_mode == DGAC_ANCHOR_EVAL_RUN_MODE:", 1)[1].split("elif run_mode == DILOCO_RUN_MODE:", 1)[0]
    assert "--diloco_mode" not in eval_branch
    assert "--push_to_hub" not in eval_branch


def test_kaggle_notebook_supports_dgac_training_mode_without_diloco_or_eval_only():
    source = _notebook_source()

    assert "DGAC_TRAIN_RUN_MODE" in source
    assert "build_dgac_training_command" in source
    assert "runs/stage3_dgac" in source
    train_branch = source.split("if run_mode == DGAC_TRAIN_RUN_MODE:", 1)[1].split("elif run_mode == DGAC_ANCHOR_EVAL_RUN_MODE:", 1)[0]
    assert "--use_halt_gate" in train_branch
    assert "--resume_from_diloco_anchor" in train_branch
    assert "--push_to_hub" in train_branch
    assert "--hf_stage_subdir" in train_branch
    assert "--eval_only" not in train_branch
    assert "--diloco_mode" not in train_branch
