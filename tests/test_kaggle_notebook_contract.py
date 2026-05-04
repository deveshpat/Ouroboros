from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "kaggle-utils.ipynb"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def test_kaggle_notebook_uses_packaged_launch_helper_instead_of_inline_training_command():
    source = _notebook_source()

    assert "from ouroboros.kaggle import" in source
    assert "build_diloco_training_command" in source
    assert "subprocess.run(command, check=True)" in source
    assert "!torchrun --standalone" not in source


def test_kaggle_notebook_describes_itself_as_thin_adapter():
    source = _notebook_source()

    assert "thin adapter" in source
    assert "Reusable training, checkpoint, DGAC, DiLoCo worker" in source
