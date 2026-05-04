from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_root_diloco_coordinator_is_thin_compatibility_adapter():
    adapter = REPO_ROOT / "diloco_coordinator.py"
    source = adapter.read_text(encoding="utf-8")

    assert "from ouroboros.diloco.coordinator import main" in source
    assert "if __name__ == \"__main__\"" in source
    assert "def weighted_average_deltas" not in source
    assert "def trigger_kaggle_workers" not in source
    assert len(source.splitlines()) <= 12


def test_root_diloco_coordinator_help_stays_cli_compatible():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "diloco_coordinator.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0
    assert "--hf_token" in result.stdout
    assert "--repo_id" in result.stdout
    assert "--kaggle_username_a" in result.stdout
    assert "--kaggle_notebook_path" in result.stdout
