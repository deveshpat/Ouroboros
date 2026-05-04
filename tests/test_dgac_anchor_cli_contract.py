from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_MONOLITH = REPO_ROOT / "jamba_coconut_finetune.py"
MODULAR_TRAIN = REPO_ROOT / "ouroboros" / "train.py"
BLUEPRINT = REPO_ROOT / "BLUEPRINT.md"
CLI_MODULE = REPO_ROOT / "ouroboros" / "cli.py"


def test_dgac_anchor_flag_is_exposed_by_bootstrap_free_help():
    completed = subprocess.run(
        [sys.executable, str(TRAINING_MONOLITH), "--help"],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[:1000]
    assert "--resume_from_diloco_anchor" in completed.stdout


def test_dgac_anchor_launch_contract_matches_root_modular_and_cli_entrypoints():
    blueprint_source = BLUEPRINT.read_text(encoding="utf-8")
    monolith_source = TRAINING_MONOLITH.read_text(encoding="utf-8")
    modular_source = MODULAR_TRAIN.read_text(encoding="utf-8")
    cli_source = CLI_MODULE.read_text(encoding="utf-8")

    assert "--use_halt_gate --resume_from_diloco_anchor" in blueprint_source
    assert '"--resume_from_diloco_anchor",' in cli_source

    assert "from ouroboros.train import run_cli" in monolith_source
    assert "from ouroboros.cli import parse_args" in monolith_source

    assert "--resume_from_diloco_anchor" in modular_source
    assert "diloco_download_anchor" in modular_source
    assert "Loading DiLoCo anchor" in modular_source
    assert "requires --use_halt_gate" in modular_source
    assert "requires an HF token" in modular_source
