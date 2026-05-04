from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ouroboros.workflow_validation import (
    CPU_SMOKE_MODE,
    build_cpu_smoke_validation_command,
    is_cpu_smoke_validation_enabled,
    run_cpu_smoke_validation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "kaggle-utils.ipynb"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def test_workflow_validation_module_import_is_bootstrap_safe_and_does_not_import_torch():
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ouroboros.workflow_validation; raise SystemExit('torch' in sys.modules)",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[:1000]


def test_cpu_smoke_validation_writes_status_and_report_without_gpu_or_hub(tmp_path):
    env = {
        "OUROBOROS_WORKFLOW_VALIDATE": CPU_SMOKE_MODE,
        "DILOCO_WORKER_ID": " b ",
        "OUROBOROS_REPO_URL": "https://example.invalid/Ouroboros.git",
        "OUROBOROS_REPO_REF": "feature/cpu-smoke",
        "OUROBOROS_REPO_COMMIT": "abc123",
        "OUROBOROS_VALIDATION_STAGE_K": "8",
        "OUROBOROS_VALIDATION_ROUND_N": "0",
        "HF_TOKEN": "hf_fake",
        "WANDB_KEY": "wandb_fake",
        "GH_TOKEN": "gh_fake",
    }
    lines = []

    report = run_cpu_smoke_validation(env, output_dir=tmp_path, emit=lines.append)

    assert is_cpu_smoke_validation_enabled(env) is True
    assert report.mode == CPU_SMOKE_MODE
    assert report.worker_id == "B"
    assert report.repo_ref == "feature/cpu-smoke"
    assert report.repo_commit == "abc123"
    assert report.gpu_requested is False
    assert report.torchrun_requested is False
    assert report.push_to_hub_requested is False
    assert "--use_4bit" not in report.command
    assert "--push_to_hub" not in report.command
    assert report.command[:3] == [sys.executable, "-m", "ouroboros.workflow_validation_worker"]

    status = json.loads(Path(report.status_path).read_text(encoding="utf-8"))
    assert status == {
        "worker_id": "B",
        "stage_k": 8,
        "round_n": 0,
        "samples_seen": 0,
        "status": "done",
        "timestamp": status["timestamp"],
        "weights_path": "local_validation/workers/B/round_0000_stage_8",
        "validation_mode": CPU_SMOKE_MODE,
    }
    persisted_report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert persisted_report["secret_presence"] == {
        "HF_TOKEN": True,
        "WANDB_KEY": True,
        "GITHUB_TOKEN": True,
        "DILOCO_WORKER_ID": True,
    }
    assert any("CPU smoke validation complete" in line for line in lines)


def test_cpu_smoke_validation_worker_command_is_executable_without_training_dependencies(tmp_path):
    status_path = tmp_path / "worker_status.json"
    command = build_cpu_smoke_validation_command(
        worker_id="A",
        stage_k=3,
        round_n=2,
        status_path=status_path,
    )
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT))

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[:1000]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["worker_id"] == "A"
    assert status["stage_k"] == 3
    assert status["round_n"] == 2
    assert status["samples_seen"] == 0
    assert status["validation_mode"] == CPU_SMOKE_MODE


def test_kaggle_notebook_validation_branch_exits_before_real_torchrun():
    source = _notebook_source()

    assert "OUROBOROS_WORKFLOW_VALIDATE" in source
    assert "run_cpu_smoke_validation" in source
    assert "raise SystemExit(0)" in source
    assert source.index("run_cpu_smoke_validation") < source.index("!torchrun --standalone")
    assert source.index("raise SystemExit(0)") < source.index("!torchrun --standalone")
