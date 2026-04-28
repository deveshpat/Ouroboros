from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_runtime_entrypoints_stay_thin() -> None:
    root = _repo_root()
    limits = {
        root / "ouroboros" / "diloco" / "coordinator_runtime.py": 80,
        root / "ouroboros" / "coconut" / "finetune_runtime.py": 40,
    }
    for path, limit in limits.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= limit, f"{path} drifted back to {len(lines)} lines"


def test_diloco_runtime_env_payload_roundtrips(monkeypatch) -> None:
    import argparse
    import base64
    import json
    import zlib

    from ouroboros.diloco.runtime_env import build_worker_runtime_env, encode_runtime_env_payload

    monkeypatch.setenv("OUROBOROS_RUNTIME_REPO_REF", "main")
    args = argparse.Namespace(
        hub_repo_id="owner/model",
        hf_token="hf_x",
        github_token=None,
        runtime_repo_url="https://github.com/example/repo",
        runtime_repo_ref=None,
        runtime_repo_commit="abc123",
        stage=2,
        round=3,
        total_samples=90,
        total_samples_seen=30,
        outer_lr=0.7,
    )
    env = build_worker_runtime_env(args, "b")
    assert env["OUROBOROS_DILOCO_WORKER_ID"] == "B"
    assert env["OUROBOROS_HF_REPO_ID"] == "owner/model"
    payload = encode_runtime_env_payload(env)
    decoded = json.loads(zlib.decompress(base64.urlsafe_b64decode(payload)).decode("utf-8"))
    assert decoded == env


def test_coconut_training_runtime_public_seam() -> None:
    from ouroboros.coconut.training_runtime import TrainingConfig, run_training

    result = run_training(TrainingConfig(stage_k=4, output_dir="out"), {"rank": 0})
    assert result.stage_k == 4
    assert result.output_dir == "out"
    assert result.metrics["runtime_keys"] == 1.0
