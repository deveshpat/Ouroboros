from __future__ import annotations

import argparse

import pytest

from ouroboros.diloco.runtime_env import build_worker_runtime_env


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        hf_token="hf_test",
        repo_id="WeirdRunner/Ouroboros",
        outer_lr=0.7,
        wandb_project="ouroboros-stage3-jamba",
        wandb_entity="weirdrunner",
        wandb_key="wb_test",
    )


def test_build_worker_runtime_env_injects_worker_and_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUROBOROS_CUSTOM_FLAG", "on")
    monkeypatch.setenv("GITHUB_REPOSITORY", "weirdrunner/Ouroboros")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_SHA", "abc123")

    env = build_worker_runtime_env(_args(), "b")

    assert env["DILOCO_WORKER_ID"] == "B"
    assert env["WORKER_ID"] == "B"
    assert env["OUROBOROS_AUTO_TRIGGERED"] == "1"
    assert env["HF_TOKEN"] == "hf_test"
    assert env["WANDB_API_KEY"] == "wb_test"
    assert env["OUROBOROS_CUSTOM_FLAG"] == "on"
    assert env["OUROBOROS_REPO_URL"] == "https://github.com/weirdrunner/Ouroboros.git"
    assert env["OUROBOROS_REPO_COMMIT"] == "abc123"


def test_build_worker_runtime_env_rejects_unknown_worker() -> None:
    with pytest.raises(ValueError):
        build_worker_runtime_env(_args(), "z")
