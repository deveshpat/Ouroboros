from __future__ import annotations

import argparse
import base64
import json
import os
import zlib
from pathlib import Path

from ouroboros.diloco import dispatch


def _decode_payload(payload: str) -> dict[str, str]:
    return json.loads(zlib.decompress(base64.b64decode(payload)).decode("utf-8"))


def _minimal_notebook(cells):
    return {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_build_kaggle_kernel_metadata_preserves_gpu_and_internet_contract():
    metadata = dispatch._build_kaggle_kernel_metadata(
        slug="weirdrunner/kaggle-utils",
        notebook_filename="kaggle-utils.ipynb",
    )

    assert metadata["id"] == "weirdrunner/kaggle-utils"
    assert metadata["title"] == "kaggle-utils"
    assert metadata["code_file"] == "kaggle-utils.ipynb"
    assert metadata["kernel_type"] == "notebook"
    assert metadata["enable_gpu"] is True
    assert metadata["accelerator"] == "NvidiaTeslaT4"
    assert metadata["enable_internet"] is True




def test_build_kaggle_kernel_metadata_disables_gpu_for_cpu_smoke_validation():
    metadata = dispatch._build_kaggle_kernel_metadata(
        slug="weirdrunner/kaggle-utils",
        notebook_filename="kaggle-utils.ipynb",
        enable_gpu=False,
    )

    assert metadata["enable_gpu"] is False
    assert "accelerator" not in metadata


def test_kaggle_push_success_requires_explicit_success_marker():
    assert dispatch._is_successful_kaggle_push(
        0,
        "Kernel version 38 successfully pushed. Please check progress at https://www.kaggle.com/code/x/y",
        "",
    ) is True
    assert dispatch._is_successful_kaggle_push(0, "ok", "") is False
    assert dispatch._is_successful_kaggle_push(1, "Kernel version 38 successfully pushed.", "") is False
    assert dispatch._is_successful_kaggle_push(
        0,
        "Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.",
        "",
    ) is False


def test_runtime_env_payload_round_trips_and_runtime_env_uses_expected_aliases(monkeypatch):
    # Keep this contract test hermetic when run from a developer shell or CI
    # where real tokens/runtime overrides may already be exported.
    for key in (
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "WANDB_API_KEY",
        "WANDB_KEY",
        "OUROBOROS_DILOCO_STATE_REPO",
        "OUROBOROS_DILOCO_SIGNAL_REPO",
        "OUROBOROS_DILOCO_OUTER_LR",
        "OUROBOROS_WANDB_PROJECT",
        "OUROBOROS_WANDB_ENTITY",
        "OUROBOROS_DILOCO_OUTPUT_DIR",
        "OUROBOROS_KAGGLE_RUN_MODE",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("OUROBOROS_REPO_URL", "https://example.invalid/repo.git")
    monkeypatch.setenv("OUROBOROS_REPO_REF", "feature/ref")
    monkeypatch.setenv("OUROBOROS_REPO_COMMIT", "abc123")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GH_TOKEN", "gh_fake")

    args = argparse.Namespace(
        hf_token="hf_fake",
        wandb_key="wandb_fake",
        repo_id="state/repo",
        outer_lr=0.9,
        wandb_project="project",
        wandb_entity="entity",
        workflow_validate=None,
        workflow_validation_run_id=None,
        kaggle_run_mode="diloco",
    )

    env = dispatch._build_worker_runtime_env(args, " b ")
    payload = dispatch._encode_runtime_env_payload(env)

    decoded = _decode_payload(payload)
    assert decoded["DILOCO_WORKER_ID"] == "B"
    assert decoded["OUROBOROS_DILOCO_WORKER_ID"] == "B"
    assert decoded["WORKER_ID"] == "B"
    assert decoded["OUROBOROS_AUTO_TRIGGERED"] == "1"
    assert decoded["OUROBOROS_KAGGLE_RUN_MODE"] == "diloco"
    assert decoded["HF_TOKEN"] == "hf_fake"
    assert decoded["HUGGINGFACE_HUB_TOKEN"] == "hf_fake"
    assert decoded["WANDB_API_KEY"] == "wandb_fake"
    assert decoded["WANDB_KEY"] == "wandb_fake"
    assert decoded["GITHUB_TOKEN"] == "gh_fake"
    assert decoded["OUROBOROS_REPO_URL"] == "https://example.invalid/repo.git"
    assert decoded["OUROBOROS_REPO_REF"] == "feature/ref"
    assert decoded["OUROBOROS_REPO_COMMIT"] == "abc123"
    assert decoded["OUROBOROS_DILOCO_STATE_REPO"] == "state/repo"
    assert decoded["OUROBOROS_DILOCO_SIGNAL_REPO"] == "owner/repo"
    assert decoded["OUROBOROS_DILOCO_OUTER_LR"] == "0.9"
    assert decoded["OUROBOROS_WANDB_PROJECT"] == "project"
    assert decoded["OUROBOROS_WANDB_ENTITY"] == "entity"



def test_runtime_env_includes_remote_cpu_smoke_publish_contract(monkeypatch):
    monkeypatch.delenv("OUROBOROS_WORKFLOW_VALIDATE", raising=False)
    monkeypatch.delenv("OUROBOROS_WORKFLOW_VALIDATION_RUN_ID", raising=False)
    monkeypatch.delenv("OUROBOROS_WORKFLOW_VALIDATION_STATE_REPO", raising=False)

    env = dispatch._build_worker_runtime_env(
        argparse.Namespace(
            hf_token="hf_fake",
            wandb_key=None,
            repo_id="state/repo",
            outer_lr=0.7,
            wandb_project=None,
            wandb_entity=None,
            workflow_validate="cpu-smoke",
            workflow_validation_run_id="gh-789-1",
            kaggle_run_mode="diloco",
        ),
        "A",
    )

    assert env["OUROBOROS_WORKFLOW_VALIDATE"] == "cpu-smoke"
    assert env["OUROBOROS_WORKFLOW_VALIDATION_PUBLISH"] == "1"
    assert env["OUROBOROS_WORKFLOW_VALIDATION_RUN_ID"] == "gh-789-1"
    assert env["OUROBOROS_WORKFLOW_VALIDATION_STATE_REPO"] == "state/repo"
    assert env["OUROBOROS_DILOCO_STATE_REPO"] == "state/repo"

def test_runtime_env_can_select_dgac_anchor_eval_notebook_mode(monkeypatch):
    monkeypatch.delenv("OUROBOROS_KAGGLE_RUN_MODE", raising=False)

    env = dispatch._build_worker_runtime_env(
        argparse.Namespace(
            hf_token="hf_fake",
            wandb_key=None,
            repo_id="state/repo",
            outer_lr=0.7,
            wandb_project=None,
            wandb_entity=None,
            workflow_validate=None,
            workflow_validation_run_id=None,
            kaggle_run_mode="dgac-anchor-eval",
        ),
        "A",
    )

    assert env["OUROBOROS_KAGGLE_RUN_MODE"] == "dgac-anchor-eval"
    assert env["OUROBOROS_DILOCO_STATE_REPO"] == "state/repo"


def test_stage_local_kaggle_kernel_inserts_dispatch_after_initial_markdown_and_writes_metadata(tmp_path):
    notebook_path = tmp_path / "kaggle-utils.ipynb"
    notebook_path.write_text(
        json.dumps(
            _minimal_notebook(
                [
                    {"cell_type": "markdown", "metadata": {}, "source": ["# title\n"]},
                    {"cell_type": "code", "metadata": {}, "source": ["print('launch')\n"]},
                ]
            )
        ),
        encoding="utf-8",
    )
    staging = tmp_path / "stage"
    staging.mkdir()

    staged = dispatch._stage_local_kaggle_kernel(
        notebook_path,
        "weirdrunner/kaggle-utils",
        staging,
        worker_id="A",
        runtime_env={"DILOCO_WORKER_ID": "A"},
    )

    cells = json.loads(staged.read_text(encoding="utf-8"))["cells"]
    assert cells[0]["cell_type"] == "markdown"
    assert cells[1]["metadata"]["tags"] == ["diloco-dispatch"]
    assert "AUTO-GENERATED BY DILOCO COORDINATOR" in "".join(cells[1]["source"])
    metadata = json.loads((staging / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["accelerator"] == "NvidiaTeslaT4"
    assert metadata["code_file"] == "kaggle-utils.ipynb"


def test_stage_local_kaggle_kernel_replaces_existing_dispatch_cell(tmp_path):
    notebook_path = tmp_path / "kaggle-utils.ipynb"
    notebook_path.write_text(
        json.dumps(
            _minimal_notebook(
                [
                    {
                        "cell_type": "code",
                        "metadata": {"tags": ["diloco-dispatch"]},
                        "source": ["old dispatch\n"],
                    },
                    {"cell_type": "code", "metadata": {}, "source": ["print('launch')\n"]},
                ]
            )
        ),
        encoding="utf-8",
    )
    staging = tmp_path / "stage"
    staging.mkdir()

    staged = dispatch._stage_local_kaggle_kernel(
        notebook_path,
        "weirdrunner/kaggle-utils",
        staging,
        worker_id="C",
        runtime_env={"DILOCO_WORKER_ID": "C"},
    )

    cells = json.loads(staged.read_text(encoding="utf-8"))["cells"]
    assert len(cells) == 2
    assert cells[0]["metadata"]["tags"] == ["diloco-dispatch"]
    assert "old dispatch" not in "".join(cells[0]["source"])
    assert "worker C" in "".join(cells[0]["source"])


def test_trigger_single_worker_pushes_with_t4_accelerator(monkeypatch, tmp_path):
    notebook_path = tmp_path / "kaggle-utils.ipynb"
    notebook_path.write_text(json.dumps(_minimal_notebook([])), encoding="utf-8")
    seen = {}

    class Completed:
        returncode = 0
        stdout = "Kernel version 38 successfully pushed. Please check progress at https://www.kaggle.com/code/x/y"
        stderr = ""

    def fake_run(args, **kwargs):
        seen["args"] = args
        seen["env"] = kwargs["env"]
        return Completed()

    monkeypatch.setattr(dispatch.subprocess, "run", fake_run)

    assert dispatch._trigger_single_worker(
        "A",
        "weirdrunner",
        "secret",
        "weirdrunner/kaggle-utils",
        notebook_path,
        injected_env={"DILOCO_WORKER_ID": "A"},
    ) is True
    assert seen["args"][-2:] == ["--accelerator", "NvidiaTeslaT4"]
    assert seen["env"]["KAGGLE_USERNAME"] == "weirdrunner"
    assert seen["env"]["KAGGLE_KEY"] == "secret"
    assert "KAGGLE_CONFIG_DIR" not in seen["env"]


def test_trigger_single_worker_treats_quota_output_as_failed_even_with_zero_returncode(monkeypatch, tmp_path):
    notebook_path = tmp_path / "kaggle-utils.ipynb"
    notebook_path.write_text(json.dumps(_minimal_notebook([])), encoding="utf-8")

    class Completed:
        returncode = 0
        stdout = "Kernel push error: Maximum weekly GPU quota of 30.00 hours reached."
        stderr = ""

    monkeypatch.setattr(dispatch.subprocess, "run", lambda *args, **kwargs: Completed())

    assert dispatch._trigger_single_worker(
        "B",
        "weirdrunner007",
        "secret",
        "weirdrunner007/kaggle-utils",
        notebook_path,
        injected_env={"DILOCO_WORKER_ID": "B"},
    ) is False


def test_trigger_kaggle_workers_reports_manual_failed_and_success(monkeypatch, tmp_path):
    notebook_path = tmp_path / "kaggle-utils.ipynb"
    notebook_path.write_text(json.dumps(_minimal_notebook([])), encoding="utf-8")
    calls = []

    def fake_trigger(worker_id, username, key, slug, *, notebook_path, injected_env=None, validation_mode=None):
        calls.append((worker_id, username, key, slug, injected_env, validation_mode))
        return worker_id == "A"

    monkeypatch.setattr(dispatch, "_trigger_single_worker", fake_trigger)
    monkeypatch.setattr(dispatch, "_build_worker_runtime_env", lambda args, worker_id: {"WORKER_ID": worker_id})

    results = dispatch.trigger_kaggle_workers(
        {
            "A": ("weirdrunner", "key-a"),
            "B": ("weirdrunner007", "key-b"),
            "C": (None, None),
        },
        active_workers=["A", "B", "C"],
        notebook_path=notebook_path,
        coordinator_args=argparse.Namespace(),
    )

    assert results == {"A": "success", "B": "failed", "C": "manual"}
    assert [call[0] for call in calls] == ["A", "B"]
    assert calls[0][-2] == {"WORKER_ID": "A"}
    assert calls[0][-1] is None


def test_trigger_single_worker_cpu_smoke_validation_does_not_request_accelerator(monkeypatch, tmp_path):
    notebook_path = tmp_path / "kaggle-utils.ipynb"
    notebook_path.write_text(json.dumps(_minimal_notebook([])), encoding="utf-8")
    seen = {}

    class Completed:
        returncode = 0
        stdout = "Kernel version 39 successfully pushed. Please check progress at https://www.kaggle.com/code/x/y"
        stderr = ""

    def fake_run(args, **kwargs):
        seen["args"] = args
        return Completed()

    monkeypatch.setattr(dispatch.subprocess, "run", fake_run)

    assert dispatch._trigger_single_worker(
        "A",
        "weirdrunner",
        "secret",
        "weirdrunner/kaggle-utils",
        notebook_path,
        injected_env={"DILOCO_WORKER_ID": "A", "OUROBOROS_WORKFLOW_VALIDATE": "cpu-smoke"},
        validation_mode="cpu-smoke",
    ) is True
    assert "--accelerator" not in seen["args"]
    assert "NvidiaTeslaT4" not in seen["args"]


def test_trigger_kaggle_workers_forwards_cpu_smoke_validation_mode(monkeypatch, tmp_path):
    notebook_path = tmp_path / "kaggle-utils.ipynb"
    notebook_path.write_text(json.dumps(_minimal_notebook([])), encoding="utf-8")
    calls = []

    def fake_trigger(worker_id, username, key, slug, *, notebook_path, injected_env=None, validation_mode=None):
        calls.append((worker_id, injected_env, validation_mode))
        return True

    monkeypatch.setattr(dispatch, "_trigger_single_worker", fake_trigger)
    monkeypatch.setattr(
        dispatch,
        "_build_worker_runtime_env",
        lambda args, worker_id: {"WORKER_ID": worker_id, "OUROBOROS_WORKFLOW_VALIDATE": "cpu-smoke"},
    )

    results = dispatch.trigger_kaggle_workers(
        {"A": ("weirdrunner", "key-a")},
        active_workers=["A"],
        notebook_path=notebook_path,
        coordinator_args=argparse.Namespace(),
    )

    assert results == {"A": "success"}
    assert calls == [("A", {"WORKER_ID": "A", "OUROBOROS_WORKFLOW_VALIDATE": "cpu-smoke"}, "cpu-smoke")]


def test_trigger_kaggle_workers_forwards_dgac_eval_run_mode_without_cpu_validation(monkeypatch, tmp_path):
    notebook_path = tmp_path / "kaggle-utils.ipynb"
    notebook_path.write_text(json.dumps(_minimal_notebook([])), encoding="utf-8")
    calls = []

    def fake_trigger(worker_id, username, key, slug, *, notebook_path, injected_env=None, validation_mode=None):
        calls.append((worker_id, injected_env, validation_mode))
        return True

    monkeypatch.setattr(dispatch, "_trigger_single_worker", fake_trigger)
    monkeypatch.setattr(
        dispatch,
        "_build_worker_runtime_env",
        lambda args, worker_id: {"WORKER_ID": worker_id, "OUROBOROS_KAGGLE_RUN_MODE": "dgac-anchor-eval"},
    )

    results = dispatch.trigger_kaggle_workers(
        {"A": ("weirdrunner", "key-a")},
        active_workers=["A"],
        notebook_path=notebook_path,
        coordinator_args=argparse.Namespace(),
    )

    assert results == {"A": "success"}
    assert calls == [("A", {"WORKER_ID": "A", "OUROBOROS_KAGGLE_RUN_MODE": "dgac-anchor-eval"}, None)]
