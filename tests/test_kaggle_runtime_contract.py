from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from ouroboros.utils.kaggle_runtime import (
    build_authenticated_git_env,
    copy_runtime_files,
    fetch_and_checkout,
    resolve_kaggle_repo_spec,
)


class Completed:
    def __init__(self, returncode: int = 0, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


def test_resolve_kaggle_repo_spec_uses_runtime_env_over_defaults(tmp_path):
    spec = resolve_kaggle_repo_spec(
        {
            "OUROBOROS_REPO_URL": " https://example.invalid/repo.git ",
            "OUROBOROS_REPO_REF": " feature/validation ",
            "OUROBOROS_REPO_COMMIT": " abc123 ",
        },
        repo_dir=tmp_path / "repo",
        target_dir=tmp_path / "working",
    )

    assert spec.repo_url == "https://example.invalid/repo.git"
    assert spec.repo_ref == "feature/validation"
    assert spec.repo_commit == "abc123"
    assert spec.repo_dir == tmp_path / "repo"
    assert spec.target_dir == tmp_path / "working"
    assert spec.files_to_copy == ("ouroboros/",)


def test_build_authenticated_git_env_adds_github_header_without_mutating_source_env():
    env = {"GITHUB_TOKEN": "gh_secret", "OTHER": "1"}

    git_env = build_authenticated_git_env("https://github.com/deveshpat/Ouroboros.git", env)

    expected_basic = base64.b64encode(b"x-access-token:gh_secret").decode("ascii")
    assert git_env["GIT_CONFIG_COUNT"] == "1"
    assert git_env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert git_env["GIT_CONFIG_VALUE_0"] == f"AUTHORIZATION: basic {expected_basic}"
    assert "GIT_CONFIG_COUNT" not in env


def test_fetch_and_checkout_prefers_requested_ref_then_exact_commit(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    calls = []

    def fake_runner(args, **kwargs):
        calls.append((list(args), kwargs))
        if args == ["git", "rev-parse", "--short", "HEAD"]:
            return Completed(stdout="abc123\n")
        return Completed()

    head = fetch_and_checkout(
        repo_dir,
        "feature/validation",
        "deadbeef",
        {"ENV": "1"},
        runner=fake_runner,
    )

    commands = [args for args, _ in calls]
    assert head == "abc123"
    assert commands[0] == ["git", "fetch", "--depth", "1", "origin", "feature/validation"]
    assert ["git", "checkout", "--force", "--detach", "FETCH_HEAD"] in commands
    assert ["git", "checkout", "--force", "--detach", "deadbeef"] in commands
    assert commands[-1] == ["git", "rev-parse", "--short", "HEAD"]


def test_fetch_and_checkout_falls_back_to_full_fetch_when_ref_fetches_fail(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    calls = []

    def fake_runner(args, **kwargs):
        calls.append(list(args))
        if args[:4] == ["git", "fetch", "--depth", "1"]:
            return Completed(returncode=1)
        if args == ["git", "rev-parse", "--short", "HEAD"]:
            return Completed(stdout="fed456\n")
        return Completed()

    assert fetch_and_checkout(repo_dir, "missing-ref", "", {}, runner=fake_runner) == "fed456"
    assert calls[:3] == [
        ["git", "fetch", "--depth", "1", "origin", "missing-ref"],
        ["git", "fetch", "--depth", "1", "origin", "refs/heads/missing-ref"],
        ["git", "fetch", "--depth", "1", "origin", "refs/tags/missing-ref"],
    ]
    assert ["git", "fetch", "origin"] in calls


def test_copy_runtime_files_replaces_package_without_pycache(tmp_path):
    repo = tmp_path / "repo"
    target = tmp_path / "working"
    (repo / "ouroboros" / "__pycache__").mkdir(parents=True)
    (repo / "ouroboros" / "__init__.py").write_text("# package\n", encoding="utf-8")
    (repo / "ouroboros" / "__pycache__" / "stale.pyc").write_text("stale", encoding="utf-8")
    (target / "ouroboros").mkdir(parents=True)
    (target / "ouroboros" / "old.py").write_text("old\n", encoding="utf-8")

    copy_runtime_files(repo, target)

    assert (target / "ouroboros" / "__init__.py").exists()
    assert not (target / "ouroboros" / "old.py").exists()
    assert not (target / "ouroboros" / "__pycache__").exists()
