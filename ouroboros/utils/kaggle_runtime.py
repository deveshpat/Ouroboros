"""Kaggle notebook runtime helpers for repo sync and checkout selection.

This module keeps the Kaggle notebook as a thin adapter while making the
runtime checkout/ref behavior testable without Kaggle, CUDA, GitHub, or shell
secrets. Only stdlib imports are allowed here because the notebook imports it
before the training dependency bootstrap runs.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Optional, Sequence

DEFAULT_REPO_URL = "https://github.com/deveshpat/Ouroboros.git"
DEFAULT_REPO_REF = "main"
DEFAULT_REPO_DIR = Path("/kaggle/working/ouroboros_repo")
DEFAULT_TARGET_DIR = Path("/kaggle/working")
DEFAULT_FILES_TO_COPY = ("ouroboros/",)

Runner = Callable[..., subprocess.CompletedProcess]
Emitter = Callable[[str], None]


@dataclass(frozen=True)
class KaggleRepoSpec:
    """Resolved repository selection for a Kaggle worker runtime."""

    repo_url: str
    repo_ref: str
    repo_commit: str
    repo_dir: Path = DEFAULT_REPO_DIR
    target_dir: Path = DEFAULT_TARGET_DIR
    files_to_copy: tuple[str, ...] = DEFAULT_FILES_TO_COPY


def _normalize_text(value: object | None, *, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def resolve_kaggle_repo_spec(
    env: Mapping[str, str] | None = None,
    *,
    default_repo_url: str = DEFAULT_REPO_URL,
    default_repo_ref: str = DEFAULT_REPO_REF,
    repo_dir: Path | str = DEFAULT_REPO_DIR,
    target_dir: Path | str = DEFAULT_TARGET_DIR,
    files_to_copy: Sequence[str] = DEFAULT_FILES_TO_COPY,
) -> KaggleRepoSpec:
    """Resolve the repo URL/ref/commit the Kaggle notebook should checkout."""
    env = os.environ if env is None else env
    return KaggleRepoSpec(
        repo_url=_normalize_text(env.get("OUROBOROS_REPO_URL"), default=default_repo_url),
        repo_ref=_normalize_text(env.get("OUROBOROS_REPO_REF"), default=default_repo_ref),
        repo_commit=_normalize_text(env.get("OUROBOROS_REPO_COMMIT"), default=""),
        repo_dir=Path(repo_dir),
        target_dir=Path(target_dir),
        files_to_copy=tuple(files_to_copy),
    )


def build_authenticated_git_env(
    repo_url: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a git env that authenticates GitHub HTTPS fetches without logging tokens."""
    source = os.environ if env is None else env
    git_env = dict(source)
    token = _normalize_text(git_env.get("GITHUB_TOKEN") or git_env.get("GH_TOKEN"))
    if token and repo_url.startswith("https://") and "github.com" in repo_url:
        basic = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
        git_env["GIT_CONFIG_COUNT"] = "1"
        git_env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
        git_env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {basic}"
    return git_env


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    runner: Runner = subprocess.run,
) -> subprocess.CompletedProcess:
    """Run a subprocess command through an injectable runner."""
    return runner(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env is not None else None,
        check=check,
        text=True,
        capture_output=True,
    )


def ensure_repo(repo_url: str, repo_dir: Path, git_env: Mapping[str, str], *, runner: Runner = subprocess.run) -> None:
    """Clone or refresh the local repo directory used by the Kaggle worker."""
    if not repo_dir.exists():
        run_command(["git", "clone", "--filter=blob:none", repo_url, str(repo_dir)], env=git_env, runner=runner)
        return
    run_command(["git", "remote", "set-url", "origin", repo_url], cwd=repo_dir, env=git_env, runner=runner)
    run_command(["git", "clean", "-fd"], cwd=repo_dir, env=git_env, runner=runner)


def fetch_and_checkout(repo_dir: Path, repo_ref: str, repo_commit: str, git_env: Mapping[str, str], *, runner: Runner = subprocess.run) -> str:
    """Fetch a ref, checkout FETCH_HEAD, optionally pin an exact commit, and return HEAD."""
    ref = _normalize_text(repo_ref)
    commit = _normalize_text(repo_commit)

    fetched = False
    if ref:
        for fetch_cmd in (
            ["git", "fetch", "--depth", "1", "origin", ref],
            ["git", "fetch", "--depth", "1", "origin", f"refs/heads/{ref}"],
            ["git", "fetch", "--depth", "1", "origin", f"refs/tags/{ref}"],
        ):
            result = run_command(fetch_cmd, cwd=repo_dir, env=git_env, check=False, runner=runner)
            if result.returncode == 0:
                fetched = True
                break

    if not fetched:
        run_command(["git", "fetch", "origin"], cwd=repo_dir, env=git_env, runner=runner)

    run_command(["git", "checkout", "--force", "--detach", "FETCH_HEAD"], cwd=repo_dir, env=git_env, runner=runner)

    if commit:
        commit_result = run_command(
            ["git", "checkout", "--force", "--detach", commit],
            cwd=repo_dir,
            env=git_env,
            check=False,
            runner=runner,
        )
        if commit_result.returncode != 0:
            run_command(["git", "fetch", "origin"], cwd=repo_dir, env=git_env, runner=runner)
            commit_result = run_command(
                ["git", "checkout", "--force", "--detach", commit],
                cwd=repo_dir,
                env=git_env,
                check=False,
                runner=runner,
            )
        if commit_result.returncode != 0:
            raise RuntimeError(f"Unable to checkout OUROBOROS_REPO_COMMIT={commit!r}")

    head = run_command(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_dir,
        env=git_env,
        runner=runner,
    )
    return str(head.stdout).strip()


def copy_runtime_files(repo_dir: Path, target_dir: Path, files_to_copy: Sequence[str] = DEFAULT_FILES_TO_COPY) -> None:
    """Copy the runtime adapter files from the checked-out repo into /kaggle/working."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for relative in files_to_copy:
        source = repo_dir / relative.rstrip("/")
        destination = target_dir / relative.rstrip("/")
        if source.is_dir():
            shutil.rmtree(destination, ignore_errors=True)
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def sync_repo_to_kaggle_working(
    env: Mapping[str, str] | None = None,
    *,
    runner: Runner = subprocess.run,
    emit: Emitter = print,
) -> str:
    """Checkout the requested repo/ref/commit and copy runtime files into Kaggle working."""
    env = os.environ if env is None else env
    spec = resolve_kaggle_repo_spec(env)
    git_env = build_authenticated_git_env(spec.repo_url, env)
    ensure_repo(spec.repo_url, spec.repo_dir, git_env, runner=runner)
    checked_out = fetch_and_checkout(spec.repo_dir, spec.repo_ref, spec.repo_commit, git_env, runner=runner)
    copy_runtime_files(spec.repo_dir, spec.target_dir, spec.files_to_copy)
    emit(
        "[kaggle-runtime] synced "
        f"repo={spec.repo_url} ref={spec.repo_ref or '<none>'} "
        f"commit={spec.repo_commit or '<none>'} head={checked_out}"
    )
    return checked_out


__all__ = [
    "DEFAULT_FILES_TO_COPY",
    "DEFAULT_REPO_REF",
    "DEFAULT_REPO_URL",
    "KaggleRepoSpec",
    "build_authenticated_git_env",
    "copy_runtime_files",
    "ensure_repo",
    "fetch_and_checkout",
    "resolve_kaggle_repo_spec",
    "run_command",
    "sync_repo_to_kaggle_working",
]
