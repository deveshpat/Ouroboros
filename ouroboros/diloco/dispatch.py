"""Kaggle dispatch helpers for the DiLoCo coordinator."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WORKER_IDS = ["A", "B", "C"]
WORKER_KAGGLE_SLUGS: Dict[str, Tuple[str, str]] = {
    "A": ("weirdrunner", "weirdrunner/kaggle-utils"),
    "B": ("weirdrunner007", "weirdrunner007/kaggle-utils"),
    "C": ("weirdrunner008", "weirdrunner008/kaggle-utils"),
}

_KAGGLE_PUSH_SUCCESS_MARKERS = (
    "successfully pushed",
)
_KAGGLE_PUSH_FAILURE_MARKERS = (
    "kernel push error",
    "maximum weekly gpu quota",
    "gpu quota",
    "quota reached",
    "error",
)


def _format_kaggle_output(stdout: Optional[str], stderr: Optional[str]) -> str:
    return "\n".join(part.strip() for part in (stdout or "", stderr or "") if part and part.strip())


def _is_successful_kaggle_push(returncode: int, stdout: Optional[str], stderr: Optional[str]) -> bool:
    """Return True only for an explicitly successful Kaggle push.

    Kaggle can print a human-readable `Kernel push error: ...` while still
    returning 0. Treating any zero exit code as success leaves workers in
    `triggered_workers` even though they never launched, which then forces the
    coordinator to wait for the long worker timeout.
    """
    combined = _format_kaggle_output(stdout, stderr).lower()
    if returncode != 0:
        return False
    if any(marker in combined for marker in _KAGGLE_PUSH_FAILURE_MARKERS):
        return False
    return any(marker in combined for marker in _KAGGLE_PUSH_SUCCESS_MARKERS)


def _normalize_optional_text(value: Optional[Any], *, uppercase: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.upper() if uppercase else text


def _first_nonempty_text(*values: Optional[Any], uppercase: bool = False) -> Optional[str]:
    for value in values:
        text = _normalize_optional_text(value, uppercase=uppercase)
        if text is not None:
            return text
    return None


def _set_env_if_present(
    target: Dict[str, str],
    key: str,
    value: Optional[Any],
    *,
    uppercase: bool = False,
) -> None:
    text = _normalize_optional_text(value, uppercase=uppercase)
    if text is not None:
        target[key] = text


def _infer_runtime_repo_url(default: str = "https://github.com/deveshpat/Ouroboros.git") -> str:
    explicit = _first_nonempty_text(os.environ.get("OUROBOROS_REPO_URL"))
    if explicit:
        return explicit
    repo = _first_nonempty_text(os.environ.get("GITHUB_REPOSITORY"))
    server = _first_nonempty_text(os.environ.get("GITHUB_SERVER_URL"), "https://github.com")
    if repo:
        return f"{server.rstrip('/')}/{repo}.git"
    return default


def _infer_runtime_repo_ref(default: str = "main") -> str:
    return (
        _first_nonempty_text(
            os.environ.get("OUROBOROS_REPO_REF"),
            os.environ.get("GITHUB_REF_NAME"),
            default,
        )
        or default
    )


def _infer_runtime_repo_commit() -> Optional[str]:
    return _first_nonempty_text(
        os.environ.get("OUROBOROS_REPO_COMMIT"),
        os.environ.get("GITHUB_SHA"),
    )


def _build_worker_runtime_env(args: argparse.Namespace, worker_id: str) -> Dict[str, str]:
    worker = _normalize_optional_text(worker_id, uppercase=True)
    if worker not in WORKER_IDS:
        raise ValueError(f"Invalid worker id for runtime env injection: {worker_id!r}")

    runtime_env: Dict[str, str] = {}

    for name, value in os.environ.items():
        if name.startswith("OUROBOROS_"):
            _set_env_if_present(runtime_env, name, value)

    _set_env_if_present(runtime_env, "DILOCO_WORKER_ID", worker, uppercase=True)
    _set_env_if_present(runtime_env, "OUROBOROS_DILOCO_WORKER_ID", worker, uppercase=True)
    _set_env_if_present(runtime_env, "WORKER_ID", worker, uppercase=True)
    runtime_env["OUROBOROS_AUTO_TRIGGERED"] = "1"

    hf_token = _first_nonempty_text(
        getattr(args, "hf_token", None),
        os.environ.get("HF_TOKEN"),
        os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    if hf_token:
        runtime_env["HF_TOKEN"] = hf_token
        runtime_env["HUGGINGFACE_HUB_TOKEN"] = hf_token

    wandb_key = _first_nonempty_text(
        getattr(args, "wandb_key", None),
        os.environ.get("WANDB_API_KEY"),
        os.environ.get("WANDB_KEY"),
    )
    if wandb_key:
        runtime_env["WANDB_API_KEY"] = wandb_key
        runtime_env["WANDB_KEY"] = wandb_key

    github_token = _first_nonempty_text(
        os.environ.get("GITHUB_TOKEN"),
        os.environ.get("GH_TOKEN"),
    )
    if github_token:
        runtime_env["GITHUB_TOKEN"] = github_token
        runtime_env["GH_TOKEN"] = github_token

    _set_env_if_present(runtime_env, "OUROBOROS_REPO_URL", _infer_runtime_repo_url())
    _set_env_if_present(runtime_env, "OUROBOROS_REPO_REF", _infer_runtime_repo_ref())
    _set_env_if_present(runtime_env, "OUROBOROS_REPO_COMMIT", _infer_runtime_repo_commit())
    _set_env_if_present(
        runtime_env,
        "OUROBOROS_DILOCO_STATE_REPO",
        _first_nonempty_text(os.environ.get("OUROBOROS_DILOCO_STATE_REPO"), getattr(args, "repo_id", None)),
    )
    _set_env_if_present(
        runtime_env,
        "OUROBOROS_DILOCO_SIGNAL_REPO",
        _first_nonempty_text(
            os.environ.get("OUROBOROS_DILOCO_SIGNAL_REPO"),
            os.environ.get("GITHUB_REPOSITORY"),
        ),
    )
    _set_env_if_present(
        runtime_env,
        "OUROBOROS_DILOCO_OUTER_LR",
        _first_nonempty_text(
            os.environ.get("OUROBOROS_DILOCO_OUTER_LR"),
            f"{float(getattr(args, 'outer_lr', 0.7)):g}",
        ),
    )
    _set_env_if_present(
        runtime_env,
        "OUROBOROS_WANDB_PROJECT",
        _first_nonempty_text(os.environ.get("OUROBOROS_WANDB_PROJECT"), getattr(args, "wandb_project", None)),
    )
    _set_env_if_present(
        runtime_env,
        "OUROBOROS_WANDB_ENTITY",
        _first_nonempty_text(os.environ.get("OUROBOROS_WANDB_ENTITY"), getattr(args, "wandb_entity", None)),
    )
    _set_env_if_present(
        runtime_env,
        "OUROBOROS_DILOCO_OUTPUT_DIR",
        _first_nonempty_text(os.environ.get("OUROBOROS_DILOCO_OUTPUT_DIR"), "runs/diloco"),
    )
    return runtime_env


def _encode_runtime_env_payload(runtime_env: Dict[str, str]) -> str:
    payload = json.dumps(runtime_env, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(zlib.compress(payload, level=9)).decode("ascii")


def _build_kaggle_kernel_metadata(*, slug: str, notebook_filename: str) -> Dict[str, object]:
    title = slug.split("/", 1)[-1]
    return {
        "id": slug,
        "title": title,
        "code_file": notebook_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "accelerator": "NvidiaTeslaT4",  # official ID per kaggle-cli docs; belt-and-suspenders # ← ADD: pins T4, not just any GPU
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": ["weirdrunner007/ouroboros-cache"],  # ← ADD: attaches cache
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "keywords": [],
    }


def _build_worker_dispatch_cell(
    worker_id: str,
    runtime_env: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    worker_id = worker_id.strip().upper()
    if worker_id not in WORKER_IDS:
        raise ValueError(f"Invalid worker id for notebook dispatch: {worker_id!r}")

    payload = _encode_runtime_env_payload(runtime_env or {
        "DILOCO_WORKER_ID": worker_id,
        "OUROBOROS_DILOCO_WORKER_ID": worker_id,
        "WORKER_ID": worker_id,
        "OUROBOROS_AUTO_TRIGGERED": "1",
    })

    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "diloco-dispatch-worker-id",
        "metadata": {"tags": ["diloco-dispatch"]},
        "outputs": [],
        "source": [
            "# AUTO-GENERATED BY DILOCO COORDINATOR. DO NOT EDIT IN REPO.\n",
            "import base64\n",
            "import json\n",
            "import os\n",
            "import zlib\n",
            f"_DISPATCH_PAYLOAD = {json.dumps(payload)}\n",
            "_DISPATCH_ENV = json.loads(zlib.decompress(base64.b64decode(_DISPATCH_PAYLOAD)).decode('utf-8'))\n",
            "for _dispatch_key, _dispatch_value in _DISPATCH_ENV.items():\n",
            "    if _dispatch_value is None:\n",
            "        continue\n",
            "    os.environ[str(_dispatch_key)] = str(_dispatch_value)\n",
            "del _DISPATCH_ENV, _DISPATCH_PAYLOAD\n",
            f"print('[dispatch] Bound notebook to DiLoCo worker {worker_id}')\n",
        ],
    }


def _stage_local_kaggle_kernel(
    notebook_path: Path,
    slug: str,
    staging_dir: Path,
    *,
    worker_id: Optional[str] = None,
    runtime_env: Optional[Dict[str, str]] = None,
) -> Path:
    if not notebook_path.exists():
        raise FileNotFoundError(
            f"Notebook source not found at {notebook_path}. "
            "Auto-trigger requires a local kaggle-utils.ipynb checkout."
        )

    staged_notebook = staging_dir / notebook_path.name
    shutil.copy2(notebook_path, staged_notebook)

    if worker_id:
        notebook = json.loads(staged_notebook.read_text(encoding="utf-8"))
        cells = notebook.get("cells")
        if not isinstance(cells, list):
            raise ValueError("Kaggle notebook JSON is missing a valid 'cells' list")

        dispatch_cell = _build_worker_dispatch_cell(worker_id, runtime_env=runtime_env)
        replaced = False
        for idx, cell in enumerate(cells):
            if not isinstance(cell, dict):
                continue
            source = "".join(cell.get("source", []))
            tags = cell.get("metadata", {}).get("tags", [])
            if (
                "AUTO-GENERATED BY DILOCO COORDINATOR" in source
                or "diloco-dispatch" in tags
            ):
                cells[idx] = dispatch_cell
                replaced = True
                break
        if not replaced:
            insert_at = 1 if cells and isinstance(cells[0], dict) and cells[0].get("cell_type") == "markdown" else 0
            cells.insert(insert_at, dispatch_cell)

        staged_notebook.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    metadata_path = staging_dir / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps(
            _build_kaggle_kernel_metadata(slug=slug, notebook_filename=staged_notebook.name),
            indent=2,
        ),
        encoding="utf-8",
    )
    return staged_notebook


def _trigger_single_worker(
    worker_id: str,
    username: str,
    key: str,
    slug: str,
    notebook_path: Path,
    *,
    injected_env: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Trigger a Kaggle kernel by pushing the repo-tracked notebook with generated
    metadata, instead of pulling the live kernel back from Kaggle first.

    Why this path is safer:
      - the coordinator logs show `kaggle kernels pull` failing with
        `Permission 'kernels.get' was denied`, which blocks the old pull→push flow
        even when the worker has already completed successfully.
      - `kaggle kernels push` is the supported CLI path for updating and running a
        kernel; staging the checked-in notebook locally avoids the fragile readback
        permission entirely.
    """
    import os
    import subprocess
    import tempfile

    if not username or not key:
        print(f"[coordinator] No credentials for Worker {worker_id} — skipping trigger.")
        return False

    expected_owner, _ = WORKER_KAGGLE_SLUGS[worker_id]
    if username.strip().lower() != expected_owner.lower():
        print(
            f"[coordinator] WARNING: Worker {worker_id} expects Kaggle owner "
            f"{expected_owner}, but received username {username!r}. Trigger may fail."
        )

    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key
    env.pop("KAGGLE_CONFIG_DIR", None)

    def _run_kaggle(args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["kaggle"] + args,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _stage_local_kaggle_kernel(notebook_path, slug, tmp_path, worker_id=worker_id, runtime_env=injected_env)

            # --accelerator requires kaggle>=1.8.4 (added in PR #907).
            # "NvidiaTeslaT4" is the official accelerator ID per kaggle-cli docs.
            # This is distinct from the JSON metadata field (also present, belt-and-suspenders).
            push_args = ["kernels", "push", "-p", str(tmp_path), "--accelerator", "NvidiaTeslaT4"]
            push = _run_kaggle(push_args)
            out = _format_kaggle_output(push.stdout, push.stderr)
            if not _is_successful_kaggle_push(push.returncode, push.stdout, push.stderr):
                detail = out or f"returncode={push.returncode}"
                print(f"[coordinator] WARNING: kernels push failed for Worker {worker_id} ({slug}): {detail}")
                return False

        print(f"[coordinator] Triggered Worker {worker_id}: {slug}  ({out})")
        return True

    except subprocess.TimeoutExpired:
        print(f"[coordinator] WARNING: kaggle CLI timed out for Worker {worker_id} ({slug})")
        return False
    except FileNotFoundError as exc:
        print(f"[coordinator] WARNING: Auto-trigger prerequisites missing for {slug}: {exc}")
        return False
    except Exception as exc:
        print(f"[coordinator] WARNING: Failed to trigger {slug}: {exc}")
        return False


def trigger_kaggle_workers(
    kaggle_creds: Dict[str, Tuple[Optional[str], Optional[str]]],
    *,
    active_workers: List[str],
    notebook_path: Path,
    coordinator_args: Optional[argparse.Namespace] = None,
) -> Dict[str, str]:
    """
    Trigger only the specified active_workers using their Kaggle credentials.

    Returns:
        Mapping of worker_id -> "success" | "failed" | "manual".
        "manual" means no credentials were present, so the worker remains an
        outstanding manual dispatch rather than a confirmed auto-trigger.
    """
    results: Dict[str, str] = {}
    for worker_id in active_workers:
        username, key = kaggle_creds.get(worker_id, (None, None))
        _, slug = WORKER_KAGGLE_SLUGS[worker_id]

        if not username or not key:
            print(
                f"[coordinator] No credentials for Worker {worker_id} ({slug}) - "
                "skipping automatic trigger. Start this worker manually."
            )
            results[worker_id] = "manual"
            continue

        injected_env = (
            _build_worker_runtime_env(coordinator_args, worker_id)
            if coordinator_args is not None
            else None
        )
        results[worker_id] = (
            "success"
            if _trigger_single_worker(
                worker_id,
                username,
                key,
                slug,
                notebook_path=notebook_path,
                injected_env=injected_env,
            )
            else "failed"
        )
    return results
