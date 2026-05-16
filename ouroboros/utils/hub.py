"""Hugging Face Hub checkpoint utilities."""

from __future__ import annotations

import os
import re as _re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ouroboros.bootstrap import _resolve_hf_token_common
from ouroboros.models import _is_main_process

def _parse_stage_dir_name(name: str) -> Optional[int]:
    if not name.startswith("stage_"):
        return None
    suffix = name.split("stage_", 1)[1]
    return int(suffix) if suffix.isdigit() else None


def _read_training_state(ckpt_dir: Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(ckpt_dir / "training_state.pt", map_location=map_location)


def _resolve_hf_token(cli_value: Optional[str]) -> Optional[str]:
    return _resolve_hf_token_common(cli_value)


def _hub_upload_checkpoint(
    ckpt_dir: Path,
    hf_repo_id: str,
    hf_token: str,
    remote_prefix: str = "runs/stage3",
    timeout_s: float = 300.0,
) -> bool:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False

    remote_name = f"{remote_prefix.strip('/')}/{ckpt_dir.name}".strip("/")
    try:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hf_repo_id, private=True, exist_ok=True, token=hf_token)
        future = api.upload_folder(
            repo_id=hf_repo_id,
            folder_path=str(ckpt_dir),
            path_in_repo=remote_name,
            token=hf_token,
            commit_message=f"Upload {ckpt_dir.name}",
            run_as_future=True,
        )
        future.result(timeout=timeout_s)
        if _is_main_process():
            print(f"  [hub] uploaded {remote_name} -> {hf_repo_id}")
        return True
    except Exception as exc:
        if _is_main_process():
            print(f"  [hub] upload failed for {remote_name}: {exc}")
        return False


def _hub_download_checkpoint(
    ckpt_name: str,
    local_dir: Path,
    hf_repo_id: str,
    hf_token: str,
    remote_prefix: str = "runs/stage3",
) -> Optional[Path]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None

    local_dir.mkdir(parents=True, exist_ok=True)
    remote_path = f"{remote_prefix.strip('/')}/{ckpt_name}".strip("/")
    dest = local_dir / remote_path

    try:
        snapshot_download(
            repo_id=hf_repo_id,
            local_dir=str(local_dir),
            token=hf_token,
            force_download=True,
            allow_patterns=[f"{remote_path}/*"],
        )
        return dest if dest.exists() else None
    except Exception as exc:
        if _is_main_process():
            print(f"  [hub] download failed for {remote_path}: {exc}")
        return None


def _list_hub_stage_checkpoints(
    hf_repo_id: str,
    hf_token: str,
    remote_prefix: str = "runs/stage3",
) -> List[Tuple[int, int, str]]:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []
    try:
        api = HfApi(token=hf_token)
        files = list(api.list_repo_files(repo_id=hf_repo_id, token=hf_token))
    except Exception:
        return []

    prefix = remote_prefix.strip("/")
    found: set = set()
    for f in files:
        parts = f.split("/")
        try:
            prefix_parts = prefix.split("/")
            if parts[:len(prefix_parts)] != prefix_parts:
                continue
            rest = parts[len(prefix_parts):]
            if len(rest) < 2:
                continue
            stage_dir = rest[0]
            ckpt_dir  = rest[1]
            stage_k = _parse_stage_dir_name(stage_dir)
            if stage_k is None:
                continue
            if not (ckpt_dir.startswith("checkpoint-") or ckpt_dir == "best"):
                continue
            if ckpt_dir == "best":
                step = 0
            else:
                tail = ckpt_dir.split("-")[-1]
                step = int(tail) if tail.isdigit() else 0
            rel = "/".join(rest[:2])
            found.add((stage_k, step, rel))
        except Exception:
            continue

    return sorted(found, key=lambda x: (x[0], x[1]), reverse=True)