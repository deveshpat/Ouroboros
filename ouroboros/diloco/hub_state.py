"""Hub state module for DiLoCo coordinator/worker persistence."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .protocol import RoundState, WorkerStatus

ROUND_STATE_PATH = "diloco_state/round_state.json"
ANCHOR_PREFIX = "diloco_state/anchor"


def worker_status_path(worker_id: str) -> str:
    return f"diloco_state/workers/{str(worker_id).strip().upper()}/status.json"


def worker_weights_prefix(worker_id: str, stage_k: int, round_n: int) -> str:
    return f"diloco_state/workers/{str(worker_id).strip().upper()}/stage_{int(stage_k)}/round_{int(round_n)}"


@dataclass(frozen=True)
class HubPaths:
    round_state: str = ROUND_STATE_PATH
    anchor_prefix: str = ANCHOR_PREFIX

    def status(self, worker_id: str) -> str:
        return worker_status_path(worker_id)

    def weights(self, worker_id: str, stage_k: int, round_n: int) -> str:
        return worker_weights_prefix(worker_id, stage_k, round_n)


class InMemoryHubStateStore:
    """Test adapter for the Hub state seam."""

    def __init__(self, files: Optional[Mapping[str, Mapping[str, Any]]] = None) -> None:
        self.files: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in (files or {}).items()}
        self.paths = HubPaths()

    def load_round_state(self) -> Optional[RoundState]:
        payload = self.files.get(ROUND_STATE_PATH)
        return RoundState.from_mapping(payload) if payload is not None else None

    def save_round_state(self, state: Mapping[str, Any], message: str = "") -> None:
        self.files[ROUND_STATE_PATH] = dict(state)

    def load_worker_statuses(self, worker_ids: Sequence[str]) -> list[WorkerStatus]:
        statuses: list[WorkerStatus] = []
        for worker_id in worker_ids:
            payload = self.files.get(worker_status_path(worker_id))
            if payload is not None:
                statuses.append(WorkerStatus.from_mapping(payload))
        return statuses

    def upload_worker_status(self, status: Mapping[str, Any]) -> None:
        self.files[worker_status_path(str(status.get("worker_id", "")))] = dict(status)


class HuggingFaceHubStateStore:
    """Production adapter around Hugging Face Hub JSON state.

    Weight artifacts are intentionally kept as explicit methods so heavy tensor
    dependencies stay outside tests that only exercise JSON state.
    """

    def __init__(self, *, repo_id: str, token: str) -> None:
        self.repo_id = repo_id
        self.token = token
        self.paths = HubPaths()

    def _download_json(self, path: str) -> Optional[Dict[str, Any]]:
        from huggingface_hub import hf_hub_download

        try:
            local = hf_hub_download(repo_id=self.repo_id, filename=path, token=self.token)
        except Exception:
            return None
        return json.loads(Path(local).read_text(encoding="utf-8"))

    def _upload_json(self, path: str, data: Mapping[str, Any], message: str) -> None:
        from huggingface_hub import HfApi

        api = HfApi(token=self.token)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(dict(data), tf, indent=2)
            tmp = tf.name
        try:
            api.upload_file(
                path_or_fileobj=tmp,
                path_in_repo=path,
                repo_id=self.repo_id,
                token=self.token,
                commit_message=message,
            )
        finally:
            Path(tmp).unlink(missing_ok=True)

    def load_round_state(self) -> Optional[RoundState]:
        payload = self._download_json(ROUND_STATE_PATH)
        return RoundState.from_mapping(payload) if payload is not None else None

    def save_round_state(self, state: Mapping[str, Any], message: str) -> None:
        self._upload_json(ROUND_STATE_PATH, state, message)

    def load_worker_statuses(self, worker_ids: Sequence[str]) -> list[WorkerStatus]:
        statuses: list[WorkerStatus] = []
        for worker_id in worker_ids:
            payload = self._download_json(worker_status_path(worker_id))
            if payload is not None:
                statuses.append(WorkerStatus.from_mapping(payload))
        return statuses
