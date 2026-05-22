"""Provider adapters for coordinator side effects.

Decision/state modules stay pure.  Coordinator effects that touch Hub, clocks, or
optional external runtimes live behind these small provider seams so local fake
providers can exercise orchestration without network calls.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ouroboros.coordinator.shared import retry_io


class ClockProvider:
    """Clock seam used by coordinator orchestration."""

    def now(self) -> float:
        return time.time()


class HubProvider:
    """Hugging Face Hub JSON/text side-effect adapter."""

    def __init__(self, *, repo_id: str, token: str, attempts: int = 3, base_delay_s: float = 1.5):
        self.repo_id = repo_id
        self.token = token
        self.attempts = int(attempts)
        self.base_delay_s = float(base_delay_s)

    def download_json(self, path: str, *, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        from huggingface_hub import hf_hub_download

        def _download() -> Dict[str, Any]:
            local = hf_hub_download(repo_id=self.repo_id, filename=path, token=self.token)
            with open(local, encoding="utf-8") as f:
                return json.load(f)

        return retry_io(
            f"Download JSON {path}",
            _download,
            attempts=self.attempts,
            base_delay_s=self.base_delay_s,
            swallow=True,
            default=default,
        )

    def upload_json(self, path: str, data: Dict[str, Any], *, message: str) -> None:
        from huggingface_hub import HfApi

        api = HfApi(token=self.token)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(data, tf, indent=2)
            tmp = tf.name
        try:
            retry_io(
                f"Upload JSON {path}",
                lambda: api.upload_file(
                    path_or_fileobj=tmp,
                    path_in_repo=path,
                    repo_id=self.repo_id,
                    token=self.token,
                    commit_message=message,
                ),
                attempts=self.attempts,
                base_delay_s=self.base_delay_s,
            )
        finally:
            Path(tmp).unlink(missing_ok=True)

    def download_text(self, path: str) -> str:
        from huggingface_hub import hf_hub_download

        def _download() -> str:
            local = hf_hub_download(repo_id=self.repo_id, filename=path, token=self.token)
            return Path(local).read_text(encoding="utf-8")

        result = retry_io(
            f"Download text {path}",
            _download,
            attempts=self.attempts,
            base_delay_s=self.base_delay_s,
        )
        assert result is not None
        return result


class KaggleProvider:
    """Kaggle dispatch adapter placeholder for fake-provider validation."""

    def trigger_workers(self, *args: Any, **kwargs: Any) -> Any:
        from ouroboros.coordinator.dispatch import trigger_kaggle_workers

        return trigger_kaggle_workers(*args, **kwargs)


class WandbProvider:
    """W&B lifecycle adapter."""

    def finish(self) -> None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            return


__all__ = ["ClockProvider", "HubProvider", "KaggleProvider", "WandbProvider"]
