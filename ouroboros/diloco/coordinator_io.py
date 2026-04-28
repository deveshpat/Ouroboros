"""Hub and artifact I/O helpers for the DiLoCo coordinator."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

ANCHOR_PREFIX = "diloco_state/anchor"


def retry_io(fn: Callable[[], Any], *, attempts: int = 3, delay_seconds: float = 2.0) -> Any:
    last: BaseException | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return fn()
        except BaseException as exc:  # pragma: no cover - exercised against Hub
            last = exc
            if attempt >= attempts:
                raise
            time.sleep(delay_seconds * attempt)
    if last is not None:
        raise last
    raise RuntimeError("retry_io called with no attempts")


def hub_download_json(repo_id: str, path: str, *, token: str | None = None, default: Any = None) -> Any:
    from huggingface_hub import hf_hub_download  # type: ignore

    def _load() -> Any:
        local = hf_hub_download(repo_id=repo_id, filename=path, repo_type="model", token=token)
        return json.loads(Path(local).read_text(encoding="utf-8"))

    try:
        return retry_io(_load)
    except Exception:
        if default is not None:
            return default
        raise


def hub_upload_json(repo_id: str, path: str, payload: Mapping[str, Any], *, token: str | None = None) -> None:
    from huggingface_hub import upload_file  # type: ignore

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / Path(path).name
        local.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        retry_io(lambda: upload_file(path_or_fileobj=str(local), path_in_repo=path, repo_id=repo_id, repo_type="model", token=token))


def load_adapter_weights_cpu(path: str | Path) -> dict[str, Any]:
    from safetensors.torch import load_file  # type: ignore

    return {k: v.cpu() for k, v in load_file(str(path), device="cpu").items()}


def weighted_average_deltas(worker_weights: Iterable[tuple[Mapping[str, Any], float]]) -> dict[str, Any]:
    weighted = list(worker_weights)
    if not weighted:
        raise ValueError("no worker weights supplied")
    total = sum(float(weight) for _, weight in weighted)
    if total <= 0:
        raise ValueError("total worker weight must be positive")
    result: dict[str, Any] = {}
    for weights, weight in weighted:
        scale = float(weight) / total
        for name, tensor in weights.items():
            result[name] = tensor * scale if name not in result else result[name] + tensor * scale
    return result


def save_and_upload_anchor(repo_id: str, path_in_repo: str, weights: Mapping[str, Any], *, token: str | None = None) -> None:
    from huggingface_hub import upload_file  # type: ignore
    from safetensors.torch import save_file  # type: ignore

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "anchor.safetensors"
        save_file(dict(weights), str(local))
        retry_io(lambda: upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo, repo_id=repo_id, repo_type="model", token=token))


_retry_io = retry_io
_hub_download_json = hub_download_json
_hub_upload_json = hub_upload_json
_load_adapter_weights_cpu = load_adapter_weights_cpu
_weighted_average_deltas = weighted_average_deltas
_save_and_upload_anchor = save_and_upload_anchor


__all__ = [
    "ANCHOR_PREFIX",
    "hub_download_json",
    "hub_upload_json",
    "load_adapter_weights_cpu",
    "retry_io",
    "save_and_upload_anchor",
    "weighted_average_deltas",
]
