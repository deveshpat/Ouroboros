"""CPU aggregation helpers for the DiLoCo coordinator.

This module is intentionally import-light: tensor and Hub dependencies are
imported inside the functions that need them so coordinator contract tests can
run without network/GPU dependencies.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")

ANCHOR_PREFIX = "diloco_state/anchor"
DEFAULT_IO_RETRIES = 3
DEFAULT_IO_RETRY_BASE_DELAY_S = 1.5


def _retry_io(
    label: str,
    fn: Callable[[], T],
    *,
    attempts: int = DEFAULT_IO_RETRIES,
    base_delay_s: float = DEFAULT_IO_RETRY_BASE_DELAY_S,
    swallow: bool = False,
    default: Optional[T] = None,
) -> Optional[T]:
    """Retry transient coordinator I/O with exponential backoff."""
    last_exc: Optional[Exception] = None
    attempts = max(int(attempts), 1)
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - coordinator must keep going on transient I/O errors
            last_exc = exc
            if attempt >= attempts:
                if swallow:
                    print(
                        f"[coordinator] {label} failed after {attempts} attempts: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    return default
                raise
            delay = base_delay_s * (2 ** (attempt - 1))
            print(
                f"[coordinator] {label} failed (attempt {attempt}/{attempts}): "
                f"{type(exc).__name__}: {exc}. Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
    if swallow:
        return default
    assert last_exc is not None
    raise last_exc



def load_adapter_weights_cpu(repo_id: str, weights_path: str, token: str) -> Dict:
    """Load safetensors adapter weights to CPU tensors."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    def _download() -> Dict:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=f"{weights_path}/adapter_model.safetensors",
            token=token,
        )
        return load_file(local, device="cpu")

    result = _retry_io(f"Download adapter weights {weights_path}", _download)
    assert result is not None
    return result




def load_torch_state_cpu(repo_id: str, file_path: str, token: str) -> Optional[Dict]:
    """Load a torch state-dict artifact from Hub to CPU, returning None if absent."""
    from huggingface_hub import hf_hub_download
    import torch

    def _download() -> Dict:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            token=token,
        )
        return torch.load(local, map_location="cpu")

    result = _retry_io(
        f"Download torch state {file_path}",
        _download,
        swallow=True,
        default=None,
    )
    return result


def zero_like_state(reference: Dict) -> Dict:
    """Create a zero-valued state dict matching a worker state dict."""
    import torch

    return {key: torch.zeros_like(value) for key, value in reference.items()}

def weighted_average_deltas(
    anchor_weights: Dict,
    worker_weights: List[Dict],
    worker_samples: List[int],
    outer_lr: float,
) -> Dict:
    """
    DiLoCo outer update:
      pseudo_grad_i = anchor - worker_i
      outer_grad = weighted_mean(pseudo_grad_i, weights=samples_i)
      new_anchor = anchor - outer_lr * outer_grad

    All operations run on CPU tensors.
    """
    import torch

    total_samples = sum(worker_samples)
    if total_samples <= 0:
        raise ValueError("total_samples must be > 0 for aggregation")

    new_weights = {}
    for key in anchor_weights:
        anchor_tensor = anchor_weights[key].float()
        outer_grad = torch.zeros_like(anchor_tensor)
        for weights, n_samples in zip(worker_weights, worker_samples):
            if key not in weights:
                continue
            delta = anchor_tensor - weights[key].float()
            outer_grad += delta * (float(n_samples) / float(total_samples))
        new_weights[key] = (anchor_tensor - outer_lr * outer_grad).to(anchor_weights[key].dtype)
    return new_weights


def aggregate_worker_updates(
    anchor_weights: Dict,
    worker_weights: List[Dict],
    worker_samples: List[int],
    outer_lr: float,
    *,
    mode: str = "diloco",
) -> Dict:
    """Return the next anchor weights for one coordinator aggregation step.

    This preserves the coordinator contract: a single contributor, or a round
    explicitly marked as solo, promotes that worker's weights directly instead
    of computing a weighted delta average.
    """
    if not worker_weights:
        raise ValueError("worker_weights must contain at least one worker for aggregation")
    if len(worker_weights) == 1 or mode == "solo":
        return worker_weights[0]
    return weighted_average_deltas(anchor_weights, worker_weights, worker_samples, outer_lr)


def save_and_upload_anchor(
    new_weights: Dict,
    anchor_adapter_config: Dict,
    repo_id: str,
    token: str,
    message: str,
    halt_gate_state: Optional[Dict] = None,
) -> None:
    from huggingface_hub import HfApi
    from safetensors.torch import save_file

    api = HfApi(token=token)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        weights_path = tmp_path / "adapter_model.safetensors"
        config_path = tmp_path / "adapter_config.json"
        save_file(new_weights, str(weights_path))
        config_path.write_text(json.dumps(anchor_adapter_config, indent=2), encoding="utf-8")
        upload_files = ["adapter_model.safetensors", "adapter_config.json"]
        if halt_gate_state is not None:
            import torch

            torch.save(halt_gate_state, tmp_path / "halt_gate.pt")
            upload_files.append("halt_gate.pt")

        for fname in upload_files:
            _retry_io(
                f"Upload anchor artifact {fname}",
                lambda fname=fname: api.upload_file(
                    path_or_fileobj=str(tmp_path / fname),
                    path_in_repo=f"{ANCHOR_PREFIX}/{fname}",
                    repo_id=repo_id,
                    token=token,
                    commit_message=message,
                ),
            )
    print(f"[coordinator] New anchor uploaded: {message}")
