from __future__ import annotations

import math
import os
import random
import shutil
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

HF_UPLOAD_TIMEOUT_SECONDS = 300.0


def resolve_hf_token(cli_value: Optional[str]) -> Optional[str]:
    """Return the HF write token from CLI or environment, never from disk."""
    if cli_value:
        return cli_value
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs for deterministic control flow."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vram_gb(device: torch.device) -> float:
    """Return allocated VRAM in GiB or 0.0 on CPU."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / (1024 ** 3)


def checkpoint_step_from_name(name: str) -> int:
    """Extract the numeric step from a checkpoint directory name."""
    prefix = "checkpoint-"
    if prefix not in name:
        return -1
    tail = name.split(prefix, 1)[1]
    digits: List[str] = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else -1


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    """Linear warmup followed by cosine decay to min_lr_ratio times base LR."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        if total_steps <= warmup_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


class ModelEMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        one_minus = 1.0 - self.decay
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=one_minus)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.decay = float(sd.get("decay", self.decay))
        for name, tensor in sd.get("shadow", {}).items():
            if name in self.shadow and self.shadow[name].shape == tensor.shape:
                self.shadow[name].copy_(tensor)


@contextmanager
def ema_scope(model: torch.nn.Module, ema: ModelEMA) -> Iterator[None]:
    """Temporarily swap EMA weights into a model and restore live weights on exit."""
    live_backup: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name in ema.shadow:
            live_backup[name] = param.data.clone()
            param.data.copy_(ema.shadow[name].to(device=param.device, dtype=param.dtype))
    try:
        yield
    finally:
        for name, param in model.named_parameters():
            if name in live_backup:
                param.data.copy_(live_backup[name])


def autocast_context(device: torch.device, dtype: torch.dtype):
    """Return an autocast context that only activates on CUDA."""
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda"))


def cleanup_temporary_checkpoints(output_dir: Path) -> None:
    """Remove stale .tmp checkpoint directories left by interrupted runs."""
    if not output_dir.exists():
        return
    for entry in output_dir.iterdir():
        if entry.is_dir() and entry.name.endswith(".tmp"):
            print(f"  [cleanup] removing stale tmp checkpoint: {entry.name}")
            shutil.rmtree(entry, ignore_errors=True)


def sync_checkpoint_to_hub(
    checkpoint_dir: Path,
    repo_id: str,
    token: str,
    timeout_seconds: float = HF_UPLOAD_TIMEOUT_SECONDS,
) -> bool:
    """Upload one checkpoint directory to the Hugging Face Hub without raising."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[hub] huggingface_hub not installed - skipping Hub sync.")
        return False

    remote_name = checkpoint_dir.name[:-4] if checkpoint_dir.name.endswith(".tmp") else checkpoint_dir.name
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, token=token, private=True, exist_ok=True)
        print(f"  [hub] uploading {remote_name} -> {repo_id} ...")
        future = api.upload_folder(
            repo_id=repo_id,
            folder_path=str(checkpoint_dir),
            path_in_repo=remote_name,
            token=token,
            commit_message=f"Upload {remote_name}",
            run_as_future=True,
        )
        commit_info = future.result(timeout=timeout_seconds)
        oid = getattr(commit_info, "oid", "?")
        oid_short = oid[:8] if isinstance(oid, str) and oid != "?" else "?"
        print(f"  [hub] uploaded  {remote_name} (commit={oid_short})")
        return True
    except FutureTimeoutError:
        print(f"  [hub] upload timed out for {remote_name}; local copy retained.")
        return False
    except Exception as exc:
        print(f"  [hub] upload failed for {remote_name}: {exc}")
        return False


def list_local_checkpoints(output_dir: Path) -> List[Path]:
    """Return finalized local checkpoint directories sorted newest-first."""
    if not output_dir.exists() or not output_dir.is_dir():
        return []
    candidates: List[Path] = []
    for entry in output_dir.iterdir():
        if entry.is_dir() and not entry.name.endswith(".tmp"):
            step = checkpoint_step_from_name(entry.name)
            if step >= 0:
                candidates.append(entry)
    return sorted(candidates, key=lambda entry: checkpoint_step_from_name(entry.name), reverse=True)


def try_load_state(path: Path, device: torch.device) -> Optional[Dict[str, Any]]:
    """Load training_state.pt from a checkpoint directory, returning None on corruption."""
    state_path = path / "training_state.pt"
    if not state_path.exists():
        return None
    try:
        return torch.load(state_path, map_location=device)
    except Exception as exc:
        print(f"  [resume] corrupt checkpoint {path.name}: {exc} - skipping")
        return None


def list_remote_checkpoint_names(repo_id: str, token: str) -> List[str]:
    """Return Hub checkpoint directory names sorted newest-first."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []

    try:
        api = HfApi(token=token)
        repo_files = list(api.list_repo_files(repo_id=repo_id, token=token))
    except Exception:
        return []

    names = set()
    for file_name in repo_files:
        parts = Path(file_name).parts
        top = parts[0] if parts else ""
        if checkpoint_step_from_name(top) >= 0:
            names.add(top)
    return sorted(names, key=checkpoint_step_from_name, reverse=True)


def download_checkpoint_from_hub(
    checkpoint_name: str,
    output_dir: Path,
    repo_id: str,
    token: str,
) -> Optional[Path]:
    """Download a single checkpoint directory from the Hub into output_dir."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None

    local_dir = output_dir / checkpoint_name
    if local_dir.exists():
        shutil.rmtree(local_dir, ignore_errors=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_dir),
            token=token,
            force_download=True,
            allow_patterns=[f"{checkpoint_name}/*"],
        )
    except Exception as exc:
        print(f"  [hub]  download failed for {checkpoint_name}: {exc}")
        return None

    return local_dir if local_dir.exists() else None


def build_adamw_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    prefer_fused: bool = True,
) -> tuple[AdamW, bool]:
    """Build AdamW with standard decay/no-decay parameter grouping."""
    decay_params = [param for param in model.parameters() if param.requires_grad and param.ndim >= 2]
    no_decay_params = [param for param in model.parameters() if param.requires_grad and param.ndim < 2]
    base_kwargs = {"lr": lr, "betas": betas, "eps": eps}
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if prefer_fused:
        try:
            return AdamW(param_groups, fused=True, **base_kwargs), True
        except TypeError:
            pass
    return AdamW(param_groups, **base_kwargs), False
