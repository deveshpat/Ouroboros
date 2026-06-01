"""Checkpoint, resume, and Hub-sync helpers for training sessions."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ouroboros.coconut.dgac import HaltGate
from ouroboros.models import barrier
from ouroboros.utils.runtime_env import is_main_process


def _parse_stage_dir_name(name: str) -> Optional[int]:
    if not name.startswith("stage_"):
        return None
    suffix = name.split("stage_", 1)[1]
    return int(suffix) if suffix.isdigit() else None


def read_training_state(ckpt_dir: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(ckpt_dir / "training_state.pt", map_location=map_location)


def _upload_checkpoint_folder(
    ckpt_dir: Path,
    hf_repo_id: str,
    hf_token: str,
    remote_prefix: str,
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
        if is_main_process():
            print(f"  [hub] uploaded {remote_name} -> {hf_repo_id}")
        return True
    except Exception as exc:
        if is_main_process():
            print(f"  [hub] upload failed for {remote_name}: {exc}")
        return False


def _download_checkpoint_folder(
    ckpt_name: str,
    local_dir: Path,
    hf_repo_id: str,
    hf_token: str,
    remote_prefix: str,
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
        if is_main_process():
            print(f"  [hub] download failed for {remote_path}: {exc}")
        return None


def _list_remote_stage_checkpoints(
    hf_repo_id: str,
    hf_token: str,
    remote_prefix: str,
) -> List[Tuple[int, int, str]]:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []

    try:
        files = list(HfApi(token=hf_token).list_repo_files(repo_id=hf_repo_id, token=hf_token))
    except Exception:
        return []

    prefix_parts = remote_prefix.strip("/").split("/") if remote_prefix.strip("/") else []
    found: set[Tuple[int, int, str]] = set()
    for filename in files:
        parts = filename.split("/")
        if prefix_parts and parts[: len(prefix_parts)] != prefix_parts:
            continue
        rest = parts[len(prefix_parts):]
        if len(rest) < 2:
            continue
        stage_k = _parse_stage_dir_name(rest[0])
        ckpt_dir = rest[1]
        if stage_k is None or not (ckpt_dir.startswith("checkpoint-") or ckpt_dir == "best"):
            continue
        if ckpt_dir == "best":
            step = 0
        else:
            tail = ckpt_dir.split("-")[-1]
            step = int(tail) if tail.isdigit() else 0
        found.add((stage_k, step, "/".join(rest[:2])))
    return sorted(found, key=lambda x: (x[0], x[1]), reverse=True)


def save_checkpoint(
    output_dir: Path,
    step: int,
    epoch: int,
    step_in_epoch: int,
    step_in_phase: int,
    stage_k: int,
    model,
    halt_gate: Optional[HaltGate],
    optimizer: Optional[AdamW],
    scheduler: Optional[LambdaLR],
    args: argparse.Namespace,
    val_ce: Optional[float],
    val_acc: Optional[float],
    tag: str = "",
) -> Path:
    stage_dir = output_dir / f"stage_{stage_k}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    name = "best" if tag == "best" else f"checkpoint-{step:07d}"
    ckpt = stage_dir / name
    tmp = stage_dir / f"{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(tmp / "adapter_model"))
    if halt_gate is not None:
        torch.save(halt_gate.state_dict(), tmp / "halt_gate.pt")
    torch.save(
        {
            "stage_k": stage_k,
            "step": step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "step_in_phase": step_in_phase,
            "val_ce": val_ce,
            "val_acc": val_acc,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "use_halt_gate": args.use_halt_gate,
            "model_id": args.model_id,
        },
        tmp / "training_state.pt",
    )

    if ckpt.exists():
        shutil.rmtree(ckpt, ignore_errors=True)
    tmp.replace(ckpt)
    label = "best" if tag == "best" else "saved"
    if is_main_process():
        print(f"  [ckpt] {label} -> {ckpt}  token_acc={val_acc}  ce={val_ce}")

    hf_token = getattr(args, "_resolved_hf_token", None)
    push = getattr(args, "push_to_hub", False)
    repo_id = getattr(args, "hf_repo_id", "WeirdRunner/Ouroboros")
    subdir = getattr(args, "hf_stage_subdir", "runs/stage3")
    if push and hf_token and is_main_process():
        stage_remote_prefix = f"{subdir.strip('/')}/stage_{stage_k}"
        _upload_checkpoint_folder(ckpt, repo_id, hf_token, remote_prefix=stage_remote_prefix)
    return ckpt


def load_checkpoint(
    ckpt_dir: Path,
    model,
    halt_gate: Optional[HaltGate],
    optimizer: Optional[AdamW],
    scheduler: Optional[LambdaLR],
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, Any]:
    state = read_training_state(ckpt_dir, map_location=device)
    adapter_dir = ckpt_dir / "adapter_model"
    if adapter_dir.exists():
        from peft import set_peft_model_state_dict

        loaded = False
        for fname in ["adapter_model.safetensors", "adapter_model.bin"]:
            fpath = adapter_dir / fname
            if not fpath.exists():
                continue
            if fname.endswith(".safetensors"):
                from safetensors.torch import load_file
                weights = load_file(str(fpath))
            else:
                weights = torch.load(fpath, map_location=device)
            set_peft_model_state_dict(model, weights)
            loaded = True
            break
        if not loaded:
            raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")

    halt_gate_path = ckpt_dir / "halt_gate.pt"
    if halt_gate is not None and halt_gate_path.exists():
        halt_gate.load_state_dict(torch.load(halt_gate_path, map_location=device))

    if optimizer is not None and state.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception as exc:
            if verbose:
                print(f"  [resume] optimizer state mismatch ({exc}); resetting optimizer.")
    if scheduler is not None and state.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(state["scheduler"])
        except Exception as exc:
            if verbose:
                print(f"  [resume] scheduler state mismatch ({exc}); resetting scheduler.")

    if verbose:
        print(
            f"  [resume] step={int(state.get('step', 0))} "
            f"epoch={int(state.get('epoch', 0))} "
            f"stage_k={int(state.get('stage_k', 0))} "
            f"val_acc={state.get('val_acc')}"
        )
    return state


def prune_epoch_checkpoints(stage_dir: Path, keep: int) -> None:
    keep = max(int(keep), 0)
    checkpoints = sorted(
        [p for p in stage_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    for old in checkpoints[:-keep] if keep > 0 else checkpoints:
        shutil.rmtree(old, ignore_errors=True)


def find_latest_resume_checkpoint(
    output_dir: Path,
    hf_token: Optional[str] = None,
    hf_repo_id: str = "WeirdRunner/Ouroboros",
    hf_stage_subdir: str = "runs/stage3",
) -> Optional[Path]:
    best_path: Optional[Path] = None
    best_key:  Optional[Tuple[int, int, int, int]] = None

    if output_dir.exists():
        for stage_dir in output_dir.iterdir():
            stage_k = _parse_stage_dir_name(stage_dir.name)
            if stage_k is None or not stage_dir.is_dir():
                continue
            for ckpt in stage_dir.iterdir():
                if not ckpt.is_dir() or not ckpt.name.startswith("checkpoint-"):
                    continue
                state_path = ckpt / "training_state.pt"
                if not state_path.exists():
                    continue
                try:
                    state = read_training_state(ckpt, map_location="cpu")
                except Exception:
                    continue
                key = (
                    stage_k,
                    int(state.get("epoch", -1)),
                    int(state.get("step_in_epoch", -1)),
                    int(state.get("step", -1)),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_path = ckpt

    if best_path is not None:
        return best_path

    if not hf_token:
        return None

    if is_main_process():
        print("  [resume] No local checkpoints found. Scanning Hub...")

    hub_candidates = _list_remote_stage_checkpoints(hf_repo_id, hf_token, hf_stage_subdir)
    hub_candidates = [(k, s, n) for k, s, n in hub_candidates if not n.endswith("/best")]
    hub_resume_dir = output_dir / ".hub_resume"

    for stage_k, step, rel_name in hub_candidates:
        ckpt_name = rel_name.split("/")[-1]
        if is_main_process():
            print(f"  [hub] downloading {rel_name} ...")
        downloaded = _download_checkpoint_folder(
            ckpt_name=ckpt_name,
            local_dir=hub_resume_dir,
            hf_repo_id=hf_repo_id,
            hf_token=hf_token,
            remote_prefix=f"{hf_stage_subdir}/stage_{stage_k}",
        )
        if downloaded is not None and (downloaded / "training_state.pt").exists():
            if is_main_process():
                print(f"  [hub] using {rel_name} as resume checkpoint")
            return downloaded

    return None


def startup_hub_sync_and_prune(
    output_dir: Path,
    resume_path: Optional[Path],
    hf_token: str,
    hf_repo_id: str,
    hf_stage_subdir: str,
) -> None:
    """
    Called once at session start (rank 0 only, before training).
    1. Upload every local checkpoint that exists to Hub (best + numbered).
    2. Delete all local numbered checkpoints EXCEPT the one we are resuming from.
       Always preserve best/ dirs.

    This prevents Kaggle disk overflow across sessions. Upload failures are
    logged, but local pruning still follows the keep policy so stale numbered
    checkpoints do not accumulate across sessions.
    """
    if not is_main_process():
        return

    all_ckpts: List[Tuple[Path, bool]] = []  # (path, is_resume)
    if output_dir.exists():
        for stage_dir in sorted(output_dir.iterdir()):
            if _parse_stage_dir_name(stage_dir.name) is None or not stage_dir.is_dir():
                continue
            for ckpt in sorted(stage_dir.iterdir()):
                if not ckpt.is_dir():
                    continue
                if not (ckpt / "training_state.pt").exists():
                    continue
                is_resume = resume_path is not None and ckpt.resolve() == resume_path.resolve()
                all_ckpts.append((ckpt, is_resume))

    if not all_ckpts:
        print("  [startup] No local checkpoints found; nothing to sync/prune.")
        return

    print(f"  [startup] Found {len(all_ckpts)} local checkpoint(s). Uploading to Hub before pruning...")
    for ckpt, is_resume in all_ckpts:
        stage_dir_name = ckpt.parent.name
        remote_prefix = f"{hf_stage_subdir.strip('/')}/{stage_dir_name}"
        ok = _upload_checkpoint_folder(ckpt, hf_repo_id, hf_token, remote_prefix=remote_prefix)
        resume_marker = "  (resume)" if is_resume else ""
        status = "✓" if ok else "✗ (upload failed)"
        print(f"  [startup]   {stage_dir_name}/{ckpt.name}{resume_marker}  {status}")

    pruned = 0
    for ckpt, is_resume in all_ckpts:
        if is_resume:
            continue
        shutil.rmtree(ckpt, ignore_errors=True)
        print(f"  [startup]   pruned {ckpt.parent.name}/{ckpt.name}")
        pruned += 1

    print(f"  [startup] Sync+prune complete. Pruned {pruned} checkpoint(s) locally.")


def _distributed_resume_marker(output_dir: Path) -> Path:
    return output_dir / ".resolved_resume_path.txt"


def _resolve_resume_checkpoint_for_all_ranks(
    *,
    output_dir: Path,
    requested_resume: Optional[Path],
    hf_token: Optional[str],
    hf_repo_id: str,
    hf_stage_subdir: str,
    distributed: bool,
    is_main: bool,
) -> Optional[Path]:
    marker_path = _distributed_resume_marker(output_dir)
    if is_main and marker_path.exists():
        marker_path.unlink(missing_ok=True)

    resolved: Optional[Path] = requested_resume
    if distributed:
        if is_main and resolved is None:
            resolved = find_latest_resume_checkpoint(
                output_dir,
                hf_token=hf_token,
                hf_repo_id=hf_repo_id,
                hf_stage_subdir=hf_stage_subdir,
            )
            if resolved is not None:
                print(f"  [resume] discovered latest checkpoint: {resolved}")
        if is_main:
            marker_path.write_text(
                str(resolved.resolve()) if resolved is not None else "",
                encoding="utf-8",
            )
        barrier()
        if not is_main:
            raw = marker_path.read_text(encoding="utf-8").strip() if marker_path.exists() else ""
            resolved = Path(raw) if raw else None
        barrier()
        return resolved

    if resolved is None:
        resolved = find_latest_resume_checkpoint(
            output_dir,
            hf_token=hf_token,
            hf_repo_id=hf_repo_id,
            hf_stage_subdir=hf_stage_subdir,
        )
        if resolved is not None and is_main:
            print(f"  [resume] discovered latest checkpoint: {resolved}")
    return resolved


def _cleanup_distributed_resume_artifacts(
    output_dir: Path,
    hub_resume_dir: Path,
    distributed: bool,
    is_main: bool,
) -> None:
    if distributed:
        barrier()
    if is_main:
        marker_path = _distributed_resume_marker(output_dir)
        marker_path.unlink(missing_ok=True)
        if hub_resume_dir.exists():
            shutil.rmtree(hub_resume_dir, ignore_errors=True)
    if distributed:
        barrier()
