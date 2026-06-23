"""
checkpointing.py
================
Cross-session resume + Hub sync/prune for the training session.

Intra-stage resume is delegated to Trainer.train(resume_from_checkpoint=) —
this module owns only what Trainer can't: finding the latest checkpoint across
local dirs + the Hub, agreeing on it across DDP ranks, reading the stage sidecar
to know which stage to resume, and the startup Hub sync+prune that keeps Kaggle
disk from overflowing across sessions (rank-0 uploads every local checkpoint,
then deletes all numbered checkpoints except the one being resumed, always
preserving best).

JSON sidecars (ouroboros_stage.json + trainer_state.json), not torch pickles,
so startup reads them without importing torch — faster Kaggle boot, no pickle
risk. One file because resume + Hub-sync are one concern: locating and
maintaining the set of checkpoints.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional


def _distributed_resume_marker(output_dir: Path) -> Path:
    """Rank-0 writes the resolved resume path here so other ranks agree."""
    return output_dir / ".resolved_resume_path.txt"


def _parse_stage(name: str) -> Optional[int]:
    if not name.startswith("stage_"):
        return None
    suffix = name.split("stage_", 1)[1]
    return int(suffix) if suffix.isdigit() else None


def read_stage_from_sidecar(ckpt_dir: Path) -> Optional[int]:
    """Read the stage_k a checkpoint belongs to, from the JSON sidecar."""
    sidecar = Path(ckpt_dir) / "ouroboros_stage.json"
    if sidecar.exists():
        try:
            return int(json.loads(sidecar.read_text(encoding="utf-8")).get("stage_k"))
        except Exception:
            pass
    # Fall back to the parent stage_N dir name.
    parent = Path(ckpt_dir).parent.name
    return _parse_stage(parent)


def find_latest_resume_checkpoint(
    output_dir: Path,
    hf_token: Optional[str] = None,
    hf_repo_id: str = "WeirdRunner/Ouroboros",
    hf_stage_subdir: str = "runs/stage3",
) -> Optional[Path]:
    """
    Latest checkpoint by (stage_k, epoch, step_in_epoch, global_step), local
    first, then a Hub scan. A checkpoint is resumable iff it has a
    trainer_state.json (Trainer's own) — the sidecar is auxiliary.
    """
    best_path: Optional[Path] = None
    best_key: tuple = ()

    if output_dir.exists():
        for stage_dir in output_dir.iterdir():
            stage_k = _parse_stage(stage_dir.name)
            if stage_k is None or not stage_dir.is_dir():
                continue
            for ckpt in stage_dir.iterdir():
                if not ckpt.is_dir() or not (ckpt / "trainer_state.json").exists():
                    continue
                try:
                    state = json.loads((ckpt / "trainer_state.json").read_text(encoding="utf-8"))
                except Exception:
                    continue
                key = (stage_k, int(state.get("epoch", -1)),
                       int(state.get("global_step", -1)))
                if key > best_key:
                    best_key, best_path = key, ckpt

    if best_path is not None:
        return best_path
    if not hf_token:
        return None

    # Hub fallback: list remote stage checkpoints, download the newest.
    print("  [resume] No local checkpoints. Scanning Hub...")
    candidates = _list_remote_checkpoints(hf_repo_id, hf_token, hf_stage_subdir)
    if not candidates:
        return None
    hub_resume_dir = output_dir / ".hub_resume"
    for stage_k, step, rel_name in candidates:
        ckpt_name = rel_name.split("/")[-1]
        print(f"  [hub] downloading {rel_name} ...")
        downloaded = _download_remote(ckpt_name, hub_resume_dir, hf_repo_id, hf_token,
                                      f"{hf_stage_subdir}/stage_{stage_k}")
        if downloaded is not None and (downloaded / "trainer_state.json").exists():
            print(f"  [hub] using {rel_name} as resume checkpoint")
            return downloaded
    return None


def _list_remote_checkpoints(hf_repo_id: str, hf_token: str, hf_stage_subdir: str):
    try:
        from huggingface_hub import HfApi
        files = list(HfApi(token=hf_token).list_repo_files(repo_id=hf_repo_id, token=hf_token))
    except Exception:
        return []
    prefix = [p for p in hf_stage_subdir.strip("/").split("/") if p]
    found = set()
    for filename in files:
        parts = filename.split("/")
        if prefix and parts[:len(prefix)] != prefix:
            continue
        rest = parts[len(prefix):]
        if len(rest) < 2:
            continue
        stage_k = _parse_stage(rest[0])
        name = rest[1]
        if stage_k is None or not (name.startswith("checkpoint-") or name == "best"):
            continue
        step = int(name.split("-")[-1]) if name.split("-")[-1].isdigit() else 0
        found.add((stage_k, step, "/".join(rest[:2])))
    return sorted(found, key=lambda x: (x[0], x[1]), reverse=True)


def _download_remote(ckpt_name: str, local_dir: Path, hf_repo_id: str, hf_token: str, remote_prefix: str):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_path = f"{remote_prefix.strip('/')}/{ckpt_name}".strip("/")
    try:
        snapshot_download(repo_id=hf_repo_id, local_dir=str(local_dir), token=hf_token,
                          allow_patterns=[f"{remote_path}/*"])
        dest = local_dir / remote_path
        return dest if dest.exists() else None
    except Exception as exc:
        print(f"  [hub] download failed for {remote_path}: {exc}")
        return None


def load_adapter_into_model(model, ckpt_dir: Path, device) -> None:
    """Load a saved adapter (LoRA + halt_gate + resized embed/lm_head) back into an Ouroboros."""
    from peft import set_peft_model_state_dict
    adapter_dir = ckpt_dir / "adapter_model" if (ckpt_dir / "adapter_model").exists() else ckpt_dir
    for fname in ("adapter_model.safetensors", "adapter_model.bin"):
        path = adapter_dir / fname
        if not path.exists():
            continue
        if fname.endswith(".safetensors"):
            from safetensors.torch import load_file
            weights = load_file(str(path))
        else:
            import torch
            weights = torch.load(path, map_location="cpu")
        set_peft_model_state_dict(model, weights)
        # Resized embed/lm_head rows (modules_to_save) aren't lora_ keys; load them
        # separately, best-effort.
        try:
            model.load_state_dict(
                {k: v for k, v in weights.items() if "embed_tokens" in k or "lm_head" in k},
                strict=False,
            )
        except Exception:
            pass
        break
    gate_path = ckpt_dir / "halt_gate.pt"
    if model.halt_gate is not None and gate_path.exists():
        import torch
        model.halt_gate.load_state_dict(torch.load(gate_path, map_location=device))


def startup_hub_sync_and_prune(
    output_dir: Path,
    resume_path: Optional[Path],
    hf_token: str,
    hf_repo_id: str,
    hf_stage_subdir: str,
    is_main: bool,
) -> None:
    """
    Rank-0, before training: upload every local checkpoint to the Hub, then
    delete all local numbered checkpoints except the resume one (always keep
    best). Prevents Kaggle disk overflow across sessions — a single epoch's
    full-model-equivalent checkpoints would otherwise fill the disk.
    """
    if not is_main:
        return
    all_ckpts: list[tuple[Path, bool]] = []
    if output_dir.exists():
        for stage_dir in sorted(output_dir.iterdir()):
            if _parse_stage(stage_dir.name) is None or not stage_dir.is_dir():
                continue
            for ckpt in sorted(stage_dir.iterdir()):
                if not ckpt.is_dir() or not (ckpt / "trainer_state.json").exists():
                    continue
                is_resume = resume_path is not None and ckpt.resolve() == resume_path.resolve()
                all_ckpts.append((ckpt, is_resume))

    if not all_ckpts:
        print("  [startup] No local checkpoints; nothing to sync/prune.")
        return

    print(f"  [startup] {len(all_ckpts)} local checkpoint(s). Uploading to Hub before pruning...")
    for ckpt, is_resume in all_ckpts:
        remote_prefix = f"{hf_stage_subdir.strip('/')}/{ckpt.parent.name}"
        ok = _upload_folder(ckpt, hf_repo_id, hf_token, remote_prefix)
        tag = "  (resume)" if is_resume else ""
        print(f"  [startup]   {ckpt.parent.name}/{ckpt.name}{tag}  {'✓' if ok else '✗'}")

    pruned = 0
    for ckpt, is_resume in all_ckpts:
        if is_resume:
            continue
        shutil.rmtree(ckpt, ignore_errors=True)
        pruned += 1
    print(f"  [startup] Sync+prune complete. Pruned {pruned} locally.")


def _upload_folder(ckpt_dir: Path, hf_repo_id: str, hf_token: str, remote_prefix: str) -> bool:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hf_repo_id, private=True, exist_ok=True, token=hf_token)
        remote_name = f"{remote_prefix.strip('/')}/{ckpt_dir.name}".strip("/")
        future = api.upload_folder(repo_id=hf_repo_id, folder_path=str(ckpt_dir),
                                   path_in_repo=remote_name, token=hf_token,
                                   commit_message=f"Upload {ckpt_dir.name}", run_as_future=True)
        future.result(timeout=300.0)
        return True
    except Exception as exc:
        print(f"  [hub] upload failed for {ckpt_dir.name}: {exc}")
        return False
