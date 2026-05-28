"""Checkpoint, release bundle, Hub upload, and optional W&B helpers."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch

from ouroboros.config import DEFAULT_BASE_MODEL, DEFAULT_LATENT_TOKEN
from ouroboros.latent import HaltGate, OuroborosCoconutForCausalLM
from ouroboros.utils import default_hf_token


def write_model_card(
    bundle_dir: Path,
    *,
    base_model: str,
    stage: int,
    latent_token: str,
    has_halt_gate: bool,
) -> None:
    card = f"""---
library_name: peft
base_model: {base_model}
tags:
- peft
- lora
- coconut
- latent-reasoning
---

# Ouroboros

This repo contains the PEFT adapter, tokenizer, and lightweight runtime files for
the simplified Ouroboros latent-reasoning experiment.

Runtime facts:

- Base model: `{base_model}`
- Latent token: `{latent_token}`
- Export stage: `{stage}`
- DGAC HaltGate exported: `{has_halt_gate}`

Load with `ouroboros.latent.load_published_coconut` and generate with
`ouroboros.generation.generate`.
"""
    (bundle_dir / "README.md").write_text(card, encoding="utf-8")


def save_release_bundle(
    model: OuroborosCoconutForCausalLM,
    tokenizer,
    bundle_dir: str | Path,
    *,
    base_model: str,
    stage: int,
    latent_token: str = DEFAULT_LATENT_TOKEN,
    halt_gate: HaltGate | None = None,
) -> Path:
    bundle = Path(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(bundle, safe_serialization=True)
    tokenizer.save_pretrained(bundle)
    if halt_gate is not None:
        torch.save(halt_gate.state_dict(), bundle / "halt_gate.pt")
    write_model_card(
        bundle,
        base_model=base_model,
        stage=stage,
        latent_token=latent_token,
        has_halt_gate=halt_gate is not None,
    )
    for name in ("config.py", "data.py", "latent.py", "generation.py"):
        shutil.copyfile(Path(__file__).with_name(name), bundle / name)
    return bundle


def push_release_bundle(
    bundle_dir: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = True,
    commit_message: str = "Release Ouroboros bundle",
) -> str:
    from huggingface_hub import HfApi

    token = default_hf_token(token)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, token=token)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(bundle_dir),
        token=token,
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"


def maybe_init_wandb(args: argparse.Namespace):
    if getattr(args, "wandb_mode", "disabled") == "disabled":
        return None
    import wandb

    return wandb.init(
        project=getattr(args, "wandb_project", "ouroboros"),
        name=getattr(args, "wandb_run_name", None),
        mode=getattr(args, "wandb_mode", "online"),
        config=vars(args),
    )


def build_publish_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Push an existing Ouroboros release bundle to Hugging Face Hub.")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--hub-model-id", required=True)
    parser.add_argument("--hub-token", default=None)
    parser.add_argument("--hub-public", dest="hub_private", action="store_false")
    parser.set_defaults(hub_private=True)
    return parser


def publish_main(argv: list[str] | None = None) -> None:
    args = build_publish_parser().parse_args(argv)
    url = push_release_bundle(
        args.bundle_dir,
        args.hub_model_id,
        token=args.hub_token or os.environ.get("HF_TOKEN"),
        private=bool(args.hub_private),
    )
    print(f"[ouroboros] pushed -> {url}")
