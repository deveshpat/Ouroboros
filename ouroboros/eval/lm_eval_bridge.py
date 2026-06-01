"""Thin lm-evaluation-harness launcher.

This bridge intentionally uses EleutherAI's stock CLI instead of reimplementing
benchmark plumbing. It is for standard HF/PEFT benchmark smoke tests. The
latent-aware generated-answer harness remains in ``compare-coconut-val`` until a
faithful lm-eval model wrapper supports Coconut latent passes and loglikelihood.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ouroboros.utils.runtime_env import resolve_hf_token


def _resolve_adapter_for_lm_eval(args: argparse.Namespace) -> str | None:
    """Return a PEFT path/repo that lm-eval can pass directly to PEFT.

    lm-eval's stock HF backend is intentionally kept in charge of benchmark
    execution. When an adapter lives in a Hub subfolder, resolve that subfolder
    first with Hugging Face Hub so the harness receives a normal local PEFT
    adapter directory containing ``adapter_config.json``.
    """
    adapter = (args.adapter or "").strip()
    if not adapter:
        return None

    subfolder = (getattr(args, "adapter_subfolder", "") or "").strip().strip("/")
    local_adapter = Path(adapter).expanduser()
    if local_adapter.exists():
        resolved = local_adapter / subfolder if subfolder else local_adapter
    elif subfolder:
        from huggingface_hub import snapshot_download

        cache_dir = Path(getattr(args, "adapter_cache_dir", "") or "runs/lm_eval_adapter").expanduser()
        snapshot_download(
            repo_id=adapter,
            token=resolve_hf_token(env=os.environ),
            local_dir=str(cache_dir),
            allow_patterns=[f"{subfolder}/*"],
        )
        resolved = cache_dir / subfolder
    else:
        return adapter

    if not (resolved / "adapter_config.json").exists():
        raise SystemExit(f"No PEFT adapter_config.json found at {resolved}.")
    return str(resolved)


def _model_args(args: argparse.Namespace, resolved_adapter: str | None) -> str:
    pairs = [
        ("pretrained", args.model_id),
        ("trust_remote_code", "True" if args.trust_remote_code else "False"),
    ]
    if resolved_adapter:
        pairs.append(("peft", resolved_adapter))
    if args.dtype:
        pairs.append(("dtype", args.dtype))
    if args.load_in_4bit:
        pairs.append(("load_in_4bit", "True"))
    if args.extra_model_args:
        raw = args.extra_model_args.strip().strip(",")
        extra = f",{raw}" if raw else ""
    else:
        extra = ""
    return ",".join(f"{key}={value}" for key, value in pairs) + extra


def _write_run_config(args: argparse.Namespace, command: list[str], resolved_adapter: str | None) -> None:
    if not args.output_path:
        return
    output = Path(args.output_path)
    output.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "runtime": "lm-evaluation-harness stock hf backend",
        "model_id": args.model_id,
        "adapter": args.adapter,
        "adapter_subfolder": getattr(args, "adapter_subfolder", ""),
        "resolved_adapter": resolved_adapter,
        "tasks": args.tasks,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "device": args.device,
        "dtype": args.dtype,
        "load_in_4bit": bool(args.load_in_4bit),
        "command": command,
        "boundary": (
            "This is a standard HF/PEFT lm-eval smoke path. It does not execute "
            "Ouroboros Coconut latent passes."
        ),
    }
    (output / "ouroboros_lm_eval_run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def run_lm_eval_hf(args: argparse.Namespace) -> None:
    resolved_adapter = _resolve_adapter_for_lm_eval(args)
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        _model_args(args, resolved_adapter),
        "--tasks",
        args.tasks,
        "--batch_size",
        str(args.batch_size),
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    if args.output_path:
        command.extend(["--output_path", args.output_path])
    _write_run_config(args, command, resolved_adapter)
    print("[lm-eval] " + " ".join(command))
    subprocess.run(command, check=True)


__all__ = ("run_lm_eval_hf",)
