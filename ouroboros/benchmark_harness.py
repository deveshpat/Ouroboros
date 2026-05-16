"""Launch EleutherAI lm-evaluation-harness for Ouroboros checkpoints.

This module stays intentionally thin: it prepares the private Hub adapter path,
constructs a deterministic ``lm_eval`` command, runs it, and optionally persists
results back to the Hugging Face state repo. The benchmark implementation itself
remains the upstream lm-evaluation-harness package.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

DEFAULT_BASE_MODEL = "ai21labs/AI21-Jamba-Reasoning-3B"
DEFAULT_ADAPTER_REPO = "WeirdRunner/Ouroboros"
DEFAULT_ADAPTER_SUBFOLDER = "diloco_state/anchor"
DEFAULT_TASKS = "arc_easy,hellaswag,winogrande"
DEFAULT_OUTPUT_DIR = "runs/lm_eval_benchmark"
DEFAULT_BATCH_SIZE = "1"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_DTYPE = "float16"
DEFAULT_HUB_PREFIX = "benchmarks/lm_eval"


def _normalize_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _env(env: Mapping[str, str], name: str, default: str) -> str:
    return _normalize_text(env.get(name)) or default


def _truthy(value: object | None) -> bool:
    text = _normalize_text(value)
    return text is not None and text.lower() in {"1", "true", "yes", "y", "on"}


def _resolve_hf_token(env: Mapping[str, str]) -> str | None:
    return _normalize_text(env.get("HF_TOKEN")) or _normalize_text(env.get("HUGGINGFACE_HUB_TOKEN"))


def parse_args(argv: Iterable[str] | None = None, *, env: Mapping[str, str] | None = None) -> argparse.Namespace:
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(description="Run Ouroboros lm-evaluation-harness benchmarks")
    parser.add_argument("--tasks", default=_env(env, "OUROBOROS_BENCHMARK_TASKS", DEFAULT_TASKS))
    parser.add_argument("--limit", default=_normalize_text(env.get("OUROBOROS_BENCHMARK_LIMIT")))
    parser.add_argument("--output_dir", default=_env(env, "OUROBOROS_BENCHMARK_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base_model", default=_env(env, "OUROBOROS_BENCHMARK_BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--adapter_repo", default=_env(env, "OUROBOROS_BENCHMARK_ADAPTER_REPO", DEFAULT_ADAPTER_REPO))
    parser.add_argument("--adapter_subfolder", default=_env(env, "OUROBOROS_BENCHMARK_ADAPTER_SUBFOLDER", DEFAULT_ADAPTER_SUBFOLDER))
    parser.add_argument("--batch_size", default=_env(env, "OUROBOROS_BENCHMARK_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    parser.add_argument("--device", default=_env(env, "OUROBOROS_BENCHMARK_DEVICE", DEFAULT_DEVICE))
    parser.add_argument("--dtype", default=_env(env, "OUROBOROS_BENCHMARK_DTYPE", DEFAULT_DTYPE))
    parser.add_argument("--model_args", default=_normalize_text(env.get("OUROBOROS_BENCHMARK_MODEL_ARGS")))
    parser.add_argument("--adapter_cache_dir", default=_env(env, "OUROBOROS_BENCHMARK_ADAPTER_CACHE_DIR", "/kaggle/working/ouroboros_benchmark_adapter"))
    parser.add_argument("--hub_results_prefix", default=_env(env, "OUROBOROS_BENCHMARK_HUB_PREFIX", DEFAULT_HUB_PREFIX))
    parser.add_argument("--publish_to_hub", action="store_true", default=_truthy(env.get("OUROBOROS_BENCHMARK_PUBLISH_TO_HUB")))
    parser.add_argument("--skip_install", action="store_true", default=_truthy(env.get("OUROBOROS_BENCHMARK_SKIP_INSTALL")))
    return parser.parse_args(list(argv) if argv is not None else None)


def install_lm_eval_if_needed(*, skip_install: bool = False) -> None:
    """Install the harness only when the runtime does not already have it."""
    if importlib.util.find_spec("lm_eval") is not None:
        return
    if skip_install:
        raise SystemExit("lm_eval is not installed and --skip_install was set.")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lm-eval"], check=True)


def download_adapter_snapshot(
    *,
    repo_id: str,
    subfolder: str,
    token: str | None,
    cache_dir: str,
) -> Path:
    """Download the PEFT adapter subfolder and return its local path."""
    from huggingface_hub import snapshot_download

    target = Path(cache_dir)
    subfolder = (subfolder or "").strip().strip("/")
    allow_patterns = [f"{subfolder}/*"] if subfolder and subfolder != "." else None
    snapshot_download(
        repo_id=repo_id,
        token=token,
        local_dir=str(target),
        allow_patterns=allow_patterns,
    )
    adapter_dir = target / subfolder if subfolder and subfolder != "." else target
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(
            f"No PEFT adapter_config.json found at {adapter_dir}. "
            "Check OUROBOROS_BENCHMARK_ADAPTER_REPO/SUBFOLDER."
        )
    return adapter_dir


def build_model_args(*, base_model: str, adapter_path: str | None, dtype: str, override: str | None = None) -> str:
    """Build lm-eval HF model args, unless the caller supplied an override."""
    override = _normalize_text(override)
    if override is not None:
        return override
    parts = [
        f"pretrained={base_model}",
        "trust_remote_code=True",
        f"dtype={dtype}",
    ]
    adapter_path = _normalize_text(adapter_path)
    if adapter_path is not None:
        parts.append(f"peft={adapter_path}")
    return ",".join(parts)


def build_lm_eval_argv(
    *,
    tasks: str,
    output_dir: str,
    model_args: str,
    batch_size: str,
    device: str,
    limit: str | None = None,
) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--batch_size",
        str(batch_size),
        "--device",
        device,
        "--output_path",
        output_dir,
    ]
    limit = _normalize_text(limit)
    if limit is not None:
        argv.extend(["--limit", limit])
    return argv


def publish_results_to_hub(
    *,
    output_dir: str,
    repo_id: str,
    token: str | None,
    prefix: str,
) -> str | None:
    if not token:
        print("[benchmark] HF token unavailable; leaving results local only.")
        return None
    from huggingface_hub import HfApi

    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"[benchmark] Output directory {output_path} does not exist; nothing to upload.")
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    remote_path = f"{prefix.strip('/')}/{stamp}".strip("/")
    HfApi(token=token).upload_folder(
        repo_id=repo_id,
        folder_path=str(output_path),
        path_in_repo=remote_path,
        token=token,
        commit_message=f"Upload lm-eval benchmark {stamp}",
    )
    print(f"[benchmark] uploaded {output_path} -> {repo_id}/{remote_path}")
    return remote_path


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    token = _resolve_hf_token(os.environ)
    install_lm_eval_if_needed(skip_install=bool(args.skip_install))
    adapter_path = download_adapter_snapshot(
        repo_id=args.adapter_repo,
        subfolder=args.adapter_subfolder,
        token=token,
        cache_dir=args.adapter_cache_dir,
    )
    model_args = build_model_args(
        base_model=args.base_model,
        adapter_path=str(adapter_path),
        dtype=args.dtype,
        override=args.model_args,
    )
    command = build_lm_eval_argv(
        tasks=args.tasks,
        output_dir=args.output_dir,
        model_args=model_args,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )
    print("[benchmark] " + " ".join(command))
    subprocess.run(command, check=True)
    if args.publish_to_hub:
        publish_results_to_hub(
            output_dir=args.output_dir,
            repo_id=args.adapter_repo,
            token=token,
            prefix=args.hub_results_prefix,
        )


if __name__ == "__main__":  # pragma: no cover - exercised by Kaggle runtime
    main()
