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
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

import torch

from ouroboros.runtime_env import normalize_benchmark_limit

DEFAULT_BASE_MODEL = "ai21labs/AI21-Jamba-Reasoning-3B"
DEFAULT_ADAPTER_REPO = "WeirdRunner/Ouroboros"
DEFAULT_ADAPTER_SUBFOLDER = "diloco_state/anchor"
DEFAULT_TASKS = "arc_easy,hellaswag,winogrande"
DEFAULT_OUTPUT_DIR = "runs/lm_eval_benchmark"
DEFAULT_BATCH_SIZE = "1"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_DTYPE = "float16"
DEFAULT_HUB_PREFIX = "benchmarks/lm_eval"
DEFAULT_BOOTSTRAP_LM_EVAL = True
_VOCAB_WEIGHT_SUFFIXES = ("embed_tokens.weight", "lm_head.weight")


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


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    text = _normalize_text(env.get(name))
    if text is None:
        return default
    return text.lower() in {"1", "true", "yes", "y", "on"}


def _resolve_hf_token(env: Mapping[str, str]) -> str | None:
    return _normalize_text(env.get("HF_TOKEN")) or _normalize_text(env.get("HUGGINGFACE_HUB_TOKEN"))


def parse_args(argv: Iterable[str] | None = None, *, env: Mapping[str, str] | None = None) -> argparse.Namespace:
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(description="Run Ouroboros lm-evaluation-harness benchmarks")
    parser.add_argument("--tasks", default=_env(env, "OUROBOROS_BENCHMARK_TASKS", DEFAULT_TASKS))
    parser.add_argument("--limit", default=normalize_benchmark_limit(env.get("OUROBOROS_BENCHMARK_LIMIT")))
    parser.add_argument("--output_dir", default=_env(env, "OUROBOROS_BENCHMARK_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base_model", default=_env(env, "OUROBOROS_BENCHMARK_BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--adapter_repo", default=_env(env, "OUROBOROS_BENCHMARK_ADAPTER_REPO", DEFAULT_ADAPTER_REPO))
    parser.add_argument("--adapter_subfolder", default=_env(env, "OUROBOROS_BENCHMARK_ADAPTER_SUBFOLDER", DEFAULT_ADAPTER_SUBFOLDER))
    parser.add_argument("--batch_size", default=_env(env, "OUROBOROS_BENCHMARK_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    parser.add_argument("--device", default=_env(env, "OUROBOROS_BENCHMARK_DEVICE", DEFAULT_DEVICE))
    parser.add_argument("--dtype", default=_env(env, "OUROBOROS_BENCHMARK_DTYPE", DEFAULT_DTYPE))
    parser.add_argument("--model_args", default=_normalize_text(env.get("OUROBOROS_BENCHMARK_MODEL_ARGS")))
    parser.add_argument(
        "--bootstrap_lm_eval",
        dest="bootstrap_lm_eval",
        action="store_true",
        default=_env_bool(env, "OUROBOROS_BENCHMARK_BOOTSTRAP_LM_EVAL", DEFAULT_BOOTSTRAP_LM_EVAL),
        help=(
            "Run lm-eval through the Ouroboros bootstrap wrapper so Jamba CUDA "
            "Mamba kernels are installed and patched inside the child process. "
            "Enabled by default for the default Jamba benchmark path."
        ),
    )
    parser.add_argument(
        "--no_bootstrap_lm_eval",
        dest="bootstrap_lm_eval",
        action="store_false",
        help="Call upstream python -m lm_eval directly without the Ouroboros bootstrap wrapper.",
    )
    parser.add_argument("--adapter_cache_dir", default=_env(env, "OUROBOROS_BENCHMARK_ADAPTER_CACHE_DIR", "/kaggle/working/ouroboros_benchmark_adapter"))
    parser.add_argument("--hub_results_prefix", default=_env(env, "OUROBOROS_BENCHMARK_HUB_PREFIX", DEFAULT_HUB_PREFIX))
    parser.add_argument("--publish_to_hub", action="store_true", default=_truthy(env.get("OUROBOROS_BENCHMARK_PUBLISH_TO_HUB")))
    parser.add_argument("--skip_install", action="store_true", default=_truthy(env.get("OUROBOROS_BENCHMARK_SKIP_INSTALL")))
    parser.add_argument(
        "--sanitize_vocab_mismatch",
        dest="sanitize_vocab_mismatch",
        action="store_true",
        default=_env_bool(env, "OUROBOROS_BENCHMARK_SANITIZE_VOCAB_MISMATCH", True),
        help=(
            "Create an lm-eval adapter copy without saved embedding/lm_head tensors "
            "when the DiLoCo anchor vocab includes Ouroboros' extra <|lat|> token. "
            "Enabled by default because external text benchmarks do not use <|lat|>."
        ),
    )
    parser.add_argument(
        "--no_sanitize_vocab_mismatch",
        dest="sanitize_vocab_mismatch",
        action="store_false",
        help="Disable adapter vocab-mismatch sanitization for raw PEFT loading experiments.",
    )
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



def _adapter_weight_path(adapter_dir: Path) -> Path | None:
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter_dir / filename
        if candidate.exists():
            return candidate
    return None


def _infer_base_vocab_size(base_model: str, token: str | None) -> int | None:
    """Read the base model vocab size without loading model weights."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(base_model, token=token, trust_remote_code=True)
        vocab_size = getattr(config, "vocab_size", None)
        return int(vocab_size) if vocab_size is not None else None
    except Exception as exc:  # noqa: BLE001 - benchmark should still try raw PEFT if metadata is unavailable
        print(f"[benchmark] Could not infer base vocab size for {base_model}: {exc}")
        return None


def _filter_vocab_mismatched_weights(
    weights: Mapping[str, torch.Tensor],
    *,
    base_vocab_size: int | None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Drop saved token-embedding tensors that cannot fit the base lm-eval model.

    Ouroboros training adds one latent token, so DiLoCo PEFT anchors can contain
    65537-row ``embed_tokens``/``lm_head`` tensors while upstream Jamba starts at
    65536 rows. lm-evaluation-harness initializes the base model before applying
    PEFT and does not know to resize for the private ``<|lat|>`` token. External
    text benchmarks never emit that token, so keeping the normal base embeddings
    and loading only the LoRA adapter weights is the safe adapter-only baseline.
    """
    if base_vocab_size is None:
        return dict(weights), []

    filtered: dict[str, torch.Tensor] = {}
    removed: list[str] = []
    for key, value in weights.items():
        is_vocab_weight = any(key.endswith(suffix) for suffix in _VOCAB_WEIGHT_SUFFIXES)
        if is_vocab_weight and getattr(value, "ndim", 0) >= 2 and int(value.shape[0]) != int(base_vocab_size):
            removed.append(key)
            continue
        filtered[key] = value
    return filtered, removed


def prepare_adapter_for_lm_eval(
    *,
    adapter_dir: Path,
    base_model: str,
    token: str | None,
    enabled: bool = True,
) -> Path:
    """Return a PEFT adapter path safe for vanilla lm-eval HF loading."""
    if not enabled:
        return adapter_dir

    weight_path = _adapter_weight_path(adapter_dir)
    if weight_path is None:
        return adapter_dir

    base_vocab_size = _infer_base_vocab_size(base_model, token)
    if base_vocab_size is None:
        return adapter_dir

    if weight_path.suffix == ".safetensors":
        from safetensors.torch import load_file, save_file

        weights = load_file(str(weight_path), device="cpu")
        filtered, removed = _filter_vocab_mismatched_weights(weights, base_vocab_size=base_vocab_size)
        if not removed:
            return adapter_dir

        sanitized_dir = adapter_dir.parent / f"{adapter_dir.name}_lm_eval_sanitized"
        if sanitized_dir.exists():
            shutil.rmtree(sanitized_dir)
        shutil.copytree(adapter_dir, sanitized_dir)
        save_file(filtered, str(sanitized_dir / weight_path.name), metadata={"format": "pt"})
    else:
        weights = torch.load(weight_path, map_location="cpu")
        if not isinstance(weights, Mapping):
            return adapter_dir
        filtered, removed = _filter_vocab_mismatched_weights(weights, base_vocab_size=base_vocab_size)
        if not removed:
            return adapter_dir

        sanitized_dir = adapter_dir.parent / f"{adapter_dir.name}_lm_eval_sanitized"
        if sanitized_dir.exists():
            shutil.rmtree(sanitized_dir)
        shutil.copytree(adapter_dir, sanitized_dir)
        torch.save(filtered, sanitized_dir / weight_path.name)

    print(
        "[benchmark] Created lm-eval adapter copy without vocab-resized tensors: "
        f"{sanitized_dir} (removed: {', '.join(removed)})"
    )
    return sanitized_dir


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
    bootstrap_lm_eval: bool = True,
) -> list[str]:
    module = "ouroboros.lm_eval_bootstrap" if bootstrap_lm_eval else "lm_eval"
    argv = [
        sys.executable,
        "-m",
        module,
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
    limit = normalize_benchmark_limit(limit)
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
    adapter_path = prepare_adapter_for_lm_eval(
        adapter_dir=adapter_path,
        base_model=args.base_model,
        token=token,
        enabled=bool(args.sanitize_vocab_mismatch),
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
        bootstrap_lm_eval=bool(args.bootstrap_lm_eval),
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
