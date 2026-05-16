"""Run Ouroboros lm-eval benchmarks across multiple Kaggle GPUs.

This wrapper keeps each lm-evaluation-harness process single-GPU, then shards
whole tasks across devices. That is safer for PEFT/Jamba than trying to make one
lm-eval process data-parallel, and it lets Kaggle's T4 x2 allocation run two
independent benchmark shards at once.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import threading
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ouroboros.benchmark_harness import (
    DEFAULT_ADAPTER_REPO,
    DEFAULT_ADAPTER_SUBFOLDER,
    DEFAULT_BASE_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DTYPE,
    DEFAULT_HUB_PREFIX,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TASKS,
    _normalize_text,
    _resolve_hf_token,
    _truthy,
    install_lm_eval_if_needed,
    publish_results_to_hub,
)
from ouroboros.kaggle import build_lm_eval_benchmark_command
from ouroboros.runtime_env import normalize_benchmark_limit

DEFAULT_DEVICES = "auto"


def _env(env: Mapping[str, str], name: str, default: str) -> str:
    return _normalize_text(env.get(name)) or default


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _visible_cuda_count_from_env(env: Mapping[str, str]) -> int:
    visible = _normalize_text(env.get("CUDA_VISIBLE_DEVICES"))
    if visible is None or visible.lower() in {"none", "void", "-1"}:
        return 0
    return len([part for part in _split_csv(visible) if part != "-1"])


def _torch_cuda_device_count() -> int:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return 0
    try:
        if not bool(torch.cuda.is_available()):
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _nvidia_smi_device_count() -> int:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return 0
    if completed.returncode != 0:
        return 0
    return len([line for line in completed.stdout.splitlines() if line.strip().startswith("GPU ")])


def detect_cuda_devices(env: Mapping[str, str] | None = None) -> list[str]:
    """Detect runtime CUDA devices using Kaggle-visible process state.

    Kaggle/GitHub should not have to pass ``cuda:0,cuda:1`` through workflow
    inputs. The benchmark launcher resolves visible devices inside the runtime,
    matching the training path's notebook-local hardware assumptions.
    """
    env = os.environ if env is None else env
    count = _torch_cuda_device_count() or _visible_cuda_count_from_env(env) or _nvidia_smi_device_count()
    if count <= 0:
        fallback = _normalize_text(env.get("OUROBOROS_BENCHMARK_DEVICE")) or "cuda:0"
        return [fallback]
    return [f"cuda:{index}" for index in range(count)]


def resolve_benchmark_devices(
    requested: str | Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Resolve explicit, env, or auto-detected benchmark devices."""
    env = os.environ if env is None else env
    if requested is not None and not isinstance(requested, str):
        devices = [str(device).strip() for device in requested if str(device).strip()]
    else:
        text = _normalize_text(requested)
        if text is None:
            text = _normalize_text(env.get("OUROBOROS_BENCHMARK_DEVICES")) or DEFAULT_DEVICES
        devices = [] if text.lower() == "auto" else _split_csv(text)
    if devices:
        return devices
    return detect_cuda_devices(env)


def _safe_device_label(device: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in device).strip("-") or "device"


def shard_tasks(tasks: str, devices: Sequence[str]) -> list[list[str]]:
    """Split comma-separated lm-eval tasks into non-empty device shards.

    Round-robin gives the default benchmark a useful balance:
    ``arc_easy,winogrande`` on GPU 0 and ``hellaswag`` on GPU 1.
    """
    task_names = _split_csv(tasks)
    device_names = [device for device in devices if str(device).strip()]
    if not task_names:
        raise ValueError("At least one benchmark task is required.")
    if not device_names:
        raise ValueError("At least one benchmark device is required.")

    shards: list[list[str]] = [[] for _ in range(min(len(task_names), len(device_names)))]
    for index, task in enumerate(task_names):
        shards[index % len(shards)].append(task)
    return shards


def build_sharded_lm_eval_benchmark_commands(
    *,
    tasks: str = DEFAULT_TASKS,
    devices: str | Sequence[str] | None = None,
    limit: str | int | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    base_model: str = DEFAULT_BASE_MODEL,
    adapter_repo: str = DEFAULT_ADAPTER_REPO,
    adapter_subfolder: str = DEFAULT_ADAPTER_SUBFOLDER,
    batch_size: str = DEFAULT_BATCH_SIZE,
    dtype: str = DEFAULT_DTYPE,
    model_args: str | None = None,
    publish_to_hub: bool = False,
) -> list[list[str]]:
    """Build one benchmark command per active GPU/task shard."""
    device_names = resolve_benchmark_devices(devices)
    shards = shard_tasks(tasks, device_names)
    root = Path(output_dir)
    commands: list[list[str]] = []
    for index, shard in enumerate(shards):
        device = device_names[index]
        label = _safe_device_label(device)
        commands.append(
            build_lm_eval_benchmark_command(
                tasks=",".join(shard),
                limit=limit,
                output_dir=str(root / f"shard-{index}-{label}"),
                base_model=base_model,
                adapter_repo=adapter_repo,
                adapter_subfolder=adapter_subfolder,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                model_args=model_args,
                publish_to_hub=publish_to_hub,
                adapter_cache_dir=f"/kaggle/working/ouroboros_benchmark_adapter_{index}_{label}",
            )
        )
    return commands


def parse_args(argv: Iterable[str] | None = None, *, env: Mapping[str, str] | None = None) -> argparse.Namespace:
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(description="Run Ouroboros lm-eval benchmarks across multiple GPUs")
    parser.add_argument("--tasks", default=_env(env, "OUROBOROS_BENCHMARK_TASKS", DEFAULT_TASKS))
    parser.add_argument("--devices", default=_env(env, "OUROBOROS_BENCHMARK_DEVICES", DEFAULT_DEVICES), help="Comma-separated devices or auto")
    parser.add_argument("--limit", default=normalize_benchmark_limit(env.get("OUROBOROS_BENCHMARK_LIMIT")))
    parser.add_argument("--output_dir", default=_env(env, "OUROBOROS_BENCHMARK_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base_model", default=_env(env, "OUROBOROS_BENCHMARK_BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--adapter_repo", default=_env(env, "OUROBOROS_BENCHMARK_ADAPTER_REPO", DEFAULT_ADAPTER_REPO))
    parser.add_argument("--adapter_subfolder", default=_env(env, "OUROBOROS_BENCHMARK_ADAPTER_SUBFOLDER", DEFAULT_ADAPTER_SUBFOLDER))
    parser.add_argument("--batch_size", default=_env(env, "OUROBOROS_BENCHMARK_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    parser.add_argument("--dtype", default=_env(env, "OUROBOROS_BENCHMARK_DTYPE", DEFAULT_DTYPE))
    parser.add_argument("--model_args", default=_normalize_text(env.get("OUROBOROS_BENCHMARK_MODEL_ARGS")))
    parser.add_argument("--hub_results_prefix", default=_env(env, "OUROBOROS_BENCHMARK_HUB_PREFIX", DEFAULT_HUB_PREFIX))
    parser.add_argument("--publish_to_hub", action="store_true", default=_truthy(env.get("OUROBOROS_BENCHMARK_PUBLISH_TO_HUB")))
    parser.add_argument("--skip_install", action="store_true", default=_truthy(env.get("OUROBOROS_BENCHMARK_SKIP_INSTALL")))
    return parser.parse_args(list(argv) if argv is not None else None)


def _stream_output(prefix: str, pipe: object) -> None:
    if pipe is None:
        return
    for line in pipe:  # type: ignore[union-attr]
        print(f"[{prefix}] {line}", end="", flush=True)


def run_commands_parallel(commands: Sequence[Sequence[str]]) -> None:
    processes: list[tuple[str, subprocess.Popen[str], threading.Thread]] = []
    for index, command in enumerate(commands):
        prefix = f"bench-shard-{index}"
        print(f"[{prefix}] launch: {' '.join(command)}", flush=True)
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        thread = threading.Thread(target=_stream_output, args=(prefix, process.stdout), daemon=True)
        thread.start()
        processes.append((prefix, process, thread))

    failures: list[tuple[str, int]] = []
    for prefix, process, thread in processes:
        return_code = process.wait()
        thread.join(timeout=5)
        if return_code != 0:
            failures.append((prefix, return_code))
    if failures:
        details = ", ".join(f"{prefix}={code}" for prefix, code in failures)
        raise SystemExit(f"Benchmark shard failure(s): {details}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    devices = resolve_benchmark_devices(args.devices)
    commands = build_sharded_lm_eval_benchmark_commands(
        tasks=args.tasks,
        devices=devices,
        limit=args.limit,
        output_dir=args.output_dir,
        base_model=args.base_model,
        adapter_repo=args.adapter_repo,
        adapter_subfolder=args.adapter_subfolder,
        batch_size=args.batch_size,
        dtype=args.dtype,
        model_args=args.model_args,
        publish_to_hub=False,
    )
    print(
        "[benchmark-multi-gpu] "
        f"tasks={args.tasks} devices={','.join(devices)} shards={len(commands)} output_dir={args.output_dir}",
        flush=True,
    )
    install_lm_eval_if_needed(skip_install=bool(args.skip_install))
    run_commands_parallel(commands)
    if args.publish_to_hub:
        publish_results_to_hub(
            output_dir=args.output_dir,
            repo_id=args.adapter_repo,
            token=_resolve_hf_token(os.environ),
            prefix=args.hub_results_prefix,
        )


if __name__ == "__main__":  # pragma: no cover - exercised by Kaggle runtime
    main()
