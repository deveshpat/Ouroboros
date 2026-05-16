from __future__ import annotations

import pytest

from ouroboros.eval.benchmark_multi_gpu import (
    build_sharded_lm_eval_benchmark_commands,
    resolve_benchmark_devices,
    shard_tasks,
)


def _arg_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_shard_tasks_round_robins_default_benchmark_across_two_gpus():
    assert shard_tasks("arc_easy,hellaswag,winogrande", ["cuda:0", "cuda:1"]) == [
        ["arc_easy", "winogrande"],
        ["hellaswag"],
    ]


def test_sharded_commands_use_unique_devices_outputs_and_adapter_caches():
    commands = build_sharded_lm_eval_benchmark_commands(
        tasks="arc_easy,hellaswag,winogrande",
        devices="cuda:0,cuda:1",
        limit="100",
        output_dir="runs/lm_eval_benchmark",
        publish_to_hub=False,
    )

    assert len(commands) == 2
    assert _arg_value(commands[0], "--tasks") == "arc_easy,winogrande"
    assert _arg_value(commands[0], "--device") == "cuda:0"
    assert _arg_value(commands[0], "--output_dir") == "runs/lm_eval_benchmark/shard-0-cuda-0"
    assert _arg_value(commands[0], "--adapter_cache_dir") == "/kaggle/working/ouroboros_benchmark_adapter_0_cuda-0"
    assert _arg_value(commands[1], "--tasks") == "hellaswag"
    assert _arg_value(commands[1], "--device") == "cuda:1"
    assert _arg_value(commands[1], "--output_dir") == "runs/lm_eval_benchmark/shard-1-cuda-1"
    assert _arg_value(commands[1], "--adapter_cache_dir") == "/kaggle/working/ouroboros_benchmark_adapter_1_cuda-1"
    assert all("--publish_to_hub" not in command for command in commands)


def test_shard_tasks_rejects_empty_tasks_or_devices():
    with pytest.raises(ValueError, match="At least one benchmark task"):
        shard_tasks("", ["cuda:0"])
    with pytest.raises(ValueError, match="At least one benchmark device"):
        shard_tasks("arc_easy", [])


def test_resolve_benchmark_devices_auto_detects_cuda_visible_devices(monkeypatch):
    monkeypatch.setattr("ouroboros.eval.benchmark_multi_gpu._torch_cuda_device_count", lambda: 0)
    monkeypatch.setattr("ouroboros.eval.benchmark_multi_gpu._nvidia_smi_device_count", lambda: 0)

    assert resolve_benchmark_devices("auto", env={"CUDA_VISIBLE_DEVICES": "0,1"}) == [
        "cuda:0",
        "cuda:1",
    ]


def test_resolve_benchmark_devices_keeps_explicit_runtime_override():
    assert resolve_benchmark_devices("cuda:0", env={"CUDA_VISIBLE_DEVICES": "0,1"}) == ["cuda:0"]
