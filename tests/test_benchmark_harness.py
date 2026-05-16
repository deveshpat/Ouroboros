from __future__ import annotations

import sys

import torch

from ouroboros.benchmark_harness import (
    _filter_vocab_mismatched_weights,
    build_lm_eval_argv,
    build_model_args,
    parse_args,
)


def test_benchmark_harness_parses_env_defaults_without_importing_lm_eval():
    args = parse_args(
        [],
        env={
            "OUROBOROS_BENCHMARK_TASKS": "arc_easy",
            "OUROBOROS_BENCHMARK_LIMIT": "10",
            "OUROBOROS_BENCHMARK_OUTPUT_DIR": "runs/bench",
            "OUROBOROS_BENCHMARK_BASE_MODEL": "base/model",
            "OUROBOROS_BENCHMARK_ADAPTER_REPO": "adapter/repo",
            "OUROBOROS_BENCHMARK_ADAPTER_SUBFOLDER": "anchor",
            "OUROBOROS_BENCHMARK_BATCH_SIZE": "auto",
            "OUROBOROS_BENCHMARK_DEVICE": "cuda:0",
            "OUROBOROS_BENCHMARK_DTYPE": "bfloat16",
            "OUROBOROS_BENCHMARK_MODEL_ARGS": "pretrained=merged/model",
            "OUROBOROS_BENCHMARK_PUBLISH_TO_HUB": "1",
        },
    )

    assert args.tasks == "arc_easy"
    assert args.limit == "10"
    assert args.output_dir == "runs/bench"
    assert args.base_model == "base/model"
    assert args.adapter_repo == "adapter/repo"
    assert args.adapter_subfolder == "anchor"
    assert args.batch_size == "auto"
    assert args.dtype == "bfloat16"
    assert args.model_args == "pretrained=merged/model"
    assert args.bootstrap_lm_eval is True
    assert args.publish_to_hub is True


def test_build_model_args_targets_base_model_plus_local_peft_adapter():
    model_args = build_model_args(
        base_model="base/model",
        adapter_path="/tmp/anchor",
        dtype="float16",
    )

    assert model_args == "pretrained=base/model,trust_remote_code=True,dtype=float16,peft=/tmp/anchor"


def test_build_model_args_respects_raw_override():
    assert (
        build_model_args(
            base_model="ignored",
            adapter_path="ignored",
            dtype="ignored",
            override="pretrained=merged/model,trust_remote_code=True",
        )
        == "pretrained=merged/model,trust_remote_code=True"
    )


def test_benchmark_harness_can_disable_lm_eval_bootstrap():
    args = parse_args(["--no_bootstrap_lm_eval"], env={})

    assert args.bootstrap_lm_eval is False


def test_build_lm_eval_argv_is_reproducible_and_keeps_limit_optional():
    command = build_lm_eval_argv(
        tasks="arc_easy,hellaswag",
        output_dir="runs/bench",
        model_args="pretrained=base,peft=/tmp/adapter",
        batch_size="1",
        device="cuda:0",
        limit="100",
    )

    assert command[:3] == [sys.executable, "-m", "ouroboros.lm_eval_bootstrap"]
    assert command[command.index("--model") + 1] == "hf"
    assert command[command.index("--tasks") + 1] == "arc_easy,hellaswag"
    assert command[command.index("--output_path") + 1] == "runs/bench"
    assert command[command.index("--limit") + 1] == "100"

    no_limit = build_lm_eval_argv(
        tasks="arc_easy",
        output_dir="runs/bench",
        model_args="pretrained=base",
        batch_size="1",
        device="cuda:0",
        limit="",
        bootstrap_lm_eval=False,
    )
    assert no_limit[:3] == [sys.executable, "-m", "lm_eval"]
    assert "--limit" not in no_limit


def test_build_lm_eval_argv_treats_full_limit_sentinels_as_unlimited():
    command = build_lm_eval_argv(
        tasks="arc_easy",
        output_dir="runs/bench",
        model_args="pretrained=base",
        batch_size="1",
        device="cuda:0",
        limit="full",
    )

    assert "--limit" not in command


def test_benchmark_harness_sanitizes_only_vocab_resized_adapter_tensors():
    weights = {
        "base_model.model.model.embed_tokens.weight": torch.zeros(65537, 4),
        "base_model.model.lm_head.weight": torch.zeros(65537, 4),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(8, 4),
    }

    filtered, removed = _filter_vocab_mismatched_weights(weights, base_vocab_size=65536)

    assert removed == [
        "base_model.model.model.embed_tokens.weight",
        "base_model.model.lm_head.weight",
    ]
    assert set(filtered) == {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"}


def test_benchmark_harness_keeps_matching_vocab_tensors():
    weights = {
        "base_model.model.model.embed_tokens.weight": torch.zeros(65536, 4),
        "base_model.model.lm_head.weight": torch.zeros(65536, 4),
    }

    filtered, removed = _filter_vocab_mismatched_weights(weights, base_vocab_size=65536)

    assert removed == []
    assert set(filtered) == set(weights)
