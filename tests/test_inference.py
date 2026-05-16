from __future__ import annotations

from types import SimpleNamespace

import torch

from ouroboros.inference import (
    build_generation_args,
    format_prompt,
    parse_args,
    resolve_device,
    run_single_prompt,
)
from tests.fakes.eval_fakes import FakeCausalLM, FakeHaltGate, FakeTokenizer


def test_inference_parses_env_defaults_without_loading_transformers():
    args = parse_args(
        [],
        env={
            "OUROBOROS_INFERENCE_PROMPT": "What is 2+2?",
            "OUROBOROS_INFERENCE_BASE_MODEL": "base/model",
            "OUROBOROS_INFERENCE_ADAPTER_REPO": "adapter/repo",
            "OUROBOROS_INFERENCE_ADAPTER_SUBFOLDER": "anchor",
            "OUROBOROS_INFERENCE_STAGE_K": "7",
            "OUROBOROS_INFERENCE_MAX_NEW_TOKENS": "12",
            "OUROBOROS_INFERENCE_USE_HALT_GATE": "0",
        },
    )

    assert args.prompt == "What is 2+2?"
    assert args.base_model == "base/model"
    assert args.adapter_repo == "adapter/repo"
    assert args.adapter_subfolder == "anchor"
    assert args.stage_k == 7
    assert args.max_new_tokens == 12
    assert args.use_halt_gate is False


def test_build_generation_args_provides_latent_decode_contract():
    args = SimpleNamespace(max_new_tokens=5, max_seq_len=64, halt_threshold=0.25)

    built = build_generation_args(args)

    assert built.gen_max_tokens == 5
    assert built.max_seq_len == 64
    assert built.halt_threshold == 0.25
    assert built.latent_cache is False


def test_format_prompt_uses_chat_template_when_requested():
    tokenizer = FakeTokenizer()

    formatted = format_prompt(tokenizer, "Hello", use_chat_template=True)

    assert "Hello" in formatted
    assert "Assistant:" in formatted


def test_run_single_prompt_executes_fixed_depth_latent_inference_on_cpu():
    model = FakeCausalLM()
    tokenizer = FakeTokenizer()
    args = SimpleNamespace(max_new_tokens=3, max_seq_len=64, halt_threshold=0.5)

    result = run_single_prompt(
        model=model,
        tokenizer=tokenizer,
        halt_gate=None,
        prompt="Answer briefly",
        stage_k=3,
        device=torch.device("cpu"),
        args=args,
        use_chat_template=False,
    )

    assert result.prompt == "Answer briefly"
    assert result.stage_k == 3
    assert result.actual_latents == 3
    assert result.used_halt_gate is False
    assert isinstance(result.text, str)
    assert model.eval_calls == 0  # public helper does not mutate caller's mode.


def test_run_single_prompt_routes_halt_gate_when_supplied():
    model = FakeCausalLM()
    tokenizer = FakeTokenizer()
    halt_gate = FakeHaltGate()
    args = SimpleNamespace(max_new_tokens=3, max_seq_len=64, halt_threshold=0.5)

    result = run_single_prompt(
        model=model,
        tokenizer=tokenizer,
        halt_gate=halt_gate,
        prompt="Answer briefly",
        stage_k=3,
        device=resolve_device("cpu"),
        args=args,
        use_chat_template=False,
    )

    assert result.used_halt_gate is True
    assert result.actual_latents == 3
