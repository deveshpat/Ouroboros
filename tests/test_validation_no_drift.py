from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tests.fakes.eval_fakes import FakeCausalLM, FakeHaltGate, FakeTokenizer
from ouroboros import coconut
from ouroboros.coconut import evaluation as evaluation_module
from ouroboros.coconut.latent import DecodeResult

REPO_ROOT = Path(__file__).resolve().parents[1]


def _decorator_prefix(function_name: str, source: str) -> str:
    needle = f"def {function_name}("
    index = source.index(needle)
    return source[max(0, index - 160):index]


def test_evaluation_and_generation_keep_no_grad_contracts():
    source = (REPO_ROOT / "ouroboros" / "coconut" / "evaluation.py").read_text(encoding="utf-8")
    for function_name in ("evaluate_stage", "run_generation_callback"):
        assert "@torch.no_grad()" in _decorator_prefix(function_name, source)


def test_validation_and_generation_route_through_latent_execution_seam():
    source = (REPO_ROOT / "ouroboros" / "coconut" / "evaluation.py").read_text(encoding="utf-8")
    assert "prepare_latent_runtime" in source
    assert "run_latent_passes" in source
    assert "decode_from_latent_context" in source
    for forbidden in [
        "_get_backbone",
        "_get_embed_tokens",
        "_get_lm_head",
        "_autocast_ctx",
        "_extract_last_hidden_state",
    ]:
        assert forbidden not in source


def test_evaluate_stage_uses_no_grad_restores_train_modes_and_prints_progress_on_cpu(capsys, monkeypatch):
    model = FakeCausalLM()
    halt_gate = FakeHaltGate()
    tokenizer = FakeTokenizer()
    args = argparse.Namespace(
        max_seq_len=24,
        val_batch_size=2,
        gen_max_tokens=2,
        halt_threshold=0.9,
        dgac_warmup_steps=0,
        dgac_ramp_steps=1,
        dgac_lambda_ponder_max=0.0,
        dgac_tau=0.95,
        dgac_lambda_diversity=0.0,
        eval_progress_every=1,
    )
    val_samples = [
        {
            "question": "What is 1 plus 1?",
            "steps": ["Add the numbers."],
            "answer_full": "2",
            "answer_norm": "2",
        }
    ]

    device = torch.device("cpu")
    model.train()
    halt_gate.train()
    monkeypatch.setattr(
        evaluation_module,
        "decode_from_latent_context",
        lambda **kwargs: DecodeResult(token_ids=[2], text="2"),
    )

    ce, acc = evaluation_module.evaluate_stage(
        model=model,
        val_samples=val_samples,
        tokenizer=tokenizer,
        lat_token_id=3,
        stage_k=1,
        device=device,
        args=args,
        halt_gate=halt_gate,
    )

    assert isinstance(ce, float)
    assert isinstance(acc, float)
    assert model.eval_calls >= 1
    assert model.train_calls >= 1
    assert halt_gate.eval_calls >= 1
    assert halt_gate.train_calls >= 1
    assert model.training is True
    assert halt_gate.training is True
    assert model.model.grad_enabled_observations
    assert model.model.device_type_observations
    assert set(model.model.device_type_observations) == {"cpu"}
    assert model.model.grad_enabled_observations == [False] * len(model.model.grad_enabled_observations)
    out = capsys.readouterr().out
    assert "[eval] validation CE start" in out
    assert "[eval CE rank0] 1/1 (100.0%)" in out
    assert "[eval] accuracy decode start" in out
    assert "[eval acc rank0] 1/1 (100.0%)" in out


def test_run_generation_callback_uses_no_grad_and_restores_train_modes_on_cpu(capsys, monkeypatch):
    model = FakeCausalLM()
    halt_gate = FakeHaltGate()
    tokenizer = FakeTokenizer()
    args = argparse.Namespace(
        max_seq_len=24,
        gen_max_tokens=2,
        halt_threshold=0.9,
        dgac_warmup_steps=0,
        dgac_ramp_steps=1,
        dgac_lambda_ponder_max=0.0,
        dgac_tau=0.95,
        dgac_lambda_diversity=0.0,
    )

    model.train()
    halt_gate.train()
    monkeypatch.setattr(
        evaluation_module,
        "decode_from_latent_context",
        lambda **kwargs: DecodeResult(token_ids=[2], text="2"),
    )
    mean_uwr = evaluation_module.run_generation_callback(
        model=model,
        tokenizer=tokenizer,
        halt_gate=halt_gate,
        stage_k=1,
        device=torch.device("cpu"),
        args=args,
        step=7,
        wandb_run=None,
    )

    assert isinstance(mean_uwr, float)
    assert model.eval_calls >= 1
    assert model.train_calls >= 1
    assert halt_gate.eval_calls >= 1
    assert halt_gate.train_calls >= 1
    assert model.training is True
    assert halt_gate.training is True
    assert set(model.model.device_type_observations) == {"cpu"}
    assert model.model.grad_enabled_observations == [False] * len(model.model.grad_enabled_observations)
    assert "Generation @ step 7" in capsys.readouterr().out
