from __future__ import annotations

import argparse
import ast
from pathlib import Path

import torch

from tests.fakes.eval_fakes import FakeCausalLM, FakeHaltGate, FakeTokenizer
from ouroboros import train as train_module

REPO_ROOT = Path(__file__).resolve().parents[1]


def _decorator_prefix(function_name: str, source: str) -> str:
    needle = f"def {function_name}("
    index = source.index(needle)
    return source[max(0, index - 160):index]


def test_evaluation_and_generation_preserve_monolith_no_grad_decorators():
    monolith_source = (REPO_ROOT / "tests" / "fixtures" / "training_monolith_source.py").read_text(encoding="utf-8")
    modular_source = (REPO_ROOT / "ouroboros" / "train.py").read_text(encoding="utf-8")

    for function_name in ("evaluate_stage", "run_generation_callback"):
        assert "@torch.no_grad()" in _decorator_prefix(function_name, monolith_source)
        assert "@torch.no_grad()" in _decorator_prefix(function_name, modular_source), (
            f"{function_name} must keep the monolith no-grad decorator; dropping it reopens validation OOM drift"
        )


def _function_ast_dump(path: Path, function_name: str) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.dump(node, include_attributes=False)
    raise AssertionError(f"{function_name} not found in {path}")


def test_validation_and_generation_ast_match_monolith_source_of_truth():
    monolith_path = REPO_ROOT / "tests" / "fixtures" / "training_monolith_source.py"
    modular_path = REPO_ROOT / "ouroboros" / "train.py"

    for function_name in ("evaluate_stage", "run_generation_callback"):
        assert _function_ast_dump(modular_path, function_name) == _function_ast_dump(monolith_path, function_name)


def test_evaluate_stage_uses_no_grad_and_restores_train_modes_on_cpu():
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

    ce, acc = train_module.evaluate_stage(
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


def test_run_generation_callback_uses_no_grad_and_restores_train_modes_on_cpu(capsys):
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
    mean_uwr = train_module.run_generation_callback(
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
