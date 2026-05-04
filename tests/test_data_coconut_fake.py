from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import torch

from ouroboros.data import build_sample_at_stage, collate_stage_k
from ouroboros.dgac import coconut_forward, compute_dgac_lambda1, normalize_pred
from tests.fakes.eval_fakes import FakeCausalLM, FakeHaltGate, FakeTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _function_ast_dump(path: Path, function_name: str) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.dump(node, include_attributes=False)
    raise AssertionError(f"{function_name} not found in {path}")


def test_data_and_coconut_core_ast_match_monolith_source_of_truth():
    monolith_path = REPO_ROOT / "tests" / "fixtures" / "training_monolith_source.py"
    for module_rel, function_names in {
        "ouroboros/data.py": ("build_sample_at_stage", "collate_stage_k"),
        "ouroboros/dgac.py": ("compute_dgac_lambda1", "coconut_forward", "normalize_pred"),
    }.items():
        module_path = REPO_ROOT / module_rel
        for function_name in function_names:
            assert _function_ast_dump(module_path, function_name) == _function_ast_dump(monolith_path, function_name)


def test_stage_sample_construction_handles_native_and_json_steps_and_masks_question_latents():
    tokenizer = FakeTokenizer()
    native_sample = {
        "question": "What is 2 plus 3?",
        "steps": ["Compute 2+3.", "It equals 5."],
        "answer_full": "5",
        "answer_norm": "5",
    }
    json_sample = {**native_sample, "steps": json.dumps(native_sample["steps"])}

    native = build_sample_at_stage(tokenizer, native_sample, stage_k=1, lat_token_id=7, max_seq_len=64)
    encoded = build_sample_at_stage(tokenizer, json_sample, stage_k=1, lat_token_id=7, max_seq_len=64)

    assert native is not None and encoded is not None
    assert torch.equal(native["full_ids"], encoded["full_ids"])
    assert torch.equal(native["labels"], encoded["labels"])
    assert native["n_latent"] == 1
    assert native["answer_norm"] == "5"

    q_len = int(native["q_len"])
    n_latent = int(native["n_latent"])
    assert native["full_ids"][q_len : q_len + n_latent].tolist() == [7]
    assert native["labels"][: q_len + n_latent].tolist() == [-100] * (q_len + n_latent)
    assert any(label != -100 for label in native["labels"][q_len + n_latent :].tolist())


def test_collation_shapes_dtypes_attention_and_padding_contracts():
    tokenizer = FakeTokenizer()
    samples = [
        build_sample_at_stage(
            tokenizer,
            {"question": "Q1", "steps": ["s1"], "answer_full": "a1", "answer_norm": "a1"},
            stage_k=1,
            lat_token_id=5,
            max_seq_len=64,
        ),
        build_sample_at_stage(
            tokenizer,
            {"question": "Longer Q2", "steps": ["s1", "s2"], "answer_full": "a2", "answer_norm": "a2"},
            stage_k=2,
            lat_token_id=5,
            max_seq_len=64,
        ),
    ]
    assert all(sample is not None for sample in samples)
    batch = collate_stage_k(samples, pad_id=tokenizer.pad_token_id)

    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["q_lens"].dtype == torch.long
    assert batch["n_latents"].dtype == torch.long
    assert batch["pad_id"].shape == torch.Size([])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[0] == 2

    for row, sample in enumerate(samples):
        seq_len = int(sample["full_ids"].numel())
        assert batch["attention_mask"][row, :seq_len].all()
        assert not batch["attention_mask"][row, seq_len:].any()
        assert batch["labels"][row, seq_len:].tolist() == [-100] * (batch["labels"].shape[1] - seq_len)


def _dgac_args() -> argparse.Namespace:
    return argparse.Namespace(
        halt_threshold=0.9,
        dgac_warmup_steps=0,
        dgac_ramp_steps=2,
        dgac_lambda_ponder_max=0.02,
        dgac_tau=0.95,
        dgac_lambda_diversity=0.1,
    )


def test_fake_coconut_forward_runs_on_cpu_and_produces_deterministic_metrics():
    tokenizer = FakeTokenizer()
    model = FakeCausalLM()
    sample = build_sample_at_stage(
        tokenizer,
        {
            "question": "What is 4 plus 1?",
            "steps": ["Add one.", "Return the number."],
            "answer_full": "5",
            "answer_norm": "5",
        },
        stage_k=2,
        lat_token_id=6,
        max_seq_len=64,
    )
    assert sample is not None
    batch = collate_stage_k([sample], pad_id=tokenizer.pad_token_id)

    loss1, metrics1 = coconut_forward(
        model=model,
        batch=batch,
        stage_k=2,
        device=torch.device("cpu"),
        halt_gate=None,
        args=_dgac_args(),
        step_in_phase=0,
    )
    loss2, metrics2 = coconut_forward(
        model=model,
        batch=batch,
        stage_k=2,
        device=torch.device("cpu"),
        halt_gate=None,
        args=_dgac_args(),
        step_in_phase=0,
    )

    assert loss1.requires_grad
    assert torch.isclose(loss1.detach(), loss2.detach())
    assert metrics1 == metrics2
    assert metrics1["ce"] >= 0.0
    assert set(model.model.device_type_observations) == {"cpu"}


def test_halt_gate_metrics_and_answer_normalization_contracts():
    assert compute_dgac_lambda1(step=0, warmup=2, ramp=3, lmax=0.9) == 0.0
    assert compute_dgac_lambda1(step=4, warmup=2, ramp=4, lmax=1.0) == 0.5
    assert compute_dgac_lambda1(step=99, warmup=2, ramp=4, lmax=1.0) == 1.0

    tokenizer = FakeTokenizer()
    model = FakeCausalLM()
    halt_gate = FakeHaltGate()
    sample = build_sample_at_stage(
        tokenizer,
        {"question": "Q", "steps": ["s1", "s2"], "answer_full": "9", "answer_norm": "9"},
        stage_k=2,
        lat_token_id=6,
        max_seq_len=64,
    )
    assert sample is not None
    loss, metrics = coconut_forward(
        model=model,
        batch=collate_stage_k([sample], pad_id=tokenizer.pad_token_id),
        stage_k=2,
        device=torch.device("cpu"),
        halt_gate=halt_gate,
        args=_dgac_args(),
        step_in_phase=3,
    )
    assert loss.requires_grad
    assert {"ce", "ponder", "diversity", "halt_step_mean", "lambda1"}.issubset(metrics)
    assert metrics["lambda1"] > 0.0

    assert normalize_pred(r"final result is \\boxed{1,234}") == "1234"
    assert normalize_pred("Therefore, answer is **-3.5") == "-3.5"
    assert normalize_pred("last line\nFinal Answer: blue.") == "blue"
