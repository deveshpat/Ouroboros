from __future__ import annotations

import argparse

import torch
from torch import nn

from ouroboros.data import build_sample_at_stage, collate_stage_k
from ouroboros.latent import (
    decode_from_latent_context,
    forward_latent_batch,
    prepare_latent_runtime,
    run_latent_passes,
)
from tests.fakes.eval_fakes import FakeCausalLM, FakeTokenizer


class AlwaysHaltAfterOneGate(nn.Module):
    def forward(self, h_curr, h_prev):
        return torch.ones(h_curr.size(0), device=h_curr.device)


def _args(**overrides) -> argparse.Namespace:
    values = dict(
        halt_threshold=0.5,
        gen_max_tokens=3,
        max_seq_len=64,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_latent_passes_preserves_fixed_depth_context_contract_on_cpu():
    model = FakeCausalLM(hidden_size=8)
    runtime = prepare_latent_runtime(model, torch.device("cpu"))
    input_ids = torch.tensor([[3, 4]], dtype=torch.long)
    ctx = runtime.embed_tokens(input_ids)
    ctx_mask = torch.ones((1, 2), dtype=torch.bool)

    latent_ctx, latent_mask, actual_k = run_latent_passes(
        runtime=runtime,
        ctx=ctx,
        ctx_mask=ctx_mask,
        n_latent=2,
        halt_gate=None,
        args=_args(),
    )

    assert actual_k == 2
    assert latent_ctx.shape == (1, 4, 8)
    assert latent_mask.tolist() == [[True, True, True, True]]


def test_run_latent_passes_allows_halt_gate_to_reduce_actual_depth():
    model = FakeCausalLM(hidden_size=8)
    runtime = prepare_latent_runtime(model, torch.device("cpu"))
    input_ids = torch.tensor([[3, 4]], dtype=torch.long)
    ctx = runtime.embed_tokens(input_ids)
    ctx_mask = torch.ones((1, 2), dtype=torch.bool)

    latent_ctx, latent_mask, actual_k = run_latent_passes(
        runtime=runtime,
        ctx=ctx,
        ctx_mask=ctx_mask,
        n_latent=3,
        halt_gate=AlwaysHaltAfterOneGate(),
        args=_args(),
    )

    assert actual_k == 1
    assert latent_ctx.shape == (1, 3, 8)
    assert latent_mask.tolist() == [[True, True, True]]


def test_forward_latent_batch_returns_ce_and_row_accounting():
    tokenizer = FakeTokenizer()
    sample = build_sample_at_stage(
        tokenizer,
        {
            "question": "What is 2 plus 2?",
            "steps": ["Think one.", "Think two."],
            "answer_full": "4",
            "answer_norm": "4",
        },
        stage_k=2,
        lat_token_id=6,
        max_seq_len=64,
    )
    assert sample is not None
    batch = collate_stage_k([sample], pad_id=tokenizer.pad_token_id)
    model = FakeCausalLM(hidden_size=8)
    runtime = prepare_latent_runtime(model, torch.device("cpu"))

    result = forward_latent_batch(
        runtime=runtime,
        batch=batch,
        args=_args(),
        include_hidden_sequences=True,
    )

    assert result["n_valid"] > 0
    assert result["ce_sum"].ndim == 0
    assert result["ce_by_row"].shape == (1,)
    assert result["valid_by_row"].tolist() == [result["n_valid"]]
    assert result["actual_n_latents"].tolist() == [2]
    assert len(result["hidden_sequences"][0]) == 2


def test_decode_from_latent_context_emits_deterministic_fake_model_tokens():
    tokenizer = FakeTokenizer()
    model = FakeCausalLM(hidden_size=8)
    with torch.no_grad():
        model.model.embed_tokens.weight.fill_(0.2)
        model.lm_head.weight.zero_()
        model.lm_head.weight[3].fill_(1.0)
    runtime = prepare_latent_runtime(model, torch.device("cpu"))
    input_ids = torch.tensor([[3, 4]], dtype=torch.long)
    ctx = runtime.embed_tokens(input_ids)
    ctx_mask = torch.ones((1, 2), dtype=torch.bool)

    decoded = decode_from_latent_context(
        runtime=runtime,
        ctx=ctx,
        ctx_mask=ctx_mask,
        tokenizer=tokenizer,
        args=_args(gen_max_tokens=3),
        context="latent decode contract",
    )

    assert decoded.token_ids == [3, 3, 3]
    assert decoded.text == "3 3 3"
