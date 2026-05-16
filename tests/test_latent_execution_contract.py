from __future__ import annotations

import argparse

import pytest
import torch
from torch import nn

from ouroboros.coconut import build_sample_at_stage, collate_stage_k
from ouroboros.coconut import (
    compute_ce_from_hidden,
    compute_ce_mean_by_row,
    compute_ce_sum_and_count,
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


def test_cached_latent_pass_matches_jamba_prefix_recompute_and_restores_checkpointing():
    transformers = pytest.importorskip("transformers")
    JambaConfig = transformers.JambaConfig
    JambaForCausalLM = transformers.JambaForCausalLM
    config = JambaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        expert_layer_period=2,
        expert_layer_offset=1,
        attn_layer_period=2,
        attn_layer_offset=0,
        mamba_d_state=4,
        mamba_d_conv=2,
        mamba_expand=1,
        use_mamba_kernels=False,
    )
    torch.manual_seed(1234)
    model = JambaForCausalLM(config)
    model.eval()
    runtime = prepare_latent_runtime(model, torch.device("cpu"))
    input_ids = torch.tensor([[3, 4, 5, 6], [7, 8, 0, 0]], dtype=torch.long)
    ctx = runtime.embed_tokens(input_ids).detach().requires_grad_(True)
    ctx_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    target_steps = torch.tensor([3, 1], dtype=torch.long)

    recomputed_ctx, recomputed_mask, recomputed_k = run_latent_passes(
        runtime=runtime,
        ctx=ctx,
        ctx_mask=ctx_mask,
        n_latent=target_steps,
        halt_gate=None,
        args=_args(latent_cache=False),
    )
    recomputed_loss = recomputed_ctx.sum()
    recomputed_grad = torch.autograd.grad(recomputed_loss, ctx, retain_graph=True)[0]

    model.gradient_checkpointing_enable()
    cached_ctx, cached_mask, cached_k = run_latent_passes(
        runtime=runtime,
        ctx=ctx,
        ctx_mask=ctx_mask,
        n_latent=target_steps,
        halt_gate=None,
        args=_args(latent_cache=True),
    )
    cached_loss = cached_ctx.sum()
    cached_grad = torch.autograd.grad(cached_loss, ctx)[0]

    assert cached_k.tolist() == recomputed_k.tolist() == [3, 1]
    assert cached_mask.tolist() == recomputed_mask.tolist()
    assert torch.allclose(cached_ctx, recomputed_ctx, atol=1e-5, rtol=1e-5)
    assert torch.allclose(cached_grad, recomputed_grad, atol=1e-4, rtol=1e-4)
    assert model.is_gradient_checkpointing is True


def test_sparse_lm_head_ce_matches_dense_logits():
    model = FakeCausalLM(hidden_size=8, vocab_size=16)
    runtime = prepare_latent_runtime(model, torch.device("cpu"))
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    labels = torch.tensor(
        [
            [-100, -100, 3, 4, -100],
            [-100, 5, -100, 6, 7],
        ],
        dtype=torch.long,
    )
    dense_logits = runtime.lm_head(hidden).float()
    dense_sum, dense_count = compute_ce_sum_and_count(dense_logits, labels)
    dense_by_row, dense_valid_by_row = compute_ce_mean_by_row(dense_logits, labels)

    sparse_sum, sparse_count, sparse_by_row, sparse_valid_by_row = compute_ce_from_hidden(
        runtime=runtime,
        hidden=hidden,
        labels=labels,
    )

    assert sparse_count == dense_count
    assert sparse_valid_by_row.tolist() == dense_valid_by_row.tolist()
    assert torch.allclose(sparse_sum, dense_sum, atol=1e-6, rtol=1e-6)
    assert torch.allclose(sparse_by_row, dense_by_row, atol=1e-6, rtol=1e-6)


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
