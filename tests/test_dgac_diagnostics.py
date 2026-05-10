from __future__ import annotations

import argparse

import torch
from torch import nn

from ouroboros.train import run_dgac_diagnostics
from tests.fakes.eval_fakes import FakeCausalLM, FakeTokenizer


class AlwaysHaltAfterOneGate(nn.Module):
    def forward(self, h_curr, h_prev):
        return torch.ones(h_curr.size(0), device=h_curr.device)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        max_seq_len=64,
        val_batch_size=2,
        halt_threshold=0.5,
        dgac_warmup_steps=0,
        dgac_ramp_steps=1,
        dgac_lambda_ponder_max=0.01,
        dgac_tau=0.95,
        dgac_lambda_diversity=0.1,
    )


def test_dgac_diagnostics_reports_halt_histogram_and_ce_comparison(capsys):
    tokenizer = FakeTokenizer()
    samples = [
        {"question": "Q one", "steps": ["s1", "s2", "s3"], "answer_full": "1", "answer_norm": "1"},
        {"question": "Q two", "steps": ["s1", "s2", "s3"], "answer_full": "2", "answer_norm": "2"},
    ]

    metrics = run_dgac_diagnostics(
        model=FakeCausalLM(),
        tokenizer=tokenizer,
        halt_gate=AlwaysHaltAfterOneGate(),
        val_samples=samples,
        lat_token_id=6,
        stage_k=3,
        device=torch.device("cpu"),
        args=_args(),
        step=0,
        wandb_run=None,
        val_ce_forced_kmax=0.123,
    )

    assert metrics["dgac_diag/samples"] == 2.0
    assert metrics["dgac_diag/hist_k1"] == 2.0
    assert metrics["dgac_diag/hist_k2"] == 0.0
    assert metrics["dgac_diag/hist_k3"] == 0.0
    assert metrics["dgac_diag/k_mean"] == 1.0
    assert metrics["dgac_diag/k_p50"] == 1.0
    assert metrics["dgac_diag/k_p90"] == 1.0
    assert metrics["dgac_diag/pct_at_1"] == 1.0
    assert metrics["dgac_diag/val_ce_forced_k3"] == 0.123
    assert metrics["dgac_diag/val_ce_gated"] == metrics["dgac_diag/val_ce_forced_k1"]

    out = capsys.readouterr().out
    assert "[DGAC diagnostic] stage=3 samples=2" in out
    assert "k_actual histogram: k1=2 k2=0 k3=0" in out
    assert "val_ce_forced_k1=" in out
    assert "val_ce_gated=" in out
    assert "val_ce_forced_k3=0.1230" in out
