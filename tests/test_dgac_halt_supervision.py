from __future__ import annotations

import argparse

import torch

from ouroboros.data import build_sample_at_stage, collate_stage_k
from ouroboros.dgac import (
    HaltGate,
    build_dgac_halt_probe_depths,
    build_halt_supervision_labels,
    coconut_forward,
    construct_dgac_halt_targets,
)
from ouroboros.cli import parse_args
from tests.fakes.eval_fakes import FakeCausalLM, FakeTokenizer


def _args(**overrides) -> argparse.Namespace:
    values = dict(
        halt_threshold=0.5,
        dgac_warmup_steps=0,
        dgac_ramp_steps=1,
        dgac_lambda_ponder_max=0.0,
        dgac_tau=0.95,
        dgac_lambda_diversity=0.0,
        dgac_halt_supervision_weight=0.5,
        dgac_halt_ce_tolerance=0.02,
        dgac_halt_probe_steps="1,2,4,stage_k",
        dgac_halt_target_mode="ce_within_tolerance",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_probe_depths_are_unique_clipped_and_include_full_stage():
    assert build_dgac_halt_probe_depths(10, "1,2,4,stage_k") == [1, 2, 4, 10]
    assert build_dgac_halt_probe_depths(3, "0,1,4,full,2") == [1, 2, 3]
    assert build_dgac_halt_probe_depths(0, "1,2,stage_k") == []


def test_halt_target_is_smallest_depth_within_full_ce_tolerance():
    full_depths = torch.tensor([4, 4, 4])
    full_ce = torch.tensor([1.00, 1.00, 1.00])
    targets = construct_dgac_halt_targets(
        ce_by_probe_depth={
            1: torch.tensor([1.01, 1.20, 1.30]),
            2: torch.tensor([1.02, 1.01, 1.25]),
            4: torch.tensor([1.00, 1.00, 1.00]),
        },
        full_ce=full_ce,
        full_depths=full_depths,
        tolerance=0.02,
    )

    assert targets.tolist() == [1, 2, 4]


def test_halt_targets_clip_probe_depths_to_sample_available_depth():
    targets = construct_dgac_halt_targets(
        ce_by_probe_depth={
            1: torch.tensor([1.30, 1.20]),
            4: torch.tensor([1.00, 1.00]),
        },
        full_ce=torch.tensor([1.00, 1.00]),
        full_depths=torch.tensor([2, 3]),
        tolerance=0.02,
    )

    assert targets.tolist() == [2, 3]


def test_halt_supervision_labels_continue_before_target_and_do_not_reward_early_halt():
    labels, mask = build_halt_supervision_labels(torch.tensor([1, 2, 4]), max_depth=4)

    assert labels.tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert mask.tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]


def test_supervised_halt_loss_reaches_halt_gate_gradients_on_cpu():
    tokenizer = FakeTokenizer()
    model = FakeCausalLM(hidden_size=8)
    halt_gate = HaltGate(d_model=8)
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

    loss, metrics = coconut_forward(
        model=model,
        batch=collate_stage_k([sample], pad_id=tokenizer.pad_token_id),
        stage_k=2,
        device=torch.device("cpu"),
        halt_gate=halt_gate,
        args=_args(),
        step_in_phase=1,
    )
    loss.backward()

    assert "dgac_halt_loss" in metrics
    assert "dgac_ponder" in metrics
    assert metrics["dgac_halt_supervised_count"] > 0.0
    gate_grads = [p.grad for p in halt_gate.parameters()]
    assert all(grad is not None for grad in gate_grads)
    assert sum(float(grad.abs().sum().item()) for grad in gate_grads if grad is not None) > 0.0


def test_cli_exposes_correctness_aware_halt_supervision_defaults():
    args = parse_args(["--use_halt_gate"])

    assert args.dgac_halt_supervision_weight == 0.1
    assert args.dgac_halt_ce_tolerance == 0.02
    assert args.dgac_halt_probe_steps == "1,2,4,stage_k"
    assert args.dgac_halt_target_mode == "ce_within_tolerance"
