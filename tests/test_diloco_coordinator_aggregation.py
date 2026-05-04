from __future__ import annotations

import pytest
import torch

from ouroboros.diloco import aggregation


def _monolith_weighted_average(anchor_weights, worker_weights, worker_samples, outer_lr):
    total_samples = sum(worker_samples)
    if total_samples <= 0:
        raise ValueError("total_samples must be > 0 for aggregation")

    new_weights = {}
    for key in anchor_weights:
        anchor_tensor = anchor_weights[key].float()
        outer_grad = torch.zeros_like(anchor_tensor)
        for weights, n_samples in zip(worker_weights, worker_samples):
            if key not in weights:
                continue
            delta = anchor_tensor - weights[key].float()
            outer_grad += delta * (float(n_samples) / float(total_samples))
        new_weights[key] = (anchor_tensor - outer_lr * outer_grad).to(anchor_weights[key].dtype)
    return new_weights


def test_weighted_average_deltas_matches_monolith_formula_and_preserves_dtype():
    anchor = {
        "lora_a": torch.tensor([10.0, 20.0], dtype=torch.float16),
        "lora_b": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    }
    workers = [
        {
            "lora_a": torch.tensor([8.0, 24.0], dtype=torch.float32),
            "lora_b": torch.tensor([[2.0, 0.0]], dtype=torch.float32),
        },
        {
            "lora_a": torch.tensor([16.0, 10.0], dtype=torch.float32),
            "lora_b": torch.tensor([[0.0, 4.0]], dtype=torch.float32),
        },
    ]
    samples = [2, 6]

    actual = aggregation.weighted_average_deltas(anchor, workers, samples, outer_lr=0.5)
    expected = _monolith_weighted_average(anchor, workers, samples, outer_lr=0.5)

    assert set(actual) == set(expected) == set(anchor)
    assert actual["lora_a"].dtype == torch.float16
    assert actual["lora_b"].dtype == torch.float32
    torch.testing.assert_close(actual["lora_a"], expected["lora_a"])
    torch.testing.assert_close(actual["lora_b"], expected["lora_b"])


def test_weighted_average_deltas_preserves_missing_worker_key_behavior():
    anchor = {
        "present": torch.tensor([4.0], dtype=torch.float32),
        "missing_from_one_worker": torch.tensor([10.0], dtype=torch.float32),
    }
    workers = [
        {"present": torch.tensor([2.0]), "missing_from_one_worker": torch.tensor([8.0])},
        {"present": torch.tensor([8.0])},
    ]
    samples = [1, 3]

    actual = aggregation.weighted_average_deltas(anchor, workers, samples, outer_lr=1.0)
    expected = _monolith_weighted_average(anchor, workers, samples, outer_lr=1.0)

    torch.testing.assert_close(actual["present"], expected["present"])
    torch.testing.assert_close(actual["missing_from_one_worker"], expected["missing_from_one_worker"])


def test_weighted_average_deltas_rejects_non_positive_total_samples():
    with pytest.raises(ValueError, match="total_samples must be > 0"):
        aggregation.weighted_average_deltas(
            {"x": torch.tensor([1.0])},
            [{"x": torch.tensor([1.0])}],
            [0],
            outer_lr=0.7,
        )


def test_aggregate_worker_updates_promotes_single_or_solo_worker_directly():
    anchor = {"x": torch.tensor([1.0])}
    worker_a = {"x": torch.tensor([100.0])}
    worker_b = {"x": torch.tensor([-100.0])}

    assert aggregation.aggregate_worker_updates(anchor, [worker_a], [5], 0.7) is worker_a
    assert aggregation.aggregate_worker_updates(anchor, [worker_a, worker_b], [5, 5], 0.7, mode="solo") is worker_a


def test_aggregate_worker_updates_rejects_empty_contributor_list():
    with pytest.raises(ValueError, match="worker_weights must contain"):
        aggregation.aggregate_worker_updates({"x": torch.tensor([1.0])}, [], [], 0.7)
