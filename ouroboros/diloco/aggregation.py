"""DiLoCo aggregation module.

CPU tensor aggregation is a deep module because callers should not need to know
pseudo-gradient details, dtype restoration, or sample weighting rules.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence


def weighted_average_deltas(
    anchor_weights: Mapping[str, object],
    worker_weights: Sequence[Mapping[str, object]],
    worker_samples: Sequence[int],
    outer_lr: float,
) -> Dict[str, object]:
    """Apply the DiLoCo outer update to adapter weights on CPU tensors."""

    import torch

    total_samples = sum(int(n) for n in worker_samples)
    if total_samples <= 0:
        raise ValueError("total_samples must be > 0 for aggregation")
    if len(worker_weights) != len(worker_samples):
        raise ValueError("worker_weights and worker_samples must have the same length")

    new_weights: Dict[str, object] = {}
    for key, anchor_value in anchor_weights.items():
        anchor_tensor = anchor_value.float()  # type: ignore[attr-defined]
        outer_grad = torch.zeros_like(anchor_tensor)
        for weights, n_samples in zip(worker_weights, worker_samples):
            if key not in weights:
                continue
            worker_tensor = weights[key].float()  # type: ignore[index,attr-defined]
            delta = anchor_tensor - worker_tensor
            outer_grad += delta * (float(n_samples) / float(total_samples))
        new_weights[key] = (anchor_tensor - float(outer_lr) * outer_grad).to(anchor_value.dtype)  # type: ignore[attr-defined]
    return new_weights
