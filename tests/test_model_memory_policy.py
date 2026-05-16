from __future__ import annotations

from types import SimpleNamespace

from ouroboros.models import _should_auto_disable_gradient_checkpointing


def test_high_vram_small_workload_auto_disables_gradient_checkpointing(monkeypatch):
    monkeypatch.delenv("OUROBOROS_FORCE_GRAD_CHECKPOINT", raising=False)
    args = SimpleNamespace(batch_size=2, max_seq_len=1024, max_stage=1, use_halt_gate=False)

    assert _should_auto_disable_gradient_checkpointing(args, 80.0) is True


def test_high_depth_dgac_keeps_gradient_checkpointing_on_h100(monkeypatch):
    monkeypatch.delenv("OUROBOROS_FORCE_GRAD_CHECKPOINT", raising=False)
    args = SimpleNamespace(batch_size=4, max_seq_len=2048, max_stage=10, use_halt_gate=True)

    assert _should_auto_disable_gradient_checkpointing(args, 100.0) is False


def test_force_gradient_checkpointing_env_overrides_auto_disable(monkeypatch):
    monkeypatch.setenv("OUROBOROS_FORCE_GRAD_CHECKPOINT", "1")
    args = SimpleNamespace(batch_size=1, max_seq_len=1024, max_stage=1, use_halt_gate=False)

    assert _should_auto_disable_gradient_checkpointing(args, 80.0) is False
