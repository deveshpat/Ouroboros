from __future__ import annotations

from ouroboros.wandb_runtime import _wandb_init_timeout_seconds, wandb_init_kwargs


class _FakeWandb:
    class Settings:
        def __init__(self, *, init_timeout):
            self.init_timeout = init_timeout


def test_wandb_init_timeout_defaults_to_longer_local_timeout():
    assert _wandb_init_timeout_seconds({}) == 300.0


def test_wandb_init_timeout_uses_project_override_but_never_below_wandb_default():
    assert _wandb_init_timeout_seconds({"OUROBOROS_WANDB_INIT_TIMEOUT": "120"}) == 120.0
    assert _wandb_init_timeout_seconds({"OUROBOROS_WANDB_INIT_TIMEOUT": "30"}) == 90.0


def test_wandb_init_kwargs_builds_settings_when_available():
    kwargs = wandb_init_kwargs(_FakeWandb, {"WANDB_INIT_TIMEOUT": "240"})

    assert set(kwargs) == {"settings"}
    assert kwargs["settings"].init_timeout == 240.0
