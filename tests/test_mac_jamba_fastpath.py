from __future__ import annotations

import pytest
import torch

from ouroboros.mac_jamba_fastpath import install_mac_jamba_mps_fastpath


def _tiny_jamba_config():
    transformers = pytest.importorskip("transformers")
    config_module = pytest.importorskip("transformers.models.jamba.configuration_jamba")
    del transformers
    return config_module.JambaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_experts=1,
        num_experts_per_tok=1,
        expert_layer_period=2,
        expert_layer_offset=1,
        attn_layer_period=8,
        attn_layer_offset=4,
        mamba_d_state=4,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank=4,
        use_mamba_kernels=False,
    )


def test_mac_jamba_fastpath_installs_when_selective_scan_is_available():
    pytest.importorskip("mamba_ssm.ops.selective_scan_interface")
    pytest.importorskip("transformers.models.jamba.modeling_jamba")

    assert install_mac_jamba_mps_fastpath()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Silicon MPS")
def test_mac_jamba_fastpath_matches_original_mps_forward_and_input_gradients():
    pytest.importorskip("mamba_ssm.ops.selective_scan_interface")
    jamba_modeling = pytest.importorskip("transformers.models.jamba.modeling_jamba")
    assert install_mac_jamba_mps_fastpath()

    mixer = jamba_modeling.JambaMambaMixer(_tiny_jamba_config(), layer_idx=0).to(
        device="mps",
        dtype=torch.float16,
    )
    mixer.train()

    original = getattr(jamba_modeling.JambaMambaMixer, "_ouroboros_original_slow_forward")
    torch.manual_seed(17)
    x_original = torch.randn(1, 16, 32, device="mps", dtype=torch.float16, requires_grad=True)
    x_patched = x_original.detach().clone().requires_grad_(True)

    y_original = original(mixer, x_original)
    loss_original = y_original.float().square().mean()
    loss_original.backward()
    original_grad = x_original.grad.detach().clone()

    mixer.zero_grad(set_to_none=True)
    y_patched = mixer.slow_forward(x_patched)
    loss_patched = y_patched.float().square().mean()
    loss_patched.backward()
    patched_grad = x_patched.grad.detach().clone()

    torch.mps.synchronize()
    assert torch.allclose(y_patched, y_original, atol=1e-3, rtol=1e-3)
    assert torch.allclose(patched_grad, original_grad, atol=1e-5, rtol=1e-3)
