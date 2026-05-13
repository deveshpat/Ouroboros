from __future__ import annotations

from types import SimpleNamespace

import ouroboros.bootstrap as bootstrap


def _fake_torch(*, mps_available: bool, cuda_available: bool):
    return SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available),
        ),
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
    )


def test_strict_mac_mps_bootstrap_detector_requires_flag_darwin_mps_and_no_cuda(monkeypatch):
    monkeypatch.setattr(bootstrap.sys, "platform", "darwin")

    assert bootstrap._bootstrap_strict_mac_mps_requested(
        torch_module=_fake_torch(mps_available=True, cuda_available=False),
        argv=["--mac_mps_mamba_kernels"],
    )
    assert not bootstrap._bootstrap_strict_mac_mps_requested(
        torch_module=_fake_torch(mps_available=True, cuda_available=False),
        argv=[],
    )
    assert not bootstrap._bootstrap_strict_mac_mps_requested(
        torch_module=_fake_torch(mps_available=True, cuda_available=True),
        argv=["--mac_mps_mamba_kernels"],
    )


def test_strict_mac_process_finalize_skips_cuda_kernel_export_and_verify(monkeypatch, capsys):
    monkeypatch.setattr(bootstrap, "_bootstrap_env_rank", lambda: 0)
    monkeypatch.setattr(bootstrap, "_bootstrap_env_world_size", lambda: 1)
    monkeypatch.setattr(bootstrap, "_bootstrap_prepare_local_cuda_device", lambda _torch: None)
    monkeypatch.setattr(bootstrap, "_bootstrap_strict_mac_mps_requested", lambda **_: True)
    monkeypatch.setattr(
        bootstrap,
        "_patch_kernel_top_level_exports",
        lambda: (_ for _ in ()).throw(AssertionError("must skip CUDA export shim")),
    )
    monkeypatch.setattr(
        bootstrap,
        "_bootstrap_verify_fast_path",
        lambda: (_ for _ in ()).throw(AssertionError("must skip CUDA verification")),
    )

    bootstrap._bootstrap_process_local_finalize()

    assert "Strict Mac MPS fallback: skipping CUDA kernel export shim" in capsys.readouterr().out
