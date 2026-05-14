from __future__ import annotations

import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType
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


def test_flash_attention_bootstrap_is_gated_to_ampere_or_newer():
    assert bootstrap._FLASH_ATTN_HUB_WHEEL_BASE not in bootstrap._bootstrap_wheel_bases_for_cuda_arch((7, 5))
    assert bootstrap._FLASH_ATTN_HUB_WHEEL_BASE in bootstrap._bootstrap_wheel_bases_for_cuda_arch((8, 0))
    assert bootstrap._FLASH_ATTN_HUB_WHEEL_BASE in bootstrap._bootstrap_wheel_bases_for_cuda_arch((9, 0))
    for base in bootstrap._MAMBA_HUB_WHEEL_BASES:
        assert base in bootstrap._bootstrap_wheel_bases_for_cuda_arch((7, 5))
        assert base in bootstrap._bootstrap_wheel_bases_for_cuda_arch((9, 0))


def test_flash_attention_wheel_base_uses_pypi_distribution_name():
    pkg_name, pip_spec = bootstrap._bootstrap_pip_spec_for_wheel_base(
        bootstrap._FLASH_ATTN_HUB_WHEEL_BASE
    )

    assert pkg_name == "flash_attn"
    assert pip_spec == "flash-attn==2.8.3"


def test_triton_log1p_patch_restores_missing_math_symbol(monkeypatch):
    triton_module = ModuleType("triton")
    language_module = ModuleType("triton.language")
    language_module.math = SimpleNamespace()
    language_module.log = lambda value: ("log", value)
    jit_calls = []

    def fake_jit(fn):
        jit_calls.append(fn.__name__)
        fn._triton_jit_wrapped = True
        return fn

    triton_module.jit = fake_jit
    triton_module.language = language_module
    monkeypatch.setitem(sys.modules, "triton", triton_module)
    monkeypatch.setitem(sys.modules, "triton.language", language_module)

    assert bootstrap._patch_triton_math_log1p() == ["triton.language.math.log1p"]
    assert jit_calls == ["_log1p"]
    assert language_module.math.log1p._triton_jit_wrapped is True
    assert language_module.math.log1p(2.0) == ("log", 3.0)
    assert bootstrap._patch_triton_math_log1p() == []


def test_mamba_triton_log1p_source_patch_rewrites_installed_kernel(monkeypatch, tmp_path):
    package_root = tmp_path / "mamba_ssm"
    triton_dir = package_root / "ops" / "triton"
    triton_dir.mkdir(parents=True)
    kernel = triton_dir / "selective_state_update.py"
    kernel.write_text(
        "dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)\n",
        encoding="utf-8",
    )

    spec = importlib.machinery.ModuleSpec("mamba_ssm", loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_root)]
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda name: spec if name == "mamba_ssm" else None)

    assert bootstrap._patch_mamba_triton_log1p_source() == [str(kernel)]
    assert kernel.read_text(encoding="utf-8") == (
        "dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)\n"
    )
    assert bootstrap._patch_mamba_triton_log1p_source() == []
