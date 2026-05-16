"""Bootstrap and credential helpers for the Ouroboros training entrypoint."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ouroboros.utils.runtime_env import (
    WANDB_KEY_ALIASES,
    normalize_text,
    require_known_worker_id,
    resolve_env_alias,
    resolve_github_token,
    resolve_hf_token,
    resolve_worker_id,
)


def _normalize_text(value: Optional[Any], *, uppercase: bool = False) -> Optional[str]:
    return normalize_text(value, uppercase=uppercase)


_MAMBA_HUB_WHEEL_BASES = [
    "causal_conv1d-1.6.1-cp312-cp312-linux_x86_64",
    "mamba_ssm-1.2.2-cp312-cp312-linux_x86_64",
]
_FLASH_ATTN_HUB_WHEEL_BASE = "flash_attn-2.8.3-cp312-cp312-linux_x86_64"
_HUB_WHEEL_BASES = [*_MAMBA_HUB_WHEEL_BASES, _FLASH_ATTN_HUB_WHEEL_BASE]


_HUB_REPO_ID = "WeirdRunner/Ouroboros"


_KNOWN_ARCH_SUFFIXES = [
    "sm60", "sm70", "sm72", "sm75", "sm80", "sm86", "sm87", "sm89",
    "sm90", "sm100", "sm120", "smunknown",
]


def _bootstrap_flash_attention_supported(cuda_capability: Tuple[int, int]) -> bool:
    """FlashAttention 2 is for Ampere/Ada/Hopper; keep T4 on eager attention."""
    return tuple(cuda_capability) >= (8, 0)


def _bootstrap_wheel_bases_for_cuda_arch(cuda_capability: Tuple[int, int]) -> List[str]:
    bases = list(_MAMBA_HUB_WHEEL_BASES)
    if _bootstrap_flash_attention_supported(cuda_capability):
        bases.append(_FLASH_ATTN_HUB_WHEEL_BASE)
    return bases


def _patch_triton_math_log1p() -> List[str]:
    """
    Restore the Triton math.log1p symbol expected by mamba-ssm 1.2.2.

    Triton 3.2 removed/relocated ``tl.math.log1p`` while the Mamba kernels still
    reference it during JIT compilation. A small alias to ``tl.log(1 + x)`` keeps
    the verified fast path alive without changing model math.
    """
    try:
        import triton as _triton
        import triton.language as _tl
    except Exception:
        return []
    _math = getattr(_tl, "math", None)
    if _math is None or getattr(_math, "log1p", None) is not None:
        return []

    def _log1p(x):
        return _tl.log(1.0 + x)

    _jit = getattr(_triton, "jit", None)
    if callable(_jit):
        _log1p = _jit(_log1p)

    setattr(_math, "log1p", _log1p)
    return ["triton.language.math.log1p"]


_MAMBA_TRITON_LOG1P_REPLACEMENTS = {
    "tl.math.log1p(tl.exp(dt))": "tl.log(1.0 + tl.exp(dt))",
}


def _patch_mamba_triton_log1p_source() -> List[str]:
    """
    Patch mamba-ssm 1.2.2 Triton source for Triton 3.x log1p compatibility.

    The runtime monkeypatch above must be a JIT function, but source-level repair is
    stronger: it keeps mamba's own kernels and the Jamba fast path on the same code
    path before Triton parses the kernel AST.
    """
    import importlib.util as _ilu

    try:
        spec = _ilu.find_spec("mamba_ssm")
    except Exception:
        return []
    search_locations = list(getattr(spec, "submodule_search_locations", None) or [])
    if not search_locations:
        return []

    patched: List[str] = []
    for package_root in search_locations:
        triton_dir = Path(package_root) / "ops" / "triton"
        if not triton_dir.exists():
            continue
        for source_path in triton_dir.glob("*.py"):
            try:
                source = source_path.read_text(encoding="utf-8")
            except OSError:
                continue
            updated = source
            for old, new in _MAMBA_TRITON_LOG1P_REPLACEMENTS.items():
                updated = updated.replace(old, new)
            if updated == source:
                continue
            source_path.write_text(updated, encoding="utf-8")
            patched.append(str(source_path))
    return patched


def _maybe_get_kaggle_secret(label: str) -> Optional[str]:
    try:
        mod = importlib.import_module("kaggle_secrets")
        client_cls = getattr(mod, "UserSecretsClient", None)
        if client_cls is None:
            return None
        value = client_cls().get_secret(label)
    except Exception:
        return None
    return _normalize_text(value)


def _maybe_get_colab_secret(label: str) -> Optional[str]:
    try:
        colab_mod = importlib.import_module("google.colab")
        userdata = getattr(colab_mod, "userdata", None)
        if userdata is None:
            return None
        value = userdata.get(label)
    except Exception:
        return None
    return _normalize_text(value)


def _resolve_hf_token_common(cli_value: Optional[str] = None) -> Optional[str]:
    """
    Resolve HF token without importing heavy third-party ML libraries.

    Resolution order:
      1. Explicit CLI override
      2. HF_TOKEN / HUGGINGFACE_HUB_TOKEN env vars
      3. Kaggle secret HF_TOKEN
      4. Colab userdata HF_TOKEN
    """
    token = resolve_hf_token(cli_value)
    if token:
        return token

    for secret_name, resolver in (
        ("HF_TOKEN", _maybe_get_kaggle_secret),
        ("HF_TOKEN", _maybe_get_colab_secret),
    ):
        token = _normalize_text(resolver(secret_name))
        if token:
            return token
    return None


def _resolve_github_token_common(cli_value: Optional[str] = None) -> Optional[str]:
    """
    Resolve the GitHub token used for worker->coordinator signaling.

    Resolution order:
      1. Explicit CLI override
      2. GITHUB_TOKEN / GH_TOKEN env vars
      3. Kaggle secret GITHUB_TOKEN / GH_TOKEN
      4. Colab userdata GITHUB_TOKEN / GH_TOKEN
    """
    token = resolve_github_token(cli_value)
    if token:
        return token

    for secret_name, resolver in (
        ("GITHUB_TOKEN", _maybe_get_kaggle_secret),
        ("GH_TOKEN", _maybe_get_kaggle_secret),
        ("GITHUB_TOKEN", _maybe_get_colab_secret),
        ("GH_TOKEN", _maybe_get_colab_secret),
    ):
        token = _normalize_text(resolver(secret_name))
        if token:
            return token
    return None


def _resolve_diloco_worker_id_common(cli_value: Optional[str] = None) -> Optional[str]:
    """
    Resolve the DiLoCo worker id from CLI, environment, or notebook secrets.

    Resolution order:
      1. Explicit CLI override
      2. DILOCO_WORKER_ID / OUROBOROS_DILOCO_WORKER_ID / WORKER_ID env vars
      3. Kaggle secret DILOCO_WORKER_ID
      4. Colab userdata DILOCO_WORKER_ID
    """
    worker_id = resolve_worker_id(cli_value=cli_value)
    if worker_id:
        return worker_id

    for secret_name, resolver in (
        ("DILOCO_WORKER_ID", _maybe_get_kaggle_secret),
        ("DILOCO_WORKER_ID", _maybe_get_colab_secret),
    ):
        worker_id = _normalize_text(resolver(secret_name), uppercase=True)
        if worker_id:
            return worker_id
    return None


def _require_valid_diloco_worker_id(value: Optional[str]) -> str:
    worker_id = _resolve_diloco_worker_id_common(value)
    if worker_id is None:
        raise ValueError(
            "DiLoCo mode requires a worker id. Provide --diloco_worker_id, set "
            "DILOCO_WORKER_ID / OUROBOROS_DILOCO_WORKER_ID, or define a "
            "Kaggle/Colab secret named DILOCO_WORKER_ID."
        )
    return require_known_worker_id(worker_id)


def _wandb_credentials_available() -> bool:
    if resolve_env_alias(os.environ, WANDB_KEY_ALIASES):
        return True

    try:
        import netrc as _netrc

        auth = _netrc.netrc().authenticators("api.wandb.ai")
        if auth is not None and any(auth):
            return True
    except Exception:
        pass
    return False


def _bootstrap_resolve_token() -> Optional[str]:
    return _resolve_hf_token_common()


def _load_mamba_fast_path_symbols() -> Dict[str, Any]:
    """Load the exact fast-path symbols Jamba expects, via the stable submodule paths."""
    import importlib as _il
    import warnings as _warnings

    _patch_mamba_triton_log1p_source()
    _patch_triton_math_log1p()
    _il.invalidate_caches()

    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", category=FutureWarning, module=r"mamba_ssm.*")
        from mamba_ssm.ops.selective_scan_interface import (  # type: ignore
            selective_scan_fn,
            mamba_inner_fn,
        )
        from mamba_ssm.ops.triton.selective_state_update import (  # type: ignore
            selective_state_update,
        )
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update  # type: ignore

    return {
        "selective_scan_fn": selective_scan_fn,
        "selective_state_update": selective_state_update,
        "mamba_inner_fn": mamba_inner_fn,
        "causal_conv1d_fn": causal_conv1d_fn,
        "causal_conv1d_update": causal_conv1d_update,
    }


def _patch_kernel_top_level_exports() -> List[str]:
    """
    Export the fast-path symbols on the top-level packages.

    Older / current Jamba implementations in transformers resolve kernels from
    top-level modules (for example via lazy_load_kernel("mamba-ssm")) and look
    for attributes such as ``selective_state_update`` directly on ``mamba_ssm``.
    The upstream mamba_ssm==1.2.2 package exposes that symbol only from the
    Triton submodule, so Jamba can incorrectly conclude that the fast path is
    unavailable even though the compiled kernels import and execute correctly.
    """
    import importlib as _il

    patched: List[str] = []
    symbols = _load_mamba_fast_path_symbols()

    mamba_ssm = _il.import_module("mamba_ssm")
    causal_conv1d = _il.import_module("causal_conv1d")

    mamba_exports = {
        "selective_scan_fn": symbols["selective_scan_fn"],
        "selective_state_update": symbols["selective_state_update"],
        "mamba_inner_fn": symbols["mamba_inner_fn"],
    }
    conv_exports = {
        "causal_conv1d_fn": symbols["causal_conv1d_fn"],
        "causal_conv1d_update": symbols["causal_conv1d_update"],
    }

    for name, value in mamba_exports.items():
        if getattr(mamba_ssm, name, None) is not value:
            setattr(mamba_ssm, name, value)
            patched.append(f"mamba_ssm.{name}")
    for name, value in conv_exports.items():
        if getattr(causal_conv1d, name, None) is not value:
            setattr(causal_conv1d, name, value)
            patched.append(f"causal_conv1d.{name}")

    return patched


def _bootstrap_verify_fast_path() -> bool:
    """
    Hard verification of the Jamba Mamba fast path:
      1. Import the exact 5 symbols that transformers.models.jamba checks.
      2. Assert none are None (catches API / packaging mismatches).
      3. Run tiny real CUDA/Triton ops for causal_conv1d, selective_scan_fn,
         and selective_state_update.
    """
    import torch as _torch
    import triton.language as _tl

    _patch_triton_math_log1p()
    _syms = _load_mamba_fast_path_symbols()
    selective_scan_fn = _syms["selective_scan_fn"]
    selective_state_update = _syms["selective_state_update"]
    mamba_inner_fn = _syms["mamba_inner_fn"]
    causal_conv1d_fn = _syms["causal_conv1d_fn"]
    causal_conv1d_update = _syms["causal_conv1d_update"]

    _none = [k for k, v in _syms.items() if v is None]
    if _none:
        raise ImportError(
            "Fast-path symbols imported but resolved to None: "
            f"{_none}. This usually means an API / ABI mismatch."
        )

    _x = _torch.randn(1, 4, 8, device="cuda", dtype=_torch.float32)
    _w = _torch.randn(4, 4, device="cuda", dtype=_torch.float32)
    _out = causal_conv1d_fn(_x, _w)
    assert _out.shape == _x.shape, f"Unexpected causal_conv1d_fn shape: {_out.shape}"

    _u = _torch.randn(1, 4, 8, device="cuda", dtype=_torch.float32)
    _delta = _torch.randn(1, 4, 8, device="cuda", dtype=_torch.float32)
    _A = _torch.randn(4, 3, device="cuda", dtype=_torch.float32)
    _B = _torch.randn(1, 3, 8, device="cuda", dtype=_torch.float32)
    _C = _torch.randn(1, 3, 8, device="cuda", dtype=_torch.float32)
    _D = _torch.randn(4, device="cuda", dtype=_torch.float32)
    _scan_out = selective_scan_fn(_u, _delta, _A, _B, _C, _D, delta_softplus=True)
    assert _scan_out.shape == _u.shape, f"Unexpected selective_scan_fn shape: {_scan_out.shape}"

    _state = _torch.randn(1, 4, 3, device="cuda", dtype=_torch.float32)
    _x_step = _torch.randn(1, 4, device="cuda", dtype=_torch.float32)
    _dt = _torch.randn(1, 4, device="cuda", dtype=_torch.float32)
    _dt_bias = _torch.zeros(4, device="cuda", dtype=_torch.float32)
    _A_step = -_torch.rand(4, 3, device="cuda", dtype=_torch.float32) - 1.0
    _B_step = _torch.randn(1, 3, device="cuda", dtype=_torch.float32)
    _C_step = _torch.randn(1, 3, device="cuda", dtype=_torch.float32)
    _has_tl_math_log1p = hasattr(getattr(_tl, "math", None), "log1p")
    if _has_tl_math_log1p:
        _step_out = selective_state_update(
            _state, _x_step, _dt, _A_step, _B_step, _C_step, _D,
            dt_bias=_dt_bias, dt_softplus=True
        )
    else:
        _dt_pos = _torch.rand(1, 4, device="cuda", dtype=_torch.float32) + 0.05
        _step_out = selective_state_update(
            _state, _x_step, _dt_pos, _A_step, _B_step, _C_step, _D,
            dt_bias=_dt_bias, dt_softplus=False
        )
    assert _step_out.shape == _x_step.shape, (
        f"Unexpected selective_state_update shape: {_step_out.shape}"
    )

    _torch.cuda.synchronize()
    return True


def _bootstrap_verify_flash_attention() -> bool:
    """Run a tiny real FlashAttention 2 forward/backward op on the active CUDA device."""
    import torch as _torch
    from flash_attn import flash_attn_func  # type: ignore

    if not _torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    _cc = _torch.cuda.get_device_capability()
    if not _bootstrap_flash_attention_supported(_cc):
        raise RuntimeError(f"FlashAttention 2 is unsupported on sm{_cc[0]}{_cc[1]}")

    _dtype = _torch.bfloat16 if _cc >= (8, 0) else _torch.float16
    _q = _torch.randn(2, 64, 4, 64, device="cuda", dtype=_dtype, requires_grad=True)
    _k = _torch.randn_like(_q, requires_grad=True)
    _v = _torch.randn_like(_q, requires_grad=True)
    _out = flash_attn_func(_q, _k, _v, dropout_p=0.0, causal=True)
    assert _out.shape == _q.shape, f"Unexpected flash_attn_func shape: {_out.shape}"
    _out.float().sum().backward()
    assert _q.grad is not None, "flash_attn_func backward did not produce q.grad"
    _torch.cuda.synchronize()
    return True


def _bootstrap_env_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except (TypeError, ValueError):
        return 0


def _bootstrap_env_world_size() -> int:
    try:
        return max(int(os.environ.get("WORLD_SIZE", "1")), 1)
    except (TypeError, ValueError):
        return 1


def _bootstrap_env_local_rank() -> int:
    raw = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
    try:
        return max(int(raw), 0)
    except (TypeError, ValueError):
        return 0


def _bootstrap_launch_key() -> str:
    override = _normalize_text(os.environ.get("OUROBOROS_BOOTSTRAP_LAUNCH_KEY"))
    if override is not None:
        return override

    import hashlib as _hashlib

    launch_components = [
        os.environ.get("TORCHELASTIC_RUN_ID", ""),
        os.environ.get("MASTER_ADDR", ""),
        os.environ.get("MASTER_PORT", ""),
        os.environ.get("WORLD_SIZE", "1"),
        str(Path(sys.argv[0]).resolve()) if sys.argv else "interactive",
        sys.executable,
        ",".join(_HUB_WHEEL_BASES),
    ]
    raw = "|".join(launch_components)
    return _hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _bootstrap_sync_paths() -> Tuple[Path, Path, Path]:
    root = Path("/tmp/ouroboros_bootstrap_sync") / _bootstrap_launch_key()
    return root, root / "install.ok.json", root / "install.failed.txt"


def _bootstrap_prepare_local_cuda_device(_torch) -> None:
    if not _torch.cuda.is_available():
        return
    local_rank = _bootstrap_env_local_rank()
    device_count = _torch.cuda.device_count()
    if device_count <= 0:
        return
    device_index = local_rank if 0 <= local_rank < device_count else 0
    _torch.cuda.set_device(device_index)


def _bootstrap_shared_install_phases() -> None:
    """
    Run the filesystem-mutating install phases once per launch.
    """
    import importlib as _il
    import torch as _torch

    _bootstrap_prepare_local_cuda_device(_torch)

    print("[bootstrap] Phase 1: pure-Python deps...")
    _r1 = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "transformers>=4.54.0",
         "peft",
         "datasets",
         "tqdm",
         "wandb",
         "bitsandbytes>=0.46.1",
         "accelerate",
         "huggingface_hub",
         "ninja",
         "einops",
         "safetensors"],
        check=False,
    )
    if _r1.returncode != 0:
        print("[bootstrap] WARNING: Phase 1 pip returned non-zero — check output above.")


    print("[bootstrap] Phase 2: arch-aware Hub wheel install...")
    _hf_token = _bootstrap_resolve_token()
    if not _hf_token:
        print("[bootstrap] FATAL: No HF_TOKEN found.")
        print("            Set HF_TOKEN as a Kaggle Secret or environment variable.")
        sys.exit(1)

    _il.invalidate_caches()
    from huggingface_hub import hf_hub_download  # installed in Phase 1

    _cc = _torch.cuda.get_device_capability() if _torch.cuda.is_available() else (0, 0)
    _arch_suffix = f"sm{_cc[0]}{_cc[1]}"
    _arch_list   = f"{_cc[0]}.{_cc[1]}+PTX"
    print(f"[bootstrap]   GPU arch: {_arch_suffix}  (TORCH_CUDA_ARCH_LIST={_arch_list} if build needed)")

    _wheel_dir = Path("/tmp/ouroboros_wheels")
    _wheel_dir.mkdir(exist_ok=True)

    _kaggle_wheel_dir = Path("/kaggle/input/ouroboros-cache/wheels")
    _kaggle_wheel_cache_active = _arch_suffix == "sm75" and _kaggle_wheel_dir.exists()
    if _kaggle_wheel_cache_active:
        print("[bootstrap]   Kaggle dataset wheel cache detected for sm75 fast path ✓")
    elif _kaggle_wheel_dir.exists():
        print(f"[bootstrap]   Kaggle dataset wheel cache present but skipped on {_arch_suffix} (sm75-only cache)")

    _wheel_bases = _bootstrap_wheel_bases_for_cuda_arch(_cc)
    if _FLASH_ATTN_HUB_WHEEL_BASE not in _wheel_bases:
        print(
            f"[bootstrap]   flash-attn skipped on {_arch_suffix}: "
            "FlashAttention 2 requires sm80+."
        )

    for _base in _wheel_bases:
        _hub_filename   = f"{_base}-{_arch_suffix}.whl"
        _local_filename = f"{_base}.whl"
        _local_path     = _wheel_dir / _local_filename

        _downloaded = False
        if _kaggle_wheel_cache_active:
            _pkg_prefix = _base.split("-")[0]
            _matches = list(_kaggle_wheel_dir.glob(f"{_pkg_prefix}*.whl"))
            if _matches:
                _kaggle_whl = max(_matches, key=lambda p: p.stat().st_mtime)
                shutil.copy2(str(_kaggle_whl), str(_local_path))
                print(f"[bootstrap]   Loaded {_local_path.name} from Kaggle dataset cache ✓")
                _downloaded = True

        if not _downloaded:
            try:
                _dl = hf_hub_download(
                    repo_id=_HUB_REPO_ID,
                    filename=_hub_filename,
                    token=_hf_token,
                    local_dir=str(_wheel_dir),
                )
                shutil.copy2(_dl, str(_local_path))
                print(f"[bootstrap]   Downloaded {_hub_filename} ✓")
                _downloaded = True
            except Exception as _dl_err:
                print(f"[bootstrap]   {_hub_filename} not on Hub "
                      f"({type(_dl_err).__name__}).")

        if not _downloaded:
            print(
                f"[bootstrap] FATAL: {_hub_filename} not found on Hub for {_arch_suffix}. "
                "No source compilation fallback. Add the prebuilt wheel to the Hub repo."
            )
            sys.exit(1)
        _r = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--force-reinstall", "--no-deps", str(_local_path)],
            check=False,
        )
        if _r.returncode != 0:
            print(f"[bootstrap] FATAL: pip install failed for {_local_filename}.")
            sys.exit(1)
        print(f"[bootstrap]   Installed {_local_filename} ✓")


def _bootstrap_shared_install_requested() -> bool:
    text = _normalize_text(os.environ.get("OUROBOROS_BOOTSTRAP_SHARED_INSTALL"))
    return text is not None and text.lower() in {"1", "true", "yes", "y", "on"}


def _bootstrap_wait_for_shared_install() -> None:
    world_size = _bootstrap_env_world_size()
    if world_size <= 1 and not _bootstrap_shared_install_requested():
        _bootstrap_shared_install_phases()
        return

    rank = _bootstrap_env_rank()
    root, success_path, failure_path = _bootstrap_sync_paths()
    root.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        if success_path.exists():
            print("[bootstrap] DDP guard: shared install already completed for this launch.")
            return
        failure_path.unlink(missing_ok=True)
        print("[bootstrap] DDP guard: rank 0 performing shared bootstrap install; other ranks will wait.")
        try:
            _bootstrap_shared_install_phases()
        except BaseException:
            import traceback as _traceback
            failure_path.write_text(_traceback.format_exc(), encoding="utf-8")
            raise
        else:
            success_path.write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "completed_at": time.time(),
                        "python": sys.executable,
                        "wheel_bases": _HUB_WHEEL_BASES,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print("[bootstrap] DDP guard: shared bootstrap install complete.")
        return

    prefix = f"[bootstrap][rank={rank}]"
    print(f"{prefix} Waiting for rank 0 shared bootstrap install...")
    deadline = time.time() + (2 * 60 * 60)
    while True:
        if success_path.exists():
            print(f"{prefix} Shared bootstrap install ready; continuing with local shims.")
            return
        if failure_path.exists():
            failure_text = failure_path.read_text(encoding="utf-8", errors="replace").strip()
            print(f"{prefix} FATAL: rank 0 shared bootstrap install failed.")
            if failure_text:
                print(failure_text)
            sys.exit(1)
        if time.time() > deadline:
            print(f"{prefix} FATAL: timed out waiting for rank 0 shared bootstrap install.")
            sys.exit(1)
        time.sleep(2.0)


def _bootstrap_process_local_finalize() -> None:
    """
    Run the process-local bootstrap phases on every rank.
    """
    import importlib as _il
    import torch as _torch

    rank = _bootstrap_env_rank()
    verbose = _bootstrap_env_world_size() == 1 or rank == 0
    prefix = "[bootstrap]" if rank == 0 else f"[bootstrap][rank={rank}]"

    def _info(message: str) -> None:
        if verbose:
            print(f"{prefix} {message}")

    def _always(message: str) -> None:
        print(f"{prefix} {message}")

    _bootstrap_prepare_local_cuda_device(_torch)
    _il.invalidate_caches()

    try:
        import importlib as _il2
        _il2.invalidate_caches()
        import transformers.generation as _tg_mod

        _GENERATION_COMPAT_ALIASES = {
            "GreedySearchDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
            "SampleDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
            "ContrastiveSearchDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
            "BeamSearchDecoderOnlyOutput": "GenerateBeamDecoderOnlyOutput",
            "BeamSampleDecoderOnlyOutput": "GenerateBeamDecoderOnlyOutput",
            "GreedySearchEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
            "SampleEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
            "ContrastiveSearchEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
            "BeamSearchEncoderDecoderOutput": "GenerateBeamEncoderDecoderOutput",
            "BeamSampleEncoderDecoderOutput": "GenerateBeamEncoderDecoderOutput",
        }
        _patched = []
        for _old, _new in _GENERATION_COMPAT_ALIASES.items():
            if getattr(_tg_mod, _old, None) is None:
                _repl = getattr(_tg_mod, _new, None)
                if _repl is not None:
                    setattr(_tg_mod, _old, _repl)
                    _patched.append(_old)
        if _patched:
            _info(f"Shim: patched {len(_patched)} removed transformers.generation names ✓")
        else:
            _info("Shim: all generation names present (no patch needed)")
    except ImportError:
        pass
    except Exception as _shim_err:
        _always(f"WARNING: transformers shim failed: {_shim_err}")


    try:
        _mamba_source_patches = _patch_mamba_triton_log1p_source()
        if _mamba_source_patches:
            _info(
                "Mamba Triton source shim: "
                + ", ".join(Path(path).name for path in _mamba_source_patches)
                + " ✓"
            )
        else:
            _info("Mamba Triton source shim: already aligned")
    except Exception as _mamba_source_patch_err:
        _always(f"WARNING: Mamba Triton source shim failed: {_mamba_source_patch_err}")

    try:
        _patched_exports = _patch_kernel_top_level_exports()
        if _patched_exports:
            _info("Kernel export shim: " + ", ".join(_patched_exports) + " ✓")
        else:
            _info("Kernel export shim: already aligned")
    except Exception as _kernel_patch_err:
        _always(f"WARNING: kernel export shim failed: {_kernel_patch_err}")

    _triton_math_patches = _patch_triton_math_log1p()
    if _triton_math_patches:
        _info("Triton math shim: " + ", ".join(_triton_math_patches) + " ✓")

    _info("Phase 3: verifying mamba fast path (symbol + CUDA op)...")

    try:
        current_device = _torch.cuda.current_device() if _torch.cuda.is_available() else None
        _cc = _torch.cuda.get_device_capability(current_device) if current_device is not None else (0, 0)
        _arch_suffix = f"sm{_cc[0]}{_cc[1]}"
        _gpu_name = _torch.cuda.get_device_name(current_device) if current_device is not None else "no-gpu"
        _info(
            f"  ABI fingerprint: GPU={_gpu_name} {_arch_suffix} | "
            f"CUDA={_torch.version.cuda} | "
            f"PyTorch={_torch.__version__} | "
            f"Python=cp{sys.version_info.major}{sys.version_info.minor}"
        )
        _info(f"  Wheel bases: {_HUB_WHEEL_BASES}")
    except Exception:
        pass

    try:
        _bootstrap_verify_fast_path()
        _info("Mamba fast path: ACTIVE ✓ — profile step time empirically.")
    except Exception as _ve:
        _always(f"FATAL: Mamba fast path verification FAILED: {_ve}")
        _always("         Exiting now (no slow-path fallback — 500s/step is unusable).")
        sys.exit(1)

    try:
        _flash_device = _torch.cuda.current_device() if _torch.cuda.is_available() else None
        _flash_cc = (
            _torch.cuda.get_device_capability(_flash_device)
            if _flash_device is not None
            else (0, 0)
        )
    except Exception:
        _flash_cc = (0, 0)

    if _bootstrap_flash_attention_supported(_flash_cc):
        _info("Phase 4: verifying flash-attn (symbol + CUDA forward/backward)...")
        try:
            _bootstrap_verify_flash_attention()
            _info("flash-attn: ACTIVE ✓ — transformers can use flash_attention_2.")
        except Exception as _flash_err:
            _always(f"FATAL: flash-attn verification FAILED: {_flash_err}")
            _always("         Exiting now; rerun after fixing flash-attn build/cache.")
            sys.exit(1)
    else:
        _info(
            f"flash-attn: skipped on sm{_flash_cc[0]}{_flash_cc[1]} "
            "(requires sm80+); attention loader will use eager fallback."
        )


def _bootstrap_run_local_finalize_padded() -> None:
    world_size = _bootstrap_env_world_size()
    if world_size <= 1:
        _bootstrap_process_local_finalize()
        return

    rank = _bootstrap_env_rank()
    prefix = "[bootstrap]" if rank == 0 else f"[bootstrap][rank={rank}]"
    root, _, _ = _bootstrap_sync_paths()
    phase_dir = root / "local_finalize"
    phase_dir.mkdir(parents=True, exist_ok=True)
    ok_path = phase_dir / f"rank_{rank}.ok"
    fail_path = phase_dir / f"rank_{rank}.failed.txt"
    ok_path.unlink(missing_ok=True)
    fail_path.unlink(missing_ok=True)

    try:
        _bootstrap_process_local_finalize()
    except BaseException:
        import traceback as _traceback
        fail_path.write_text(_traceback.format_exc(), encoding="utf-8")
        raise
    else:
        ok_path.write_text(str(time.time()), encoding="utf-8")

    deadline = time.time() + (30 * 60)
    while True:
        failure_files = sorted(phase_dir.glob("rank_*.failed.txt"))
        if failure_files:
            failure_text = failure_files[0].read_text(encoding="utf-8", errors="replace").strip()
            print(f"{prefix} FATAL: bootstrap local finalize failed on {failure_files[0].name}.")
            if failure_text:
                print(failure_text)
            sys.exit(1)
        if len(list(phase_dir.glob("rank_*.ok"))) >= world_size:
            return
        if time.time() > deadline:
            print(f"{prefix} FATAL: timed out waiting for all ranks to finish bootstrap local finalize.")
            sys.exit(1)
        time.sleep(1.0)


def _bootstrap() -> None:
    """
    Install all dependencies before any third-party imports.
    """
    _bootstrap_wait_for_shared_install()
    _bootstrap_run_local_finalize_padded()

_BOOTSTRAP_COMPLETE = False


def ensure_environment() -> None:
    """Run the original startup bootstrap once, unless argparse is only printing help."""
    global _BOOTSTRAP_COMPLETE
    if _BOOTSTRAP_COMPLETE:
        return
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return
    _bootstrap()
    _BOOTSTRAP_COMPLETE = True
