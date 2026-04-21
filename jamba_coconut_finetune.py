#!/usr/bin/env python3
"""
Stage 3 Coconut-Ouroboros Fine-tuning for Jamba Reasoning 3B
=============================================================
Implements Meta's Coconut progressive replacement curriculum + DGAC halt gate.

Curriculum (run in sequence, each resuming from previous best checkpoint):
  Stage 0:   standard CoT fine-tune; labels on all reasoning steps + answer
  Stage 1..K: replace first k reasoning steps with k latent passes;
              labels shift to supervise only the remaining (k+1..n) steps + answer

Phase 3.4 (--use_halt_gate): DGAC adaptive halt gate added on top of Stage K.

Dataset: read from --data_dir (output of prepare_coconut_dataset.py).
         Canonical format: {id, source, question, steps, answer_full, answer_norm, n_steps}
         Note: 'steps' column from Hub is JSON-encoded; local JSONL has native lists.

References:
  Coconut (Meta, arXiv:2412.06769)
  Jamba Reasoning 3B (AI21, ai21labs/AI21-Jamba-Reasoning-3B, Oct 2025)

Install:
  Self-contained. _bootstrap() runs on every startup.
  Under torchrun, Phase 1/2 are DDP-coordinated so rank 0 performs the shared
  pip / wheel install once while the other ranks wait; process-local shims and
  fast-path verification still run on every rank after the shared install.
    Phase 1: pure-Python deps (transformers, peft, bitsandbytes>=0.46.1, huggingface_hub, ...)
    Phase 2: arch-aware Hub wheel install.
             Detects current GPU (e.g. sm100), tries to download the matching
             arch-encoded wheel (e.g. mamba_ssm-1.2.2-...-sm100.whl) from Hub.
             On 404: source-compiles with correct TORCH_CUDA_ARCH_LIST, uploads
             the arch-encoded wheel to Hub for future sessions, then installs.
             Over time Hub accumulates one wheel per GPU arch — compilation
             becomes rare after the first session on each arch.
    Phase 3: functional verification — imports all 5 mamba fast-path symbols AND
             runs a tiny causal_conv1d CUDA op on real tensors.
             On failure: prints full ABI fingerprint and sys.exit(1).
             There is NO fallback to slow path — a broken kernel wastes 500s/step.

Run (smoke test, Kaggle):
  !python jamba_coconut_finetune.py \
    --data_dir data/coconut_v1 --use_4bit \
    --epochs_per_stage 1 --max_stage 2 --max_samples 200 \
    --max_seq_len 1024 --max_grad_norm 0.3 \
    --session_timeout_hours 1.5 --wandb_mode disabled --output_dir runs/smoke

Run (Phase 3.1 through 3.K, Kaggle Dual GPU):
  !torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
    --data_dir data/coconut_v1 --use_4bit \
    --epochs_per_stage 3 --max_stage 10 --batch_size 2 --grad_accum 8 \
    --max_grad_norm 0.3 \
    --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \
    --push_to_hub \
    --output_dir runs/stage3_curriculum

Run (Phase 3.4, DGAC gate, from Stage K best checkpoint):
  !python jamba_coconut_finetune.py \
    --data_dir data/coconut_v1 --use_4bit \
    --use_halt_gate --resume_from runs/stage3_curriculum/stage_10/best \
    --epochs_per_stage 3 --output_dir runs/stage3_dgac
"""

import argparse
import contextlib
import functools
import json
import importlib
import math
import os
import random
import re as _re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Set critical env vars BEFORE any torch/nccl import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", str(4 * 3600))  # match init_process_group
os.environ.setdefault("NCCL_TIMEOUT", str(4 * 3600))

# Wall-clock start for Kaggle timeout accounting. This must be captured
# before _bootstrap() so dependency install and model load time are included
# in the graceful-exit budget.
_SCRIPT_START = time.perf_counter()

# ── Hub wheel config ──────────────────────────────────────────────────────────
_HUB_WHEEL_BASES = [
    "causal_conv1d-1.6.1-cp312-cp312-linux_x86_64",
    "mamba_ssm-1.2.2-cp312-cp312-linux_x86_64",
]
_HUB_REPO_ID = "WeirdRunner/Ouroboros"

_KNOWN_ARCH_SUFFIXES = [
    "sm70", "sm72", "sm75", "sm80", "sm86", "sm87", "sm89",
    "sm90", "sm100", "sm120", "smunknown",
]




def _resolve_hf_token_common(cli_value: Optional[str] = None) -> Optional[str]:
    """
    Resolve HF token without importing heavy third-party ML libraries.

    Resolution order:
      1. Explicit CLI override
      2. Kaggle secret HF_TOKEN
      3. Colab userdata HF_TOKEN
      4. HF_TOKEN / HUGGINGFACE_HUB_TOKEN env vars
    """
    if cli_value:
        return cli_value

    def _maybe_get_kaggle_secret() -> Optional[str]:
        try:
            mod = importlib.import_module("kaggle_secrets")
            client_cls = getattr(mod, "UserSecretsClient", None)
            if client_cls is None:
                return None
            tok = client_cls().get_secret("HF_TOKEN")
            return tok or None
        except Exception:
            return None

    def _maybe_get_colab_secret() -> Optional[str]:
        try:
            colab_mod = importlib.import_module("google.colab")
            userdata = getattr(colab_mod, "userdata", None)
            if userdata is None:
                return None
            tok = userdata.get("HF_TOKEN")
            return tok or None
        except Exception:
            return None

    for resolver in (_maybe_get_kaggle_secret, _maybe_get_colab_secret):
        tok = resolver()
        if tok:
            return tok
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _bootstrap_resolve_token() -> Optional[str]:
    return _resolve_hf_token_common()

def _load_mamba_fast_path_symbols() -> Dict[str, Any]:
    """Load the exact fast-path symbols Jamba expects, via the stable submodule paths."""
    import importlib as _il
    import warnings as _warnings

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
    import hashlib as _hashlib

    launch_components = [
        os.environ.get("TORCHELASTIC_RUN_ID", ""),
        os.environ.get("MASTER_ADDR", ""),
        os.environ.get("MASTER_PORT", ""),
        os.environ.get("WORLD_SIZE", "1"),
        str(Path(__file__).resolve()) if "__file__" in globals() else "interactive",
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
    from huggingface_hub import hf_hub_download, HfApi  # installed in Phase 1

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

    for _base in _HUB_WHEEL_BASES:
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
                      f"({type(_dl_err).__name__}). Compiling from source...")

        if not _downloaded:
            _parts    = _base.split("-")
            _pkg_name = _parts[0]
            _pkg_ver  = _parts[1]
            if _pkg_name == "mamba_ssm":
                _pip_spec = f"git+https://github.com/state-spaces/mamba.git@v{_pkg_ver}"
            else:
                _pip_spec = f"{_pkg_name.replace('_', '-')}=={_pkg_ver}"

            _env_vars = os.environ.copy()
            _env_vars["MAX_JOBS"] = "4"
            _env_vars["TORCH_CUDA_ARCH_LIST"] = _arch_list

            print(f"[bootstrap]   Building {_pip_spec} "
                  f"(TORCH_CUDA_ARCH_LIST={_arch_list}) ...")
            _build_result = subprocess.run(
                [sys.executable, "-m", "pip", "wheel", _pip_spec,
                 "--no-build-isolation", "--no-deps",
                 "-w", str(_wheel_dir), "--verbose"],
                env=_env_vars, check=False,
            )
            if _build_result.returncode != 0:
                print(f"[bootstrap] FATAL: Source build failed for {_pip_spec}.")
                sys.exit(1)

            _found = [
                f for f in _wheel_dir.glob(f"{_pkg_name}*.whl")
                if not any(f.name.endswith(f"-{s}.whl") for s in _KNOWN_ARCH_SUFFIXES)
            ]
            if not _found:
                print(f"[bootstrap] FATAL: pip wheel succeeded but no .whl found "
                      f"for {_pkg_name}.")
                sys.exit(1)
            _built_whl = max(_found, key=lambda p: p.stat().st_mtime)

            if _built_whl.resolve() != _local_path.resolve():
                shutil.copy2(str(_built_whl), str(_local_path))

            print(f"[bootstrap]   Build succeeded: {_built_whl.name}")

            _arch_whl_path = _wheel_dir / _hub_filename
            shutil.copy2(str(_local_path), str(_arch_whl_path))
            try:
                _api = HfApi(token=_hf_token)
                _api.create_repo(repo_id=_HUB_REPO_ID, repo_type="model",
                                 private=True, exist_ok=True, token=_hf_token)
                _api.upload_file(
                    path_or_fileobj=str(_arch_whl_path),
                    path_in_repo=_hub_filename,
                    repo_id=_HUB_REPO_ID,
                    repo_type="model",
                    token=_hf_token,
                    commit_message=f"Add wheel: {_hub_filename}",
                )
                print(f"[bootstrap]   Uploaded {_hub_filename} to Hub ✓ "
                      f"(future sessions on {_arch_suffix} skip compilation)")
            except Exception as _up_err:
                print(f"[bootstrap]   [warn] Hub upload failed: {_up_err}")

        _r = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--force-reinstall", "--no-deps", str(_local_path)],
            check=False,
        )
        if _r.returncode != 0:
            print(f"[bootstrap] FATAL: pip install failed for {_local_filename}.")
            sys.exit(1)
        print(f"[bootstrap]   Installed {_local_filename} ✓")


def _bootstrap_wait_for_shared_install() -> None:
    world_size = _bootstrap_env_world_size()
    if world_size <= 1:
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
        _patched_exports = _patch_kernel_top_level_exports()
        if _patched_exports:
            _info("Kernel export shim: " + ", ".join(_patched_exports) + " ✓")
        else:
            _info("Kernel export shim: already aligned")
    except Exception as _kernel_patch_err:
        _always(f"WARNING: kernel export shim failed: {_kernel_patch_err}")

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


_bootstrap()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import transformers as _tf

    _TF_VERSION = tuple(int(x) for x in _tf.__version__.split(".")[:2])
    from peft import LoraConfig, get_peft_model
    from tqdm.auto import tqdm
except ImportError as exc:
    sys.exit(
        f"Missing dependency after bootstrap: {exc}\n"
        "Check pip install output above for errors."
    )

MODEL_ID = "ai21labs/AI21-Jamba-Reasoning-3B"

# conv1d intentionally excluded - shape incompatible with standard LoRA
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "x_proj",
    "dt_proj",
    "out_proj",
]

GEN_PROMPTS = [
    "What is 15 + 27?",
    "Write a Python function that returns the factorial of n.",
    "What is the capital of Japan?",
    "Explain what a neural network is in simple terms.",
    "Solve for x: 3x + 6 = 21.",
]

# Fallback estimated optimiser steps per worker per DiLoCo round.
# Actual value is computed at runtime in run_diloco_worker().
_DILOCO_SHARD_STEP_FALLBACK = 385

_LAST_NUM = _re.compile(r"[\d,]+(?:\.\d+)?")
_CHAT_TEMPLATE_WARNED = False


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", str(_rank())))


def _parse_stage_dir_name(name: str) -> Optional[int]:
    if not name.startswith("stage_"):
        return None
    suffix = name.split("stage_", 1)[1]
    return int(suffix) if suffix.isdigit() else None


def _read_training_state(ckpt_dir: Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(ckpt_dir / "training_state.pt", map_location=map_location)


def _maybe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@functools.lru_cache(maxsize=None)
def _amp_dtype(device: torch.device) -> torch.dtype:
    """
    Select autocast dtype based on hardware capability.

    BF16 requires native tensor core support (sm80+ / Ampere).
    On sm75 (T4) and earlier, torch.cuda.is_bf16_supported() returns True under
    CUDA 12 due to software emulation, but matrix multiplications fall back to
    FP32 paths (~8 TFLOPS) rather than FP16 tensor cores (~65 TFLOPS on T4).

    lru_cache: result is memoised per device — safe because the GPU assigned to
    each rank never changes within a process lifetime. Avoids repeated
    get_device_capability() calls in the hot training loop.
    """
    if device.type == "cuda":
        cc = torch.cuda.get_device_capability(device)
        if cc >= (8, 0):  # Ampere+ (A100, H100, RTX 3090+): native BF16 tensor cores
            return torch.bfloat16
        return torch.float16  # T4 (sm75), V100 (sm70): FP16 tensor cores; BF16 is FP32 fallback
    return torch.float32


def _autocast_ctx(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _extract_last_hidden_state(outputs, context: str) -> torch.Tensor:
    last_hidden = getattr(outputs, "last_hidden_state", None)
    assert last_hidden is not None, (
        f"{context}: model backbone returned None for last_hidden_state. "
        "Pass output_hidden_states=True and use out.hidden_states[-1] instead."
    )
    return last_hidden


def _unwrap_peft_model(model):
    try:
        base = model.get_base_model()
        if base is not None:
            return base
    except Exception:
        pass
    return model




def _cache_model_lookup(model, cache_name: str, resolver):
    cached = getattr(model, cache_name, None)
    if cached is not None:
        return cached
    resolved = resolver()
    setattr(model, cache_name, resolved)
    return resolved




def _get_backbone(model):
    def _resolve():
        base = _unwrap_peft_model(model)
        candidates = [
            getattr(model, "model", None),
            getattr(base, "model", None),
            getattr(getattr(base, "model", None), "model", None),
        ]
        for cand in candidates:
            if cand is not None and hasattr(cand, "forward") and hasattr(cand, "embed_tokens"):
                return cand
        raise AttributeError(
            "Cannot locate backbone model. Inspect:\n"
            "  print(type(model))\n"
            "  print([n for n, _ in model.named_modules()][:40])"
        )

    return _cache_model_lookup(model, "_ouro_cache_backbone", _resolve)


def _get_embed_tokens(model):
    def _resolve():
        backbone = _get_backbone(model)
        if getattr(backbone, "embed_tokens", None) is not None:
            return backbone.embed_tokens

        base = _unwrap_peft_model(model)
        for obj in [model, base]:
            if obj is None:
                continue
            try:
                emb = obj.get_input_embeddings()
                if emb is not None:
                    return emb
            except Exception:
                continue
        raise AttributeError(
            "Cannot locate embed_tokens. Inspect:\n"
            "  print([n for n, _ in model.named_modules()][:40])"
        )

    return _cache_model_lookup(model, "_ouro_cache_embed_tokens", _resolve)

def _get_lm_head(model):
    def _resolve():
        base = _unwrap_peft_model(model)
        for obj in [model, base, getattr(base, "model", None)]:
            if obj is None:
                continue
            head = getattr(obj, "lm_head", None)
            if head is not None:
                return head
        raise AttributeError("Cannot locate lm_head. Inspect model.named_modules().")

    return _cache_model_lookup(model, "_ouro_cache_lm_head", _resolve)

def _maybe_apply_chat_template(tokenizer, question: str) -> str:
    global _CHAT_TEMPLATE_WARNED
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        if not _CHAT_TEMPLATE_WARNED and _is_main_process():
            print("  [warn] tokenizer.apply_chat_template failed; using plain prompt fallback.")
            _CHAT_TEMPLATE_WARNED = True
        return f"User: {question}\nAssistant: "


def _patch_transformers_jamba_fast_path_globals() -> bool:
    """
    Patch transformers.models.jamba.modeling_jamba module globals so the
    runtime fast-path check sees the verified kernel symbols.
    """
    try:
        import importlib as _il
        _il.invalidate_caches()
        import transformers.models.jamba.modeling_jamba as _jamba_mod
        symbols = _load_mamba_fast_path_symbols()
        changed = False
        for name, value in symbols.items():
            if getattr(_jamba_mod, name, None) is not value:
                setattr(_jamba_mod, name, value)
                changed = True
        is_available = all(symbols.values())
        if getattr(_jamba_mod, "is_fast_path_available", None) != is_available:
            _jamba_mod.is_fast_path_available = is_available
            changed = True
        return is_available
    except Exception:
        return False


def _probe_jamba_runtime_fast_path(model, device: torch.device, amp_dtype: torch.dtype) -> None:
    if device.type != "cuda":
        return

    backbone = _get_backbone(model)
    probe_ids = torch.tensor([[1, 2]], dtype=torch.long, device=device)
    probe_mask = torch.ones_like(probe_ids, dtype=torch.bool, device=device)

    def _run_once() -> None:
        with torch.no_grad():
            with _autocast_ctx(device, amp_dtype):
                outputs = backbone(input_ids=probe_ids, attention_mask=probe_mask, use_cache=False)
                _ = _extract_last_hidden_state(outputs, "post-load Jamba runtime probe")
        torch.cuda.synchronize()

    try:
        _run_once()
    except ValueError as exc:
        if "Fast Mamba kernels are not available" not in str(exc):
            raise
        if _is_main_process():
            print("  [warn] Jamba runtime probe hit stale fast-path globals; patching transformers Jamba module and retrying once.")
        if not _patch_transformers_jamba_fast_path_globals():
            raise SystemExit(
                "Jamba runtime probe failed and transformers Jamba globals could not be refreshed. "
                "This environment would fall back to an unusably slow path."
            ) from exc
        _run_once()
        if _is_main_process():
            print("  [ok] Jamba runtime probe passed after fast-path refresh.")


def _safe_from_pretrained(model_id: str, load_kwargs: Dict[str, Any]):
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception as exc:
        message = str(exc)
        retry_kwargs = dict(load_kwargs)
        changed = False
        for key in ["use_mamba_kernels", "attn_implementation"]:
            if key in retry_kwargs and key in message:
                retry_kwargs.pop(key, None)
                changed = True
                if _is_main_process():
                    print(f"  [warn] model load rejected '{key}'; retrying without it.")
        if changed:
            return AutoModelForCausalLM.from_pretrained(model_id, **retry_kwargs)
        raise


def _distributed_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def all_reduce_gradients(parameters: Iterable[torch.nn.Parameter], world_size: int) -> None:
    if world_size <= 1 or not _distributed_is_initialized():
        return
    for param in parameters:
        if param.grad is None:
            continue
        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
        param.grad.div_(world_size)


def broadcast_parameters(parameters: Iterable[torch.nn.Parameter], src: int = 0) -> None:
    if _world_size() <= 1 or not _distributed_is_initialized():
        return
    for param in parameters:
        torch.distributed.broadcast(param.data, src=src)


def broadcast_bool(value: bool, device: torch.device) -> bool:
    if not _distributed_is_initialized() or _world_size() <= 1:
        return value
    tensor = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
    torch.distributed.broadcast(tensor, src=0)
    return bool(tensor.item())


def _ddp_sum(values: List[float], device: torch.device) -> List[float]:
    if not _distributed_is_initialized() or _world_size() <= 1:
        return [float(v) for v in values]
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.tolist()


def barrier() -> None:
    if not (_distributed_is_initialized() and _world_size() > 1):
        return
    if torch.cuda.is_available():
        try:
            torch.distributed.barrier(device_ids=[_local_rank()])
            return
        except TypeError:
            pass
    torch.distributed.barrier()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stage_grad_clip_norm(args: argparse.Namespace, stage_k: int) -> float:
    clip_norm = float(args.max_grad_norm)
    if stage_k >= 2:
        return min(clip_norm, 0.3)
    return clip_norm


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    action = getattr(argparse, "BooleanOptionalAction", None)
    if action is not None:
        parser.add_argument(name, action=action, default=default, help=help_text)
    else:
        if default:
            parser.add_argument(name, action="store_true", default=default, help=help_text)
        else:
            parser.add_argument(name, action="store_false", default=default, help=help_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jamba Reasoning 3B Coconut-Ouroboros fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--max_seq_len", type=int, default=1024)

    # LoRA / QLoRA
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="QLoRA (4-bit NF4). Requires CUDA + bitsandbytes.",
    )
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Dataset
    parser.add_argument("--data_dir", default="data/coconut_v1")
    parser.add_argument("--max_samples", type=int, default=None)

    # Curriculum
    parser.add_argument(
        "--max_stage",
        type=int,
        default=None,
        help="Override K. None = read n_steps_median from stats.json.",
    )
    parser.add_argument("--epochs_per_stage", type=int, default=3)
    parser.add_argument("--stage_0_epochs", type=int, default=None)

    # DGAC halt gate (Phase 3.4)
    parser.add_argument("--use_halt_gate", action="store_true")
    parser.add_argument("--halt_threshold", type=float, default=0.5)
    parser.add_argument("--dgac_lambda_ponder_max", type=float, default=0.01)
    parser.add_argument("--dgac_lambda_diversity", type=float, default=0.1)
    parser.add_argument("--dgac_tau", type=float, default=0.9)
    parser.add_argument("--dgac_warmup_steps", type=int, default=200)
    parser.add_argument("--dgac_ramp_steps", type=int, default=300)

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Base gradient clip norm. Stages k>=2 are additionally capped at 0.3.",
    )
    _add_bool_arg(parser, "--grad_checkpoint", True, "Enable gradient checkpointing.")
    parser.add_argument("--seed", type=int, default=42)

    # Session timeout (MANDATORY for Kaggle)
    parser.add_argument("--session_timeout_hours", type=float, default=11.0)
    parser.add_argument("--graceful_exit_buffer_minutes", type=float, default=20.0)
    parser.add_argument(
        "--val_skip_buffer_minutes",
        type=float,
        default=60.0,
        help=(
            "Skip val+gen if remaining session time is below this threshold (minutes). "
            "With DDP val on Dual T4 (val_batch_size=2, 50 acc samples), val takes ~37min. "
            "Default 60min provides a 23min safety margin."
        ),
    )

    # Hub checkpoint sync
    parser.add_argument("--push_to_hub", action="store_true",
        help="Push checkpoints to HF Hub after each epoch save.")
    parser.add_argument("--hf_token", default=None,
        help="HF write token. Falls back to HF_TOKEN env var.")
    parser.add_argument("--hf_repo_id", default="WeirdRunner/Ouroboros",
        help="HF model repo to sync checkpoints to.")
    parser.add_argument("--hf_stage_subdir", default="runs/stage3",
        help="Remote subdirectory inside the HF repo for Stage 3 checkpoints.")

    # DiLoCo
    parser.add_argument("--diloco_mode", action="store_true",
        help="Enable DiLoCo parallel training mode.")
    parser.add_argument("--diloco_worker_id", default=None, choices=["A", "B", "C"],
        help="This worker's identity. Required when --diloco_mode is set.")
    parser.add_argument("--diloco_outer_lr", type=float, default=0.7,
        help="Outer SGD learning rate for DiLoCo aggregation. Default: 0.7 (DiLoCo paper).")
    parser.add_argument("--diloco_min_workers", type=int, default=2,
        help="Minimum workers needed for coordinator to aggregate (default: 2 of 3).")
    parser.add_argument("--diloco_state_repo", default="WeirdRunner/Ouroboros",
        help="HF Hub repo used as shared state store.")
    parser.add_argument("--diloco_signal_repo", default="deveshpat/Ouroboros",
        help="GitHub repo to push coordinator trigger signals to.")
    parser.add_argument("--diloco_run_val", action="store_true",
        help="Run val pass before training begins (used by the first worker of a new stage).")

    # I/O
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--output_dir", default="runs/stage3")
    parser.add_argument("--keep_checkpoints_per_stage", type=int, default=2)

    # Monitoring
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=1,
        help="Batch size for val forward passes. Keep at 1 to avoid OOM.",
    )
    _add_bool_arg(parser, "--gen_every_stage", True, "Run generation callback at stage end.")
    parser.add_argument("--gen_max_tokens", type=int, default=200)

    # wandb
    parser.add_argument("--wandb_project", default="ouroboros-stage3-jamba")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )

    return parser.parse_args()


def _wandb_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(vars(args))
    for key in ["hf_token", "_resolved_hf_token"]:
        if key in config and config[key] is not None:
            config[key] = "***"
    return config


class HaltGate(nn.Module):
    """
    Halt gate for DGAC. Zero-initialized: outputs ~0.5 at start of Phase 3.4.
    Input: h_curr [B, D] + h_prev [B, D] at question-end position.
    Output: halt_prob [B].
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h_curr: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(torch.cat([h_curr, h_prev], dim=-1))).squeeze(-1)


def compute_dgac_lambda1(step: int, warmup: int, ramp: int, lmax: float) -> float:
    """lambda_1=0 for warmup steps, then linearly ramps to lmax over ramp steps."""
    if step < warmup:
        return 0.0
    return lmax * min((step - warmup) / max(ramp, 1), 1.0)


def load_model_and_tokenizer(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, Any, int, int]:
    is_main = _is_main_process()
    rank = _rank()
    amp_dtype = _amp_dtype(device)

    # ── [perf] Prominent GPU capability log — visible every session ──────────
    if is_main and device.type == "cuda":
        _cc = torch.cuda.get_device_capability(device)
        _gpu_name = torch.cuda.get_device_name(device)
        _vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(
            f"  [GPU] {_gpu_name}  cc=sm{_cc[0]}{_cc[1]}  "
            f"VRAM={_vram_gb:.0f}GB  amp_dtype={str(amp_dtype).replace('torch.', '')}"
        )

    if args.use_4bit and device.type != "cuda":
        raise SystemExit("--use_4bit requires CUDA + bitsandbytes.")

    if is_main:
        print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lat_token = "<|lat|>"
    existing_id = tokenizer.convert_tokens_to_ids(lat_token)
    if existing_id is None or existing_id == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [lat_token]})
    lat_token_id = tokenizer.convert_tokens_to_ids(lat_token)
    if is_main:
        print(f"  <|lat|> token id: {lat_token_id}  vocab: {len(tokenizer)}")

    attn_impl = "eager"
    if device.type == "cuda":
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            if is_main:
                print("  flash-attn available: using flash_attention_2")
        except ImportError:
            if is_main:
                print("  flash-attn not installed: falling back to eager attention")
    elif is_main:
        print("  non-CUDA device: using eager attention")

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_impl,
    }

    if device.type == "cuda":
        load_kwargs["device_map"] = {"": device.index if device.index is not None else rank}

    _mamba_fast_path = device.type == "cuda"
    if not _mamba_fast_path:
        load_kwargs["use_mamba_kernels"] = False
    elif is_main:
        print("  mamba CUDA kernels: fast path ACTIVE (verified at bootstrap)")

    if args.use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=amp_dtype if amp_dtype != torch.float32 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = amp_dtype if device.type == "cuda" else torch.float32

    if is_main:
        print(f"Loading model: {args.model_id}")
        print(f"  device={device} amp_dtype={str(amp_dtype).replace('torch.', '')}")

    if _mamba_fast_path:
        _patch_transformers_jamba_fast_path_globals()

    model = _safe_from_pretrained(args.model_id, load_kwargs)
    model.config.use_cache = False

    if _mamba_fast_path:
        _patch_transformers_jamba_fast_path_globals()
        _probe_jamba_runtime_fast_path(model, device, amp_dtype)

    embed_module = _get_embed_tokens(model)
    if hasattr(embed_module, "num_embeddings"):
        embed_size = int(embed_module.num_embeddings)
    else:
        embed_size = int(embed_module.weight.shape[0])
    if len(tokenizer) > embed_size:
        if is_main:
            print(f"  Resizing embed_tokens: {embed_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # ── [perf] Auto-disable gradient checkpointing on high-VRAM GPUs ────────
    # GC is mandatory on T4 (16GB) to avoid OOM at k>=2, but wastes 20-40%
    # compute on A100 (80GB) where the full model fits in VRAM without recomputation.
    if args.grad_checkpoint and device.type == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        if total_vram_gb >= 40.0:
            args.grad_checkpoint = False
            if is_main:
                print(
                    f"  [perf] {total_vram_gb:.0f}GB VRAM detected: "
                    "disabling gradient checkpointing (not needed, saves ~20-40%)."
                )

    if args.use_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.grad_checkpoint,
        )
    else:
        if args.grad_checkpoint:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        elif device.type != "cuda":
            model = model.to(device)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if device.type != "cuda" and not args.use_4bit:
        model = model.to(device)

    if is_main and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    _ = _get_embed_tokens(model)
    _ = _get_lm_head(model)
    _ = _get_backbone(model)
    if is_main:
        print("  embed_tokens and lm_head paths verified after PEFT wrap.")

    d_model = int(getattr(model.config, "hidden_size"))
    if is_main:
        num_layers = getattr(model.config, "num_hidden_layers", "?")
        print(f"  d_model={d_model}  layers={num_layers}")
    return model, tokenizer, d_model, lat_token_id


def _download_dataset_from_hub(
    data_dir: Path,
    hf_repo_id: str = "WeirdRunner/Ouroboros",
    hf_config: str = "coconut-v1",
) -> None:
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    is_main = _is_main_process()
    if is_main:
        print(f"  [data] Local files missing. Downloading {hf_repo_id}[{hf_config}] from Hub...")

    data_dir.mkdir(parents=True, exist_ok=True)

    def _write_split(split_name: str, hf_split: str, out_path: Path) -> List[Dict[str, Any]]:
        try:
            ds = hf_load_dataset(hf_repo_id, hf_config, split=hf_split, token=True)
        except Exception as exc:
            if is_main:
                print(f"  [data] Could not load split '{hf_split}': {exc}")
            return []

        rows: List[Dict[str, Any]] = []
        with out_path.open("w", encoding="utf-8") as fh:
            for row in ds:
                steps = row.get("steps", [])
                if isinstance(steps, str):
                    try:
                        steps = json.loads(steps)
                    except json.JSONDecodeError:
                        steps = [steps]
                sample = {
                    "id":          row.get("id", ""),
                    "source":      row.get("source", ""),
                    "question":    row.get("question", ""),
                    "steps":       steps,
                    "answer_full": row.get("answer_full", ""),
                    "answer_norm": row.get("answer_norm", ""),
                    "n_steps":     int(row.get("n_steps", len(steps))),
                }
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                rows.append(sample)

        if is_main:
            print(f"  [data] {split_name}: {len(rows)} samples -> {out_path}")
        return rows

    train_rows = _write_split("train", "train",      data_dir / "train.jsonl")
    val_rows   = _write_split("val",   "validation", data_dir / "val.jsonl")

    def _quick_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {}
        n_steps = [r["n_steps"] for r in rows]
        by_source: Dict[str, int] = {}
        for r in rows:
            by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        sorted_steps = sorted(n_steps)
        return {
            "n_samples":      len(rows),
            "n_steps_mean":   round(sum(n_steps) / len(n_steps), 2),
            "n_steps_min":    min(n_steps),
            "n_steps_max":    max(n_steps),
            "n_steps_median": sorted_steps[len(sorted_steps) // 2],
            "by_source":      by_source,
        }

    stats = {"train": _quick_stats(train_rows), "val": _quick_stats(val_rows)}
    with (data_dir / "stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    if is_main:
        t = stats.get("train", {})
        print(f"  [data] stats.json written. "
              f"median_steps={t.get('n_steps_median')}  "
              f"recommended --max_stage={t.get('n_steps_median')}")


def load_canonical_dataset(
    data_dir: Path,
    max_samples: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    train_path = data_dir / "train.jsonl"
    val_path   = data_dir / "val.jsonl"
    stats_path = data_dir / "stats.json"

    if not train_path.exists():
        _download_dataset_from_hub(data_dir)

    if not train_path.exists():
        raise FileNotFoundError(
            f"train.jsonl not found at {train_path} and Hub download failed.\n"
            "Run: python prepare_coconut_dataset.py --output_dir data/coconut_v1 --push_to_hub"
        )

    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                steps = row.get("steps")
                if isinstance(steps, str):
                    try:
                        row["steps"] = json.loads(steps)
                    except json.JSONDecodeError:
                        row["steps"] = [steps]
                rows.append(row)
        return rows

    train = _load_jsonl(train_path)
    val   = _load_jsonl(val_path) if val_path.exists() else []
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

    if max_samples is not None:
        n_val   = max(1, max_samples // 20) if val else 0
        n_train = max(max_samples - n_val, 0)
        train   = train[:n_train]
        val     = val[:n_val] if n_val else []

    if _is_main_process():
        print(f"  Loaded {len(train)} train / {len(val)} val from {data_dir}")
        if stats:
            t = stats.get("train", {})
            print(
                f"  Step stats: median={t.get('n_steps_median')} "
                f"mean={t.get('n_steps_mean')} max={t.get('n_steps_max')}"
            )
    return train, val, stats


def get_max_stage(args: argparse.Namespace, stats: Dict[str, Any]) -> int:
    if args.max_stage is not None:
        return int(args.max_stage)
    median = stats.get("train", {}).get("n_steps_median")
    if median is not None:
        if _is_main_process():
            print(f"  --max_stage not set; using n_steps_median={median} from stats.json")
        return int(median)
    if _is_main_process():
        print("  [warn] --max_stage not set and stats.json absent; defaulting to 10")
    return 10


def build_sample_at_stage(
    tokenizer,
    sample: Dict[str, Any],
    stage_k: int,
    lat_token_id: int,
    max_seq_len: int,
) -> Optional[Dict[str, Any]]:
    question = str(sample.get("question", "")).strip()
    if not question:
        return None

    prefix_text = _maybe_apply_chat_template(tokenizer, question)
    q_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    steps_raw = sample.get("steps") or []
    if isinstance(steps_raw, str):
        try:
            steps_raw = json.loads(steps_raw)
        except json.JSONDecodeError:
            steps_raw = [steps_raw]
    steps = [str(s) for s in steps_raw if str(s).strip()]

    n_latent = min(int(stage_k), len(steps))
    remaining_steps = steps[n_latent:]

    supervised_ids: List[int] = []
    for step_text in remaining_steps:
        supervised_ids.extend(tokenizer.encode(step_text + "\n", add_special_tokens=False))

    answer_text = str(sample.get("answer_full", ""))
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer_ids.append(int(tokenizer.eos_token_id))
    supervised_ids.extend(answer_ids)

    if not supervised_ids:
        return None

    total = len(q_ids) + n_latent + len(supervised_ids)
    if total > max_seq_len:
        allowed = max_seq_len - len(q_ids) - n_latent
        if allowed < 4:
            return None
        supervised_ids = supervised_ids[:allowed]

    full_ids = q_ids + [lat_token_id] * n_latent + supervised_ids
    labels = [-100] * len(q_ids) + [-100] * n_latent + supervised_ids
    assert len(full_ids) == len(labels)

    return {
        "full_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "q_len": len(q_ids),
        "n_latent": n_latent,
        "answer_norm": str(sample.get("answer_norm", "")),
    }


def collate_stage_k(samples: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(s["full_ids"].size(0) for s in samples)
    batch_size = len(samples)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    q_lens = torch.zeros(batch_size, dtype=torch.long)
    n_latents = torch.zeros(batch_size, dtype=torch.long)
    for i, sample in enumerate(samples):
        seq_len = sample["full_ids"].size(0)
        input_ids[i, :seq_len] = sample["full_ids"]
        labels[i, :seq_len] = sample["labels"]
        attn_mask[i, :seq_len] = True
        q_lens[i] = int(sample["q_len"])
        n_latents[i] = int(sample["n_latent"])
    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "labels": labels,
        "q_lens": q_lens,
        "n_latents": n_latents,
        "pad_id": torch.tensor(int(pad_id), dtype=torch.long),
    }


def _compute_ce_sum_and_count(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1)
    valid = shift_labels != -100
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return logits.new_zeros((), dtype=torch.float32), 0
    ce_sum = F.cross_entropy(shift_logits[valid], shift_labels[valid], reduction="sum")
    return ce_sum, n_valid


def _build_question_context(
    all_embeds: torch.Tensor,
    q_lens: torch.Tensor,
    pad_embed: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(all_embeds.size(0))
    if q_lens.numel() == 0:
        empty_ctx = all_embeds.new_empty((batch_size, 0, all_embeds.size(-1)))
        empty_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=all_embeds.device)
        return empty_ctx, empty_mask

    max_q_len = int(q_lens.max().item())
    if max_q_len <= 0:
        empty_ctx = all_embeds.new_empty((batch_size, 0, all_embeds.size(-1)))
        empty_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=all_embeds.device)
        return empty_ctx, empty_mask

    ctx = all_embeds[:, :max_q_len, :].clone()
    positions = torch.arange(max_q_len, device=all_embeds.device).unsqueeze(0)
    ctx_mask = positions < q_lens.unsqueeze(1)
    pad_value = pad_embed.to(device=all_embeds.device, dtype=ctx.dtype).view(1, 1, -1)
    ctx = torch.where(ctx_mask.unsqueeze(-1), ctx, pad_value)
    return ctx, ctx_mask


def _run_latent_passes(
    model,
    ctx: torch.Tensor,
    ctx_mask: torch.Tensor,
    n_latent,
    halt_gate: Optional[HaltGate],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    backbone = _get_backbone(model)

    if isinstance(n_latent, int):
        target_steps = torch.full((ctx.size(0),), int(n_latent), device=device, dtype=torch.long)
        scalar_input = True
    elif isinstance(n_latent, torch.Tensor):
        target_steps = n_latent.to(device=device, dtype=torch.long).view(-1)
        scalar_input = False
    else:
        target_steps = torch.tensor(list(n_latent), device=device, dtype=torch.long)
        scalar_input = False

    batch_size = int(ctx.size(0))
    actual_k = torch.zeros(batch_size, dtype=torch.long, device=device)
    prev_hidden = ctx.new_zeros((batch_size, ctx.size(-1)))
    halted = torch.zeros(batch_size, dtype=torch.bool, device=device)

    max_steps = int(target_steps.max().item()) if target_steps.numel() > 0 else 0
    for latent_step in range(max_steps):
        active_indices = ((target_steps > latent_step) & (~halted)).nonzero(as_tuple=False).flatten()
        if active_indices.numel() == 0:
            break

        prefix_lens = ctx_mask[active_indices].sum(dim=1).to(dtype=torch.long)
        max_prefix_len = int(prefix_lens.max().item())
        prefix_embeds = ctx[active_indices, :max_prefix_len, :].clone()
        prefix_positions = torch.arange(max_prefix_len, device=device).unsqueeze(0)
        prefix_mask = prefix_positions < prefix_lens.unsqueeze(1)
        if max_prefix_len > 0:
            prefix_embeds = torch.where(
                prefix_mask.unsqueeze(-1),
                prefix_embeds,
                prefix_embeds.new_zeros((1, 1, prefix_embeds.size(-1))),
            )

        with _autocast_ctx(device, amp_dtype):
            outputs = backbone(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_mask,
                use_cache=False,
            )
        hidden = _extract_last_hidden_state(outputs, f"latent pass step={latent_step}")
        last_positions = prefix_lens - 1
        h_step = hidden[torch.arange(active_indices.numel(), device=device), last_positions, :]

        append_mask = torch.ones(active_indices.numel(), dtype=torch.bool, device=device)
        if halt_gate is not None:
            has_prev = actual_k[active_indices] > 0
            if bool(has_prev.any().item()):
                halt_probs = halt_gate(
                    h_step[has_prev].to(dtype=torch.float32),
                    prev_hidden[active_indices[has_prev]].to(dtype=torch.float32),
                )
                halt_now = halt_probs > args.halt_threshold
                if bool(halt_now.any().item()):
                    blocked_local = has_prev.nonzero(as_tuple=False).flatten()[halt_now]
                    append_mask[blocked_local] = False
                    halted[active_indices[blocked_local]] = True

        next_col = ctx.new_zeros((batch_size, 1, ctx.size(-1)))
        next_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        if bool(append_mask.any().item()):
            append_indices = active_indices[append_mask]
            append_hidden = h_step[append_mask].to(dtype=ctx.dtype)
            next_col[append_indices, 0, :] = append_hidden
            next_mask[append_indices, 0] = True
            actual_k[append_indices] += 1
            prev_hidden[append_indices] = append_hidden

        ctx = torch.cat([ctx, next_col], dim=1)
        ctx_mask = torch.cat([ctx_mask, next_mask], dim=1)

    if ctx_mask.numel() > 0 and ctx_mask.size(1) > 0:
        active_cols = ctx_mask.any(dim=0)
        if bool(active_cols.any().item()):
            last_active_col = int(active_cols.nonzero(as_tuple=False).max().item()) + 1
            ctx = ctx[:, :last_active_col, :]
            ctx_mask = ctx_mask[:, :last_active_col]
        else:
            ctx = ctx[:, :0, :]
            ctx_mask = ctx_mask[:, :0]

    return_k: Any = int(actual_k[0].item()) if scalar_input and batch_size == 1 else actual_k
    return ctx, ctx_mask, return_k


def _collect_latent_hidden_sequences(
    latent_ctx: torch.Tensor,
    max_q_len: int,
    actual_n_latents: torch.Tensor,
) -> List[List[torch.Tensor]]:
    hidden_sequences: List[List[torch.Tensor]] = [[] for _ in range(int(latent_ctx.size(0)))]
    max_steps = int(actual_n_latents.max().item()) if actual_n_latents.numel() > 0 else 0
    for latent_step in range(max_steps):
        active_indices = (actual_n_latents > latent_step).nonzero(as_tuple=False).flatten()
        if active_indices.numel() == 0:
            break
        step_hidden = latent_ctx[active_indices, max_q_len + latent_step, :]
        for local_idx, sample_idx in enumerate(active_indices.tolist()):
            hidden_sequences[sample_idx].append(step_hidden[local_idx : local_idx + 1])
    return hidden_sequences


def _compute_batched_halt_metrics(
    hidden_sequences: List[List[torch.Tensor]],
    actual_n_latents: torch.Tensor,
    halt_gate: HaltGate,
    device: torch.device,
    args: argparse.Namespace,
    step_in_phase: int,
) -> Optional[Dict[str, Any]]:
    lam1 = compute_dgac_lambda1(
        step_in_phase,
        args.dgac_warmup_steps,
        args.dgac_ramp_steps,
        args.dgac_lambda_ponder_max,
    )
    one = torch.ones(1, device=device, dtype=torch.float32)
    ponder_terms: List[torch.Tensor] = []
    diversity_terms: List[torch.Tensor] = []
    halt_terms: List[torch.Tensor] = []

    for row, hidden_at_q_end in enumerate(hidden_sequences):
        if len(hidden_at_q_end) < 2:
            continue

        ponder = torch.zeros(1, device=device, dtype=torch.float32)
        div_loss = torch.zeros(1, device=device, dtype=torch.float32)
        remainder = one.clone()
        halt_steps = torch.zeros(1, device=device, dtype=torch.float32)

        for idx in range(1, len(hidden_at_q_end)):
            h_curr = hidden_at_q_end[idx].to(dtype=torch.float32)
            h_prev = hidden_at_q_end[idx - 1].to(dtype=torch.float32)
            halt_prob = halt_gate(h_curr, h_prev)
            ponder = ponder + remainder
            if idx < len(hidden_at_q_end) - 1:
                remainder = remainder * (1.0 - halt_prob)
            div_loss = div_loss + F.relu(F.cosine_similarity(h_curr, h_prev, dim=-1) - args.dgac_tau)
            with torch.no_grad():
                halted = (halt_prob > args.halt_threshold) & (halt_steps == 0)
                halt_steps = torch.where(halted, torch.full_like(halt_steps, float(idx)), halt_steps)

        halt_steps = torch.where(
            halt_steps == 0,
            torch.full_like(halt_steps, float(int(actual_n_latents[row].item()))),
            halt_steps,
        )
        ponder_terms.append(ponder.mean())
        diversity_terms.append(div_loss.mean())
        halt_terms.append(halt_steps.mean())

    if not diversity_terms:
        return None

    return {
        "ponder": torch.stack(ponder_terms).mean(),
        "diversity": torch.stack(diversity_terms).mean(),
        "halt_step_mean": torch.stack(halt_terms).mean(),
        "lambda1": lam1,
    }


def _forward_batched_latent(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    q_lens: torch.Tensor,
    n_latents: torch.Tensor,
    pad_id: torch.Tensor,
    device: torch.device,
    halt_gate: Optional[HaltGate],
    args: argparse.Namespace,
    step_in_phase: int,
    amp_dtype: torch.dtype,
) -> Dict[str, Any]:
    backbone = _get_backbone(model)
    embed_fn = _get_embed_tokens(model)
    lm_head_fn = _get_lm_head(model)

    with _autocast_ctx(device, amp_dtype):
        all_embeds = embed_fn(input_ids)
        pad_embed = embed_fn(pad_id.view(1)).squeeze(0)

    q_ctx, q_ctx_mask = _build_question_context(all_embeds, q_lens, pad_embed)
    latent_ctx, _, actual_k = _run_latent_passes(
        model=model,
        ctx=q_ctx,
        ctx_mask=q_ctx_mask,
        n_latent=n_latents,
        halt_gate=None,
        args=args,
        device=device,
        amp_dtype=amp_dtype,
    )
    actual_n_latents = actual_k if isinstance(actual_k, torch.Tensor) else torch.full_like(n_latents, int(actual_k))

    patched = all_embeds.clone()
    max_q_len = int(q_ctx.size(1))
    max_n_latent = int(actual_n_latents.max().item()) if actual_n_latents.numel() > 0 else 0
    for latent_step in range(max_n_latent):
        active_indices = (actual_n_latents > latent_step).nonzero(as_tuple=False).flatten()
        if active_indices.numel() == 0:
            break
        inject_pos = q_lens[active_indices] + latent_step
        valid_inject = inject_pos < patched.size(1)
        if not bool(torch.all(valid_inject).item()):
            active_indices = active_indices[valid_inject]
            inject_pos = inject_pos[valid_inject]
            if active_indices.numel() == 0:
                continue
        h_step = latent_ctx[active_indices, max_q_len + latent_step, :]
        patched_next = patched.clone()
        patched_next[active_indices, inject_pos, :] = h_step.to(dtype=patched_next.dtype)
        patched = patched_next

    with _autocast_ctx(device, amp_dtype):
        outputs = backbone(inputs_embeds=patched, attention_mask=attention_mask, use_cache=False)
        hidden = _extract_last_hidden_state(outputs, "coconut batched full forward")
        logits = lm_head_fn(hidden).float()

    ce_sum, n_valid = _compute_ce_sum_and_count(logits, labels)
    ce_value = float(ce_sum.item() / max(n_valid, 1))
    result: Dict[str, Any] = {
        "ce_sum": ce_sum,
        "n_valid": n_valid,
        "ce": ce_value,
        "ponder": None,
        "diversity": None,
        "halt_step_mean": None,
        "lambda1": 0.0,
    }

    if halt_gate is None:
        return result

    hidden_sequences = _collect_latent_hidden_sequences(latent_ctx, max_q_len, actual_n_latents)
    halt_metrics = _compute_batched_halt_metrics(
        hidden_sequences=hidden_sequences,
        actual_n_latents=actual_n_latents,
        halt_gate=halt_gate,
        device=device,
        args=args,
        step_in_phase=step_in_phase,
    )
    if halt_metrics is None:
        return result

    result.update(halt_metrics)
    return result


def coconut_forward(
    model,
    batch: Dict[str, torch.Tensor],
    stage_k: int,
    device: torch.device,
    halt_gate: Optional[HaltGate],
    args: argparse.Namespace,
    step_in_phase: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    q_lens = batch["q_lens"].to(device)
    n_latents = batch["n_latents"].to(device)
    pad_id = batch["pad_id"].to(device)
    amp_dtype = _amp_dtype(device)

    result = _forward_batched_latent(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        q_lens=q_lens,
        n_latents=n_latents,
        pad_id=pad_id,
        device=device,
        halt_gate=halt_gate,
        args=args,
        step_in_phase=step_in_phase,
        amp_dtype=amp_dtype,
    )

    if result["n_valid"] == 0:
        zero = torch.zeros((), device=device, requires_grad=True)
        return zero, {"ce": 0.0}

    ce = result["ce_sum"] / result["n_valid"]
    total_loss = ce
    metrics: Dict[str, float] = {"ce": float(ce.item())}

    if halt_gate is not None and result["diversity"] is not None:
        total_loss = total_loss + result["lambda1"] * result["ponder"] + args.dgac_lambda_diversity * result["diversity"]
        metrics.update(
            {
                "ponder": float(result["ponder"].item()),
                "diversity": float(result["diversity"].item()),
                "halt_step_mean": float(result["halt_step_mean"].item()),
                "lambda1": float(result["lambda1"]),
            }
        )

    return total_loss, metrics
def normalize_pred(text: str) -> str:
    boxed = _re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed.group(1).strip().replace(",", "")
    numeric = _re.search(r"(?:answer is|=)\s*\**\s*([\d,\.\-]+)", text, _re.IGNORECASE)
    if numeric:
        return numeric.group(1).strip().replace(",", "")
    nums = _LAST_NUM.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    stripped = text.strip()
    if not stripped:
        return ""
    last_line = stripped.splitlines()[-1].strip()
    last_line = _re.sub(r"^(?:final answer|answer)\s*[:\-]\s*", "", last_line, flags=_re.IGNORECASE)
    return last_line.strip(" .,:;!*")


@torch.no_grad()
def evaluate_stage(
    model,
    val_samples: List[Dict[str, Any]],
    tokenizer,
    lat_token_id: int,
    stage_k: int,
    device: torch.device,
    args: argparse.Namespace,
    halt_gate: Optional[HaltGate] = None,
) -> Tuple[float, float]:
    """
    Runs on ALL DDP ranks. Each rank processes its interleaved shard of val_samples,
    then all-reduces CE and accuracy counts.
    """
    _maybe_empty_cuda_cache()
    model.eval()
    if halt_gate is not None:
        halt_gate.eval()

    rank = _rank()
    world_size = _world_size()
    pad_id = tokenizer.pad_token_id or 0
    embed_fn = _get_embed_tokens(model)
    lm_head_fn = _get_lm_head(model)
    backbone = _get_backbone(model)
    amp_dtype = _amp_dtype(device)
    batch_size = max(int(args.val_batch_size), 1)

    local_val_samples = val_samples[rank::world_size]

    ce_numer = 0.0
    ce_denom = 0

    for start in range(0, len(local_val_samples), batch_size):
        batch_raw = local_val_samples[start : start + batch_size]
        built = [
            build_sample_at_stage(tokenizer, sample, stage_k, lat_token_id, args.max_seq_len)
            for sample in batch_raw
        ]
        built = [sample for sample in built if sample is not None]
        if not built:
            continue
        batch = collate_stage_k(built, pad_id)
        loss, _ = coconut_forward(
            model,
            batch,
            stage_k,
            device,
            halt_gate=None,
            args=args,
            step_in_phase=0,
        )
        valid_tokens = int((batch["labels"][:, 1:].contiguous().view(-1) != -100).sum().item())
        ce_numer += float(loss.item()) * valid_tokens
        ce_denom += valid_tokens

    ce_numer, ce_denom = _ddp_sum([ce_numer, ce_denom], device)
    ce_denom = int(round(ce_denom))

    acc_samples = val_samples[:50]
    local_acc_samples = acc_samples[rank::world_size]

    n_correct = 0
    n_total = 0

    for sample in local_acc_samples:
        built = build_sample_at_stage(tokenizer, sample, stage_k, lat_token_id, args.max_seq_len)
        if built is None:
            continue
        q_len = int(built["q_len"])
        n_latent = int(built["n_latent"])
        q_ids = built["full_ids"][:q_len]
        q_tensor = q_ids.unsqueeze(0).to(device)
        ctx = embed_fn(q_tensor)
        ctx_mask = torch.ones((1, ctx.size(1)), dtype=torch.bool, device=device)
        ctx, ctx_mask, _ = _run_latent_passes(
            model=model,
            ctx=ctx,
            ctx_mask=ctx_mask,
            n_latent=n_latent,
            halt_gate=halt_gate,
            args=args,
            device=device,
            amp_dtype=amp_dtype,
        )

        generated: List[int] = []
        eos_id = tokenizer.eos_token_id
        for _ in range(args.gen_max_tokens):
            if ctx.size(1) > args.max_seq_len:
                ctx = ctx[:, -args.max_seq_len :, :]
                ctx_mask = ctx_mask[:, -args.max_seq_len :]
            with _autocast_ctx(device, amp_dtype):
                outputs = backbone(inputs_embeds=ctx, attention_mask=ctx_mask, use_cache=False)
                hidden = _extract_last_hidden_state(outputs, "eval decode")
                logits = lm_head_fn(hidden)
            next_id = int(logits[:, -1, :].argmax(-1).item())
            if eos_id is not None and next_id == eos_id:
                break
            generated.append(next_id)
            next_embed = embed_fn(torch.tensor([[next_id]], device=device))
            ctx = torch.cat([ctx, next_embed], dim=1)
            ctx_mask = torch.cat(
                [ctx_mask, torch.ones((1, 1), dtype=torch.bool, device=device)],
                dim=1,
            )

        pred = normalize_pred(tokenizer.decode(generated, skip_special_tokens=True)).strip()
        gold = str(sample.get("answer_norm", "")).strip()
        if pred and gold and (pred == gold or pred.lower() == gold.lower()):
            n_correct += 1
        n_total += 1

    n_correct, n_total = _ddp_sum([n_correct, n_total], device)
    n_correct = int(round(n_correct))
    n_total = int(round(n_total))

    model.train()
    if halt_gate is not None:
        halt_gate.train()

    return ce_numer / max(ce_denom, 1), n_correct / max(n_total, 1)
def _resolve_hf_token(cli_value: Optional[str]) -> Optional[str]:
    return _resolve_hf_token_common(cli_value)


def _hub_upload_checkpoint(
    ckpt_dir: Path,
    hf_repo_id: str,
    hf_token: str,
    remote_prefix: str = "runs/stage3",
    timeout_s: float = 300.0,
) -> bool:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False

    remote_name = f"{remote_prefix.strip('/')}/{ckpt_dir.name}".strip("/")
    try:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hf_repo_id, private=True, exist_ok=True, token=hf_token)
        future = api.upload_folder(
            repo_id=hf_repo_id,
            folder_path=str(ckpt_dir),
            path_in_repo=remote_name,
            token=hf_token,
            commit_message=f"Upload {ckpt_dir.name}",
            run_as_future=True,
        )
        future.result(timeout=timeout_s)
        if _is_main_process():
            print(f"  [hub] uploaded {remote_name} -> {hf_repo_id}")
        return True
    except Exception as exc:
        if _is_main_process():
            print(f"  [hub] upload failed for {remote_name}: {exc}")
        return False


def _hub_download_checkpoint(
    ckpt_name: str,
    local_dir: Path,
    hf_repo_id: str,
    hf_token: str,
    remote_prefix: str = "runs/stage3",
) -> Optional[Path]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None

    local_dir.mkdir(parents=True, exist_ok=True)
    remote_path = f"{remote_prefix.strip('/')}/{ckpt_name}".strip("/")
    dest = local_dir / remote_path

    try:
        snapshot_download(
            repo_id=hf_repo_id,
            local_dir=str(local_dir),
            token=hf_token,
            force_download=True,
            allow_patterns=[f"{remote_path}/*"],
        )
        return dest if dest.exists() else None
    except Exception as exc:
        if _is_main_process():
            print(f"  [hub] download failed for {remote_path}: {exc}")
        return None


def _list_hub_stage_checkpoints(
    hf_repo_id: str,
    hf_token: str,
    remote_prefix: str = "runs/stage3",
) -> List[Tuple[int, int, str]]:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []
    try:
        api = HfApi(token=hf_token)
        files = list(api.list_repo_files(repo_id=hf_repo_id, token=hf_token))
    except Exception:
        return []

    prefix = remote_prefix.strip("/")
    found: set = set()
    for f in files:
        parts = f.split("/")
        try:
            prefix_parts = prefix.split("/")
            if parts[:len(prefix_parts)] != prefix_parts:
                continue
            rest = parts[len(prefix_parts):]
            if len(rest) < 2:
                continue
            stage_dir = rest[0]
            ckpt_dir  = rest[1]
            stage_k = _parse_stage_dir_name(stage_dir)
            if stage_k is None:
                continue
            if not (ckpt_dir.startswith("checkpoint-") or ckpt_dir == "best"):
                continue
            if ckpt_dir == "best":
                step = 0
            else:
                tail = ckpt_dir.split("-")[-1]
                step = int(tail) if tail.isdigit() else 0
            rel = "/".join(rest[:2])
            found.add((stage_k, step, rel))
        except Exception:
            continue

    return sorted(found, key=lambda x: (x[0], x[1]), reverse=True)


def save_checkpoint(
    output_dir: Path,
    step: int,
    epoch: int,
    step_in_epoch: int,
    step_in_phase: int,
    stage_k: int,
    model,
    halt_gate: Optional[HaltGate],
    optimizer: Optional[AdamW],
    scheduler: Optional[LambdaLR],
    args: argparse.Namespace,
    val_ce: Optional[float],
    val_acc: Optional[float],
    tag: str = "",
) -> Path:
    stage_dir = output_dir / f"stage_{stage_k}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    name = "best" if tag == "best" else f"checkpoint-{step:07d}"
    ckpt = stage_dir / name
    tmp = stage_dir / f"{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(tmp / "adapter_model"))
    if halt_gate is not None:
        torch.save(halt_gate.state_dict(), tmp / "halt_gate.pt")
    torch.save(
        {
            "stage_k": stage_k,
            "step": step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "step_in_phase": step_in_phase,
            "val_ce": val_ce,
            "val_acc": val_acc,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "use_halt_gate": args.use_halt_gate,
            "model_id": args.model_id,
        },
        tmp / "training_state.pt",
    )

    if ckpt.exists():
        shutil.rmtree(ckpt, ignore_errors=True)
    tmp.replace(ckpt)
    label = "best" if tag == "best" else "saved"
    if _is_main_process():
        print(f"  [ckpt] {label} -> {ckpt}  acc={val_acc}  ce={val_ce}")

    hf_token  = getattr(args, "_resolved_hf_token", None)
    push      = getattr(args, "push_to_hub", False)
    repo_id   = getattr(args, "hf_repo_id", "WeirdRunner/Ouroboros")
    subdir    = getattr(args, "hf_stage_subdir", "runs/stage3")
    if push and hf_token and _is_main_process():
      stage_remote_prefix = f"{subdir.strip('/')}/stage_{stage_k}"
      _hub_upload_checkpoint(ckpt, repo_id, hf_token, remote_prefix=stage_remote_prefix)
    return ckpt


def load_checkpoint(
    ckpt_dir: Path,
    model,
    halt_gate: Optional[HaltGate],
    optimizer: Optional[AdamW],
    scheduler: Optional[LambdaLR],
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, Any]:
    state = _read_training_state(ckpt_dir, map_location=device)
    adapter_dir = ckpt_dir / "adapter_model"
    if adapter_dir.exists():
        from peft import set_peft_model_state_dict

        loaded = False
        for fname in ["adapter_model.safetensors", "adapter_model.bin"]:
            fpath = adapter_dir / fname
            if not fpath.exists():
                continue
            if fname.endswith(".safetensors"):
                from safetensors.torch import load_file
                weights = load_file(str(fpath))
            else:
                weights = torch.load(fpath, map_location=device)
            set_peft_model_state_dict(model, weights)
            loaded = True
            break
        if not loaded:
            raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")

    halt_gate_path = ckpt_dir / "halt_gate.pt"
    if halt_gate is not None and halt_gate_path.exists():
        halt_gate.load_state_dict(torch.load(halt_gate_path, map_location=device))

    if optimizer is not None and state.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception as exc:
            if verbose:
                print(f"  [resume] optimizer state mismatch ({exc}); resetting optimizer.")
    if scheduler is not None and state.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(state["scheduler"])
        except Exception as exc:
            if verbose:
                print(f"  [resume] scheduler state mismatch ({exc}); resetting scheduler.")

    if verbose:
        print(
            f"  [resume] step={int(state.get('step', 0))} "
            f"epoch={int(state.get('epoch', 0))} "
            f"stage_k={int(state.get('stage_k', 0))} "
            f"val_acc={state.get('val_acc')}"
        )
    return state


def prune_epoch_checkpoints(stage_dir: Path, keep: int) -> None:
    keep = max(int(keep), 0)
    checkpoints = sorted(
        [p for p in stage_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    for old in checkpoints[:-keep] if keep > 0 else checkpoints:
        shutil.rmtree(old, ignore_errors=True)


def get_trainable_parameters(
    model: nn.Module,
    halt_gate: Optional[nn.Module],
) -> List[nn.Parameter]:
    params = [p for p in model.parameters() if p.requires_grad]
    if halt_gate is not None:
        params.extend(p for p in halt_gate.parameters() if p.requires_grad)
    return params


def build_optimizer_and_scheduler(
    model: nn.Module,
    halt_gate: Optional[nn.Module],
    args: argparse.Namespace,
    total_steps: int,
) -> Tuple[AdamW, LambdaLR]:
    trainable = get_trainable_parameters(model, halt_gate)
    decay    = [p for p in trainable if p.ndim >= 2]
    no_decay = [p for p in trainable if p.ndim < 2]
    optimizer = AdamW(
        [
            {"params": decay,    "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return optimizer, LambdaLR(optimizer, lr_lambda)


def make_timeout_checker(
    args: argparse.Namespace,
    rank: int,
    session_start: Optional[float] = None,
):
    if session_start is None:
        session_start = _SCRIPT_START
    timeout_limit_s  = args.session_timeout_hours * 3600.0
    timeout_buffer_s = args.graceful_exit_buffer_minutes * 60.0
    triggered = [False]

    def check() -> bool:
        if triggered[0]:
            return True
        if rank != 0:
            return False
        elapsed = time.perf_counter() - session_start
        if elapsed + timeout_buffer_s >= timeout_limit_s:
            remaining = (timeout_limit_s - elapsed) / 60.0
            print(
                f"\n  [timeout] {elapsed / 3600:.2f}h elapsed - "
                f"{remaining:.1f} min remaining "
                f"(< {args.graceful_exit_buffer_minutes:.0f} min buffer)."
            )
            triggered[0] = True
        return triggered[0]

    return check


def find_latest_resume_checkpoint(
    output_dir: Path,
    hf_token: Optional[str] = None,
    hf_repo_id: str = "WeirdRunner/Ouroboros",
    hf_stage_subdir: str = "runs/stage3",
) -> Optional[Path]:
    best_path: Optional[Path] = None
    best_key:  Optional[Tuple[int, int, int, int]] = None

    if output_dir.exists():
        for stage_dir in output_dir.iterdir():
            stage_k = _parse_stage_dir_name(stage_dir.name)
            if stage_k is None or not stage_dir.is_dir():
                continue
            for ckpt in stage_dir.iterdir():
                if not ckpt.is_dir() or not ckpt.name.startswith("checkpoint-"):
                    continue
                state_path = ckpt / "training_state.pt"
                if not state_path.exists():
                    continue
                try:
                    state = _read_training_state(ckpt, map_location="cpu")
                except Exception:
                    continue
                key = (
                    stage_k,
                    int(state.get("epoch", -1)),
                    int(state.get("step_in_epoch", -1)),
                    int(state.get("step", -1)),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_path = ckpt

    if best_path is not None:
        return best_path

    if not hf_token:
        return None

    if _is_main_process():
        print("  [resume] No local checkpoints found. Scanning Hub...")

    hub_candidates = _list_hub_stage_checkpoints(hf_repo_id, hf_token, hf_stage_subdir)
    hub_candidates = [(k, s, n) for k, s, n in hub_candidates if not n.endswith("/best")]
    hub_resume_dir = output_dir / ".hub_resume"

    for stage_k, step, rel_name in hub_candidates:
        ckpt_name = rel_name.split("/")[-1]
        if _is_main_process():
            print(f"  [hub] downloading {rel_name} ...")
        downloaded = _hub_download_checkpoint(
            ckpt_name=ckpt_name,
            local_dir=hub_resume_dir,
            hf_repo_id=hf_repo_id,
            hf_token=hf_token,
            remote_prefix=f"{hf_stage_subdir}/stage_{stage_k}",
        )
        if downloaded is not None and (downloaded / "training_state.pt").exists():
            if _is_main_process():
                print(f"  [hub] using {rel_name} as resume checkpoint")
            return downloaded

    return None


@torch.no_grad()
def run_generation_callback(
    model,
    tokenizer,
    halt_gate: Optional[HaltGate],
    stage_k: int,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    wandb_run=None,
) -> float:
    _maybe_empty_cuda_cache()
    model.eval()
    if halt_gate is not None:
        halt_gate.eval()
    print(f"\n  -- Generation @ step {step} stage={stage_k} --")
    embed_fn = _get_embed_tokens(model)
    lm_head_fn = _get_lm_head(model)
    backbone = _get_backbone(model)
    amp_dtype = _amp_dtype(device)
    mean_uwr = 0.0

    for prompt in GEN_PROMPTS:
        prefix = _maybe_apply_chat_template(tokenizer, prompt)
        q_ids = tokenizer.encode(prefix, add_special_tokens=False)
        q_tensor = torch.tensor(q_ids, device=device).unsqueeze(0)
        ctx = embed_fn(q_tensor)
        ctx_mask = torch.ones((1, ctx.size(1)), dtype=torch.bool, device=device)
        ctx, ctx_mask, actual_k = _run_latent_passes(
            model=model,
            ctx=ctx,
            ctx_mask=ctx_mask,
            n_latent=stage_k,
            halt_gate=halt_gate,
            args=args,
            device=device,
            amp_dtype=amp_dtype,
        )

        generated: List[int] = []
        eos_id = tokenizer.eos_token_id
        for _ in range(args.gen_max_tokens):
            if ctx.size(1) > args.max_seq_len:
                ctx = ctx[:, -args.max_seq_len :, :]
                ctx_mask = ctx_mask[:, -args.max_seq_len :]
            with _autocast_ctx(device, amp_dtype):
                outputs = backbone(inputs_embeds=ctx, attention_mask=ctx_mask, use_cache=False)
                hidden = _extract_last_hidden_state(outputs, "generation decode")
                logits = lm_head_fn(hidden)
            next_id = int(logits[:, -1, :].argmax(-1).item())
            if eos_id is not None and next_id == eos_id:
                break
            generated.append(next_id)
            next_embed = embed_fn(torch.tensor([[next_id]], device=device))
            ctx = torch.cat([ctx, next_embed], dim=1)
            ctx_mask = torch.cat(
                [ctx_mask, torch.ones((1, 1), dtype=torch.bool, device=device)],
                dim=1,
            )

        text = tokenizer.decode(generated, skip_special_tokens=True)
        words = text.split()
        uwr = len(set(words)) / max(len(words), 1)
        mean_uwr += uwr
        display = text[:200].replace("\n", " ")
        print(f"  Q: {prompt}")
        print(f"  A: {display}  [k_actual={int(actual_k)} uwr={uwr:.3f}]")

    mean_uwr /= max(len(GEN_PROMPTS), 1)
    print(f"  Mean UWR: {mean_uwr:.3f}\n")
    if wandb_run is not None:
        import wandb
        wandb.log({"gen/mean_uwr": mean_uwr, "gen/stage": stage_k}, step=step)
    model.train()
    if halt_gate is not None:
        halt_gate.train()
    return mean_uwr
def _best_state_for_stage(stage_dir: Path) -> Tuple[float, float, Optional[Path]]:
    best_dir = stage_dir / "best"
    if not (best_dir / "training_state.pt").exists():
        return -1.0, float("inf"), None
    state = _read_training_state(best_dir, map_location="cpu")
    val_acc = float(state.get("val_acc", -1.0) if state.get("val_acc") is not None else -1.0)
    val_ce  = float(state.get("val_ce", float("inf")) if state.get("val_ce") is not None else float("inf"))
    return val_acc, val_ce, best_dir


def startup_hub_sync_and_prune(
    output_dir: Path,
    resume_path: Optional[Path],
    hf_token: str,
    hf_repo_id: str,
    hf_stage_subdir: str,
) -> None:
    """
    Called once at session start (rank 0 only, before training).
    1. Upload every local checkpoint that exists to Hub (best + numbered).
    2. Delete all local numbered checkpoints EXCEPT the one we are resuming from.
       Always preserve best/ dirs.

    This prevents Kaggle disk overflow across sessions. Upload failures are
    logged, but local pruning still follows the keep policy so stale numbered
    checkpoints do not accumulate across sessions.
    """
    if not _is_main_process():
        return

    all_ckpts: List[Tuple[Path, bool]] = []  # (path, is_resume)
    if output_dir.exists():
        for stage_dir in sorted(output_dir.iterdir()):
            if _parse_stage_dir_name(stage_dir.name) is None or not stage_dir.is_dir():
                continue
            for ckpt in sorted(stage_dir.iterdir()):
                if not ckpt.is_dir():
                    continue
                if not (ckpt / "training_state.pt").exists():
                    continue
                is_resume = resume_path is not None and ckpt.resolve() == resume_path.resolve()
                all_ckpts.append((ckpt, is_resume))

    if not all_ckpts:
        print("  [startup] No local checkpoints found; nothing to sync/prune.")
        return

    print(f"  [startup] Found {len(all_ckpts)} local checkpoint(s). Uploading to Hub before pruning...")
    for ckpt, is_resume in all_ckpts:
        stage_dir_name = ckpt.parent.name
        remote_prefix = f"{hf_stage_subdir.strip('/')}/{stage_dir_name}"
        ok = _hub_upload_checkpoint(ckpt, hf_repo_id, hf_token, remote_prefix=remote_prefix)
        resume_marker = "  (resume)" if is_resume else ""
        status = "✓" if ok else "✗ (upload failed)"
        print(f"  [startup]   {stage_dir_name}/{ckpt.name}{resume_marker}  {status}")

    pruned = 0
    for ckpt, is_resume in all_ckpts:
        if is_resume:
            continue
        shutil.rmtree(ckpt, ignore_errors=True)
        print(f"  [startup]   pruned {ckpt.parent.name}/{ckpt.name}")
        pruned += 1

    print(f"  [startup] Sync+prune complete. Pruned {pruned} checkpoint(s) locally.")


def _distributed_resume_marker(output_dir: Path) -> Path:
    return output_dir / ".resolved_resume_path.txt"


def _resolve_resume_checkpoint_for_all_ranks(
    *,
    output_dir: Path,
    requested_resume: Optional[Path],
    hf_token: Optional[str],
    hf_repo_id: str,
    hf_stage_subdir: str,
    distributed: bool,
    is_main: bool,
) -> Optional[Path]:
    marker_path = _distributed_resume_marker(output_dir)
    if is_main and marker_path.exists():
        marker_path.unlink(missing_ok=True)

    resolved: Optional[Path] = requested_resume
    if distributed:
        if is_main and resolved is None:
            resolved = find_latest_resume_checkpoint(
                output_dir,
                hf_token=hf_token,
                hf_repo_id=hf_repo_id,
                hf_stage_subdir=hf_stage_subdir,
            )
            if resolved is not None:
                print(f"  [resume] discovered latest checkpoint: {resolved}")
        if is_main:
            marker_path.write_text(
                str(resolved.resolve()) if resolved is not None else "",
                encoding="utf-8",
            )
        barrier()
        if not is_main:
            raw = marker_path.read_text(encoding="utf-8").strip() if marker_path.exists() else ""
            resolved = Path(raw) if raw else None
        barrier()
        return resolved

    if resolved is None:
        resolved = find_latest_resume_checkpoint(
            output_dir,
            hf_token=hf_token,
            hf_repo_id=hf_repo_id,
            hf_stage_subdir=hf_stage_subdir,
        )
        if resolved is not None and is_main:
            print(f"  [resume] discovered latest checkpoint: {resolved}")
    return resolved


def _cleanup_distributed_resume_artifacts(
    output_dir: Path,
    hub_resume_dir: Path,
    distributed: bool,
    is_main: bool,
) -> None:
    if distributed:
        barrier()
    if is_main:
        marker_path = _distributed_resume_marker(output_dir)
        marker_path.unlink(missing_ok=True)
        if hub_resume_dir.exists():
            shutil.rmtree(hub_resume_dir, ignore_errors=True)
    if distributed:
        barrier()


def _partition_contiguous_range(n_items: int, n_parts: int, part_idx: int) -> Tuple[int, int]:
    if n_items <= 0:
        return 0, 0
    base = n_items // n_parts
    remainder = n_items % n_parts
    start = part_idx * base + min(part_idx, remainder)
    width = base + (1 if part_idx < remainder else 0)
    return start, start + width


def diloco_get_shard(
    train_samples: List[Dict[str, Any]],
    worker_id: str,
    stage_k: int,
    round_n: int,
    seed: int,
    samples_already_seen: int = 0,
) -> List[Dict[str, Any]]:
    """
    Deterministic shard assignment for the current DiLoCo round.

    The permutation is still stage/round dependent, but we trim the prefix that
    has already been counted in round_state.total_samples_seen for this stage.
    This keeps partially-complete stages from re-running a full 1/3 shard when
    only the stage remainder should be trained.
    """
    worker_idx = {"A": 0, "B": 1, "C": 2}[worker_id]
    n = len(train_samples)
    if n <= 0:
        return []

    rng = random.Random(seed + stage_k * 100_003 + round_n * 7)
    indices = list(range(n))
    rng.shuffle(indices)

    seen = max(0, min(int(samples_already_seen), n))
    remaining_indices = indices[seen:]
    if not remaining_indices:
        return []

    start, end = _partition_contiguous_range(len(remaining_indices), 3, worker_idx)
    shard_indices = remaining_indices[start:end]
    return [train_samples[i] for i in shard_indices]


def diloco_read_round_state(hf_token: str, repo_id: str) -> Dict[str, Any]:
    """
    Download and parse diloco_state/round_state.json from Hub.
    Returns default state if file doesn't exist (first run).
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="diloco_state/round_state.json",
            token=hf_token,
        )
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        return {
            "stage_k": int(state.get("stage_k", 0)),
            "round_n": int(state.get("round_n", 0)),
            "anchor_path": state.get("anchor_path", "diloco_state/anchor"),
            "total_samples_seen": {
                str(k): int(v) for k, v in dict(state.get("total_samples_seen", {})).items()
            },
            "completed_stages": [int(x) for x in state.get("completed_stages", [])],
            **{k: v for k, v in state.items() if k not in {"stage_k", "round_n", "anchor_path", "total_samples_seen", "completed_stages"}},
        }
    except Exception:
        return {
            "stage_k": 0,
            "round_n": 0,
            "anchor_path": "diloco_state/anchor",
            "total_samples_seen": {},
            "completed_stages": [],
        }


def diloco_upload_worker_state(
    adapter_dir: Path,
    worker_id: str,
    stage_k: int,
    round_n: int,
    samples_seen: int,
    hf_token: str,
    repo_id: str,
) -> None:
    """
    Upload worker adapter weights and status to Hub.
    Paths:
      diloco_state/workers/{worker_id}/round_{round_n:04d}_stage_{stage_k}/adapter_model.safetensors
      diloco_state/workers/{worker_id}/status.json
    """
    import tempfile
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    remote_prefix = f"diloco_state/workers/{worker_id}/round_{round_n:04d}_stage_{stage_k}"

    for fname in ["adapter_model.safetensors", "adapter_config.json"]:
        fpath = adapter_dir / fname
        if fpath.exists():
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=f"{remote_prefix}/{fname}",
                repo_id=repo_id,
                token=hf_token,
                commit_message=f"Worker {worker_id} round {round_n} stage {stage_k}",
            )

    status = {
        "worker_id": worker_id,
        "stage_k": int(stage_k),
        "round_n": int(round_n),
        "samples_seen": int(samples_seen),
        "status": "done",
        "timestamp": time.time(),
        "weights_path": remote_prefix,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(status, tf, indent=2)
        tmp_path = tf.name
    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=f"diloco_state/workers/{worker_id}/status.json",
            repo_id=repo_id,
            token=hf_token,
            commit_message=f"Worker {worker_id} status update",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def diloco_download_anchor(
    model,
    hf_token: str,
    repo_id: str,
    anchor_path: str,
    device: torch.device,
) -> None:
    """
    Download anchor adapter weights from Hub and load them into the model in-place.
    Falls back silently if no anchor exists (first round uses random init).
    """
    from huggingface_hub import hf_hub_download
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    try:
        dl_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{anchor_path}/adapter_model.safetensors",
            token=hf_token,
        )
        weights = load_file(dl_path, device=str(device))
        set_peft_model_state_dict(model, weights)
        if _is_main_process():
            print(f"  [diloco] Loaded anchor weights from {anchor_path}")
    except Exception as exc:
        if _is_main_process():
            print(f"  [diloco] No anchor found at {anchor_path} ({exc}); using current weights.")


def diloco_push_signal(
    worker_id: str,
    stage_k: int,
    round_n: int,
    github_token: str,
    github_repo: str,
) -> None:
    """
    Push a signal file to GitHub to trigger the coordinator GitHub Action.
    File: signals/worker_{id}_stage_{k}_round_{n}.json
    Uses GitHub API directly (no git clone needed).
    """
    import base64
    import requests

    signal_path = f"signals/worker_{worker_id}_stage_{stage_k}_round_{round_n}.json"
    content = json.dumps(
        {
            "worker_id": worker_id,
            "stage_k": int(stage_k),
            "round_n": int(round_n),
            "timestamp": time.time(),
        },
        indent=2,
    )
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")

    url = f"https://api.github.com/repos/{github_repo}/contents/{signal_path}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
    }

    resp = requests.get(url, headers=headers, timeout=30)
    payload = {
        "message": f"Worker {worker_id} done: stage {stage_k} round {round_n}",
        "content": encoded,
    }
    if resp.status_code == 200:
        payload["sha"] = resp.json().get("sha")

    resp = requests.put(url, headers=headers, json=payload, timeout=30)
    if resp.status_code in (200, 201):
        if _is_main_process():
            print(f"  [diloco] Signal pushed to GitHub: {signal_path}")
    else:
        if _is_main_process():
            print(f"  [diloco] WARNING: GitHub signal push failed: {resp.status_code} {resp.text[:200]}")


def _optimizer_step_sample_count(step_idx: int, batch_size: int, grad_accum: int, dataset_size: int) -> int:
    step_start = step_idx * grad_accum * batch_size
    step_end = min((step_idx + 1) * grad_accum * batch_size, dataset_size)
    return max(step_end - step_start, 0)


def run_training_stages(
    *,
    model,
    tokenizer,
    halt_gate: Optional[HaltGate],
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    lat_token_id: int,
    pad_id: int,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    session_start: float,
    wandb_run,
    stages: List[int],
    curriculum_max_stage: int,
    resume_path: Optional[Path] = None,
    resume_same_stage: bool = False,
    resume_stage: int = 0,
    resume_epoch: int = 0,
    resume_step_in_epoch: int = -1,
    global_step: int = 0,
    step_in_phase: int = 0,
    load_best_between_stages: bool = True,
    run_generation_at_stage_end: bool = True,
    run_epoch_end_val: bool = True,
) -> Dict[str, Any]:
    if not train_samples:
        raise ValueError("No training samples available for this training plan.")

    rank = _rank()
    world_size = _world_size()
    distributed = world_size > 1
    is_main = rank == 0
    local_bs = args.batch_size // world_size if distributed else args.batch_size
    check_timeout = make_timeout_checker(args, rank, session_start=session_start)

    timeout_triggered = False
    val_budget_triggered = False
    samples_seen_total = 0

    for stage_k in stages:
        if args.use_halt_gate:
            step_in_phase = step_in_phase if resume_same_stage and stage_k == resume_stage else 0

        n_epochs = (args.stage_0_epochs or args.epochs_per_stage) if stage_k == 0 else args.epochs_per_stage
        steps_per_epoch = max(
            1,
            math.ceil(len(train_samples) / max(args.batch_size * args.grad_accum, 1)),
        )
        total_stage_steps = max(1, n_epochs * steps_per_epoch)
        optimizer, scheduler = build_optimizer_and_scheduler(model, halt_gate, args, total_stage_steps)
        trainable_params = get_trainable_parameters(model, halt_gate)

        stage_start_epoch = 0
        stage_start_step_in_epoch = -1
        if resume_same_stage and stage_k == resume_stage and resume_path is not None:
            state = load_checkpoint(
                resume_path,
                model,
                halt_gate,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                verbose=is_main,
            )
            global_step = int(state.get("step", global_step))
            stage_start_epoch = int(state.get("epoch", resume_epoch))
            stage_start_step_in_epoch = int(state.get("step_in_epoch", resume_step_in_epoch))
            if args.use_halt_gate:
                step_in_phase = int(state.get("step_in_phase", step_in_phase))
        else:
            resume_same_stage = False

        stage_dir = output_dir / f"stage_{stage_k}"
        best_val_acc, best_val_ce, best_ckpt = _best_state_for_stage(stage_dir)

        if is_main:
            print()
            print("=" * 64)
            label = "(CoT warmup)" if stage_k == 0 else f"{stage_k} latent pass(es)"
            extra = "  + DGAC" if args.use_halt_gate else ""
            print(f"  Stage {stage_k}/{curriculum_max_stage}  {label}{extra}")
            print(f"  Epochs: {n_epochs}  Steps/epoch: {steps_per_epoch}  Total: {total_stage_steps}")
            if stage_start_epoch > 0 or stage_start_step_in_epoch >= 0:
                print(
                    f"  Resuming stage from epoch={stage_start_epoch} "
                    f"step_in_epoch={stage_start_step_in_epoch} global_step={global_step}"
                )
            print("=" * 64)

        timeout_triggered = False
        stage_val_budget_triggered = False
        val_budget_exhausted = False
        for epoch in range(stage_start_epoch, n_epochs):
            rng = random.Random(args.seed + stage_k * 100_003 + epoch)
            perm = list(range(len(train_samples)))
            rng.shuffle(perm)

            model.train()
            if halt_gate is not None:
                halt_gate.train()
            optimizer.zero_grad(set_to_none=True)
            val_budget_exhausted = False

            start_step = stage_start_step_in_epoch + 1 if epoch == stage_start_epoch else 0
            remaining_steps = max(steps_per_epoch - start_step, 0)
            pbar = (
                tqdm(total=remaining_steps, desc=f"S{stage_k}E{epoch}", dynamic_ncols=True)
                if is_main else None
            )

            for step_idx in range(start_step, steps_per_epoch):
                timeout_triggered = broadcast_bool(check_timeout() or timeout_triggered, device)
                if timeout_triggered:
                    if is_main:
                        print(f"  [timeout] saving emergency checkpoint at step {global_step} ...")
                        save_checkpoint(
                            output_dir=output_dir,
                            step=global_step,
                            epoch=epoch,
                            step_in_epoch=step_idx - 1,
                            step_in_phase=step_in_phase,
                            stage_k=stage_k,
                            model=model,
                            halt_gate=halt_gate,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            args=args,
                            val_ce=None,
                            val_acc=best_val_acc if best_val_acc >= 0 else None,
                        )
                    barrier()
                    break

                step_metrics_accum: Dict[str, float] = defaultdict(float)
                micro_count = 0
                did_backward = False
                step_samples_seen = _optimizer_step_sample_count(
                    step_idx,
                    args.batch_size,
                    args.grad_accum,
                    len(train_samples),
                )

                for micro in range(args.grad_accum):
                    global_micro_base = (step_idx * args.grad_accum + micro) * args.batch_size
                    rank_base = global_micro_base + rank * local_bs
                    batch_indices = [
                        perm[(rank_base + offset) % len(train_samples)]
                        for offset in range(local_bs)
                    ]
                    batch_raw = [train_samples[idx] for idx in batch_indices]
                    built = [
                        build_sample_at_stage(
                            tokenizer, sample, stage_k, lat_token_id, args.max_seq_len,
                        )
                        for sample in batch_raw
                    ]
                    built = [s for s in built if s is not None]
                    if not built:
                        continue
                    batch = collate_stage_k(built, pad_id)
                    loss, metrics = coconut_forward(
                        model=model,
                        batch=batch,
                        stage_k=stage_k,
                        device=device,
                        halt_gate=halt_gate if args.use_halt_gate else None,
                        args=args,
                        step_in_phase=step_in_phase,
                    )
                    (loss / args.grad_accum).backward()
                    did_backward = True
                    micro_count += 1
                    for k, v in metrics.items():
                        step_metrics_accum[k] += float(v)

                samples_seen_total += step_samples_seen

                if not did_backward:
                    optimizer.zero_grad(set_to_none=True)
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(skip="1")
                    continue

                if distributed:
                    all_reduce_gradients(trainable_params, world_size)

                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        _stage_grad_clip_norm(args, stage_k),
                    )
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.use_halt_gate:
                    step_in_phase += 1

                mean_metrics = {k: v / max(micro_count, 1) for k, v in step_metrics_accum.items()}
                mean_ce = mean_metrics.get("ce", 0.0)

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(ce=f"{mean_ce:.3f}", gn=f"{grad_norm:.3f}")

                if is_main and global_step % args.log_every == 0:
                    log_payload = {
                        "train/ce": mean_ce,
                        "train/grad_norm": grad_norm,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/stage": stage_k,
                        **{f"train/{k}": v for k, v in mean_metrics.items()},
                    }
                    tqdm.write(
                        f"  step={global_step:6d} s={stage_k} ep={epoch} "
                        f"ce={mean_ce:.4f} gn={grad_norm:.4f}"
                    )
                    if wandb_run is not None:
                        import wandb
                        wandb.log(log_payload, step=global_step)

            if pbar is not None:
                pbar.close()

            stage_start_step_in_epoch = -1

            if timeout_triggered:
                break

            should_budget_guard = run_epoch_end_val or run_generation_at_stage_end
            if is_main and should_budget_guard:
                elapsed = time.perf_counter() - session_start
                remaining_min = (args.session_timeout_hours * 3600 - elapsed) / 60.0
                val_budget_exhausted = check_timeout() or (remaining_min < args.val_skip_buffer_minutes)
                if val_budget_exhausted:
                    tqdm.write(
                        f"  [timeout] Skipping val/gen at epoch {epoch} - "
                        f"{remaining_min:.0f}min remaining "
                        f"(< {args.val_skip_buffer_minutes:.0f}min val budget)."
                    )
                    save_checkpoint(
                        output_dir=output_dir,
                        step=global_step,
                        epoch=epoch,
                        step_in_epoch=steps_per_epoch - 1,
                        step_in_phase=step_in_phase,
                        stage_k=stage_k,
                        model=model,
                        halt_gate=halt_gate,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        val_ce=None,
                        val_acc=None,
                    )

            val_budget_exhausted = broadcast_bool(val_budget_exhausted, device) if should_budget_guard else False
            if val_budget_exhausted:
                stage_val_budget_triggered = True
                barrier()
                break

            if is_main:
                save_checkpoint(
                    output_dir=output_dir,
                    step=global_step,
                    epoch=epoch,
                    step_in_epoch=steps_per_epoch - 1,
                    step_in_phase=step_in_phase,
                    stage_k=stage_k,
                    model=model,
                    halt_gate=halt_gate,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    val_ce=None,
                    val_acc=None,
                )
                if not run_epoch_end_val:
                    prune_epoch_checkpoints(stage_dir, args.keep_checkpoints_per_stage)

            if not run_epoch_end_val:
                barrier()
                continue

            val_ce, val_acc = evaluate_stage(
                model=model,
                val_samples=val_samples,
                tokenizer=tokenizer,
                lat_token_id=lat_token_id,
                stage_k=stage_k,
                device=device,
                args=args,
                halt_gate=halt_gate if args.use_halt_gate else None,
            )

            if is_main:
                tqdm.write(
                    f"  [val] s={stage_k} ep={epoch} "
                    f"val_ce={val_ce:.4f} val_acc={val_acc:.4f}"
                )
                if wandb_run is not None:
                    import wandb
                    wandb.log(
                        {"val/ce": val_ce, "val/acc": val_acc, "val/stage": stage_k},
                        step=global_step,
                    )

                save_checkpoint(
                    output_dir=output_dir,
                    step=global_step,
                    epoch=epoch,
                    step_in_epoch=steps_per_epoch - 1,
                    step_in_phase=step_in_phase,
                    stage_k=stage_k,
                    model=model,
                    halt_gate=halt_gate,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    val_ce=val_ce,
                    val_acc=val_acc,
                )
                prune_epoch_checkpoints(stage_dir, args.keep_checkpoints_per_stage)

                is_better = (val_acc > best_val_acc) or (
                    math.isclose(val_acc, best_val_acc) and val_ce < best_val_ce
                )
                if is_better:
                    best_val_acc = val_acc
                    best_val_ce = val_ce
                    best_ckpt = save_checkpoint(
                        output_dir=output_dir,
                        step=global_step,
                        epoch=epoch,
                        step_in_epoch=steps_per_epoch - 1,
                        step_in_phase=step_in_phase,
                        stage_k=stage_k,
                        model=model,
                        halt_gate=halt_gate,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        val_ce=val_ce,
                        val_acc=val_acc,
                        tag="best",
                    )
                    tqdm.write(f"  [best] stage={stage_k} new best acc={best_val_acc:.4f}")

            barrier()

        if timeout_triggered or stage_val_budget_triggered:
            val_budget_triggered = val_budget_triggered or stage_val_budget_triggered
            break

        if is_main and run_generation_at_stage_end:
            if not (check_timeout() or val_budget_exhausted):
                run_generation_callback(
                    model=model,
                    tokenizer=tokenizer,
                    halt_gate=halt_gate if args.use_halt_gate else None,
                    stage_k=stage_k,
                    device=device,
                    args=args,
                    step=global_step,
                    wandb_run=wandb_run,
                )
            else:
                tqdm.write("  [timeout] Skipping gen callback - insufficient time.")

        barrier()

        if load_best_between_stages and best_ckpt is not None and not args.use_halt_gate:
            if is_main:
                print(
                    f"  [stage] Stage {stage_k} done. Best acc={best_val_acc:.4f}. "
                    "Loading best ckpt before advancing."
                )
            best_dir = output_dir / f"stage_{stage_k}" / "best"
            load_checkpoint(
                best_dir,
                model=model,
                halt_gate=halt_gate,
                optimizer=None,
                scheduler=None,
                device=device,
                verbose=is_main,
            )

        barrier()
        resume_same_stage = False

    return {
        "global_step": global_step,
        "step_in_phase": step_in_phase,
        "timeout_triggered": timeout_triggered,
        "val_budget_triggered": val_budget_triggered,
        "samples_seen": int(samples_seen_total),
        "stages": list(stages),
    }


def run_diloco_worker(
    *,
    model,
    tokenizer,
    halt_gate: Optional[HaltGate],
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    curriculum_max_stage: int,
    lat_token_id: int,
    pad_id: int,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    session_start: float,
    wandb_run,
    hf_token: str,
) -> Dict[str, Any]:
    if args.use_halt_gate:
        raise ValueError(
            "DiLoCo mode currently syncs LoRA adapter weights only. "
            "DGAC halt-gate training should remain on the sequential path."
        )
    if args.diloco_worker_id is None:
        raise ValueError("--diloco_worker_id required with --diloco_mode")
    if not hf_token:
        raise ValueError("HF token required for DiLoCo mode")
    if args.resume_from and _is_main_process():
        print("  [diloco] Ignoring --resume_from; the shared anchor defines worker startup state.")

    round_state = diloco_read_round_state(hf_token, args.diloco_state_repo)
    stage_k = int(round_state.get("stage_k", 0))
    round_n = int(round_state.get("round_n", 0))
    anchor_path = round_state.get("anchor_path", "diloco_state/anchor")

    if stage_k > curriculum_max_stage:
        if _is_main_process():
            print(f"  [diloco] stage={stage_k} exceeds max configured stage={curriculum_max_stage}. Nothing to do.")
        return {"stage_k": stage_k, "round_n": round_n, "samples_seen": 0}

    if _is_main_process():
        print(f"  [diloco] Worker {args.diloco_worker_id} | stage={stage_k} round={round_n}")
        if args.push_to_hub:
            print("  [diloco] Regular stage checkpoint Hub sync is disabled in DiLoCo mode; worker uploads go to diloco_state/ only.")
        diloco_download_anchor(model, hf_token, args.diloco_state_repo, anchor_path, device)
    barrier()

    if _world_size() > 1:
        broadcast_parameters(get_trainable_parameters(model, None), src=0)
        barrier()

    stage_samples_seen = int(round_state.get("total_samples_seen", {}).get(str(stage_k), 0))
    stage_samples_seen = max(0, min(stage_samples_seen, len(train_samples)))
    remaining_stage_samples = max(len(train_samples) - stage_samples_seen, 0)
    is_new_stage = stage_samples_seen == 0

    train_shard = diloco_get_shard(
        train_samples,
        args.diloco_worker_id,
        stage_k,
        round_n,
        args.seed,
        samples_already_seen=stage_samples_seen,
    )
    if _is_main_process():
        print(
            f"  [diloco] Stage progress before round: "
            f"{stage_samples_seen}/{len(train_samples)} samples"
        )
        print(f"  [diloco] Remaining global samples: {remaining_stage_samples}")
        print(f"  [diloco] Shard size: {len(train_shard)} samples")

    if not train_shard:
        if _is_main_process():
            print("  [diloco] Empty shard — uploading passthrough status and signal.")
            # Save current adapter (anchor weights, unchanged) for status upload
            _passthrough_dir = output_dir / "diloco_worker_upload" / f"worker_{args.diloco_worker_id}_stage_{stage_k}_round_{round_n}_passthrough"
            _passthrough_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(_passthrough_dir))
            diloco_upload_worker_state(
                adapter_dir=_passthrough_dir,
                worker_id=args.diloco_worker_id,
                stage_k=stage_k,
                round_n=round_n,
                samples_seen=0,
                hf_token=hf_token,
                repo_id=args.diloco_state_repo,
            )
            github_token = os.environ.get("GITHUB_TOKEN")
            if github_token and args.diloco_signal_repo:
                diloco_push_signal(
                    args.diloco_worker_id, stage_k, round_n,
                    github_token, args.diloco_signal_repo,
                )
            else:
                print("  [diloco] No GITHUB_TOKEN — coordinator must be triggered manually.")
            print(f"  [diloco] Worker {args.diloco_worker_id} passthrough done. stage={stage_k} round={round_n}")
        barrier()
        return {
            "stage_k": stage_k,
            "round_n": round_n,
            "samples_seen": 0,
            "global_step": 0,
            "timeout_triggered": False,
            "val_budget_triggered": False,
            "stages": [stage_k],
        }

    # ── Step-offset (always computed, regardless of W&B mode) ─────────────────
    # Keep the step budget based on a full nominal 1/3 worker shard so W&B step
    # offsets remain monotonic across rounds even when the final round is a
    # trimmed remainder. Reserve one extra marker step between rounds so the
    # round-start log for round N is strictly greater than the round-complete
    # log from round N-1.
    shard_step_estimate = math.ceil(
        len(train_samples) / 3 / max(args.batch_size * args.grad_accum, 1)
    )
    if shard_step_estimate <= 0:
        shard_step_estimate = _DILOCO_SHARD_STEP_FALLBACK
    round_step_span = shard_step_estimate + 1
    global_step_offset = round_n * round_step_span

    # ── W&B init (DiLoCo path only) ──────────────────────────────────────────
    # Each worker gets one persistent run per stage, resumed across rounds.
    # Run ID is stable: diloco-{worker}-s{stage_k}  (e.g. "diloco-a-s2")
    diloco_wandb_run = None
    if _is_main_process() and args.wandb_mode != "disabled":
        try:
            import wandb as _wandb
            run_id = f"diloco-{args.diloco_worker_id.lower()}-s{stage_k}"
            diloco_wandb_run = _wandb.init(
                project=args.wandb_project,
                id=run_id,
                resume="allow",
                name=f"Worker {args.diloco_worker_id} | Stage {stage_k}",
                config={
                    **_wandb_config(args),
                    "stage_k": stage_k,
                    "round_n": round_n,
                    "worker_id": args.diloco_worker_id,
                    "mode": "diloco",
                    "shard_step_estimate": shard_step_estimate,
                    "wandb_round_step_span": round_step_span,
                    "remaining_stage_samples": remaining_stage_samples,
                    "planned_shard_samples": len(train_shard),
                },
                mode=args.wandb_mode,
            )
            _wandb.log(
                {"diloco/round": round_n, "diloco/stage": stage_k},
                step=global_step_offset,
            )
        except Exception as _we:
            print(f"  [diloco] W&B init failed: {_we}")

    should_run_pre_val = bool(
        args.diloco_run_val
        or (round_n == 0 and is_new_stage and args.diloco_worker_id == "A")
    )
    if should_run_pre_val and val_samples:
        val_ce, val_acc = evaluate_stage(
            model=model,
            val_samples=val_samples,
            tokenizer=tokenizer,
            lat_token_id=lat_token_id,
            stage_k=stage_k,
            device=device,
            args=args,
            halt_gate=None,
        )
        if _is_main_process():
            print(f"  [diloco] Pre-training val: stage={stage_k} ce={val_ce:.4f} acc={val_acc:.4f}")
            if diloco_wandb_run is not None:
                import wandb
                wandb.log(
                    {
                        "diloco/pre_val_ce": val_ce,
                        "diloco/pre_val_acc": val_acc,
                        "diloco/stage": stage_k,
                        "diloco/round": round_n,
                    },
                    step=global_step_offset,
                )
    barrier()

    original_push_to_hub = args.push_to_hub
    original_stage0_epochs = args.stage_0_epochs
    original_epochs_per_stage = args.epochs_per_stage
    original_gen_every_stage = args.gen_every_stage

    args.push_to_hub = False
    args.epochs_per_stage = 1
    if stage_k == 0:
        args.stage_0_epochs = 1
    args.gen_every_stage = False

    try:
        result = run_training_stages(
            model=model,
            tokenizer=tokenizer,
            halt_gate=None,
            train_samples=train_shard,
            val_samples=val_samples,
            lat_token_id=lat_token_id,
            pad_id=pad_id,
            args=args,
            device=device,
            output_dir=output_dir,
            session_start=session_start,
            wandb_run=diloco_wandb_run,
            stages=[stage_k],
            curriculum_max_stage=curriculum_max_stage,
            resume_path=None,
            resume_same_stage=False,
            resume_stage=stage_k,
            resume_epoch=0,
            resume_step_in_epoch=-1,
            global_step=global_step_offset,
            step_in_phase=0,
            load_best_between_stages=False,
            run_generation_at_stage_end=False,
            run_epoch_end_val=False,
        )
    finally:
        args.push_to_hub = original_push_to_hub
        args.stage_0_epochs = original_stage0_epochs
        args.epochs_per_stage = original_epochs_per_stage
        args.gen_every_stage = original_gen_every_stage

    samples_seen_this_round = int(min(result["samples_seen"], len(train_shard)))

    barrier()
    if _is_main_process():
        upload_dir = output_dir / "diloco_worker_upload" / f"worker_{args.diloco_worker_id}_stage_{stage_k}_round_{round_n}"
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
        upload_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(upload_dir))

        diloco_upload_worker_state(
            adapter_dir=upload_dir,
            worker_id=args.diloco_worker_id,
            stage_k=stage_k,
            round_n=round_n,
            samples_seen=samples_seen_this_round,
            hf_token=hf_token,
            repo_id=args.diloco_state_repo,
        )

        github_token = os.environ.get("GITHUB_TOKEN")
        if github_token and args.diloco_signal_repo:
            diloco_push_signal(
                args.diloco_worker_id,
                stage_k,
                round_n,
                github_token,
                args.diloco_signal_repo,
            )
        else:
            print("  [diloco] No GITHUB_TOKEN - coordinator must be triggered manually.")

        print(
            f"  [diloco] Worker {args.diloco_worker_id} done. "
            f"stage={stage_k} round={round_n} samples_seen={samples_seen_this_round}"
        )
    barrier()

    if diloco_wandb_run is not None:
        import wandb
        wandb.log(
            {
                "diloco/round": round_n,
                "diloco/stage": stage_k,
                "diloco/samples_seen_this_round": samples_seen_this_round,
                "diloco/round_complete": 1,
            },
            step=global_step_offset + shard_step_estimate,
        )
        wandb.finish()

    return {
        "stage_k": stage_k,
        "round_n": round_n,
        "samples_seen": samples_seen_this_round,
        **result,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    rank = _rank()
    world_size = _world_size()
    local_rank = _local_rank()
    distributed = world_size > 1
    is_main = rank == 0

    if args.diloco_mode and args.use_halt_gate:
        raise ValueError(
            "--diloco_mode and --use_halt_gate should not be combined. "
            "DiLoCo syncs LoRA adapters only; DGAC should stay on the sequential path."
        )

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        init_kwargs = dict(
            backend=backend,
            init_method="env://",
            timeout=timedelta(hours=4),
        )
        if torch.cuda.is_available():
            try:
                torch.distributed.init_process_group(
                    **init_kwargs,
                    device_id=torch.device("cuda", local_rank),
                )
            except TypeError:
                torch.distributed.init_process_group(**init_kwargs)
        else:
            torch.distributed.init_process_group(**init_kwargs)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if distributed and args.batch_size % world_size != 0:
        raise ValueError(
            f"--batch_size ({args.batch_size}) must be divisible by WORLD_SIZE ({world_size})"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_start = _SCRIPT_START

    hf_token = _resolve_hf_token(getattr(args, "hf_token", None))
    args._resolved_hf_token = hf_token

    if getattr(args, "push_to_hub", False) and not hf_token:
        if _is_main_process():
            print("[warn] --push_to_hub set but no HF token found; Hub sync disabled.")
        args.push_to_hub = False

    wandb_run = None
    if is_main and args.wandb_mode != "disabled" and not args.diloco_mode:
        # Sequential curriculum path: init wandb normally.
        # DiLoCo path defers init to run_diloco_worker() where stage_k/round_n are known.
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config=_wandb_config(args),
            )
        except ImportError:
            print("[warn] wandb not installed")

    try:
        train_samples, val_samples, stats = load_canonical_dataset(Path(args.data_dir), args.max_samples)
        if not train_samples:
            raise ValueError("No training samples were loaded. Check --data_dir / --max_samples.")
        curriculum_max_stage = get_max_stage(args, stats)

        model, tokenizer, d_model, lat_token_id = load_model_and_tokenizer(args, device)
        pad_id = tokenizer.pad_token_id or 0
        if is_main:
            tokenizer_dir = output_dir / "tokenizer"
            tokenizer.save_pretrained(tokenizer_dir)

        halt_gate: Optional[HaltGate] = None
        if args.use_halt_gate:
            halt_gate = HaltGate(d_model).to(device=device, dtype=torch.float32)
            if is_main:
                n_params = sum(p.numel() for p in halt_gate.parameters())
                print(f"  DGAC HaltGate: d_model={d_model}  params={n_params}")

        if args.diloco_mode:
            run_diloco_worker(
                model=model,
                tokenizer=tokenizer,
                halt_gate=halt_gate,
                train_samples=train_samples,
                val_samples=val_samples,
                curriculum_max_stage=curriculum_max_stage,
                lat_token_id=lat_token_id,
                pad_id=pad_id,
                args=args,
                device=device,
                output_dir=output_dir,
                session_start=session_start,
                wandb_run=wandb_run,
                hf_token=hf_token or "",
            )
            return

        requested_resume_path: Optional[Path] = Path(args.resume_from) if args.resume_from else None
        hub_resume_dir = output_dir / ".hub_resume"
        resume_path = _resolve_resume_checkpoint_for_all_ranks(
            output_dir=output_dir,
            requested_resume=requested_resume_path,
            hf_token=hf_token,
            hf_repo_id=getattr(args, "hf_repo_id", "WeirdRunner/Ouroboros"),
            hf_stage_subdir=getattr(args, "hf_stage_subdir", "runs/stage3"),
            distributed=distributed,
            is_main=is_main,
        )

        if resume_path is not None and not (resume_path / "training_state.pt").exists():
            if is_main:
                print(f"  [warn] resume checkpoint not found: {resume_path}")
            resume_path = None
            if distributed and is_main:
                _distributed_resume_marker(output_dir).write_text("", encoding="utf-8")
        if distributed:
            barrier()
            if not is_main and resume_path is None:
                raw = _distributed_resume_marker(output_dir).read_text(encoding="utf-8").strip() if _distributed_resume_marker(output_dir).exists() else ""
                resume_path = Path(raw) if raw else None
            barrier()

        if hf_token and getattr(args, "push_to_hub", False) and is_main:
            startup_hub_sync_and_prune(
                output_dir=output_dir,
                resume_path=resume_path,
                hf_token=hf_token,
                hf_repo_id=getattr(args, "hf_repo_id", "WeirdRunner/Ouroboros"),
                hf_stage_subdir=getattr(args, "hf_stage_subdir", "runs/stage3"),
            )
        barrier()

        resume_state: Optional[Dict[str, Any]] = None
        resume_same_stage = False
        resume_stage = 0
        resume_epoch = 0
        resume_step_in_epoch = -1
        global_step = 0
        step_in_phase = 0

        if resume_path is not None:
            resume_state = load_checkpoint(
                resume_path,
                model,
                halt_gate,
                optimizer=None,
                scheduler=None,
                device=device,
                verbose=is_main,
            )

            resume_stage = int(resume_state.get("stage_k", 0))
            global_step = int(resume_state.get("step", 0))
            if args.use_halt_gate:
                resume_same_stage = bool(resume_state.get("use_halt_gate", False) and resume_path.name != "best")
                if resume_same_stage:
                    resume_epoch = int(resume_state.get("epoch", 0))
                    resume_step_in_epoch = int(resume_state.get("step_in_epoch", -1))
                    step_in_phase = int(resume_state.get("step_in_phase", 0))
            else:
                resume_same_stage = resume_path.name != "best"
                if resume_same_stage:
                    resume_epoch = int(resume_state.get("epoch", 0))
                    resume_step_in_epoch = int(resume_state.get("step_in_epoch", -1))

        if args.use_halt_gate:
            gate_stage = resume_stage if resume_state is not None else curriculum_max_stage
            stages = [gate_stage]
            if resume_state is None and is_main:
                print(
                    "  [warn] --use_halt_gate without --resume_from: "
                    "training DGAC from current weights at Stage K."
                )
        else:
            if resume_state is not None and resume_path is not None and resume_path.name == "best":
                start_stage = resume_stage + 1
            else:
                start_stage = resume_stage if resume_state is not None else 0
            stages = list(range(start_stage, curriculum_max_stage + 1))

        if distributed:
            broadcast_parameters(get_trainable_parameters(model, halt_gate), src=0)

        if is_main and not stages:
            print("  No stages left to run. Nothing to do.")
        if not stages:
            _cleanup_distributed_resume_artifacts(output_dir, hub_resume_dir, distributed, is_main)
            return

        result = run_training_stages(
            model=model,
            tokenizer=tokenizer,
            halt_gate=halt_gate,
            train_samples=train_samples,
            val_samples=val_samples,
            lat_token_id=lat_token_id,
            pad_id=pad_id,
            args=args,
            device=device,
            output_dir=output_dir,
            session_start=session_start,
            wandb_run=wandb_run,
            stages=stages,
            curriculum_max_stage=curriculum_max_stage,
            resume_path=resume_path,
            resume_same_stage=resume_same_stage,
            resume_stage=resume_stage,
            resume_epoch=resume_epoch,
            resume_step_in_epoch=resume_step_in_epoch,
            global_step=global_step,
            step_in_phase=step_in_phase,
            load_best_between_stages=not args.use_halt_gate,
            run_generation_at_stage_end=bool(args.gen_every_stage),
            run_epoch_end_val=True,
        )

        _cleanup_distributed_resume_artifacts(output_dir, hub_resume_dir, distributed, is_main)

        if is_main:
            print("\n" + "=" * 64)
            if result["timeout_triggered"] or result["val_budget_triggered"]:
                if result["val_budget_triggered"] and not result["timeout_triggered"]:
                    print(
                        "  [timeout] Remaining session time fell below "
                        f"--val_skip_buffer_minutes ({args.val_skip_buffer_minutes:.0f} min) - checkpoint saved."
                    )
                else:
                    print("  [timeout] Session budget exhausted - checkpoint saved.")
                print("  Re-run the same command with the same --output_dir to auto-resume.")
            else:
                print(f"  Curriculum complete. Stages: {stages}  Global steps: {result['global_step']}")
                if not args.use_halt_gate:
                    best_k_dir = output_dir / f"stage_{curriculum_max_stage}" / "best"
                    print(
                        "  Phase 3.4 (DGAC):\n"
                        f"    python jamba_coconut_finetune.py --use_halt_gate "
                        f"--resume_from {best_k_dir} "
                        f"--output_dir {args.output_dir}_dgac [...]"
                    )
            print("=" * 64)

    finally:
        if distributed and _distributed_is_initialized():
            torch.distributed.destroy_process_group()
        if wandb_run is not None:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
