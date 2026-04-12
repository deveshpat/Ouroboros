#!/usr/bin/env python3
"""
build_wheels_kaggle.py — Mamba kernel wheel builder for Project Ouroboros
==========================================================================
Run once in a fresh Kaggle GPU session (T4 or better).
Builds environment-matched wheels for:
  - causal-conv1d>=1.4.0   (~5-10 min)
  - mamba-ssm==1.2.2       (~10-15 min)

VERSION NOTE — why mamba-ssm==1.2.2, not 2.x:
  transformers' Jamba implementation checks for five symbols at import paths
  established by the mamba_ssm 1.x API:
    selective_scan_fn, selective_state_update, causal_conv1d_fn,
    causal_conv1d_update, mamba_inner_fn
  In mamba_ssm 2.x, selective_state_update moved to a Triton-based path and
  is None at the old location. This silently disables the fast path regardless
  of whether the CUDA kernels compiled correctly. Confirmed in log 2026-04-12:
  mamba_ssm 2.3.1 built fine but selective_state_update = None.
  Pin to 1.2.2 until transformers Jamba is updated to the 2.x API.

flash-attn is deliberately EXCLUDED by default. Jamba has 26 Mamba layers and
only 2 attention layers. flash-attn covers 7% of the model; takes 2-3 hours
to compile from source on T4 (73 CUDA TUs, no PyPI wheel for cu128+).

Usage:
  # Check if already working — zero installs, pure import check:
  !python build_wheels_kaggle.py --verify_only

  # Full build + upload (~20 min):
  !python build_wheels_kaggle.py --hf_repo_id WeirdRunner/Ouroboros --hf_token YOUR_TOKEN

Re-run when Kaggle upgrades its container:
  python -c "import torch; print(torch.__version__, torch.version.cuda)"

Why --no-build-isolation:
  CUDA C++ extensions need the live PyTorch headers + CUDA toolkit during
  compilation. pip's isolated build venv exposes neither.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Pinned to 1.x — transformers Jamba uses the 1.x selective_state_update API.
# Do NOT bump to 2.x until transformers updates its Jamba fast-path imports.
MAMBA_SSM_VERSION = "mamba-ssm==1.2.2"
CAUSAL_CONV1D_SPEC = "causal-conv1d>=1.4.0"


def _run(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=False)


def detect_env() -> dict:
    import torch
    py_ver      = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_full  = torch.__version__.split("+")[0]
    torch_short = ".".join(torch_full.split(".")[:2])
    cuda_ver    = torch.version.cuda or ""
    try:
        cxx11 = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    except AttributeError:
        cxx11 = "UNKNOWN"
    env = {"py_ver": py_ver, "torch_full": torch_full,
           "torch_short": torch_short, "cuda_ver": cuda_ver, "cxx11": cxx11}
    print("\n=== Detected Environment ===")
    for k, v in env.items():
        print(f"  {k}: {v}")
    print()
    return env


def verify_imports() -> bool:
    """
    Pure import check — confirms all five symbols that transformers Jamba
    requires for its fast-path gate are non-None.
    Does NOT install anything.
    """
    print("\n=== Verifying Fast Path (import check only) ===")
    # These are the exact symbols transformers checks in JambaMambaMixer
    checks = [
        ("mamba_ssm.ops.selective_scan_interface", "selective_scan_fn"),
        ("mamba_ssm.ops.selective_scan_interface", "selective_state_update"),
        ("mamba_ssm.ops.selective_scan_interface", "mamba_inner_fn"),
        ("causal_conv1d",                          "causal_conv1d_fn"),
        ("causal_conv1d",                          "causal_conv1d_update"),
    ]
    errors = []
    for mod, attr in checks:
        try:
            m   = __import__(mod, fromlist=[attr])
            obj = getattr(m, attr, None)
            ok  = obj is not None
            print(f"  {'✓' if ok else '✗'} {mod}.{attr}: {'OK' if ok else 'None (FAIL)'}")
            if not ok:
                errors.append(attr)
        except ImportError as e:
            print(f"  ✗ {mod}.{attr}: ImportError: {e}")
            errors.append(attr)

    if not errors:
        print("\n  ✓ All 5 fast-path symbols present. ~5s/step expected.")
        return True
    else:
        print(f"\n  ✗ {len(errors)} symbol(s) missing: {errors}")
        if any("selective_state_update" in e or "mamba_inner_fn" in e for e in errors):
            print("  Likely cause: mamba-ssm 2.x installed (2.x moved these symbols).")
            print(f"  Fix: rebuild with {MAMBA_SSM_VERSION}  (run without --verify_only)")
        else:
            print("  Likely cause: ABI mismatch or wrong CUDA arch.")
            print("  Fix: rebuild wheels in this exact Kaggle session (run without --verify_only)")
        return False


def build_wheels(wheel_dir: Path, include_flash_attn: bool) -> list:
    wheel_dir.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "-m", "pip", "install", "-q",
          "ninja", "packaging", "wheel", "setuptools"])

    packages = [
        (CAUSAL_CONV1D_SPEC, "MAX_JOBS=4"),
        (MAMBA_SSM_VERSION,  "MAX_JOBS=4"),
    ]
    if include_flash_attn:
        print("\n[warn] flash-attn: ~2-3 hours on T4. Not needed for Jamba.")
        packages.append(("flash-attn", "MAX_JOBS=4"))

    built = []
    for spec, jobs_env in packages:
        name = spec.split(">=")[0].split("==")[0].strip()
        print(f"\n{'='*60}\nBuilding: {spec}\n{'='*60}")
        env_vars = os.environ.copy()
        env_vars[jobs_env.split("=")[0]] = jobs_env.split("=")[1]
        result = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", spec,
             "--no-build-isolation", "--no-deps", "-w", str(wheel_dir), "--verbose"],
            env=env_vars, check=False,
        )
        if result.returncode != 0:
            print(f"\n[ERROR] {name} build failed. Check: nvcc --version, echo $CUDA_HOME")
            continue
        found = list(wheel_dir.glob(f"{name.replace('-', '_')}*.whl"))
        if found:
            print(f"  Built: {found[-1].name}")
            built.append(found[-1])
        else:
            print(f"  [warn] No .whl found for {name}")
    return built


def upload_wheels(built: list, hf_repo_id: str, hf_token: str) -> list:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        _run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hf_repo_id, repo_type="model",
                    private=True, exist_ok=True, token=hf_token)
    urls = []
    for whl in built:
        print(f"\nUploading: {whl.name}")
        api.upload_file(
            path_or_fileobj=str(whl),
            path_in_repo=whl.name,
            repo_id=hf_repo_id,
            repo_type="model",
            token=hf_token,
            commit_message=f"Add wheel: {whl.name}",
        )
        url = f"https://huggingface.co/{hf_repo_id}/resolve/main/{whl.name}"
        urls.append((whl.name, url))
        print(f"  → {url}")
    return urls


def print_install_snippet(urls: list, env: dict) -> None:
    print("\n" + "=" * 70)
    print("COPY THIS into jamba_coconut_finetune.py Install section")
    print("=" * 70)
    print(f"# Built for: CUDA {env['cuda_ver']}  PyTorch {env['torch_short']}"
          f"  Python {env['py_ver']}  cxx11={env['cxx11']}")
    print("!pip install -q \"transformers>=4.54.0\" peft datasets tqdm wandb \\")
    print("             bitsandbytes accelerate huggingface_hub \\")
    for _, url in urls:
        print(f"  {url} \\")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hf_repo_id",  default="WeirdRunner/Ouroboros")
    parser.add_argument("--hf_token",    default=None)
    parser.add_argument("--wheel_dir",   default="/kaggle/working/wheels")
    parser.add_argument("--include_flash_attn", action="store_true")
    parser.add_argument("--skip_upload", action="store_true")
    parser.add_argument("--skip_build",  action="store_true",
                        help="Upload existing .whl files in --wheel_dir without rebuilding.")
    parser.add_argument("--verify_only", action="store_true",
                        help="Pure import check. No installs, no builds.")
    args = parser.parse_args()

    hf_token = (args.hf_token or os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    detect_env()

    if args.verify_only:
        ok = verify_imports()
        sys.exit(0 if ok else 1)

    wheel_dir = Path(args.wheel_dir)

    if not args.skip_build:
        built = build_wheels(wheel_dir, include_flash_attn=args.include_flash_attn)
        if not built:
            print("\n[FATAL] No wheels built.")
            sys.exit(1)
    else:
        built = list(wheel_dir.glob("*.whl"))
        print(f"Skipping build. Found {len(built)} wheel(s) in {wheel_dir}.")

    env = {}
    try:
        import torch
        env = {"cuda_ver": torch.version.cuda or "",
               "torch_short": ".".join(torch.__version__.split("+")[0].split(".")[:2]),
               "py_ver": f"cp{sys.version_info.major}{sys.version_info.minor}",
               "cxx11": str(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", "?")).upper()}
    except Exception:
        pass

    if not args.skip_upload:
        if not hf_token:
            print("[ERROR] No HF token. Set --hf_token or HF_TOKEN env var.")
            sys.exit(1)
        urls = upload_wheels(built, args.hf_repo_id, hf_token)
        print_install_snippet(urls, env)

    for whl in built:
        _run([sys.executable, "-m", "pip", "install", "--force-reinstall",
              "--no-deps", str(whl)], check=False)

    verify_imports()


if __name__ == "__main__":
    main()
