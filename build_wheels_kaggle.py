#!/usr/bin/env python3
"""
build_wheels_kaggle.py — One-time Mamba/flash-attn wheel builder for Project Ouroboros
========================================================================================
Run once in a fresh Kaggle GPU session (T4 or better).
Builds environment-matched wheels for:
  - flash-attn
  - causal-conv1d
  - mamba-ssm

Then uploads them to HF Hub so every subsequent session installs in ~90 seconds
instead of compiling for 30-60 minutes.

Usage (Kaggle notebook cell):
  !python build_wheels_kaggle.py --hf_repo_id WeirdRunner/Ouroboros --hf_token YOUR_TOKEN

After this runs once, update the WHEEL_URLS dict in jamba_coconut_finetune.py
with the printed URLs.

Notes:
  - All three packages require --no-build-isolation: they need the live torch/CUDA
    headers from the current env during compilation (not pip's isolated build venv).
  - flash-attn has no PyPI pre-built wheel → always compiles from source.
  - causal-conv1d and mamba-ssm have PyPI wheels for cu118/cu121 only;
    cu128+ requires source build.
  - Expected build times: flash-attn ~20-30 min, causal-conv1d ~5-10 min,
    mamba-ssm ~10-15 min. Total: ~45-60 min. Run ONCE; subsequent sessions
    use the Hub wheels.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, capture_output=False)
    return result


def detect_env() -> dict:
    """Detect Python / CUDA / PyTorch / cxx11 ABI for wheel filename matching."""
    import torch

    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"

    torch_full = torch.__version__.split("+")[0]          # e.g. "2.10.0"
    torch_short = ".".join(torch_full.split(".")[:2])      # e.g. "2.10"
    torch_tag = torch_short.replace(".", "")               # e.g. "210"

    cuda_ver = torch.version.cuda or ""                    # e.g. "12.8"
    cuda_tag = cuda_ver.replace(".", "")                   # e.g. "128"

    # Detect CXX11 ABI used when building PyTorch
    try:
        cxx11 = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()  # "TRUE" or "FALSE"
    except AttributeError:
        cxx11 = "FALSE"

    env = {
        "py_ver": py_ver,
        "torch_full": torch_full,
        "torch_short": torch_short,
        "torch_tag": torch_tag,
        "cuda_ver": cuda_ver,
        "cuda_tag": cuda_tag,
        "cxx11": cxx11,
    }
    print("\n=== Detected Environment ===")
    for k, v in env.items():
        print(f"  {k}: {v}")
    print()
    return env


def build_wheels(wheel_dir: Path, env: dict) -> list:
    """
    Build all three packages from source. Returns list of built .whl paths.
    All three use --no-build-isolation so the live torch/CUDA env is visible
    to the C++ extension compiler (torch.utils.cpp_extension needs it).
    """
    wheel_dir.mkdir(parents=True, exist_ok=True)

    # Build dependencies (ninja speeds up compilation significantly)
    _run([sys.executable, "-m", "pip", "install", "-q",
          "ninja", "packaging", "wheel", "setuptools"])

    built = []

    packages = [
        # (pip_spec, max_jobs_env_var)
        ("causal-conv1d>=1.4.0",  "MAX_JOBS=4"),
        ("mamba-ssm",              "MAX_JOBS=4"),
        ("flash-attn",             "MAX_JOBS=4"),
    ]

    for spec, jobs_env in packages:
        name = spec.split(">=")[0].split("==")[0].strip()
        print(f"\n{'='*60}")
        print(f"Building: {spec}")
        print(f"{'='*60}")
        env_vars = os.environ.copy()
        env_vars[jobs_env.split("=")[0]] = jobs_env.split("=")[1]

        cmd = [
            sys.executable, "-m", "pip", "wheel",
            spec,
            "--no-build-isolation",
            "--no-deps",
            "-w", str(wheel_dir),
            "--verbose",
        ]
        result = subprocess.run(cmd, env=env_vars, check=False)
        if result.returncode != 0:
            print(f"\n[ERROR] Failed to build {name}. Check CUDA toolkit visibility.")
            print("Ensure: nvcc --version works and CUDA_HOME is set.")
            continue

        found = list(wheel_dir.glob(f"{name.replace('-', '_')}*.whl"))
        if found:
            print(f"  Built: {found[-1].name}")
            built.append(found[-1])
        else:
            print(f"  [warn] Wheel not found after build for {name}")

    return built


def upload_wheels(wheel_dir: Path, built: list, hf_repo_id: str, hf_token: str) -> list:
    """Upload built wheels to HF Hub model repo. Returns list of direct download URLs."""
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
        print(f"\nUploading: {whl.name} -> {hf_repo_id}")
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
        print(f"  URL: {url}")

    return urls


def print_install_command(urls: list, env: dict) -> None:
    print("\n" + "=" * 70)
    print("COPY THIS — paste into jamba_coconut_finetune.py docstring")
    print("=" * 70)
    print(f"# Environment: CUDA {env['cuda_ver']}  "
          f"PyTorch {env['torch_short']}  "
          f"Python {env['py_ver']}  "
          f"cxx11ABI={env['cxx11']}")
    parts = ["!pip install -q",
             '"transformers>=4.54.0"',
             "peft datasets tqdm wandb bitsandbytes accelerate huggingface_hub",
             "\\"]
    for name, url in urls:
        parts.append(f"  {url} \\")
    print(" ".join(parts))
    print("\nPaste these WHEEL_URLS into jamba_coconut_finetune.py:")
    for name, url in urls:
        print(f'  "{name}": "{url}",')
    print("=" * 70)


def verify_imports() -> None:
    """Quick smoke-test that the CUDA fast path is now active."""
    print("\n=== Verifying Fast Path ===")
    errors = []
    for mod, attr in [
        ("mamba_ssm.ops.selective_scan_interface", "selective_scan_fn"),
        ("mamba_ssm.ops.selective_scan_interface", "selective_state_update"),
        ("causal_conv1d", "causal_conv1d_fn"),
    ]:
        try:
            m = __import__(mod, fromlist=[attr])
            obj = getattr(m, attr, None)
            status = "OK" if obj is not None else "None (FAIL)"
        except ImportError as e:
            status = f"ImportError: {e}"
            errors.append(f"{mod}.{attr}")
        print(f"  {mod}.{attr}: {status}")

    if not errors:
        print("\n  Fast path ACTIVE. use_mamba_kernels=True will work.")
    else:
        print(f"\n  [WARN] {len(errors)} import(s) failed. Fast path not available.")
        print("  Check CUDA_HOME, nvcc, and wheel compatibility.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build mamba/flash-attn wheels and upload to HF Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hf_repo_id", default="WeirdRunner/Ouroboros")
    parser.add_argument("--hf_token", default=None,
                        help="HF write token. Falls back to HF_TOKEN env var.")
    parser.add_argument("--wheel_dir", default="/kaggle/working/wheels",
                        help="Local directory to write built .whl files.")
    parser.add_argument("--skip_upload", action="store_true",
                        help="Build wheels only, don't upload.")
    parser.add_argument("--skip_build", action="store_true",
                        help="Skip build, only upload existing wheels in --wheel_dir.")
    parser.add_argument("--verify_only", action="store_true",
                        help="Only run the import verification; don't build.")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    env = detect_env()

    if args.verify_only:
        _run([sys.executable, "-m", "pip", "install", "-q",
              "causal-conv1d>=1.4.0", "--no-build-isolation"])
        _run([sys.executable, "-m", "pip", "install", "-q",
              "mamba-ssm", "--no-build-isolation"])
        verify_imports()
        return

    wheel_dir = Path(args.wheel_dir)

    if not args.skip_build:
        built = build_wheels(wheel_dir, env)
        if not built:
            print("\n[FATAL] No wheels built successfully.")
            sys.exit(1)
        print(f"\nBuilt {len(built)} wheel(s):")
        for w in built:
            print(f"  {w.name}")
    else:
        built = list(wheel_dir.glob("*.whl"))
        print(f"Skipping build. Found {len(built)} existing wheel(s) in {wheel_dir}.")

    if not args.skip_upload:
        if not hf_token:
            print("[ERROR] No HF token. Set --hf_token or HF_TOKEN env var.")
            sys.exit(1)
        urls = upload_wheels(wheel_dir, built, args.hf_repo_id, hf_token)
        print_install_command(urls, env)
    else:
        print("Skipping Hub upload (--skip_upload).")
        for whl in built:
            print(f"  Local: {whl}")

    # Install fresh so verify_imports() works
    for whl in built:
        _run([sys.executable, "-m", "pip", "install", "--force-reinstall", str(whl)], check=False)

    verify_imports()


if __name__ == "__main__":
    main()
