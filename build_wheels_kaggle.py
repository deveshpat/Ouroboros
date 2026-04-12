#!/usr/bin/env python3
"""
build_wheels_kaggle.py — Mamba kernel wheel builder for Project Ouroboros
==========================================================================
Run once in a fresh Kaggle GPU session (T4 or better).
Builds environment-matched wheels for:
  - causal-conv1d   (~5-10 min)
  - mamba-ssm       (~10-15 min)

flash-attn is deliberately EXCLUDED by default. Jamba Reasoning 3B has
26 Mamba layers and only 2 attention layers (13:1 ratio). flash-attn would
accelerate 7% of the model while costing 2-3 hours of compilation from source
(73 CUDA translation units on T4, no PyPI wheel for cu128+). The eager
attention fallback on 2 layers is negligible vs the mamba fast path on 26.
Use --include_flash_attn only if you have a specific reason and spare quota.

Usage (Kaggle notebook cell):
  !python build_wheels_kaggle.py --hf_repo_id WeirdRunner/Ouroboros --hf_token YOUR_TOKEN

After this runs (~20 min), paste the printed wheel URLs into the Install
section of jamba_coconut_finetune.py.

Re-run when Kaggle upgrades its container. Check with:
  python -c "import torch; print(torch.__version__, torch.version.cuda)"

Why --no-build-isolation for all packages:
  Both packages compile CUDA C++ extensions via torch.utils.cpp_extension.
  pip's isolated build venv has no CUDA toolkit or PyTorch headers, so
  compilation fails without this flag. Required for all source builds.
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
    torch_full = torch.__version__.split("+")[0]
    torch_short = ".".join(torch_full.split(".")[:2])
    cuda_ver = torch.version.cuda or ""

    try:
        cxx11 = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    except AttributeError:
        cxx11 = "FALSE"

    env = {
        "py_ver": py_ver,
        "torch_full": torch_full,
        "torch_short": torch_short,
        "cuda_ver": cuda_ver,
        "cxx11": cxx11,
    }
    print("\n=== Detected Environment ===")
    for k, v in env.items():
        print(f"  {k}: {v}")
    print()
    return env


def build_wheels(wheel_dir: Path, env: dict, include_flash_attn: bool) -> list:
    """
    Build mamba packages from source. Returns list of built .whl paths.
    flash-attn skipped by default (see module docstring).
    """
    wheel_dir.mkdir(parents=True, exist_ok=True)

    _run([sys.executable, "-m", "pip", "install", "-q",
          "ninja", "packaging", "wheel", "setuptools"])

    packages = [
        ("causal-conv1d>=1.4.0", "MAX_JOBS=4"),
        ("mamba-ssm",             "MAX_JOBS=4"),
    ]
    if include_flash_attn:
        print("\n[warn] flash-attn requested. Expect 2-3 hours on T4 (73 CUDA TUs).")
        print("       NOT needed for Jamba — only 2/28 layers use attention.")
        packages.append(("flash-attn", "MAX_JOBS=4"))

    built = []
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
            print(f"\n[ERROR] Failed to build {name}.")
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
    """Upload built wheels to HF Hub model repo. Returns list of (name, url) tuples."""
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
    print("COPY THIS — paste into jamba_coconut_finetune.py Install section")
    print("=" * 70)
    print(f"# Environment: CUDA {env['cuda_ver']}  "
          f"PyTorch {env['torch_short']}  "
          f"Python {env['py_ver']}  "
          f"cxx11ABI={env['cxx11']}")
    print("!pip install -q \"transformers>=4.54.0\" peft datasets tqdm wandb \\")
    print("             bitsandbytes accelerate huggingface_hub \\")
    for name, url in urls:
        print(f"  {url} \\")
    print("\nWheel URLs for jamba_coconut_finetune.py:")
    for name, url in urls:
        print(f'  "{name}": "{url}",')
    print("=" * 70)


def verify_imports() -> None:
    """Confirm mamba CUDA fast path is active after installation."""
    print("\n=== Verifying Fast Path ===")
    checks = [
        ("mamba_ssm.ops.selective_scan_interface", "selective_scan_fn"),
        ("mamba_ssm.ops.selective_scan_interface", "selective_state_update"),
        ("causal_conv1d", "causal_conv1d_fn"),
    ]
    errors = []
    for mod, attr in checks:
        try:
            m = __import__(mod, fromlist=[attr])
            obj = getattr(m, attr, None)
            status = "OK" if obj is not None else "None (FAIL)"
            if obj is None:
                errors.append(f"{mod}.{attr}")
        except ImportError as e:
            status = f"ImportError: {e}"
            errors.append(f"{mod}.{attr}")
        print(f"  {mod}.{attr}: {status}")

    if not errors:
        print("\n  Fast path ACTIVE. Training will use ~5s/step (not ~500s/step).")
    else:
        print(f"\n  [WARN] {len(errors)} import(s) failed. Slow path will be used.")
        print("  Check: nvcc --version, echo $CUDA_HOME")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build mamba CUDA wheels and upload to HF Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hf_repo_id", default="WeirdRunner/Ouroboros")
    parser.add_argument("--hf_token", default=None,
                        help="HF write token. Falls back to HF_TOKEN env var.")
    parser.add_argument("--wheel_dir", default="/kaggle/working/wheels")
    parser.add_argument("--include_flash_attn", action="store_true",
                        help="Also build flash-attn (~2-3 hrs). Not needed for Jamba.")
    parser.add_argument("--skip_upload", action="store_true",
                        help="Build only, don't upload to Hub.")
    parser.add_argument("--skip_build", action="store_true",
                        help="Skip build, upload existing wheels in --wheel_dir.")
    parser.add_argument("--verify_only", action="store_true",
                        help="Only check if fast path is importable; don't build.")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    env = detect_env()

    if args.verify_only:
        _run([sys.executable, "-m", "pip", "install", "-q",
              "causal-conv1d>=1.4.0", "mamba-ssm", "--no-build-isolation"], check=False)
        verify_imports()
        return

    wheel_dir = Path(args.wheel_dir)

    if not args.skip_build:
        built = build_wheels(wheel_dir, env, include_flash_attn=args.include_flash_attn)
        if not built:
            print("\n[FATAL] No wheels built. Check CUDA toolkit and try again.")
            sys.exit(1)
        print(f"\nBuilt {len(built)} wheel(s):")
        for w in built:
            print(f"  {w.name}")
    else:
        built = list(wheel_dir.glob("*.whl"))
        print(f"Skipping build. Found {len(built)} wheel(s) in {wheel_dir}.")

    if not args.skip_upload:
        if not hf_token:
            print("[ERROR] No HF token. Set --hf_token or HF_TOKEN env var.")
            sys.exit(1)
        urls = upload_wheels(wheel_dir, built, args.hf_repo_id, hf_token)
        print_install_command(urls, env)
    else:
        print("Skipping Hub upload.")
        for whl in built:
            print(f"  Local: {whl}")

    for whl in built:
        _run([sys.executable, "-m", "pip", "install", "--force-reinstall", str(whl)], check=False)

    verify_imports()


if __name__ == "__main__":
    main()
