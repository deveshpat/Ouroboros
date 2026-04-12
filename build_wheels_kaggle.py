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
accelerate 7% of the model while costing 2-3 hours of compilation (73 CUDA
translation units on T4, no PyPI wheel for cu128+). Use --include_flash_attn
only if you have a specific reason and spare quota.

Usage:
  # Check if already working (pure import check, zero installs):
  !python build_wheels_kaggle.py --verify_only

  # Full build + upload (run when verify_only fails):
  !python build_wheels_kaggle.py --hf_repo_id WeirdRunner/Ouroboros --hf_token YOUR_TOKEN

Re-run the full build when Kaggle upgrades its container. Check with:
  python -c "import torch; print(torch.__version__, torch.version.cuda)"

Why --no-build-isolation:
  Both packages compile CUDA C++ extensions via torch.utils.cpp_extension.
  pip's isolated build venv has no CUDA toolkit or PyTorch headers — the
  flag is always required for source builds on CUDA packages.

Why cxx11 ABI must match:
  PyTorch itself is compiled with a specific GLIBCXX ABI setting (TRUE or
  FALSE). Any C++ extension (.so) must match or it will silently return None
  at import. Check your env: python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
  Rebuild whenever this value changes (i.e. new container image).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=False)


def detect_env() -> dict:
    import torch
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_full = torch.__version__.split("+")[0]
    torch_short = ".".join(torch_full.split(".")[:2])
    cuda_ver = torch.version.cuda or ""
    try:
        cxx11 = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    except AttributeError:
        cxx11 = "UNKNOWN"
    env = {
        "py_ver":      py_ver,
        "torch_full":  torch_full,
        "torch_short": torch_short,
        "cuda_ver":    cuda_ver,
        "cxx11":       cxx11,
    }
    print("\n=== Detected Environment ===")
    for k, v in env.items():
        print(f"  {k}: {v}")
    if cxx11 == "TRUE":
        print("\n  [NOTE] cxx11=TRUE — any wheels built with cxx11abiFALSE will NOT load.")
        print("         Rebuild is required if your Hub wheels were built with FALSE.")
    print()
    return env


def verify_imports(install_first: bool = False) -> bool:
    """
    Try to import the three mamba fast-path symbols.
    Does NOT install anything — pure import check.
    Returns True if fast path is fully active.
    """
    if install_first:
        # This branch is intentionally not used in --verify_only.
        # Install should always come from explicit Hub wheels, not PyPI source.
        pass

    print("\n=== Verifying Fast Path (import check only) ===")
    checks = [
        ("mamba_ssm.ops.selective_scan_interface", "selective_scan_fn"),
        ("mamba_ssm.ops.selective_scan_interface", "selective_state_update"),
        ("causal_conv1d",                          "causal_conv1d_fn"),
    ]
    errors = []
    for mod, attr in checks:
        try:
            m = __import__(mod, fromlist=[attr])
            obj = getattr(m, attr, None)
            status = "OK" if obj is not None else "None — ABI mismatch or bad build (FAIL)"
            if obj is None:
                errors.append(attr)
        except ImportError as e:
            status = f"ImportError: {e} (FAIL)"
            errors.append(attr)
        print(f"  {mod}.{attr}: {status}")

    if not errors:
        print("\n  ✓ Fast path ACTIVE. ~5s/step expected.")
        return True
    else:
        print(f"\n  ✗ {len(errors)} symbol(s) missing.")
        print("  Cause: package not installed, ABI mismatch, or wrong CUDA arch.")
        print("  Fix:   run without --verify_only to rebuild and re-upload wheels.")
        return False


def build_wheels(wheel_dir: Path, include_flash_attn: bool) -> list:
    wheel_dir.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "-m", "pip", "install", "-q",
          "ninja", "packaging", "wheel", "setuptools"])

    packages = [
        ("causal-conv1d>=1.4.0", "MAX_JOBS=4"),
        ("mamba-ssm",             "MAX_JOBS=4"),
    ]
    if include_flash_attn:
        print("\n[warn] flash-attn build requested. ~2-3 hours on T4. Not needed for Jamba.")
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
            print(f"\n[ERROR] Failed to build {name}. Check: nvcc --version, echo $CUDA_HOME")
            continue
        found = list(wheel_dir.glob(f"{name.replace('-', '_')}*.whl"))
        if found:
            print(f"  Built: {found[-1].name}")
            built.append(found[-1])
        else:
            print(f"  [warn] No .whl found after build for {name}")
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
    parser.add_argument("--include_flash_attn", action="store_true",
                        help="Also build flash-attn (~2-3 hrs). NOT needed for Jamba.")
    parser.add_argument("--skip_upload", action="store_true")
    parser.add_argument("--skip_build",  action="store_true",
                        help="Upload existing .whl files in --wheel_dir without rebuilding.")
    parser.add_argument("--verify_only", action="store_true",
                        help="Pure import check. No installs, no builds. Just tell me if it works.")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    env = detect_env()

    # --verify_only: pure import check, zero side effects
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

    if not args.skip_upload:
        if not hf_token:
            print("[ERROR] No HF token.")
            sys.exit(1)
        urls = upload_wheels(built, args.hf_repo_id, hf_token)
        print_install_snippet(urls, env)

    # Install the freshly built wheels into the current env, then verify
    for whl in built:
        _run([sys.executable, "-m", "pip", "install", "--force-reinstall",
              "--no-deps", str(whl)], check=False)

    verify_imports()


if __name__ == "__main__":
    main()
