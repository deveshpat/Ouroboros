#!/usr/bin/env python3
"""
build_wheels_kaggle.py — Mamba kernel wheel builder for Project Ouroboros
==========================================================================
Run once in a fresh Kaggle GPU session. Builds environment-matched wheels for:
  - causal-conv1d>=1.4.0
  - mamba-ssm==1.2.2

VERSION NOTE — why mamba-ssm==1.2.2 not 2.x:
  transformers' Jamba checks five symbols at the mamba_ssm 1.x import paths:
    selective_scan_fn, selective_state_update, mamba_inner_fn  (mamba_ssm)
    causal_conv1d_fn, causal_conv1d_update                     (causal_conv1d)
  In 2.x, selective_state_update moved to a Triton path → None at old location
  → fast path silently disabled. Confirmed broken in session 2026-04-12.
  Pin to 1.2.2 until transformers updates its Jamba fast-path imports.

ARCH NOTE — TORCH_CUDA_ARCH_LIST injection + arch-encoded Hub filenames:
  mamba_ssm 1.2.2 predates newer GPUs, so its setup.py doesn't list them.
  This script auto-detects the GPU's compute capability and injects it into
  TORCH_CUDA_ARCH_LIST before every build, forcing compilation for the actual GPU.

  Built wheels are uploaded to Hub with the GPU arch encoded in the filename:
    causal_conv1d-1.6.1-cp312-cp312-linux_x86_64-sm100.whl
    mamba_ssm-1.2.2-cp312-cp312-linux_x86_64-sm100.whl
  This means multiple GPU arches accumulate on Hub without overwriting each other.
  jamba_coconut_finetune.py bootstrap detects the current GPU and fetches the
  matching wheel, falling back to source compile + upload if not yet cached.

flash-attn EXCLUDED by default: Jamba has 26 Mamba + 2 Attention layers.
flash-attn covers 7% of the model and takes 2-3h to compile on T4.
Use --include_flash_attn only if you have a specific reason.

Usage:
  # Zero-install fast path check:
  !python build_wheels_kaggle.py --verify_only

  # Full build + Hub upload (~20-30 min):
  !python build_wheels_kaggle.py --hf_repo_id WeirdRunner/Ouroboros --hf_token YOUR_TOKEN

Re-run whenever Kaggle changes your GPU or container image.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

MAMBA_SSM_VERSION  = "mamba-ssm==1.2.2"
CAUSAL_CONV1D_SPEC = "causal-conv1d>=1.4.0"

# All known arch suffixes — used to filter out arch-encoded files during glob
_KNOWN_ARCH_SUFFIXES = [
    "sm70", "sm72", "sm75", "sm80", "sm86", "sm87", "sm89",
    "sm90", "sm100", "sm120", "smunknown",
]


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

    gpu_name    = "unknown"
    cc_str      = None
    arch_tag    = None
    arch_suffix = None
    if torch.cuda.is_available():
        gpu_name    = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        cc_str      = f"{major}.{minor}"
        arch_tag    = cc_str
        arch_suffix = f"sm{major}{minor}"   # e.g. "sm100" for B100

    env = {
        "py_ver":      py_ver,
        "torch_full":  torch_full,
        "torch_short": torch_short,
        "cuda_ver":    cuda_ver,
        "cxx11":       cxx11,
        "gpu_name":    gpu_name,
        "cc":          cc_str or "unknown",
        "arch_tag":    arch_tag,
        "arch_suffix": arch_suffix,
    }

    print("\n=== Detected Environment ===")
    for k, v in env.items():
        if v is not None:
            print(f"  {k}: {v}")

    if arch_tag:
        major = int(arch_tag.split(".")[0])
        if major >= 9:
            print(f"\n  [NOTE] {arch_suffix} ({gpu_name}) is a newer GPU arch.")
            print("         TORCH_CUDA_ARCH_LIST will be injected automatically.")
    print()
    return env


def _build_env_vars(arch_tag: str | None) -> dict:
    env_vars = os.environ.copy()
    env_vars["MAX_JOBS"] = "4"
    if arch_tag:
        arch_list = f"{arch_tag}+PTX"
        env_vars["TORCH_CUDA_ARCH_LIST"] = arch_list
        print(f"  [arch] TORCH_CUDA_ARCH_LIST={arch_list}")
    return env_vars


def verify_imports() -> bool:
    """Pure import check — confirms all five symbols transformers Jamba needs."""
    print("\n=== Verifying Fast Path (import check only) ===")
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
            print(f"  {'✓' if ok else '✗'} {mod}.{attr}: {'OK' if ok else 'None — API mismatch (FAIL)'}")
            if not ok:
                errors.append(attr)
        except ImportError as e:
            print(f"  ✗ {mod}.{attr}: ImportError: {e}")
            errors.append(attr)

    if not errors:
        print("\n  ✓ All 5 symbols present. Fast path ACTIVE. ~5s/step expected.")
        return True

    print(f"\n  ✗ {len(errors)} symbol(s) missing: {errors}")
    if any(a in errors for a in ["selective_scan_fn", "selective_state_update", "mamba_inner_fn"]):
        if any("No module" in str(e) for e in errors):
            print("  Cause: mamba_ssm not installed.")
        else:
            print("  Cause: mamba_ssm 2.x API (2.x moved these to Triton path).")
        print(f"  Fix:   rebuild — run without --verify_only (will use {MAMBA_SSM_VERSION})")
    else:
        print("  Cause: ABI mismatch or wrong CUDA arch in wheel.")
        print("  Fix:   rebuild in this exact session (arch auto-injected at build time)")
    return False


def build_wheels(wheel_dir: Path, arch_tag: str | None, arch_suffix: str | None,
                 include_flash_attn: bool) -> list:
    """
    Build wheels and return arch-encoded copies ready for Hub upload.

    pip produces standard PEP 427 filenames, e.g.:
        mamba_ssm-1.2.2-cp312-cp312-linux_x86_64.whl

    We copy each to an arch-encoded filename before returning:
        mamba_ssm-1.2.2-cp312-cp312-linux_x86_64-sm100.whl

    The arch-encoded copy is uploaded to Hub. The standard-named original
    stays in wheel_dir for local pip install.
    """
    wheel_dir.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "-m", "pip", "install", "-q",
          "ninja", "packaging", "wheel", "setuptools"])

    packages = [CAUSAL_CONV1D_SPEC, MAMBA_SSM_VERSION]
    if include_flash_attn:
        print("\n[warn] flash-attn: 2-3h+ build. Not needed for Jamba (only 2/28 layers).")
        packages.append("flash-attn")

    env_vars = _build_env_vars(arch_tag)
    built = []
    suffix = arch_suffix or "smunknown"

    for spec in packages:
        name = spec.split(">=")[0].split("==")[0].strip()
        print(f"\n{'='*60}\nBuilding: {spec}\n{'='*60}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", spec,
             "--no-build-isolation", "--no-deps", "-w", str(wheel_dir), "--verbose"],
            env=env_vars, check=False,
        )
        if result.returncode != 0:
            print(f"\n[ERROR] {name} build failed.")
            print("Check: nvcc --version  |  echo $CUDA_HOME  |  echo $TORCH_CUDA_ARCH_LIST")
            continue

        # Glob for standard-named output; exclude any pre-existing arch-encoded files
        found = [
            f for f in wheel_dir.glob(f"{name.replace('-', '_')}*.whl")
            if not any(f.name.endswith(f"-{s}.whl") for s in _KNOWN_ARCH_SUFFIXES)
        ]
        if not found:
            print(f"  [warn] No .whl found for {name} after build")
            continue

        newest = max(found, key=lambda p: p.stat().st_mtime)

        # Copy to arch-encoded name for Hub upload
        arch_whl = newest.parent / f"{newest.stem}-{suffix}{newest.suffix}"
        shutil.copy2(str(newest), str(arch_whl))
        print(f"  Built:      {newest.name}")
        print(f"  Hub target: {arch_whl.name}")
        built.append(arch_whl)

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
    print(f"# Wheels uploaded for {env.get('arch_suffix','')} "
          f"({env.get('gpu_name','')}) — Hub now accumulates per-arch wheels.")
    print(f"# CUDA {env.get('cuda_ver','')}  PyTorch {env.get('torch_short','')}  "
          f"Python {env.get('py_ver','')}  cxx11={env.get('cxx11','')}")
    for _, url in urls:
        print(f"  {url}")
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
                        help="Upload existing arch-encoded .whl files in --wheel_dir.")
    parser.add_argument("--verify_only", action="store_true",
                        help="Pure import check. No installs, no builds.")
    args = parser.parse_args()

    hf_token = (args.hf_token or os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_HUB_TOKEN"))

    env = detect_env()

    if args.verify_only:
        ok = verify_imports()
        sys.exit(0 if ok else 1)

    wheel_dir = Path(args.wheel_dir)
    arch_suffix = env.get("arch_suffix") or "smunknown"

    if not args.skip_build:
        built = build_wheels(wheel_dir,
                             arch_tag=env.get("arch_tag"),
                             arch_suffix=arch_suffix,
                             include_flash_attn=args.include_flash_attn)
        if not built:
            print("\n[FATAL] No wheels built. Check CUDA toolkit and logs above.")
            sys.exit(1)
        pkg_names = {w.name.split("-")[0] for w in built}
        missing = {"causal_conv1d", "mamba_ssm"} - pkg_names
        if missing:
            print(f"\n[WARN] These packages failed to build: {missing}")
    else:
        # Prefer arch-encoded wheels for this GPU; fall back to any .whl
        built = list(wheel_dir.glob(f"*-{arch_suffix}.whl"))
        if not built:
            built = list(wheel_dir.glob("*.whl"))
        print(f"Skipping build. Found {len(built)} wheel(s) in {wheel_dir}.")

    if not args.skip_upload:
        if not hf_token:
            print("[ERROR] No HF token. Set --hf_token or HF_TOKEN env var.")
            sys.exit(1)
        urls = upload_wheels(built, args.hf_repo_id, hf_token)
        print_install_snippet(urls, env)
    else:
        print("Skipping Hub upload (--skip_upload).")

    # Install standard-named originals into current env
    for whl in built:
        standard_name = whl.name.replace(f"-{arch_suffix}.whl", ".whl")
        standard_path = whl.parent / standard_name
        install_path = standard_path if standard_path.exists() else whl
        _run([sys.executable, "-m", "pip", "install", "--force-reinstall",
              "--no-deps", str(install_path)], check=False)

    verify_imports()


if __name__ == "__main__":
    main()
