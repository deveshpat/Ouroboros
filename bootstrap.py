"""Bootstrap: dependency install, runtime patches, and Jamba/Mamba fast-path setup.

Imported by the CLI before any ML dependency exists. All stdlib imports are at
module scope. torch/transformers/mamba imports happen inside methods, after the
pip phase has completed.
"""

from __future__ import annotations

import importlib
import importlib.util as ilu
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import List, Optional


# =====================================================================
# 1. HARDWARE LAYER: CUDA Architecture Identification
# =====================================================================
class CUDAHardwareProfile:
    """GPU-specific metadata, capability checks, and platform attributes."""

    def __init__(self) -> None:
        import torch
        self.is_available = torch.cuda.is_available()
        self.capability = torch.cuda.get_device_capability() if self.is_available else (0, 0)
        self.arch_suffix = f"sm{self.capability[0]}{self.capability[1]}"
        self.torch_cuda_arch_list = f"{self.capability[0]}.{self.capability[1]}+PTX"
        self.device_name = torch.cuda.get_device_name() if self.is_available else "no-gpu"

    @property
    def flash_attention_supported(self) -> bool:
        """FlashAttention 2 requires Ampere+ (sm80+)."""
        return self.capability >= (8, 0)


# =====================================================================
# 2. RUNTIME PATCH LAYER: Shims, source fixes, and symbol promotion
# =====================================================================
class EnvironmentPatcher:
    """Runtime shims and symbol promotion for Mamba/Transformers compatibility."""

    MAMBA_TRITON_LOG1P_REPLACEMENTS = {
        "tl.math.log1p(tl.exp(dt))": "tl.log(1.0 + tl.exp(dt))",
    }

    GENERATION_COMPAT_ALIASES = {
        "GreedySearchDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
        "SampleDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
        "ContrastiveSearchDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
        "BeamSearchDecoderOnlyOutput": "GenerateBeamDecoderOnlyOutput",
        "BeamSampleDecoderOnlyOutput": "GenerateBeamDecoderOnlyOutput",
        "GreedySearchEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
        "SampleEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
        "ContrastiveSearchEncoderDecoderOutput": "GenerateEncoderDecoderOutput",
        "BeamSearchEncoderDecoderOutput": "GenerateBeamDecoderOnlyOutput",
        "BeamSampleEncoderDecoderOutput": "GenerateBeamDecoderOnlyOutput",
    }

    @staticmethod
    def patch_transformers_generation() -> List[str]:
        """Back-fill removed transformers.generation output class names."""
        try:
            import transformers.generation as tg_mod
            patched = []
            for old, new in EnvironmentPatcher.GENERATION_COMPAT_ALIASES.items():
                if getattr(tg_mod, old, None) is None:
                    replacement = getattr(tg_mod, new, None)
                    if replacement is not None:
                        setattr(tg_mod, old, replacement)
                        patched.append(old)
            return patched
        except ImportError:
            return []

    @staticmethod
    def patch_triton_math_log1p() -> List[str]:
        """Restore tl.math.log1p if absent (Triton 3.x removed it)."""
        try:
            import triton
            import triton.language as tl
        except Exception:
            return []

        math_attr = getattr(tl, "math", None)
        if math_attr is None or getattr(math_attr, "log1p", None) is not None:
            return []

        def _log1p(x):
            return tl.log(1.0 + x)

        jit_decorator = getattr(triton, "jit", None)
        if callable(jit_decorator):
            _log1p = jit_decorator(_log1p)

        setattr(math_attr, "log1p", _log1p)
        return ["triton.language.math.log1p"]

    @classmethod
    def patch_mamba_triton_source(cls) -> List[str]:
        """Replace tl.math.log1p call in mamba_ssm Triton source before JIT compilation."""
        try:
            spec = ilu.find_spec("mamba_ssm")
        except Exception:
            return []

        search_locations = list(getattr(spec, "submodule_search_locations", None) or [])
        if not search_locations:
            return []

        patched_paths: List[str] = []
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
                for old, new in cls.MAMBA_TRITON_LOG1P_REPLACEMENTS.items():
                    updated = updated.replace(old, new)
                if updated == source:
                    continue
                source_path.write_text(updated, encoding="utf-8")
                patched_paths.append(str(source_path))
        return patched_paths

    @classmethod
    def load_and_export_mamba_symbols(cls) -> List[str]:
        """Patch Mamba source, then load and promote fast-path symbols to top-level namespace."""
        cls.patch_mamba_triton_source()
        cls.patch_triton_math_log1p()
        importlib.invalidate_caches()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"mamba_ssm.*")
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
            from mamba_ssm.ops.triton.selective_state_update import selective_state_update
            from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

        mamba_ssm_mod = importlib.import_module("mamba_ssm")
        causal_conv1d_mod = importlib.import_module("causal_conv1d")

        exports = {
            mamba_ssm_mod: {
                "selective_scan_fn": selective_scan_fn,
                "selective_state_update": selective_state_update,
                "mamba_inner_fn": mamba_inner_fn,
            },
            causal_conv1d_mod: {
                "causal_conv1d_fn": causal_conv1d_fn,
                "causal_conv1d_update": causal_conv1d_update,
            },
        }

        patched_exports: List[str] = []
        for target_module, symbols in exports.items():
            for name, value in symbols.items():
                if getattr(target_module, name, None) is not value:
                    setattr(target_module, name, value)
                    patched_exports.append(f"{target_module.__name__}.{name}")
        return patched_exports


# =====================================================================
# 3. PIPELINE LAYER: Dependency Acquisition
# =====================================================================
class DependencyInstaller:
    """pip + Hub wheel installation for the Kaggle runtime."""

    PURE_PYTHON_DEPS = [
        "transformers>=4.54.0", "peft", "datasets", "tqdm", "wandb",
        "bitsandbytes>=0.46.1", "accelerate", "huggingface_hub", "ninja",
        "einops", "safetensors", "lm-eval>=0.4.5",
    ]

    MAMBA_WHEEL_BASES = [
        "causal_conv1d-1.6.1-cp312-cp312-linux_x86_64",
        "mamba_ssm-1.2.2-cp312-cp312-linux_x86_64",
    ]
    FLASH_ATTN_WHEEL_BASE = "flash_attn-2.8.3-cp312-cp312-linux_x86_64"

    def __init__(self, hardware: CUDAHardwareProfile) -> None:
        self.hardware = hardware
        self.wheel_dir = Path("/tmp/ouroboros_wheels")
        self.wheel_dir.mkdir(exist_ok=True)

    def install_pure_python_dependencies(self) -> None:
        print("[bootstrap] Phase 1: pure-Python deps...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q"] + self.PURE_PYTHON_DEPS,
            check=False,
        )
        if result.returncode != 0:
            print("[bootstrap] WARNING: Phase 1 pip returned non-zero — check output above.")

    def _resolve_target_wheels(self) -> List[str]:
        bases = list(self.MAMBA_WHEEL_BASES)
        if self.hardware.flash_attention_supported:
            bases.append(self.FLASH_ATTN_WHEEL_BASE)
        return bases

    def install_architecture_wheels(self) -> None:
        print("[bootstrap] Phase 2: arch-aware Hub wheel install...")
        importlib.invalidate_caches()
        from huggingface_hub import hf_hub_download

        print(f"[bootstrap]   GPU arch: {self.hardware.arch_suffix} (TORCH_CUDA_ARCH_LIST={self.hardware.torch_cuda_arch_list})")
        wheel_bases = self._resolve_target_wheels()
        if self.FLASH_ATTN_WHEEL_BASE not in wheel_bases:
            print(f"[bootstrap]   flash-attn skipped ({self.hardware.arch_suffix} < sm80).")

        for base in wheel_bases:
            hub_filename = f"wheels/{base}-{self.hardware.arch_suffix}.whl"
            local_path = self.wheel_dir / f"{base}.whl"
            try:
                downloaded = hf_hub_download(
                    repo_id="WeirdRunner/Ouroboros",
                    filename=hub_filename,
                    local_dir=str(self.wheel_dir),
                )
                shutil.copy2(downloaded, str(local_path))
                print(f"[bootstrap]   Downloaded {hub_filename} ✓")
            except Exception as err:
                print(f"[bootstrap]   {hub_filename} not on Hub ({type(err).__name__}).")
                sys.exit(1)

            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(local_path)],
                check=False,
            )
            if result.returncode != 0:
                print(f"[bootstrap] FATAL: pip install failed for {local_path.name}.")
                sys.exit(1)
            print(f"[bootstrap]   Installed {local_path.name} ✓")


# =====================================================================
# 4. ORCHESTRATION FACADE (Singleton)
# =====================================================================
class OuroborosBootstrap:
    """Coordinates bootstrap phases; runs exactly once per process."""

    _instance: Optional["OuroborosBootstrap"] = None
    _bootstrap_complete: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self.hardware = CUDAHardwareProfile()
            self.installer = DependencyInstaller(self.hardware)
            self._initialized = True

    def ensure_environment(self) -> None:
        """Run all bootstrap phases. Idempotent: no-ops if already complete."""
        if self.__class__._bootstrap_complete:
            return
        if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
            return

        self.installer.install_pure_python_dependencies()
        self.installer.install_architecture_wheels()

        import torch
        if torch.cuda.is_available():
            torch.cuda.init()
        importlib.invalidate_caches()

        patched_shims = EnvironmentPatcher.patch_transformers_generation()
        if patched_shims:
            print(f"Shim: patched {len(patched_shims)} removed transformers.generation names ✓")
        else:
            print("Shim: all generation names present (no patch needed)")

        mamba_patches = EnvironmentPatcher.load_and_export_mamba_symbols()
        if mamba_patches:
            print("Kernel export shim: " + ", ".join(mamba_patches) + " ✓")
        else:
            print("Kernel export shim: already aligned")

        print(
            f"  ABI fingerprint: GPU={self.hardware.device_name} {self.hardware.arch_suffix} | "
            f"CUDA={torch.version.cuda} | PyTorch={torch.__version__} | "
            f"Python=cp{sys.version_info.major}{sys.version_info.minor}"
        )

        self.__class__._bootstrap_complete = True
