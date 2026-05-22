"""Shared model runtime readiness seam.

This module owns the small, high-leverage contract every Jamba loading path must
honour: a model is not runtime-ready just because bootstrap installed wheels or
imports succeeded.  Callers only declare the Mamba fast path active after the
loaded model survives a tiny post-load forward probe.
"""

from __future__ import annotations

import contextlib
import importlib
from typing import Any, Dict

import torch

from ouroboros.bootstrap.runtime import _load_mamba_fast_path_symbols
from ouroboros.utils.runtime_env import is_main_process


def patch_transformers_jamba_fast_path_globals() -> bool:
    """Refresh Transformers Jamba globals from the verified mamba symbols."""
    try:
        importlib.invalidate_caches()
        import transformers.models.jamba.modeling_jamba as jamba_mod
        symbols = _load_mamba_fast_path_symbols()
        for name, value in symbols.items():
            if getattr(jamba_mod, name, None) is not value:
                setattr(jamba_mod, name, value)
        is_available = all(symbols.values())
        jamba_mod.is_fast_path_available = is_available
        return bool(is_available)
    except Exception:
        return False


def safe_from_pretrained(model_cls: Any, model_id: str, load_kwargs: Dict[str, Any]):
    """Load a CausalLM, retrying once when Transformers rejects optional kwargs."""
    try:
        return model_cls.from_pretrained(model_id, **load_kwargs)
    except Exception as exc:
        message = str(exc)
        retry_kwargs = dict(load_kwargs)
        changed = False
        for key in ["use_mamba_kernels", "attn_implementation"]:
            if key in retry_kwargs and key in message:
                retry_kwargs.pop(key, None)
                changed = True
                if is_main_process():
                    print(f"  [warn] model load rejected '{key}'; retrying without it.")
        if changed:
            return model_cls.from_pretrained(model_id, **retry_kwargs)
        raise


def unwrap_peft_model(model: Any) -> Any:
    try:
        base = model.get_base_model()
        if base is not None:
            return base
    except Exception:
        pass
    return model


def resolve_backbone(model: Any) -> Any:
    """Locate the backbone object used by Jamba forward probes."""
    base = unwrap_peft_model(model)
    candidates = [
        getattr(model, "model", None),
        getattr(base, "model", None),
        getattr(getattr(base, "model", None), "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "forward") and hasattr(candidate, "embed_tokens"):
            return candidate
    raise AttributeError(
        "Cannot locate backbone model. Inspect:\n"
        "  print(type(model))\n"
        "  print([n for n, _ in model.named_modules()][:40])"
    )


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device.type == "mps":
        try:
            is_available = torch.amp.autocast_mode.is_autocast_available("mps")
        except Exception:
            is_available = False
        if is_available:
            return torch.autocast(device_type="mps", dtype=dtype)
    return contextlib.nullcontext()


def extract_last_hidden_state(outputs: Any, context: str) -> torch.Tensor:
    last_hidden = getattr(outputs, "last_hidden_state", None)
    assert last_hidden is not None, (
        f"{context}: model backbone returned None for last_hidden_state. "
        "Pass output_hidden_states=True and use out.hidden_states[-1] instead."
    )
    return last_hidden


def _run_jamba_probe_once(model: Any, device: torch.device, amp_dtype: torch.dtype) -> None:
    backbone = resolve_backbone(model)
    probe_ids = torch.tensor([[1, 2]], dtype=torch.long, device=device)
    probe_mask = torch.ones_like(probe_ids, dtype=torch.bool, device=device)
    with torch.no_grad():
        with autocast_context(device, amp_dtype):
            outputs = backbone(input_ids=probe_ids, attention_mask=probe_mask, use_cache=False)
            _ = extract_last_hidden_state(outputs, "post-load Jamba runtime probe")
    torch.cuda.synchronize()


def ensure_jamba_runtime_ready(
    model: Any,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    fast_path_requested: bool,
    context: str,
) -> bool:
    """Return True only when the loaded Jamba runtime fast path has been probed.

    Non-CUDA or explicitly disabled Mamba paths return False because no fast path
    was requested.  CUDA fast-path callers patch Transformers before probing,
    retry once on the known stale-global failure, and fail before generation if
    the loaded model cannot actually run.
    """
    if device.type != "cuda" or not fast_path_requested:
        return False

    patch_transformers_jamba_fast_path_globals()
    try:
        _run_jamba_probe_once(model, device, amp_dtype)
    except ValueError as exc:
        if "Fast Mamba kernels are not available" not in str(exc):
            raise
        if is_main_process():
            print(
                f"  [warn] {context}: Jamba runtime probe hit stale fast-path globals; "
                "patching Transformers Jamba module and retrying once."
            )
        if not patch_transformers_jamba_fast_path_globals():
            raise SystemExit(
                f"{context}: Jamba runtime probe failed and Transformers Jamba globals could not be refreshed. "
                "This environment would fail during generation."
            ) from exc
        _run_jamba_probe_once(model, device, amp_dtype)

    if is_main_process():
        print("  mamba CUDA kernels: fast path ACTIVE (post-load runtime probe passed)")
    return True


__all__ = [
    "autocast_context",
    "ensure_jamba_runtime_ready",
    "extract_last_hidden_state",
    "patch_transformers_jamba_fast_path_globals",
    "resolve_backbone",
    "safe_from_pretrained",
    "unwrap_peft_model",
]
