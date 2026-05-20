"""Model loading, tokenizer helpers, and distributed process utilities."""

from __future__ import annotations

import argparse
import contextlib
import functools
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
except ImportError:
    AutoModelForCausalLM = AutoTokenizer = BitsAndBytesConfig = None  # type: ignore[assignment]
    LoraConfig = get_peft_model = None  # type: ignore[assignment]
    _TF_VERSION = (0, 0)

from ouroboros.bootstrap import _load_mamba_fast_path_symbols

MODEL_ID = "ai21labs/AI21-Jamba-Reasoning-3B"


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


_CHAT_TEMPLATE_WARNED = False


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", str(_rank())))


def _maybe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _should_auto_disable_gradient_checkpointing(args: argparse.Namespace, total_vram_gb: float) -> bool:
    """Disable GC only when the configured latent workload is small enough."""
    if total_vram_gb < 40.0:
        return False
    if _env_truthy("OUROBOROS_FORCE_GRAD_CHECKPOINT"):
        return False

    batch_size = _coerce_positive_int(getattr(args, "batch_size", None), 1)
    max_seq_len = _coerce_positive_int(getattr(args, "max_seq_len", None), 1024)
    max_stage = _coerce_positive_int(getattr(args, "max_stage", None), 1)
    latent_token_pressure = batch_size * max_seq_len * max(max_stage, 1)
    high_depth_dgac = bool(getattr(args, "use_halt_gate", False)) and max_stage >= 6
    if high_depth_dgac or latent_token_pressure >= 32768:
        return False
    return True


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
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _autocast_ctx(device: torch.device, dtype: torch.dtype):
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


def _wandb_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(vars(args))
    for key in ["hf_token", "_resolved_hf_token", "_resolved_github_token"]:
        if key in config and config[key] is not None:
            config[key] = "***"
    return config



def _arg_value(args: argparse.Namespace, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                return value
    return default


def load_base_model_and_tokenizer(
    args: argparse.Namespace,
    device: torch.device,
    *,
    add_lat_token: bool = False,
) -> Tuple[nn.Module, Any, int, Optional[int]]:
    """Load an unadapted base CausalLM and tokenizer.

    This is the release-eval baseline seam: with ``add_lat_token=False`` it does
    not add ``<|lat|>``, resize embeddings, attach LoRA adapters, or create a
    HaltGate. ``add_lat_token=True`` is reserved for adapter/candidate setup
    code that needs the tokenizer/model vocabulary prepared before loading a
    PEFT adapter.
    """
    is_main = _is_main_process()
    rank = _rank()
    amp_dtype = _amp_dtype(device)
    model_id = str(_arg_value(args, "model_id", "base_model", default=MODEL_ID))

    if bool(_arg_value(args, "use_4bit", default=False)) and device.type != "cuda":
        raise SystemExit("--use_4bit requires CUDA + bitsandbytes.")

    if is_main:
        mode = "with <|lat|>" if add_lat_token else "true base"
        print(f"Loading tokenizer: {model_id} ({mode})")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lat_token_id: Optional[int] = None
    if add_lat_token:
        lat_token = "<|lat|>"
        existing_id = tokenizer.convert_tokens_to_ids(lat_token)
        if existing_id is None or existing_id == tokenizer.unk_token_id:
            tokenizer.add_special_tokens({"additional_special_tokens": [lat_token]})
        lat_token_id = int(tokenizer.convert_tokens_to_ids(lat_token))
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

    disable_mamba = bool(_arg_value(args, "disable_mamba_kernels", default=False))
    mamba_fast_path = device.type == "cuda" and not disable_mamba
    if not mamba_fast_path:
        load_kwargs["use_mamba_kernels"] = False
    elif is_main:
        print("  mamba CUDA kernels: fast path ACTIVE (verified at bootstrap)")

    use_4bit = bool(_arg_value(args, "use_4bit", default=False))
    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=amp_dtype if amp_dtype != torch.float32 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = amp_dtype

    if is_main:
        print(f"Loading model: {model_id}")
        print(f"  device={device} amp_dtype={str(amp_dtype).replace('torch.', '')}")

    if device.type == "cuda" and mamba_fast_path:
        _patch_transformers_jamba_fast_path_globals()

    model = _safe_from_pretrained(model_id, load_kwargs)
    model.config.use_cache = False

    if add_lat_token:
        embed_module = _get_embed_tokens(model)
        embed_size = int(embed_module.num_embeddings) if hasattr(embed_module, "num_embeddings") else int(embed_module.weight.shape[0])
        if len(tokenizer) > embed_size:
            if is_main:
                print(f"  Resizing embed_tokens: {embed_size} -> {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))

    grad_checkpoint = bool(_arg_value(args, "grad_checkpoint", default=False))
    if grad_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    elif device.type != "cuda" and not use_4bit:
        model = model.to(device)

    model.eval()
    d_model = int(getattr(model.config, "hidden_size"))
    if is_main:
        num_layers = getattr(model.config, "num_hidden_layers", "?")
        print(f"  d_model={d_model}  layers={num_layers}")
    return model, tokenizer, d_model, lat_token_id

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
    if device.type == "mps":
        load_kwargs["use_mamba_kernels"] = False
    elif not _mamba_fast_path:
        load_kwargs["use_mamba_kernels"] = False

    if _mamba_fast_path and device.type == "cuda" and is_main:
        print("  mamba CUDA kernels: fast path ACTIVE (verified at bootstrap)")

    if args.use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=amp_dtype if amp_dtype != torch.float32 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = amp_dtype

    if is_main:
        print(f"Loading model: {args.model_id}")
        print(f"  device={device} amp_dtype={str(amp_dtype).replace('torch.', '')}")

    if device.type == "cuda" and _mamba_fast_path:
        _patch_transformers_jamba_fast_path_globals()

    model = _safe_from_pretrained(args.model_id, load_kwargs)
    model.config.use_cache = False

    if device.type == "cuda" and _mamba_fast_path:
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
        if _should_auto_disable_gradient_checkpointing(args, total_vram_gb):
            args.grad_checkpoint = False
            if is_main:
                print(
                    f"  [perf] {total_vram_gb:.0f}GB VRAM detected: "
                    "disabling gradient checkpointing (not needed, saves ~20-40%)."
                )
        elif is_main and total_vram_gb >= 40.0:
            print(
                f"  [perf] {total_vram_gb:.0f}GB VRAM detected, but keeping "
                "gradient checkpointing for this high-depth latent workload."
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


def get_trainable_parameters(
    model: nn.Module,
    halt_gate: Optional[nn.Module],
) -> List[nn.Parameter]:
    params = [p for p in model.parameters() if p.requires_grad]
    if halt_gate is not None:
        params.extend(p for p in halt_gate.parameters() if p.requires_grad)
    return params
