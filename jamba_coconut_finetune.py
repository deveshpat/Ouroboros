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
  Self-contained. _bootstrap() runs on every startup:
    Phase 1: pure-Python deps (transformers, peft, bitsandbytes>=0.46.1, huggingface_hub, ...)
    Phase 2: download pre-built CUDA extension wheels from the private HF Hub repo
             (WeirdRunner/Ouroboros) and install with --force-reinstall --no-deps.
             No source compilation. ~seconds on every run.
    Phase 3: functional verification — imports all 5 mamba fast-path symbols AND runs a
             tiny causal_conv1d CUDA op on real tensors to catch silent ABI mismatches.
             On failure: prints full ABI fingerprint and sys.exit(1).
             There is NO fallback to slow path — a broken kernel wastes 500s/step silently.
  HF_TOKEN must be set (Kaggle Secret or env var) so the private Hub repo is accessible.

Run (smoke test, Colab/Kaggle T4):
  !python jamba_coconut_finetune.py \
    --data_dir data/coconut_v1 --use_4bit \
    --epochs_per_stage 1 --max_stage 2 --max_samples 200 \
    --max_seq_len 1024 --max_grad_norm 0.3 \
    --session_timeout_hours 1.5 --wandb_mode disabled --output_dir runs/smoke

Run (Phase 3.1 through 3.K, Kaggle Dual T4):
  !torchrun --standalone --nproc_per_node=2 jamba_coconut_finetune.py \
    --data_dir data/coconut_v1 --use_4bit \
    --epochs_per_stage 3 --max_stage 10 --batch_size 2 --grad_accum 8 \
    --session_timeout_hours 11.0 --graceful_exit_buffer_minutes 20 \
    --output_dir runs/stage3_curriculum

Run (Phase 3.4, DGAC gate, from Stage K best checkpoint):
  !python jamba_coconut_finetune.py \
    --data_dir data/coconut_v1 --use_4bit \
    --use_halt_gate --resume_from runs/stage3_curriculum/stage_10/best \
    --epochs_per_stage 3 --output_dir runs/stage3_dgac
"""

import argparse
import contextlib
import json
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



# ── Hub wheel filenames (update here if you upload new builds) ────────────────
# These are the pre-compiled wheels stored at WeirdRunner/Ouroboros on HF Hub.
# Build new ones with build_wheels_kaggle.py whenever CUDA/PyTorch/Python changes.
_HUB_WHEEL_FILES = [
    "causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl",
    "mamba_ssm-1.2.2-cp312-cp312-linux_x86_64.whl",  # 2.x removed selective_state_update from 1.x path
]
_HUB_REPO_ID = "WeirdRunner/Ouroboros"


def _bootstrap_resolve_token() -> Optional[str]:
    """Resolve HF token before huggingface_hub is imported as a library."""
    # Try Kaggle Secrets first (most common in headless sessions)
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            return tok
    except Exception:
        pass
    # Colab
    try:
        from google.colab import userdata as _cu
        tok = _cu.get("HF_TOKEN")
        if tok:
            return tok
    except Exception:
        pass
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _bootstrap_verify_fast_path() -> bool:
    """
    Hard verification of the mamba fast path:
      1. Import all 5 symbols transformers' Jamba checks.
      2. Assert none are None (catches mamba-ssm 2.x silent API mismatch).
      3. Run a tiny causal_conv1d CUDA op on real tensors to catch ABI mismatches
         that pass the import check but crash on first kernel invocation.

    Returns True only if all three checks pass.
    Does NOT catch exceptions — caller decides what to do on failure.
    """
    import importlib as _il
    import torch as _torch

    _il.invalidate_caches()

    # Step 1+2: symbol presence and non-None check
    from mamba_ssm.ops.selective_scan_interface import (  # type: ignore
        selective_scan_fn,
        selective_state_update,
        mamba_inner_fn,
    )
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update  # type: ignore

    _syms = {
        "selective_scan_fn":      selective_scan_fn,
        "selective_state_update": selective_state_update,
        "mamba_inner_fn":         mamba_inner_fn,
        "causal_conv1d_fn":       causal_conv1d_fn,
        "causal_conv1d_update":   causal_conv1d_update,
    }
    _none = [k for k, v in _syms.items() if v is None]
    if _none:
        raise ImportError(
            f"Symbols are None — API mismatch (mamba-ssm 2.x Triton-path issue?): {_none}"
        )

    # Step 3: real tensor op through causal_conv1d CUDA kernel
    # x: (batch=1, dim=4, seqlen=8)   weight: (dim=4, 1, width=4)
    # This hits the actual .so / nvcc-compiled kernel — catches silent ABI breaks.
    _x = _torch.randn(1, 4, 8, device="cuda", dtype=_torch.float32)
    _w = _torch.randn(4, 1, 4, device="cuda", dtype=_torch.float32)
    _out = causal_conv1d_fn(_x, _w)
    assert _out.shape == _x.shape, f"Unexpected output shape: {_out.shape}"
    _torch.cuda.synchronize()
    return True


def _bootstrap() -> None:
    """
    Install all dependencies before any third-party imports.

    Phase 1 — Pure-Python deps (fast, ~seconds):
      Installs transformers, peft, bitsandbytes>=0.46.1, huggingface_hub, etc.
      bitsandbytes floor is required — Kaggle containers ship an older version.
      huggingface_hub is installed here so Phase 2 can use hf_hub_download.

    Phase 2 — Hub wheel install (fast, ~seconds):
      Downloads pre-built causal-conv1d and mamba-ssm wheels from WeirdRunner/Ouroboros
      using the HF token from Kaggle Secrets / env var.
      Installs with --force-reinstall --no-deps so the exact Hub build is always used.
      No source compilation. No TORCH_CUDA_ARCH_LIST gymnastics.

    Phase 3 — Hard functional verification:
      Imports all 5 symbols + runs a tiny causal_conv1d CUDA op on real tensors.
      If ANY check fails: prints full ABI fingerprint and calls sys.exit(1).
      There is intentionally NO fallback to the slow PyTorch path — a broken kernel
      wastes 500s/step silently and burns the entire session budget undetected.
      A hard exit here lets Kaggle allocate a fresh container with a compatible GPU.

    To update wheels: run build_wheels_kaggle.py, upload to Hub, update _HUB_WHEEL_FILES.
    """
    import importlib as _il
    import torch as _torch

    # ── Phase 1: pure-Python deps ─────────────────────────────────────────────
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
         "huggingface_hub"],
        check=False,
    )
    if _r1.returncode != 0:
        print("[bootstrap] WARNING: Phase 1 pip returned non-zero — check output above.")

    # ── Phase 2: Hub wheel install ────────────────────────────────────────────
    print("[bootstrap] Phase 2: installing Hub wheels...")
    _hf_token = _bootstrap_resolve_token()
    if not _hf_token:
        print("[bootstrap] FATAL: No HF_TOKEN found.")
        print("            Set HF_TOKEN as a Kaggle Secret or environment variable.")
        sys.exit(1)

    _il.invalidate_caches()
    from huggingface_hub import hf_hub_download  # now installed from Phase 1

    _wheel_dir = Path("/tmp/ouroboros_wheels")
    _wheel_dir.mkdir(exist_ok=True)

    for _whl_name in _HUB_WHEEL_FILES:
        print(f"[bootstrap]   Downloading {_whl_name} ...")
        try:
            _local = hf_hub_download(
                repo_id=_HUB_REPO_ID,
                filename=_whl_name,
                token=_hf_token,
                local_dir=str(_wheel_dir),
            )
        except Exception as _e:
            print(f"[bootstrap] FATAL: Could not download {_whl_name}: {_e}")
            print( "[bootstrap]        Upload a compatible wheel with build_wheels_kaggle.py.")
            sys.exit(1)

        _r = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--force-reinstall", "--no-deps", _local],
            check=False,
        )
        if _r.returncode != 0:
            print(f"[bootstrap] FATAL: pip install failed for {_whl_name}.")
            sys.exit(1)
        print(f"[bootstrap]   Installed {_whl_name} ✓")

    # ── Phase 3: hard functional verification ────────────────────────────────
    print("[bootstrap] Phase 3: verifying mamba fast path (symbol + CUDA op)...")

    # Print ABI fingerprint unconditionally — visible in logs for future debugging
    try:
        _gpu_name = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "no-gpu"
        _cc = _torch.cuda.get_device_capability(0) if _torch.cuda.is_available() else (0, 0)
        print(f"[bootstrap]   ABI fingerprint: "
              f"GPU={_gpu_name} sm_{_cc[0]}{_cc[1]} | "
              f"CUDA={_torch.version.cuda} | "
              f"PyTorch={_torch.__version__} | "
              f"Python=cp{sys.version_info.major}{sys.version_info.minor}")
        print(f"[bootstrap]   Wheels: {_HUB_WHEEL_FILES}")
    except Exception:
        pass

    try:
        _bootstrap_verify_fast_path()
        print("[bootstrap] Mamba fast path: ACTIVE ✓ — ~5s/step expected.")
    except Exception as _ve:
        print(f"\n[bootstrap] FATAL: Mamba fast path verification FAILED: {_ve}")
        print( "[bootstrap]        Root cause is almost certainly a Python API mismatch:")
        print( "[bootstrap]        mamba_ssm 2.x removed selective_state_update from the 1.x")
        print( "[bootstrap]        import path that transformers Jamba checks.")
        print( "[bootstrap]        Required: mamba_ssm-1.2.2 on Hub. Currently installed version")
        print( "[bootstrap]        may be 2.x. Run build_wheels_kaggle.py on T4 to rebuild.")
        print( "[bootstrap]        Exiting now (no slow-path fallback — 500s/step is unusable).")
        sys.exit(1)


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


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
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


def _get_backbone(model):
    """
    Robustly retrieve the backbone module that accepts input_ids / inputs_embeds and
    returns last_hidden_state.
    """
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


def _get_embed_tokens(model):
    """
    Robustly retrieve embed_tokens regardless of PEFT wrapping depth.
    Jamba Reasoning 3B: model.model.embed_tokens
    After PEFT wrap:    base_model.model.embed_tokens or base_model.model.model.embed_tokens
    """
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


def _get_lm_head(model):
    base = _unwrap_peft_model(model)
    for obj in [model, base, getattr(base, "model", None)]:
        if obj is None:
            continue
        head = getattr(obj, "lm_head", None)
        if head is not None:
            return head
    raise AttributeError("Cannot locate lm_head. Inspect model.named_modules().")


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


def barrier() -> None:
    if _distributed_is_initialized() and _world_size() > 1:
        torch.distributed.barrier()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    _add_bool_arg(parser, "--grad_checkpoint", True, "Enable gradient checkpointing.")
    parser.add_argument("--seed", type=int, default=42)

    # Session timeout (MANDATORY for Kaggle)
    parser.add_argument("--session_timeout_hours", type=float, default=11.0)
    parser.add_argument("--graceful_exit_buffer_minutes", type=float, default=20.0)

    # Hub checkpoint sync
    parser.add_argument("--push_to_hub", action="store_true",
        help="Push checkpoints to HF Hub after each epoch save.")
    parser.add_argument("--hf_token", default=None,
        help="HF write token. Falls back to HF_TOKEN env var.")
    parser.add_argument("--hf_repo_id", default="WeirdRunner/Ouroboros",
        help="HF model repo to sync checkpoints to.")
    parser.add_argument("--hf_stage_subdir", default="runs/stage3",
        help="Remote subdirectory inside the HF repo for Stage 3 checkpoints.")

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
    """
    Load Jamba Reasoning 3B with QLoRA (--use_4bit) or standard LoRA.

    Returns (model, tokenizer, d_model, lat_token_id).

    Key implementation notes:
    - attn_implementation: tries flash_attention_2, falls back to 'eager'
    - use_mamba_kernels: probed at runtime; False only set when probe fails
    - <|lat|> token: added if absent; embed_tokens resized accordingly
    - device_map: pinned to the local process GPU when using torchrun
    - amp dtype: bfloat16 when supported, else float16 on CUDA (T4-safe)
    """
    is_main = _is_main_process()
    rank = _rank()
    amp_dtype = _amp_dtype(device)

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

    # Determine attn_implementation with fallback
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

    # Probe for mamba CUDA kernels before model load.
    # Re-uses the same hard verification from _bootstrap() so the check is identical.
    # If bootstrap passed we expect this to pass too; but it's cheap and catches
    # edge cases where the session state changed (e.g. import cache was cleared).
    _mamba_fast_path = False
    if device.type == "cuda":
        try:
            _bootstrap_verify_fast_path()
            _mamba_fast_path = True
            if is_main:
                print("  mamba CUDA kernels: OK — fast path ACTIVE (~5s/step expected)")
        except Exception as _kern_exc:
            load_kwargs["use_mamba_kernels"] = False
            if is_main:
                print(f"  [WARN] mamba CUDA kernels unavailable: {_kern_exc}")
                print("         This should not happen if _bootstrap() passed.")
                print("         Slow PyTorch path forced (~500s/step).")
    else:
        # Non-CUDA devices: always slow path
        load_kwargs["use_mamba_kernels"] = False

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
    model = _safe_from_pretrained(args.model_id, load_kwargs)
    model.config.use_cache = False

    embed_module = _get_embed_tokens(model)
    if hasattr(embed_module, "num_embeddings"):
        embed_size = int(embed_module.num_embeddings)
    else:
        embed_size = int(embed_module.weight.shape[0])
    if len(tokenizer) > embed_size:
        if is_main:
            print(f"  Resizing embed_tokens: {embed_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

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

    # Verify critical paths after PEFT wrap
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
    """
    Download coconut-v1 from HF Hub and write local train.jsonl / val.jsonl / stats.json.
    Called automatically when local files are missing.
    """
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
    """
    Load train.jsonl, val.jsonl, and stats.json.
    If local files are missing, auto-downloads from HF Hub (WeirdRunner/Ouroboros, coconut-v1).
    """
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
    """
    Build a tokenized Coconut sample for curriculum stage k.

    Sequence layout (stage k > 0):
        [Q_ids] [lat_id * min(k, n_steps)] [S_{k+1}_ids ... S_n_ids] [answer_ids + eos]

    Labels:
        -100  for Q positions and latent positions
        token ids  for remaining steps and answer (supervised)

    Stage 0 (k=0): no latent slots; labels on ALL steps + answer.
    Truncation: supervised tail is truncated if total > max_seq_len.
    Returns None if < 4 supervised tokens remain after truncation.
    """
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
    """Pad a micro-batch to the longest example."""
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


def _forward_single_sample(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    q_len: int,
    n_latent: int,
    device: torch.device,
    halt_gate: Optional[HaltGate],
    args: argparse.Namespace,
    step_in_phase: int,
    amp_dtype: torch.dtype,
) -> Dict[str, Any]:
    backbone = _get_backbone(model)
    embed_fn = _get_embed_tokens(model)
    lm_head_fn = _get_lm_head(model)

    if n_latent == 0:
        with _autocast_ctx(device, amp_dtype):
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            hidden = _extract_last_hidden_state(outputs, "stage 0 forward")
            logits = lm_head_fn(hidden).float()
        ce_sum, n_valid = _compute_ce_sum_and_count(logits, labels)
        ce_value = float(ce_sum.item() / max(n_valid, 1))
        return {
            "ce_sum": ce_sum,
            "n_valid": n_valid,
            "ce": ce_value,
            "ponder": None,
            "diversity": None,
            "halt_step_mean": None,
            "lambda1": 0.0,
        }

    with _autocast_ctx(device, amp_dtype):
        all_embeds = embed_fn(input_ids)
    patched = all_embeds.clone()
    hidden_at_q_end: List[torch.Tensor] = []

    for j in range(n_latent):
        prefix_len = q_len + j
        prefix_embeds = patched[:, :prefix_len, :]
        prefix_mask = attention_mask[:, :prefix_len]
        with _autocast_ctx(device, amp_dtype):
            outputs = backbone(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_mask,
                use_cache=False,
            )
        hidden = _extract_last_hidden_state(outputs, "coconut prefix pass")
        h_j = hidden[:, -1:, :]
        if halt_gate is not None:
            hidden_at_q_end.append(h_j.squeeze(1))
        inject_pos = q_len + j
        if inject_pos >= patched.size(1):
            break
        patched = torch.cat(
            [patched[:, :inject_pos, :], h_j, patched[:, inject_pos + 1 :, :]],
            dim=1,
        )

    with _autocast_ctx(device, amp_dtype):
        outputs = backbone(inputs_embeds=patched, attention_mask=attention_mask, use_cache=False)
        hidden = _extract_last_hidden_state(outputs, "coconut full forward")
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

    if halt_gate is None or len(hidden_at_q_end) < 2:
        return result

    lam1 = compute_dgac_lambda1(
        step_in_phase,
        args.dgac_warmup_steps,
        args.dgac_ramp_steps,
        args.dgac_lambda_ponder_max,
    )
    one = torch.ones(1, device=device, dtype=torch.float32)
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
        torch.full_like(halt_steps, float(n_latent)),
        halt_steps,
    )
    result.update(
        {
            "ponder": ponder.mean(),
            "diversity": div_loss.mean(),
            "halt_step_mean": halt_steps.mean(),
            "lambda1": lam1,
        }
    )
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
    """
    Coconut forward pass for curriculum stage k.

    Stage 0 (k=0): standard forward pass. No prefix passes, no latent injection.

    Stage k > 0:
      Phase A: embed all tokens.
      Phase B: per-sample sequential prefix passes for each latent slot.
               Pass j: prefix = Q + latents[0..j-1].
               Extract h_j = last hidden state at prefix end.
               CRITICAL: assert last_hidden_state is not None.
               Patch embedding at latent slot j with h_j.
      Phase C: full forward over patched sequence. CE on supervised positions.
      Phase D: DGAC gate regularization if --use_halt_gate.

    All passes use use_cache=False. Caching causes shape mismatches with patched embeddings.
    """
    del stage_k  # stage is encoded via n_latents per sample
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    q_lens = batch["q_lens"].to(device)
    n_latents = batch["n_latents"].to(device)
    batch_size = int(input_ids.size(0))
    amp_dtype = _amp_dtype(device)

    ce_sum_total = torch.zeros((), device=device, dtype=torch.float32)
    n_valid_total = 0
    ponder_terms: List[torch.Tensor] = []
    diversity_terms: List[torch.Tensor] = []
    halt_terms: List[torch.Tensor] = []

    for row in range(batch_size):
        seq_len = int(attention_mask[row].sum().item())
        if seq_len < 2:
            continue
        result = _forward_single_sample(
            model=model,
            input_ids=input_ids[row : row + 1, :seq_len],
            attention_mask=attention_mask[row : row + 1, :seq_len],
            labels=labels[row : row + 1, :seq_len],
            q_len=int(q_lens[row].item()),
            n_latent=int(n_latents[row].item()),
            device=device,
            halt_gate=halt_gate,
            args=args,
            step_in_phase=step_in_phase,
            amp_dtype=amp_dtype,
        )
        if result["n_valid"] == 0:
            continue
        ce_sum_total = ce_sum_total + result["ce_sum"]
        n_valid_total += int(result["n_valid"])
        if result["ponder"] is not None:
            ponder_terms.append(result["ponder"])
        if result["diversity"] is not None:
            diversity_terms.append(result["diversity"])
        if result["halt_step_mean"] is not None:
            halt_terms.append(result["halt_step_mean"])

    if n_valid_total == 0:
        zero = torch.zeros((), device=device, requires_grad=True)
        return zero, {"ce": 0.0}

    ce = ce_sum_total / n_valid_total
    total_loss = ce
    metrics: Dict[str, float] = {"ce": float(ce.item())}

    if halt_gate is not None and diversity_terms:
        lam1 = compute_dgac_lambda1(
            step_in_phase,
            args.dgac_warmup_steps,
            args.dgac_ramp_steps,
            args.dgac_lambda_ponder_max,
        )
        ponder_mean = torch.stack(ponder_terms).mean() if ponder_terms else torch.zeros((), device=device)
        diversity_mean = torch.stack(diversity_terms).mean()
        halt_mean = torch.stack(halt_terms).mean() if halt_terms else torch.zeros((), device=device)
        total_loss = total_loss + lam1 * ponder_mean + args.dgac_lambda_diversity * diversity_mean
        metrics.update(
            {
                "ponder": float(ponder_mean.item()),
                "diversity": float(diversity_mean.item()),
                "halt_step_mean": float(halt_mean.item()),
                "lambda1": float(lam1),
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
    Compute val CE and exact-match accuracy at stage k using live weights.
    torch.cuda.empty_cache() called before to avoid OOM.
    Returns (val_ce, val_acc). Accuracy is primary for best-ckpt selection.
    """
    _maybe_empty_cuda_cache()
    model.eval()
    if halt_gate is not None:
        halt_gate.eval()
    pad_id = tokenizer.pad_token_id or 0
    ce_numer = 0.0
    ce_denom = 0
    n_correct = 0
    n_total = 0
    embed_fn = _get_embed_tokens(model)
    lm_head_fn = _get_lm_head(model)
    backbone = _get_backbone(model)
    amp_dtype = _amp_dtype(device)
    batch_size = max(int(args.val_batch_size), 1)

    # CE pass
    for start in range(0, len(val_samples), batch_size):
        batch_raw = val_samples[start : start + batch_size]
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

    # Accuracy pass (cap at 200 samples for speed)
    for sample in val_samples[:200]:
        built = build_sample_at_stage(tokenizer, sample, stage_k, lat_token_id, args.max_seq_len)
        if built is None:
            continue
        q_len = int(built["q_len"])
        n_latent = int(built["n_latent"])
        q_ids = built["full_ids"][:q_len]
        q_tensor = q_ids.unsqueeze(0).to(device)
        ctx = embed_fn(q_tensor)
        ctx_mask = torch.ones((1, ctx.size(1)), dtype=torch.bool, device=device)
        h_prev = None

        for _ in range(n_latent):
            with _autocast_ctx(device, amp_dtype):
                outputs = backbone(inputs_embeds=ctx, attention_mask=ctx_mask, use_cache=False)
                hidden = _extract_last_hidden_state(outputs, "eval latent pass")
                h_curr = hidden[:, -1:, :]
            if halt_gate is not None and h_prev is not None:
                hp = float(
                    halt_gate(
                        h_curr.squeeze(1).to(dtype=torch.float32),
                        h_prev.squeeze(1).to(dtype=torch.float32),
                    ).item()
                )
                if hp > args.halt_threshold:
                    break
            ctx = torch.cat([ctx, h_curr], dim=1)
            ctx_mask = torch.cat(
                [ctx_mask, torch.ones((1, 1), dtype=torch.bool, device=device)],
                dim=1,
            )
            h_prev = h_curr

        generated: List[int] = []
        eos_id = tokenizer.eos_token_id
        for _ in range(args.gen_max_tokens):
            if ctx.size(1) > args.max_seq_len:
                ctx = ctx[:, -args.max_seq_len :, :]
                ctx_mask = ctx_mask[:, -args.max_seq_len :]
            with _autocast_ctx(device, amp_dtype):
                outputs = backbone(inputs_embeds=ctx, attention_mask=ctx_mask, use_cache=False)
                hidden = _extract_last_hidden_state(outputs, "eval decode")
                logit = lm_head_fn(hidden)
            next_id = int(logit[:, -1, :].argmax(-1).item())
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

    model.train()
    if halt_gate is not None:
        halt_gate.train()
    return ce_numer / max(ce_denom, 1), n_correct / max(n_total, 1)


def _resolve_hf_token(cli_value: Optional[str]) -> Optional[str]:
    """CLI > Kaggle Secrets > Colab Secrets > env var."""
    if cli_value:
        return cli_value
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            return tok
    except Exception:
        pass
    try:
        from google.colab import userdata as _cu
        tok = _cu.get("HF_TOKEN")
        if tok:
            return tok
    except Exception:
        pass
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _hub_upload_checkpoint(
    ckpt_dir: Path,
    hf_repo_id: str,
    hf_token: str,
    remote_prefix: str = "runs/stage3",
    timeout_s: float = 300.0,
) -> bool:
    """Upload one checkpoint directory to HF Hub. Fire-and-forget — never raises."""
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
    """Download a single named checkpoint folder from HF Hub into local_dir."""
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
    """Return [(stage_k, step, ckpt_name)] sorted newest first."""
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
    """Save to output_dir/stage_{k}/{tag or checkpoint-{step}}/."""
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
        _hub_upload_checkpoint(ckpt, repo_id, hf_token, remote_prefix=subdir)

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
    """Load checkpoint weights and optional optimizer/scheduler state."""
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


def make_timeout_checker(args: argparse.Namespace, rank: int):
    """Returns callable → True when session is within graceful exit buffer."""
    session_start = time.perf_counter()
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
    """
    Find the latest in-progress checkpoint.
    Search order: local output_dir first, then HF Hub fallback.
    'best' checkpoints skipped so reruns prefer the freshest resumable state.
    """
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
        h_prev = None
        actual_k = 0

        for _ in range(stage_k):
            with _autocast_ctx(device, amp_dtype):
                outputs = backbone(inputs_embeds=ctx, attention_mask=ctx_mask, use_cache=False)
                hidden = _extract_last_hidden_state(outputs, "generation latent pass")
                h_curr = hidden[:, -1:, :]
            if halt_gate is not None and h_prev is not None:
                hp = float(
                    halt_gate(
                        h_curr.squeeze(1).to(dtype=torch.float32),
                        h_prev.squeeze(1).to(dtype=torch.float32),
                    ).item()
                )
                if hp > args.halt_threshold:
                    break
            ctx = torch.cat([ctx, h_curr], dim=1)
            ctx_mask = torch.cat(
                [ctx_mask, torch.ones((1, 1), dtype=torch.bool, device=device)],
                dim=1,
            )
            h_prev = h_curr
            actual_k += 1

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
        print(f"  A: {display}  [k_actual={actual_k} uwr={uwr:.3f}]")

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



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    rank       = _rank()
    world_size = _world_size()
    local_rank = _local_rank()
    distributed = world_size > 1
    is_main    = rank == 0

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=60),
        )

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if distributed and args.batch_size % world_size != 0:
        raise ValueError(
            f"--batch_size ({args.batch_size}) must be divisible by WORLD_SIZE ({world_size})"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = _resolve_hf_token(getattr(args, "hf_token", None))
    args._resolved_hf_token = hf_token

    if getattr(args, "push_to_hub", False) and not hf_token:
        if _is_main_process():
            print("[warn] --push_to_hub set but no HF token found; Hub sync disabled.")
        args.push_to_hub = False

    wandb_run = None
    if is_main and args.wandb_mode != "disabled":
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config=vars(args),
            )
        except ImportError:
            print("[warn] wandb not installed")

    train_samples, val_samples, stats = load_canonical_dataset(Path(args.data_dir), args.max_samples)
    if not train_samples:
        raise ValueError("No training samples were loaded. Check --data_dir / --max_samples.")
    curriculum_max_stage = get_max_stage(args, stats)

    resume_path: Optional[Path] = Path(args.resume_from) if args.resume_from else None
    hub_resume_dir = output_dir / ".hub_resume"

    if resume_path is None:
        resume_path = find_latest_resume_checkpoint(
            output_dir,
            hf_token=hf_token,
            hf_repo_id=getattr(args, "hf_repo_id", "WeirdRunner/Ouroboros"),
            hf_stage_subdir=getattr(args, "hf_stage_subdir", "runs/stage3"),
        )

        if resume_path is not None and is_main:
            print(f"  [resume] discovered latest checkpoint: {resume_path}")

    if resume_path is not None and not (resume_path / "training_state.pt").exists():
        if is_main:
            print(f"  [warn] resume checkpoint not found: {resume_path}")
        resume_path = None

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

        if hub_resume_dir.exists():
            shutil.rmtree(hub_resume_dir, ignore_errors=True)

        resume_stage = int(resume_state.get("stage_k", 0))
        global_step  = int(resume_state.get("step", 0))
        if args.use_halt_gate:
            resume_same_stage = bool(resume_state.get("use_halt_gate", False) and resume_path.name != "best")
            if resume_same_stage:
                resume_epoch          = int(resume_state.get("epoch", 0))
                resume_step_in_epoch  = int(resume_state.get("step_in_epoch", -1))
                step_in_phase         = int(resume_state.get("step_in_phase", 0))
        else:
            resume_same_stage = resume_path.name != "best"
            if resume_same_stage:
                resume_epoch         = int(resume_state.get("epoch", 0))
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
        if distributed:
            torch.distributed.destroy_process_group()
        if wandb_run is not None:
            import wandb
            wandb.finish()
        return

    local_bs = args.batch_size // world_size if distributed else args.batch_size
    check_timeout = make_timeout_checker(args, rank)
    timeout_triggered = False

    try:
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

            stage_start_epoch          = 0
            stage_start_step_in_epoch  = -1
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
                global_step               = int(state.get("step", global_step))
                stage_start_epoch         = int(state.get("epoch", resume_epoch))
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
            for epoch in range(stage_start_epoch, n_epochs):
                rng = random.Random(args.seed + stage_k * 100_003 + epoch)
                perm = list(range(len(train_samples)))
                rng.shuffle(perm)

                model.train()
                if halt_gate is not None:
                    halt_gate.train()
                optimizer.zero_grad(set_to_none=True)

                start_step      = stage_start_step_in_epoch + 1 if epoch == stage_start_epoch else 0
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
                    micro_count  = 0
                    did_backward = False

                    for micro in range(args.grad_accum):
                        global_micro_base = (step_idx * args.grad_accum + micro) * args.batch_size
                        rank_base   = global_micro_base + rank * local_bs
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
                        did_backward  = True
                        micro_count  += 1
                        for k, v in metrics.items():
                            step_metrics_accum[k] += float(v)

                    if not did_backward:
                        optimizer.zero_grad(set_to_none=True)
                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix(skip="1")
                        continue

                    if distributed:
                        all_reduce_gradients(trainable_params, world_size)

                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if args.use_halt_gate:
                        step_in_phase += 1

                    mean_metrics = {
                        k: v / max(micro_count, 1) for k, v in step_metrics_accum.items()
                    }
                    mean_ce = mean_metrics.get("ce", 0.0)

                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(ce=f"{mean_ce:.3f}", gn=f"{grad_norm:.3f}")

                    if is_main and global_step % args.log_every == 0:
                        log_payload = {
                            "train/ce":        mean_ce,
                            "train/grad_norm": grad_norm,
                            "train/lr":        scheduler.get_last_lr()[0],
                            "train/stage":     stage_k,
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

                if is_main:
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
                        best_val_ce  = val_ce
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

            if timeout_triggered:
                break

            if is_main and args.gen_every_stage:
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

            barrier()

            if best_ckpt is not None and not args.use_halt_gate:
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

        if is_main:
            print("\n" + "=" * 64)
            if timeout_triggered:
                print("  [timeout] Session budget exhausted - checkpoint saved.")
                print("  Re-run the same command with the same --output_dir to auto-resume.")
            else:
                print(f"  Curriculum complete. Stages: {stages}  Global steps: {global_step}")
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
