"""Runtime bridge used by ``compare-coconut-val``.

This module is intentionally imported only by the heavy compare subcommand.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from ouroboros.inference.generation import (
    DEFAULT_ADAPTER_CACHE_DIR,
    DEFAULT_HALT_THRESHOLD,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_STAGE_K,
    encode_prompt_ids_with_report,
    format_prompt,
    load_components,
    resolve_device,
    run_single_prompt,
)
from ouroboros.models.loading import (
    _autocast_ctx,
    _get_embed_tokens,
    _requested_amp_dtype,
    model_device_map_summary,
    module_first_device,
)
from ouroboros.models import load_base_model_and_tokenizer


@dataclass
class BaselineRuntime:
    model: Any
    tokenizer: Any
    device: torch.device
    amp_dtype: torch.dtype
    device_map: dict[str, str] | None = None


@dataclass
class CandidateRuntime:
    model: Any
    tokenizer: Any
    halt_gate: Any
    device: torch.device
    device_map: dict[str, str] | None = None


@dataclass(frozen=True)
class BaselineGenerationResult:
    text: str
    prompt_budget: dict[str, Any]


def _common_device(args) -> torch.device:
    return resolve_device(str(getattr(args, "device", "auto")))


def load_baseline_runtime(args) -> BaselineRuntime:
    device = _common_device(args)
    baseline_args = SimpleNamespace(
        model_id=args.baseline_model_id,
        use_4bit=False,
        grad_checkpoint=False,
        dtype=getattr(args, "dtype", "auto"),
        disable_mamba_kernels=bool(getattr(args, "disable_mamba_kernels", False)),
        model_device_map=getattr(args, "model_device_map", "single"),
    )
    model, tokenizer, _, lat_token_id = load_base_model_and_tokenizer(
        baseline_args,
        device,
        add_lat_token=False,
    )
    if lat_token_id is not None:
        raise RuntimeError("baseline loader unexpectedly added a latent token")
    model.eval()
    input_device = module_first_device(_get_embed_tokens(model), device)
    return BaselineRuntime(
        model=model,
        tokenizer=tokenizer,
        device=input_device,
        amp_dtype=_requested_amp_dtype(baseline_args, input_device),
        device_map=model_device_map_summary(model),
    )


def load_candidate_runtime(args) -> CandidateRuntime:
    candidate_args = SimpleNamespace(
        prompt=None,
        prompt_file=None,
        base_model=args.baseline_model_id,
        adapter_repo=args.candidate_repo_id,
        adapter_subfolder=args.candidate_subdir,
        adapter_dir=args.candidate_adapter_dir,
        adapter_cache_dir=getattr(args, "adapter_cache_dir", DEFAULT_ADAPTER_CACHE_DIR),
        device=getattr(args, "device", "auto"),
        dtype=getattr(args, "dtype", "auto"),
        stage_k=int(getattr(args, "stage_k", DEFAULT_STAGE_K)),
        max_new_tokens=int(args.gen_max_tokens),
        max_seq_len=int(getattr(args, "max_seq_len", DEFAULT_MAX_SEQ_LEN)),
        halt_threshold=float(getattr(args, "halt_threshold", DEFAULT_HALT_THRESHOLD)),
        use_chat_template=bool(getattr(args, "use_chat_template", True)),
        use_halt_gate=not bool(getattr(args, "disable_candidate_halt_gate", False)),
        require_halt_gate=bool(getattr(args, "candidate_requires_halt_gate", False)),
        disable_mamba_kernels=bool(getattr(args, "disable_mamba_kernels", False)),
        model_device_map=getattr(args, "model_device_map", "single"),
        json=False,
    )
    model, tokenizer, halt_gate, device = load_components(candidate_args)
    requires_halt_gate = bool(getattr(args, "candidate_requires_halt_gate", False))
    fixed_depth_ablation = bool(getattr(args, "disable_candidate_halt_gate", False))
    if requires_halt_gate and not fixed_depth_ablation and halt_gate is None:
        raise RuntimeError("candidate_requires_halt_gate was set, but no HaltGate was loaded")
    input_device = module_first_device(_get_embed_tokens(model), device)
    return CandidateRuntime(
        model=model,
        tokenizer=tokenizer,
        halt_gate=halt_gate,
        device=input_device,
        device_map=model_device_map_summary(model),
    )


@torch.no_grad()
def generate_baseline_result(runtime: BaselineRuntime, question: str, args) -> BaselineGenerationResult:
    prompt = format_prompt(
        runtime.tokenizer,
        question,
        use_chat_template=bool(getattr(args, "use_chat_template", True)),
    )
    max_seq_len = max(1, int(getattr(args, "max_seq_len", DEFAULT_MAX_SEQ_LEN)))
    input_ids, prompt_budget = encode_prompt_ids_with_report(
        runtime.tokenizer,
        prompt,
        max_seq_len=max_seq_len,
        context="baseline prompt",
    )
    input_tensor = torch.tensor(input_ids, device=runtime.device, dtype=torch.long).unsqueeze(0)
    generated: list[int] = []
    eos_token_id = runtime.tokenizer.eos_token_id
    for _ in range(max(0, int(args.gen_max_tokens))):
        if input_tensor.size(1) > max_seq_len:
            input_tensor = input_tensor[:, -max_seq_len:]
        attention_mask = torch.ones_like(input_tensor, dtype=torch.long, device=runtime.device)
        with _autocast_ctx(runtime.device, runtime.amp_dtype):
            outputs = runtime.model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            next_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        if eos_token_id is not None and next_id == eos_token_id:
            break
        generated.append(next_id)
        next_token = torch.tensor([[next_id]], device=runtime.device, dtype=torch.long)
        input_tensor = torch.cat([input_tensor, next_token], dim=1)
    return BaselineGenerationResult(
        text=runtime.tokenizer.decode(generated, skip_special_tokens=True).strip(),
        prompt_budget=prompt_budget.__dict__,
    )


@torch.no_grad()
def generate_candidate(runtime: CandidateRuntime, question: str, args):
    candidate_args = SimpleNamespace(
        gen_max_tokens=int(args.gen_max_tokens),
        max_new_tokens=int(args.gen_max_tokens),
        max_seq_len=int(getattr(args, "max_seq_len", DEFAULT_MAX_SEQ_LEN)),
        halt_threshold=float(getattr(args, "halt_threshold", DEFAULT_HALT_THRESHOLD)),
        dtype=getattr(args, "dtype", "auto"),
        latent_cache=False,
        mac_mps_latent_cache=False,
    )
    return run_single_prompt(
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        halt_gate=runtime.halt_gate,
        prompt=question,
        stage_k=int(getattr(args, "stage_k", DEFAULT_STAGE_K)),
        device=runtime.device,
        args=candidate_args,
        use_chat_template=bool(getattr(args, "use_chat_template", True)),
    )
