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
    format_prompt,
    load_components,
    resolve_device,
    run_single_prompt,
)
from ouroboros.models import load_base_model_and_tokenizer


@dataclass
class BaselineRuntime:
    model: Any
    tokenizer: Any
    device: torch.device


@dataclass
class CandidateRuntime:
    model: Any
    tokenizer: Any
    halt_gate: Any
    device: torch.device


def _common_device(args) -> torch.device:
    return resolve_device(str(getattr(args, "device", "auto")))


def load_baseline_runtime(args) -> BaselineRuntime:
    device = _common_device(args)
    baseline_args = SimpleNamespace(
        model_id=args.baseline_model_id,
        use_4bit=False,
        grad_checkpoint=False,
        disable_mamba_kernels=bool(getattr(args, "disable_mamba_kernels", False)),
    )
    model, tokenizer, _, lat_token_id = load_base_model_and_tokenizer(
        baseline_args,
        device,
        add_lat_token=False,
    )
    if lat_token_id is not None:
        raise RuntimeError("baseline loader unexpectedly added a latent token")
    model.eval()
    return BaselineRuntime(model=model, tokenizer=tokenizer, device=device)


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
        use_halt_gate=True,
        require_halt_gate=bool(getattr(args, "candidate_requires_halt_gate", False)),
        disable_mamba_kernels=bool(getattr(args, "disable_mamba_kernels", False)),
        json=False,
    )
    model, tokenizer, halt_gate, device = load_components(candidate_args)
    if bool(getattr(args, "candidate_requires_halt_gate", False)) and halt_gate is None:
        raise RuntimeError("candidate_requires_halt_gate was set, but no HaltGate was loaded")
    return CandidateRuntime(model=model, tokenizer=tokenizer, halt_gate=halt_gate, device=device)


@torch.no_grad()
def generate_baseline(runtime: BaselineRuntime, question: str, args) -> str:
    prompt = format_prompt(
        runtime.tokenizer,
        question,
        use_chat_template=bool(getattr(args, "use_chat_template", True)),
    )
    input_ids = runtime.tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids:
        raise ValueError("Question encoded to an empty token sequence.")
    input_tensor = torch.tensor(input_ids, device=runtime.device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_tensor, dtype=torch.long, device=runtime.device)
    output = runtime.model.generate(
        input_ids=input_tensor,
        attention_mask=attention_mask,
        max_new_tokens=int(args.gen_max_tokens),
        do_sample=False,
        pad_token_id=runtime.tokenizer.pad_token_id,
        eos_token_id=runtime.tokenizer.eos_token_id,
    )
    generated = output[0, input_tensor.size(1):].detach().cpu().tolist()
    return runtime.tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_candidate(runtime: CandidateRuntime, question: str, args):
    candidate_args = SimpleNamespace(
        gen_max_tokens=int(args.gen_max_tokens),
        max_new_tokens=int(args.gen_max_tokens),
        max_seq_len=int(getattr(args, "max_seq_len", DEFAULT_MAX_SEQ_LEN)),
        halt_threshold=float(getattr(args, "halt_threshold", DEFAULT_HALT_THRESHOLD)),
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
