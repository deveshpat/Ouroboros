"""
inference.py
============
Inference for Ouroboros: a single-prompt CLI that loads a real Ouroboros
(base Jamba + in-place LoRA + HaltGate) via Ouroboros.from_pretrained and
generates through the inherited GenerationMixin.

This is the rewrite of the old ouroboros/inference/generation.py. That module
was built around a PeftModel-wrapped base model and drove latent reasoning
through external helpers (prepare_latent_runtime / run_latent_passes /
decode_from_latent_context) that existed *only* because the wrapped model was
never an Ouroboros. Under model.Ouroboros, latent passes run unconditionally
inside forward() whenever <|lat|> tokens are present in input_ids, and
.generate() is inherited untouched — so the entire external latent-runtime
machinery is gone. What remains is: tokenize the prompt, append
config.lat_token_id stage_k times, generate, and read actual_n_latents back
from a forward probe.

Kept (reimplemented, not copy-pasted): the InferenceResult / PromptBudgetReport
dataclasses, the context-budgeting encoder (load-bearing against eval OOM on
long validation rows), device/dtype resolution, the env-driven CLI, and a
minimal true-baseline loader for the release-eval zero-shot-CoT comparison arm.

Stdlib-only at module top so `python inference.py --help` works without torch;
heavy imports (torch/transformers/peft) happen inside main() after bootstrap.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

# Stdlib-only at module top so `python inference.py --help` works without torch.

DEFAULT_ADAPTER_REPO = "WeirdRunner/Ouroboros"
DEFAULT_ADAPTER_SUBFOLDER = ""  # adapter repo root; pass a subfolder to load an experimental gate
DEFAULT_STAGE_K = 10
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_MAX_SEQ_LEN = 512


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PromptBudgetReport:
    """Outcome of context-budgeting an encoded prompt.

    Long validation rows can blow the seq budget and OOM hours into an eval.
    encode_prompt_ids_with_report keeps the TAIL of the formatted prompt
    (preserves the assistant generation prefix + most recent user content under
    chat templates) and reports whether truncation happened, so a release eval
    can tell a clean score from a truncated-input score.
    """

    context: str
    original_tokens: int
    budget_tokens: int
    max_seq_len: int
    reserve_tokens: int
    final_tokens: int
    truncated: bool
    dropped_tokens: int


@dataclass(frozen=True)
class InferenceResult:
    prompt: str
    text: str
    actual_latents: int  # <|lat|> positions actually filled by forward's latent passes
    stage_k: int
    used_halt_gate: bool
    prompt_budget: dict[str, Any]


# ── CLI (stdlib-only; torch-free for --help) ──────────────────────────────────

def _env(env: Mapping[str, str], name: str, default: str) -> str:
    return (env.get(name) or "").strip() or default


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = (env.get(name) or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def parse_args(
    argv: Optional[Iterable[str]] = None, *, env: Optional[Mapping[str, str]] = None
) -> argparse.Namespace:
    env = os.environ if env is None else env
    p = argparse.ArgumentParser(
        description="Run Ouroboros (Coconut latent-reasoning Jamba) inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Prompt source
    p.add_argument("--prompt", default=_env(env, "OUROBOROS_INFERENCE_PROMPT", ""))
    p.add_argument("--prompt_file", default=_env(env, "OUROBOROS_INFERENCE_PROMPT_FILE", ""))
    # Model / adapter
    p.add_argument("--base_model", default=_env(env, "OUROBOROS_INFERENCE_BASE_MODEL", "ai21labs/AI21-Jamba-Reasoning-3B"))
    p.add_argument("--adapter_repo", default=_env(env, "OUROBOROS_INFERENCE_ADAPTER_REPO", DEFAULT_ADAPTER_REPO))
    p.add_argument("--adapter_subfolder", default=_env(env, "OUROBOROS_INFERENCE_ADAPTER_SUBFOLDER", DEFAULT_ADAPTER_SUBFOLDER),
                   help="Subfolder of adapter_repo holding halt_gate.pt (e.g. 'diloco_state/anchor'). Empty = repo root.")
    p.add_argument("--hf_token", default=_env(env, "HF_TOKEN", ""))
    # Compute
    p.add_argument("--device", default=_env(env, "OUROBOROS_INFERENCE_DEVICE", "auto"))
    p.add_argument("--dtype", default=_env(env, "OUROBOROS_INFERENCE_DTYPE", "auto"))
    p.add_argument("--model_device_map", default=_env(env, "OUROBOROS_INFERENCE_MODEL_DEVICE_MAP", "single"),
                   choices=("single", "auto", "balanced", "balanced_low_0", "sequential"),
                   help="Inference-only model placement. balanced_low_0/auto shard large eval models across visible CUDA GPUs.")
    p.add_argument("--use_4bit", action="store_true", default=_env_bool(env, "OUROBOROS_INFERENCE_USE_4BIT", False))
    # Generation
    p.add_argument("--stage_k", type=int, default=int(_env(env, "OUROBOROS_INFERENCE_STAGE_K", str(DEFAULT_STAGE_K))),
                   help="Number of <|lat|> tokens appended to the prompt = requested latent depth.")
    p.add_argument("--max_new_tokens", type=int, default=int(_env(env, "OUROBOROS_INFERENCE_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS))))
    p.add_argument("--max_seq_len", type=int, default=int(_env(env, "OUROBOROS_INFERENCE_MAX_SEQ_LEN", str(DEFAULT_MAX_SEQ_LEN))))
    p.add_argument("--halt_threshold", type=float, default=float(_env(env, "OUROBOROS_INFERENCE_HALT_THRESHOLD", "0.9")))
    p.add_argument("--latent_cache", action="store_true", default=_env_bool(env, "OUROBOROS_INFERENCE_LATENT_CACHE", False),
                   help="P0: cache latent prefixes during the forward probe (inference-only; off by default).")
    # Flags
    p.add_argument("--use_chat_template", action="store_true", default=_env_bool(env, "OUROBOROS_INFERENCE_USE_CHAT_TEMPLATE", True))
    p.add_argument("--no_chat_template", dest="use_chat_template", action="store_false")
    p.add_argument("--use_halt_gate", action="store_true", default=_env_bool(env, "OUROBOROS_INFERENCE_USE_HALT_GATE", True))
    p.add_argument("--no_halt_gate", dest="use_halt_gate", action="store_false")
    p.add_argument("--baseline", action="store_true", default=_env_bool(env, "OUROBOROS_INFERENCE_BASELINE", False),
                   help="P2: load the true base model (no <|lat|>, no adapter, no gate) for the zero-shot-CoT comparison arm.")
    p.add_argument("--json", action="store_true", default=_env_bool(env, "OUROBOROS_INFERENCE_JSON", False))
    return p.parse_args(list(argv) if argv is not None else None)


def resolve_hf_token(cli_value: Optional[str]) -> Optional[str]:
    val = (cli_value or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    return val or None


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    prompt = (args.prompt or "").strip()
    if not prompt:
        raise SystemExit("Provide --prompt or --prompt_file for inference.")
    return prompt


# ── device / dtype resolution ─────────────────────────────────────────────────

def resolve_device(requested: str):
    """auto -> cuda:0 / mps / cpu; otherwise torch.device(requested)."""
    import torch
    requested = (requested or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def resolve_dtype(requested: str, device):
    """auto -> device-appropriate (BF16 on Ampere+, FP16 on T4/V100, FP32 off-GPU)."""
    import torch
    requested = (requested or "auto").strip().lower()
    if requested == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.get_device_capability(0) >= (8, 0) else torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    if requested not in mapping:
        raise ValueError(f"Unsupported dtype {requested!r}. Use auto, float32, float16, or bfloat16.")
    return mapping[requested]


def _resolve_device_map_arg(requested: str, device):
    """--model_device_map single (default) => None (let from_pretrained place on device);
    the named strategies pass straight through to device_map= for multi-GPU sharding."""
    requested = (requested or "single").strip().lower()
    if requested == "single":
        return None
    return requested


# ── prompt formatting / context budgeting ─────────────────────────────────────

def format_prompt(tokenizer, prompt: str, *, use_chat_template: bool) -> str:
    """Apply the tokenizer's chat template if requested and available; else plain text."""
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            print(f"[inference] chat template failed ({type(exc).__name__}); using plain prompt fallback.")
    return prompt


def encode_prompt_ids_with_report(
    tokenizer,
    prompt: str,
    *,
    max_seq_len: int,
    reserve_tokens: int = 0,
    context: str = "prompt",
) -> tuple[list[int], PromptBudgetReport]:
    """Encode a prompt and enforce the runtime context budget.

    Keep the TAIL (preserves the assistant generation prefix + most recent user
    content) when truncation is needed. reserve_tokens makes room for the
    appended <|lat|>*stage_k block so the latent tokens never push past
    max_seq_len.
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids:
        raise ValueError(f"{context} encoded to an empty token sequence.")

    max_seq_len = max(1, int(max_seq_len))
    reserve_tokens = max(0, int(reserve_tokens))
    budget = max(1, max_seq_len - reserve_tokens)
    original_tokens = len(input_ids)
    dropped = max(0, original_tokens - budget)
    if dropped:
        print(f"[inference] {context}: truncating {dropped} prompt tokens to fit max_seq_len={max_seq_len}.")
        input_ids = input_ids[-budget:]
    report = PromptBudgetReport(
        context=context,
        original_tokens=original_tokens,
        budget_tokens=budget,
        max_seq_len=max_seq_len,
        reserve_tokens=reserve_tokens,
        final_tokens=len(input_ids),
        truncated=bool(dropped),
        dropped_tokens=dropped,
    )
    return input_ids, report


# ── loading ───────────────────────────────────────────────────────────────────

def load_components(args: argparse.Namespace):
    """Load a real Ouroboros (base Jamba + in-place LoRA + HaltGate) + tokenizer.

    A single Ouroboros.from_pretrained call replaces the old PeftModel-wrap +
    external latent-runtime assembly: from_pretrained loads base weights, resizes
    to the <|lat|>-extended vocab, injects the LoRA adapter in place (so the
    object stays an Ouroboros and forward() runs the latent passes), and loads
    halt_gate.pt. Returns (model, tokenizer, device).
    """
    import torch
    from model import Ouroboros

    token = resolve_hf_token(args.hf_token)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    device_map = _resolve_device_map_arg(args.model_device_map, device)

    model = Ouroboros.from_pretrained(
        adapter_repo=args.adapter_repo,
        base_model_id=args.base_model,
        torch_dtype=dtype,
        device_map=device_map,
        load_in_4bit=bool(args.use_4bit),
        halt_threshold=float(args.halt_threshold),
        use_halt_gate=bool(args.use_halt_gate),
        halt_gate_subfolder=args.adapter_subfolder,
        token=token,
    )
    # P0: enable the inference-only latent cache only when explicitly requested.
    model.config.use_latent_cache = bool(args.latent_cache)

    # The tokenizer lives in the adapter repo (from_pretrained loaded it
    # transiently to size the vocab). Re-load it here for encoding — tokenization
    # is this CLI's concern, not the model's.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_repo, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for single-prompt generation

    return model, tokenizer, device


def load_baseline_components(args: argparse.Namespace):
    """P2: load the TRUE base model — no <|lat|>, no adapter, no HaltGate.

    The zero-shot-CoT comparison arm for release eval. Mirrors the minimal shape
    of the old load_base_model_and_tokenizer baseline seam without any of its
    PeftModel/latent scaffolding: plain AutoModelForCausalLM over base Jamba
    weights, plain tokenizer, inherited .generate() with a standard CoT prompt.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = resolve_hf_token(args.hf_token)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    device_map = _resolve_device_map_arg(args.model_device_map, device)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
        "token": token,
    }
    if bool(args.use_4bit):
        if device.type != "cuda":
            raise SystemExit("--use_4bit requires CUDA + bitsandbytes.")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = dtype
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    elif device.type == "cuda":
        load_kwargs["device_map"] = {"": 0}
    elif device.type != "cpu":
        # mps etc.
        pass

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    if device.type != "cuda" and not bool(args.use_4bit):
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


# ── generation ────────────────────────────────────────────────────────────────

def _probe_actual_latents(model, input_ids, attention_mask):
    """One forward pass with output_hidden_sequences to read how many <|lat|>
    positions forward's latent passes actually filled (<= requested stage_k when
    the HaltGate early-stopped, == stage_k with no gate). Returns an int for the
    single-prompt batch."""
    import torch
    with torch.inference_mode():
        out = model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_sequences=True,
        )
    n = getattr(out, "actual_n_latents", None)
    if n is None:
        return 0
    vals = n.detach().cpu().view(-1).tolist()
    return int(vals[0]) if vals else 0


def run_single_prompt(
    *,
    model,
    tokenizer,
    prompt: str,
    stage_k: int,
    device,
    args: argparse.Namespace,
    use_chat_template: bool = True,
) -> InferenceResult:
    """Tokenize prompt, append <|lat|>*stage_k, generate through inherited .generate().

    The latent passes happen inside forward() during the prefill step (the first
    generate() call) — they're data-driven by the <|lat|> positions in input_ids,
    so there's no separate latent-runtime step to call. A forward probe reads
    actual_n_latents for the report.
    """
    import torch

    lat_token_id = int(getattr(model.config, "lat_token_id"))
    prefix = format_prompt(tokenizer, prompt, use_chat_template=use_chat_template)
    q_ids, prompt_budget = encode_prompt_ids_with_report(
        tokenizer, prefix,
        max_seq_len=int(args.max_seq_len),
        reserve_tokens=max(0, int(stage_k)),
        context="candidate prompt",
    )

    input_ids = torch.tensor(q_ids + [lat_token_id] * int(stage_k), device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    used_halt_gate = getattr(model.config, "use_halt_gate", False) and getattr(model, "halt_gate", None) is not None
    actual_latents = _probe_actual_latents(model, input_ids, attention_mask)

    with torch.inference_mode():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Strip the prompt + latent tokens; keep only the newly generated tail.
    new_tokens = gen_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return InferenceResult(
        prompt=prompt,
        text=text,
        actual_latents=actual_latents,
        stage_k=int(stage_k),
        used_halt_gate=used_halt_gate,
        prompt_budget=asdict(prompt_budget),
    )


def run_baseline_prompt(
    *,
    model,
    tokenizer,
    prompt: str,
    device,
    args: argparse.Namespace,
    use_chat_template: bool = True,
) -> InferenceResult:
    """P2: generate from the true base model with no latent tokens.

    Standard CoT-style generation: the prompt carries its own chain-of-thought
    cue (the caller's responsibility), <|lat|> is absent, so forward() runs zero
    latent passes and behaves as plain Jamba. actual_latents is always 0.
    """
    import torch

    prefix = format_prompt(tokenizer, prompt, use_chat_template=use_chat_template)
    q_ids, prompt_budget = encode_prompt_ids_with_report(
        tokenizer, prefix,
        max_seq_len=int(args.max_seq_len),
        reserve_tokens=0,
        context="baseline prompt",
    )
    input_ids = torch.tensor(q_ids, device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = gen_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return InferenceResult(
        prompt=prompt, text=text, actual_latents=0, stage_k=0,
        used_halt_gate=False, prompt_budget=asdict(prompt_budget),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    # Defer bootstrap + heavy imports until after argparse so --help is torch-free.
    from bootstrap import OuroborosBootstrap
    OuroborosBootstrap().ensure_environment()

    prompt = resolve_prompt(args)
    if args.baseline:
        model, tokenizer, device = load_baseline_components(args)
        result = run_baseline_prompt(
            model=model, tokenizer=tokenizer, prompt=prompt,
            device=device, args=args, use_chat_template=bool(args.use_chat_template),
        )
    else:
        model, tokenizer, device = load_components(args)
        result = run_single_prompt(
            model=model, tokenizer=tokenizer, prompt=prompt,
            stage_k=int(args.stage_k), device=device, args=args,
            use_chat_template=bool(args.use_chat_template),
        )

    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(result.text)
        arm = "baseline" if args.baseline else "ouroboros"
        print(f"\n[inference:{arm}] actual_latents={result.actual_latents} "
              f"stage_k={result.stage_k} halt_gate={result.used_halt_gate}")


if __name__ == "__main__":  # pragma: no cover
    main()
