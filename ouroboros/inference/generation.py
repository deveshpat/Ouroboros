"""Ouroboros adapter/DGAC inference entrypoint.

This script is intentionally separate from lm-evaluation-harness. It loads the
actual Ouroboros runtime shape: base Jamba tokenizer + added ``<|lat|>`` token,
base model resized to that tokenizer, PEFT adapter, and optional HaltGate. That
makes it the right place to smoke-test the real checkpoint before wrapping it in
benchmarks or product-facing inference.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import torch

from ouroboros.coconut import HaltGate
from ouroboros.coconut import decode_from_latent_context, prepare_latent_runtime, run_latent_passes
from ouroboros.models import MODEL_ID, load_base_model_and_tokenizer
from ouroboros.models.loading import _amp_dtype, _get_embed_tokens, module_first_device
from ouroboros.utils.runtime_env import resolve_hf_token

DEFAULT_ADAPTER_REPO = "WeirdRunner/Ouroboros"
DEFAULT_ADAPTER_SUBFOLDER = "diloco_state/anchor"
DEFAULT_ADAPTER_CACHE_DIR = "/kaggle/working/ouroboros_inference_adapter"
DEFAULT_STAGE_K = 10
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_HALT_THRESHOLD = 0.5


@dataclass(frozen=True)
class PromptBudgetReport:
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
    actual_latents: int | list[int]
    stage_k: int
    used_halt_gate: bool
    prompt_budget: dict[str, Any]


def _normalize_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _env(env: Mapping[str, str], name: str, default: str) -> str:
    return _normalize_text(env.get(name)) or default


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    text = _normalize_text(env.get(name))
    if text is None:
        return default
    return text.lower() in {"1", "true", "yes", "y", "on"}


def parse_args(argv: Iterable[str] | None = None, *, env: Mapping[str, str] | None = None) -> argparse.Namespace:
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(description="Run Ouroboros adapter/DGAC inference")
    parser.add_argument("--prompt", default=_normalize_text(env.get("OUROBOROS_INFERENCE_PROMPT")))
    parser.add_argument("--prompt_file", default=_normalize_text(env.get("OUROBOROS_INFERENCE_PROMPT_FILE")))
    parser.add_argument("--base_model", default=_env(env, "OUROBOROS_INFERENCE_BASE_MODEL", MODEL_ID))
    parser.add_argument("--adapter_repo", default=_env(env, "OUROBOROS_INFERENCE_ADAPTER_REPO", DEFAULT_ADAPTER_REPO))
    parser.add_argument("--adapter_subfolder", default=_env(env, "OUROBOROS_INFERENCE_ADAPTER_SUBFOLDER", DEFAULT_ADAPTER_SUBFOLDER))
    parser.add_argument("--adapter_dir", default=_normalize_text(env.get("OUROBOROS_INFERENCE_ADAPTER_DIR")))
    parser.add_argument("--adapter_cache_dir", default=_env(env, "OUROBOROS_INFERENCE_ADAPTER_CACHE_DIR", DEFAULT_ADAPTER_CACHE_DIR))
    parser.add_argument("--device", default=_env(env, "OUROBOROS_INFERENCE_DEVICE", "auto"))
    parser.add_argument("--dtype", default=_env(env, "OUROBOROS_INFERENCE_DTYPE", "auto"))
    parser.add_argument(
        "--model_device_map",
        default=_env(env, "OUROBOROS_INFERENCE_MODEL_DEVICE_MAP", "single"),
        choices=("single", "auto", "balanced", "balanced_low_0", "sequential"),
        help="Inference-only model placement. Use balanced_low_0/auto to shard large eval models across visible CUDA GPUs.",
    )
    parser.add_argument("--stage_k", type=int, default=int(_env(env, "OUROBOROS_INFERENCE_STAGE_K", str(DEFAULT_STAGE_K))))
    parser.add_argument("--max_new_tokens", type=int, default=int(_env(env, "OUROBOROS_INFERENCE_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS))))
    parser.add_argument("--max_seq_len", type=int, default=int(_env(env, "OUROBOROS_INFERENCE_MAX_SEQ_LEN", str(DEFAULT_MAX_SEQ_LEN))))
    parser.add_argument("--halt_threshold", type=float, default=float(_env(env, "OUROBOROS_INFERENCE_HALT_THRESHOLD", str(DEFAULT_HALT_THRESHOLD))))
    parser.add_argument("--use_chat_template", action="store_true", default=_env_bool(env, "OUROBOROS_INFERENCE_USE_CHAT_TEMPLATE", True))
    parser.add_argument("--no_chat_template", dest="use_chat_template", action="store_false")
    parser.add_argument("--use_halt_gate", action="store_true", default=_env_bool(env, "OUROBOROS_INFERENCE_USE_HALT_GATE", True))
    parser.add_argument("--no_halt_gate", dest="use_halt_gate", action="store_false")
    parser.add_argument("--use_4bit", action="store_true", default=_env_bool(env, "OUROBOROS_INFERENCE_USE_4BIT", False))
    parser.add_argument("--disable_mamba_kernels", action="store_true", default=_env_bool(env, "OUROBOROS_INFERENCE_DISABLE_MAMBA_KERNELS", False))
    parser.add_argument("--json", action="store_true", default=_env_bool(env, "OUROBOROS_INFERENCE_JSON", False))
    return parser.parse_args(list(argv) if argv is not None else None)


def resolve_device(requested: str) -> torch.device:
    requested = (requested or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    requested = (requested or "auto").strip().lower()
    if requested == "auto":
        return _amp_dtype(device)
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if requested not in mapping:
        raise ValueError(f"Unsupported dtype {requested!r}. Use auto, float32, float16, or bfloat16.")
    return mapping[requested]


def download_adapter_snapshot(
    *,
    repo_id: str,
    subfolder: str,
    token: str | None,
    cache_dir: str,
) -> Path:
    from huggingface_hub import snapshot_download

    target = Path(cache_dir)
    subfolder = (subfolder or "").strip().strip("/")
    allow_patterns = [f"{subfolder}/*"] if subfolder and subfolder != "." else None
    snapshot_download(repo_id=repo_id, token=token, local_dir=str(target), allow_patterns=allow_patterns)
    adapter_dir = target / subfolder if subfolder and subfolder != "." else target
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"No PEFT adapter_config.json found at {adapter_dir}.")
    return adapter_dir


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    prompt = _normalize_text(args.prompt)
    if prompt is None:
        raise SystemExit("Provide --prompt or --prompt_file for inference.")
    return prompt


def _load_tokenizer(base_model: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    lat_token = "<|lat|>"
    existing_id = tokenizer.convert_tokens_to_ids(lat_token)
    if existing_id is None or existing_id == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [lat_token]})
    return tokenizer


def load_components(args: argparse.Namespace):
    """Load base model, resized tokenizer, PEFT adapter, and optional HaltGate."""
    from peft import PeftModel

    token = resolve_hf_token(env=os.environ)
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else download_adapter_snapshot(
        repo_id=args.adapter_repo,
        subfolder=args.adapter_subfolder,
        token=token,
        cache_dir=args.adapter_cache_dir,
    )
    if bool(getattr(args, "require_halt_gate", False)):
        gate_path = adapter_dir / "halt_gate.pt"
        if not gate_path.exists():
            raise FileNotFoundError(f"Required halt_gate.pt not found at {gate_path}.")

    device = resolve_device(args.device)
    _ = resolve_dtype(args.dtype, device)  # validates explicit dtype requests; model seam owns auto dtype policy.
    base_model, tokenizer, _, lat_token_id = load_base_model_and_tokenizer(
        SimpleNamespace(
            model_id=args.base_model,
            use_4bit=bool(getattr(args, "use_4bit", False)),
            grad_checkpoint=False,
            dtype=getattr(args, "dtype", "auto"),
            disable_mamba_kernels=bool(getattr(args, "disable_mamba_kernels", False)),
            model_device_map=getattr(args, "model_device_map", "single"),
        ),
        device,
        add_lat_token=True,
    )
    if lat_token_id is None:
        raise RuntimeError("inference model load did not add the required <|lat|> token")

    device = module_first_device(_get_embed_tokens(base_model), device)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    model.eval()

    halt_gate = None
    if args.use_halt_gate:
        gate_path = adapter_dir / "halt_gate.pt"
        if gate_path.exists():
            d_model = int(getattr(model.config, "hidden_size"))
            halt_gate = HaltGate(d_model).to(device=device, dtype=torch.float32)
            halt_gate.load_state_dict(torch.load(gate_path, map_location=device))
            halt_gate.eval()
        else:
            if bool(getattr(args, "require_halt_gate", False)):
                raise FileNotFoundError(f"Required halt_gate.pt not found at {gate_path}.")
            print(f"[inference] halt_gate.pt not found at {gate_path}; using fixed-depth latent inference.")

    return model, tokenizer, halt_gate, device


def build_generation_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        gen_max_tokens=int(args.max_new_tokens),
        max_seq_len=int(args.max_seq_len),
        halt_threshold=float(args.halt_threshold),
        latent_cache=False,
        mac_mps_latent_cache=False,
    )


def format_prompt(tokenizer, prompt: str, *, use_chat_template: bool) -> str:
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
    """Encode a prompt and report whether context budgeting changed scoring input.

    Full validation rows can contain long OpenHermes/code prompts.  Both the
    baseline and latent runtimes recompute full prefixes on T4-class Kaggle GPUs,
    so a single over-budget row can OOM hours into an eval.  Keep the tail of the
    formatted prompt because it preserves the assistant generation prefix and the
    most recent user content under chat templates.  The returned report is used by
    release evals to distinguish a clean score from a truncated-input score.
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
        print(f"[eval] {context}: truncating {dropped} prompt tokens to fit max_seq_len={max_seq_len}.")
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


def encode_prompt_ids(
    tokenizer,
    prompt: str,
    *,
    max_seq_len: int,
    reserve_tokens: int = 0,
    context: str = "prompt",
) -> list[int]:
    """Encode a prompt and enforce the runtime context budget before model forward."""
    input_ids, _ = encode_prompt_ids_with_report(
        tokenizer,
        prompt,
        max_seq_len=max_seq_len,
        reserve_tokens=reserve_tokens,
        context=context,
    )
    return input_ids


def _actual_latents_to_jsonable(actual_latents) -> int | list[int]:
    if isinstance(actual_latents, torch.Tensor):
        values = actual_latents.detach().cpu().view(-1).tolist()
        return int(values[0]) if len(values) == 1 else [int(v) for v in values]
    return int(actual_latents)


def run_single_prompt(
    *,
    model,
    tokenizer,
    halt_gate,
    prompt: str,
    stage_k: int,
    device: torch.device,
    args: argparse.Namespace,
    use_chat_template: bool = True,
) -> InferenceResult:
    runtime = prepare_latent_runtime(model, device, resolve_dtype(getattr(args, "dtype", "auto"), device))
    generation_args = build_generation_args(args)
    prefix = format_prompt(tokenizer, prompt, use_chat_template=use_chat_template)
    q_ids, prompt_budget = encode_prompt_ids_with_report(
        tokenizer,
        prefix,
        max_seq_len=int(getattr(generation_args, "max_seq_len", DEFAULT_MAX_SEQ_LEN)),
        reserve_tokens=max(0, int(stage_k)),
        context="candidate prompt",
    )
    q_tensor = torch.tensor(q_ids, device=device, dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        ctx = runtime.embed_tokens(q_tensor)
        ctx_mask = torch.ones((1, ctx.size(1)), dtype=torch.bool, device=device)
        ctx, ctx_mask, actual_latents = run_latent_passes(
            runtime=runtime,
            ctx=ctx,
            ctx_mask=ctx_mask,
            n_latent=int(stage_k),
            halt_gate=halt_gate,
            args=generation_args,
        )
        decoded = decode_from_latent_context(
            runtime=runtime,
            ctx=ctx,
            ctx_mask=ctx_mask,
            tokenizer=tokenizer,
            args=generation_args,
            context="ouroboros inference decode",
        )
    return InferenceResult(
        prompt=prompt,
        text=decoded.text,
        actual_latents=_actual_latents_to_jsonable(actual_latents),
        stage_k=int(stage_k),
        used_halt_gate=halt_gate is not None,
        prompt_budget=asdict(prompt_budget),
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    prompt = resolve_prompt(args)
    model, tokenizer, halt_gate, device = load_components(args)
    result = run_single_prompt(
        model=model,
        tokenizer=tokenizer,
        halt_gate=halt_gate,
        prompt=prompt,
        stage_k=args.stage_k,
        device=device,
        args=args,
        use_chat_template=bool(args.use_chat_template),
    )
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(result.text)
        print(f"\n[inference] actual_latents={result.actual_latents} halt_gate={result.used_halt_gate}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
