"""Latent-aware greedy generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from ouroboros.config import DEFAULT_BASE_MODEL, DEFAULT_LATENT_TOKEN
from ouroboros.data import format_question
from ouroboros.latent import OuroborosCoconutForCausalLM, load_published_coconut
from ouroboros.utils import json_print, resolve_device, resolve_dtype


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]


@torch.no_grad()
def generate(
    model: OuroborosCoconutForCausalLM,
    tokenizer,
    *,
    prompt: str,
    stage: int,
    max_new_tokens: int = 128,
    max_seq_len: int = 1024,
    use_chat_template: bool = True,
) -> GenerationResult:
    device = model.device
    q_ids = tokenizer.encode(
        format_question(tokenizer, prompt, use_chat_template=use_chat_template),
        add_special_tokens=False,
    )
    q_ids = q_ids[-max(1, int(max_seq_len) - int(stage)) :]
    input_ids = torch.tensor([q_ids + [model.latent_token_id] * int(stage)], device=device)
    attention_mask = torch.ones_like(input_ids)
    embeds = model._fill_latents(input_ids, attention_mask).embeds

    generated: list[int] = []
    eos_id = tokenizer.eos_token_id
    for _ in range(max(0, int(max_new_tokens))):
        outputs = model.base_causallm(
            inputs_embeds=embeds,
            attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device=device),
            return_dict=True,
        )
        next_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)
        next_embed = model.get_input_embeddings()(torch.tensor([[next_id]], device=device))
        embeds = torch.cat([embeds, next_embed], dim=1)
        if embeds.size(1) > max_seq_len:
            embeds = embeds[:, -max_seq_len:, :]
    return GenerationResult(text=tokenizer.decode(generated, skip_special_tokens=True).strip(), token_ids=generated)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one latent-aware Ouroboros generation.")
    parser.add_argument("--adapter", required=True, help="Local bundle dir or Hub repo id.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--latent-token", default=DEFAULT_LATENT_TOKEN)
    parser.add_argument("--stage", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-chat-template", dest="use_chat_template", action="store_false")
    parser.set_defaults(use_chat_template=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model, tokenizer, _latent_id = load_published_coconut(
        base_model=args.base_model,
        adapter_id_or_path=args.adapter,
        latent_token=args.latent_token,
        device=device,
        dtype=dtype,
    )
    result = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        stage=args.stage,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=args.max_seq_len,
        use_chat_template=bool(args.use_chat_template),
    )
    if args.json:
        json_print({"text": result.text, "token_ids": result.token_ids})
    else:
        print(result.text)
