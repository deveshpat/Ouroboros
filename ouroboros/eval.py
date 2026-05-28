"""Direct evaluation entrypoint: local latent loss or lm-eval bridge."""

from __future__ import annotations

import argparse
import subprocess
import sys

import torch

from ouroboros.config import DEFAULT_BASE_MODEL, DEFAULT_LATENT_TOKEN
from ouroboros.data import make_loader
from ouroboros.latent import load_published_coconut
from ouroboros.train import evaluate_loss
from ouroboros.utils import json_print, resolve_device, resolve_dtype


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Ouroboros without coordinators.")
    parser.add_argument("--adapter", required=True, help="Local bundle dir or Hub repo id.")
    parser.add_argument("--data", default="data/coconut_v1/val.jsonl")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--latent-token", default=DEFAULT_LATENT_TOKEN)
    parser.add_argument("--stage", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tasks", default=None, help="Comma-separated lm-eval tasks, e.g. hellaswag,arc_easy.")
    parser.add_argument("--lm-eval", action="store_true", help="Run EleutherAI lm-eval-harness directly.")
    parser.add_argument("--lm-eval-batch-size", default="auto")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--no-chat-template", dest="use_chat_template", action="store_false")
    parser.set_defaults(use_chat_template=True)
    return parser


@torch.no_grad()
def evaluate_teacher_forced(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model, tokenizer, latent_id = load_published_coconut(
        base_model=args.base_model,
        adapter_id_or_path=args.adapter,
        latent_token=args.latent_token,
        device=device,
        dtype=dtype,
    )
    loader = make_loader(
        path=args.data,
        tokenizer=tokenizer,
        latent_id=latent_id,
        stage=args.stage,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        use_chat_template=bool(args.use_chat_template),
    )
    loss = evaluate_loss(model, loader, halt_gate=None, dgac=None)
    json_print({"adapter": args.adapter, "examples": len(loader.dataset), "stage": args.stage, "loss": loss})


def run_lm_eval(args: argparse.Namespace) -> None:
    if not args.tasks:
        raise SystemExit("--lm-eval requires --tasks")
    model_args = f"pretrained={args.base_model},peft={args.adapter},trust_remote_code=True"
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        args.tasks,
        "--batch_size",
        str(args.lm_eval_batch_size),
    ]
    if args.output_path:
        command.extend(["--output_path", args.output_path])
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.lm_eval or args.tasks:
        run_lm_eval(args)
    else:
        evaluate_teacher_forced(args)
