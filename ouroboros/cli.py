"""Ouroboros command line.

Commands:

    python -m ouroboros train --push-to-hub --hub-model-id WeirdRunner/Ouroboros
    python -m ouroboros infer --adapter WeirdRunner/Ouroboros --prompt "..."
    python -m ouroboros publish --bundle-dir runs/ouroboros/final --hub-model-id WeirdRunner/Ouroboros

The point is to use the boring library APIs directly and keep the experiment in
two files: ``coconut.py`` for the runtime and this file for the CLI.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ouroboros.coconut import (
    DEFAULT_BASE_MODEL,
    DEFAULT_LATENT_TOKEN,
    DEFAULT_LORA_TARGETS,
    JsonlCoconutDataset,
    CoconutCollator,
    build_features,
    generate,
    load_lora_coconut,
    load_published_coconut,
    load_rows,
    push_release_bundle,
    resolve_device,
    resolve_dtype,
    save_release_bundle,
)


def _split_csv(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def _default_hf_token(cli_value: str | None = None) -> str | None:
    return cli_value or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ouroboros",
        description="Ouroboros: train, infer, eval, and publish a Coconut adapter.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a PEFT Coconut adapter and optionally publish it.")
    train.add_argument("--train", default="data/coconut_v1/train.jsonl")
    train.add_argument("--validation", default="data/coconut_v1/val.jsonl")
    train.add_argument("--output-dir", default="runs/ouroboros")
    train.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    train.add_argument("--latent-token", default=DEFAULT_LATENT_TOKEN)
    train.add_argument("--stage", type=int, default=10)
    train.add_argument("--max-seq-len", type=int, default=1024)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--grad-accum", type=int, default=8)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--max-samples", type=int, default=None)
    train.add_argument("--eval-samples", type=int, default=128)
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--device", default="auto")
    train.add_argument("--dtype", default="auto")
    train.add_argument("--load-in-4bit", action="store_true")
    train.add_argument("--lora-r", type=int, default=32)
    train.add_argument("--lora-alpha", type=int, default=64)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.add_argument("--target-modules", default=",".join(DEFAULT_LORA_TARGETS))
    train.add_argument("--no-chat-template", dest="use_chat_template", action="store_false")
    train.set_defaults(use_chat_template=True)
    train.add_argument("--push-to-hub", action="store_true")
    train.add_argument("--hub-model-id", default=None)
    train.add_argument("--hub-token", default=None)
    train.add_argument("--hub-public", dest="hub_private", action="store_false")
    train.set_defaults(hub_private=True)

    ev = sub.add_parser("eval", help="Teacher-forced loss on a JSON/JSONL split.")
    ev.add_argument("--adapter", required=True, help="Local bundle dir or Hub repo id.")
    ev.add_argument("--data", default="data/coconut_v1/val.jsonl")
    ev.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ev.add_argument("--latent-token", default=DEFAULT_LATENT_TOKEN)
    ev.add_argument("--stage", type=int, default=10)
    ev.add_argument("--max-seq-len", type=int, default=1024)
    ev.add_argument("--max-samples", type=int, default=128)
    ev.add_argument("--batch-size", type=int, default=1)
    ev.add_argument("--device", default="auto")
    ev.add_argument("--dtype", default="auto")
    ev.add_argument("--no-chat-template", dest="use_chat_template", action="store_false")
    ev.set_defaults(use_chat_template=True)

    infer = sub.add_parser("infer", help="Run a single prompt through the Coconut adapter.")
    infer.add_argument("--adapter", required=True, help="Local bundle dir or Hub repo id.")
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    infer.add_argument("--latent-token", default=DEFAULT_LATENT_TOKEN)
    infer.add_argument("--stage", type=int, default=10)
    infer.add_argument("--max-new-tokens", type=int, default=128)
    infer.add_argument("--max-seq-len", type=int, default=1024)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--dtype", default="auto")
    infer.add_argument("--json", action="store_true")
    infer.add_argument("--no-chat-template", dest="use_chat_template", action="store_false")
    infer.set_defaults(use_chat_template=True)

    publish = sub.add_parser("publish", help="Push an already-saved release bundle to Hugging Face Hub.")
    publish.add_argument("--bundle-dir", required=True)
    publish.add_argument("--hub-model-id", required=True)
    publish.add_argument("--hub-token", default=None)
    publish.add_argument("--hub-public", dest="hub_private", action="store_false")
    publish.set_defaults(hub_private=True)
    return parser


def _make_loader(
    *,
    path: str,
    tokenizer,
    latent_id: int,
    stage: int,
    max_seq_len: int,
    max_samples: int | None,
    batch_size: int,
    use_chat_template: bool,
    shuffle: bool = False,
) -> DataLoader:
    rows = load_rows(path, limit=max_samples)
    features = build_features(
        rows,
        tokenizer,
        latent_token_id=latent_id,
        stage=stage,
        max_seq_len=max_seq_len,
        use_chat_template=use_chat_template,
    )
    if not features:
        raise SystemExit(f"No usable examples built from {path}")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return DataLoader(
        JsonlCoconutDataset(features),
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        collate_fn=CoconutCollator(int(pad_id or 0)),
    )


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        output = model(**batch)
        if output.loss is not None:
            losses.append(float(output.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def train_command(args: argparse.Namespace) -> None:
    if args.push_to_hub and not args.hub_model_id:
        raise SystemExit("--push-to-hub requires --hub-model-id")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model, tokenizer, latent_id = load_lora_coconut(
        base_model=args.base_model,
        latent_token=args.latent_token,
        device=device,
        dtype=dtype,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=_split_csv(args.target_modules),
        load_in_4bit=bool(args.load_in_4bit),
    )
    try:
        model.base_causallm.print_trainable_parameters()
    except Exception:
        pass

    train_loader = _make_loader(
        path=args.train,
        tokenizer=tokenizer,
        latent_id=latent_id,
        stage=args.stage,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        use_chat_template=bool(args.use_chat_template),
        shuffle=True,
    )
    val_loader = _make_loader(
        path=args.validation,
        tokenizer=tokenizer,
        latent_id=latent_id,
        stage=args.stage,
        max_seq_len=args.max_seq_len,
        max_samples=args.eval_samples,
        batch_size=args.batch_size,
        use_chat_template=bool(args.use_chat_template),
        shuffle=False,
    )

    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(int(args.epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for micro_step, batch in enumerate(pbar, start=1):
            output = model(**batch)
            if output.loss is None:
                continue
            loss = output.loss / max(int(args.grad_accum), 1)
            loss.backward()
            if micro_step % max(int(args.grad_accum), 1) == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % max(int(args.log_every), 1) == 0:
                    pbar.set_postfix(loss=f"{float(output.loss.detach().cpu()):.4f}", step=global_step)
        if micro_step % max(int(args.grad_accum), 1) != 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        val_loss = evaluate_loss(model, val_loader)
        print(f"[ouroboros] epoch={epoch + 1} step={global_step} val_loss={val_loss:.4f}")

    bundle_dir = save_release_bundle(
        model,
        tokenizer,
        output_dir / "final",
        base_model=args.base_model,
        stage=args.stage,
        latent_token=args.latent_token,
    )
    print(f"[ouroboros] saved release bundle -> {bundle_dir}")

    if args.push_to_hub:
        url = push_release_bundle(
            bundle_dir,
            args.hub_model_id,
            token=_default_hf_token(args.hub_token),
            private=bool(args.hub_private),
        )
        print(f"[ouroboros] pushed -> {url}")


def eval_command(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model, tokenizer, latent_id = load_published_coconut(
        base_model=args.base_model,
        adapter_id_or_path=args.adapter,
        latent_token=args.latent_token,
        device=device,
        dtype=dtype,
    )
    loader = _make_loader(
        path=args.data,
        tokenizer=tokenizer,
        latent_id=latent_id,
        stage=args.stage,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        use_chat_template=bool(args.use_chat_template),
    )
    _json_print({"adapter": args.adapter, "examples": len(loader.dataset), "loss": evaluate_loss(model, loader)})


def infer_command(args: argparse.Namespace) -> None:
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
        _json_print({"text": result.text, "token_ids": result.token_ids})
    else:
        print(result.text)


def publish_command(args: argparse.Namespace) -> None:
    bundle = Path(args.bundle_dir)
    if not bundle.exists():
        raise SystemExit(f"Bundle directory not found: {bundle}")
    url = push_release_bundle(
        bundle,
        args.hub_model_id,
        token=_default_hf_token(args.hub_token),
        private=bool(args.hub_private),
    )
    print(f"[ouroboros] pushed -> {url}")


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "train":
        train_command(args)
    elif args.command == "eval":
        eval_command(args)
    elif args.command == "infer":
        infer_command(args)
    elif args.command == "publish":
        publish_command(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
