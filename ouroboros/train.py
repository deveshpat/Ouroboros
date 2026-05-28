"""Direct staged training entrypoint: config -> model -> data -> stages -> save."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from ouroboros.callbacks import maybe_init_wandb, push_release_bundle, save_release_bundle
from ouroboros.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_LATENT_TOKEN,
    DEFAULT_LORA_TARGETS,
    DgacConfig,
    parse_stage_spec,
    split_csv,
)
from ouroboros.data import make_loader
from ouroboros.latent import HaltGate, load_lora_coconut
from ouroboros.utils import default_hf_token, resolve_device, resolve_dtype, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an Ouroboros Coconut PEFT adapter.")
    parser.add_argument("--train", default="data/coconut_v1/train.jsonl")
    parser.add_argument("--validation", default="data/coconut_v1/val.jsonl")
    parser.add_argument("--output-dir", default="runs/ouroboros")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--latent-token", default=DEFAULT_LATENT_TOKEN)
    parser.add_argument("--stages", default=None, help="Comma/range stages, e.g. '0-10' or '0,2,4,8'.")
    parser.add_argument("--max-stage", type=int, default=10)
    parser.add_argument("--epochs-per-stage", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=",".join(DEFAULT_LORA_TARGETS))
    parser.add_argument("--use-halt-gate", action="store_true")
    parser.add_argument("--dgac-warmup-steps", type=int, default=200)
    parser.add_argument("--dgac-ramp-steps", type=int, default=300)
    parser.add_argument("--dgac-lambda-ponder-max", type=float, default=0.01)
    parser.add_argument("--dgac-lambda-diversity", type=float, default=0.1)
    parser.add_argument("--dgac-tau", type=float, default=0.9)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--wandb-project", default="ouroboros")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-chat-template", dest="use_chat_template", action="store_false")
    parser.set_defaults(use_chat_template=True)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--hub-token", default=None)
    parser.add_argument("--hub-public", dest="hub_private", action="store_false")
    parser.set_defaults(hub_private=True)
    return parser


@torch.no_grad()
def evaluate_loss(model, loader, *, halt_gate: HaltGate | None, dgac: DgacConfig | None, accelerator=None) -> float:
    model.eval()
    if halt_gate is not None:
        halt_gate.eval()
    losses: list[float] = []
    for batch in loader:
        output = model(**batch, halt_gate=halt_gate, dgac=dgac)
        if output.loss is not None:
            loss = output.loss.detach()
            if accelerator is not None:
                loss = accelerator.gather_for_metrics(loss.repeat(batch["input_ids"].size(0))).mean()
            losses.append(float(loss.cpu()))
    model.train()
    if halt_gate is not None:
        halt_gate.train()
    return sum(losses) / max(len(losses), 1)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.push_to_hub and not args.hub_model_id:
        raise SystemExit("--push-to-hub requires --hub-model-id")

    from accelerate import Accelerator

    seed_everything(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=max(int(args.grad_accum), 1))
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
        target_modules=split_csv(args.target_modules),
        load_in_4bit=bool(args.load_in_4bit),
    )
    halt_gate = HaltGate(model.hidden_size).to(model.device) if args.use_halt_gate else None
    dgac = DgacConfig(
        enabled=bool(args.use_halt_gate),
        warmup_steps=args.dgac_warmup_steps,
        ramp_steps=args.dgac_ramp_steps,
        lambda_ponder_max=args.dgac_lambda_ponder_max,
        lambda_diversity=args.dgac_lambda_diversity,
        tau=args.dgac_tau,
    )

    trainable = [param for param in model.parameters() if param.requires_grad]
    if halt_gate is not None:
        trainable.extend(halt_gate.parameters())
    optimizer = AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    if halt_gate is None:
        model, optimizer = accelerator.prepare(model, optimizer)
    else:
        model, halt_gate, optimizer = accelerator.prepare(model, halt_gate, optimizer)

    wandb_run = maybe_init_wandb(args) if accelerator.is_main_process else None
    stages = parse_stage_spec(args.stages, max_stage=args.max_stage)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for stage in stages:
        train_loader = make_loader(
            path=args.train,
            tokenizer=tokenizer,
            latent_id=latent_id,
            stage=stage,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            use_chat_template=bool(args.use_chat_template),
            shuffle=True,
        )
        val_loader = make_loader(
            path=args.validation,
            tokenizer=tokenizer,
            latent_id=latent_id,
            stage=stage,
            max_seq_len=args.max_seq_len,
            max_samples=args.eval_samples,
            batch_size=args.batch_size,
            use_chat_template=bool(args.use_chat_template),
            shuffle=False,
        )
        train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
        for epoch in range(int(args.epochs_per_stage)):
            model.train()
            if halt_gate is not None:
                halt_gate.train()
            progress = tqdm(
                train_loader,
                desc=f"stage {stage} epoch {epoch + 1}/{args.epochs_per_stage}",
                disable=not accelerator.is_main_process,
                dynamic_ncols=True,
            )
            for batch in progress:
                with accelerator.accumulate(model):
                    output = model(**batch, halt_gate=halt_gate, dgac=dgac, global_step=global_step)
                    loss = output.loss
                    if loss is None:
                        continue
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable, float(args.max_grad_norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    global_step += 1
                    if global_step % max(int(args.log_every), 1) == 0 and accelerator.is_main_process:
                        progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", step=global_step)
                        if wandb_run is not None:
                            wandb_run.log({"train/loss": float(loss.detach().cpu()), "stage": stage}, step=global_step)

            val_loss = evaluate_loss(model, val_loader, halt_gate=halt_gate, dgac=dgac, accelerator=accelerator)
            if accelerator.is_main_process:
                print(f"[ouroboros] stage={stage} epoch={epoch + 1} step={global_step} val_loss={val_loss:.4f}")
                if wandb_run is not None:
                    wandb_run.log({"val/loss": val_loss, "stage": stage}, step=global_step)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped_gate = accelerator.unwrap_model(halt_gate) if halt_gate is not None else None
        final_stage = stages[-1] if stages else 0
        bundle_dir = save_release_bundle(
            unwrapped,
            tokenizer,
            output_dir / "final",
            base_model=args.base_model,
            stage=final_stage,
            latent_token=args.latent_token,
            halt_gate=unwrapped_gate,
        )
        print(f"[ouroboros] saved release bundle -> {bundle_dir}")
        if args.push_to_hub:
            url = push_release_bundle(
                bundle_dir,
                args.hub_model_id,
                token=default_hf_token(args.hub_token),
                private=bool(args.hub_private),
            )
            print(f"[ouroboros] pushed -> {url}")
        if wandb_run is not None:
            wandb_run.finish()
