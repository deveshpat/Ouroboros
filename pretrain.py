#!/usr/bin/env python3
"""Stage 1 pre-training for BaselineTRMMamba on FineWeb-Edu.

Project Ouroboros uses this script to teach the non-recursive Transformer-Mamba
backbone basic language modeling before any supervised instruction tuning.
Training is plain next-token prediction on raw text: every packed token position
is real text and contributes to the objective.

Dataset
    HuggingFaceFW/fineweb-edu (default config: sample-10BT), streamed and
    tokenized on the fly.

Install requirements
    pip install "causal-conv1d>=1.4.0" mamba-ssm --no-build-isolation
    pip install transformers datasets tqdm huggingface_hub wandb

Runtime guide
    +----------------------+-------------------+----------------------------------------------+
    | mode                 | hardware          | notes                                        |
    +----------------------+-------------------+----------------------------------------------+
    | smoke_test_20_steps  | CPU or single GPU | synthetic offline pipeline                   |
    | --dry_run            | single GPU        | 10M-token sanity run                         |
    | full Stage 1         | single GPU        | original training path                       |
    | full Stage 1         | 2x T4 / multi-GPU | auto DDP when >=2 CUDA devices are visible   |
    +----------------------+-------------------+----------------------------------------------+
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import socket
import sys
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("transformers required: pip install transformers")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("datasets required: pip install datasets")

try:
    from tqdm.auto import tqdm
except ImportError:
    sys.exit("tqdm required: pip install tqdm")

try:
    from baseline_trm_mamba import BaselineConfig, BaselineTRMMamba, count_parameters
except ImportError as exc:
    sys.exit(
        f"Cannot import baseline_trm_mamba: {exc}\n"
        "baseline_trm_mamba.py must be in the same directory."
    )

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HF_UPLOAD_TIMEOUT_SECONDS = 300.0


PRESETS: Dict[str, Dict[str, Any]] = {
    "nano": dict(d_model=512, n_groups=1, n_heads=8, n_kv_heads=4),
    "small": dict(d_model=1024, n_groups=2, n_heads=16, n_kv_heads=8),
    "medium": dict(d_model=2048, n_groups=2, n_heads=16, n_kv_heads=8),
}


GEN_PROMPTS_STAGE1 = [
    "The capital of France is",
    "In mathematics, a prime number is",
    "def factorial(n):\n    \"\"\"Return n!.\"\"\"\n    if n",
    "Neural networks learn by",
    "The French Revolution began in",
]


@dataclass
class PretrainConfig:
    """Configuration for streaming Stage 1 language-model pre-training."""

    # Dataset
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    token_budget: int = 2_000_000_000
    shuffle_buffer: int = 10_000
    num_epochs: int = 1

    # Model preset
    preset: str = "nano"

    # Sequence packing
    chunk_size: int = 1024

    # Optimizer
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Batch
    batch_size: int = 8
    grad_accum: int = 4

    # EMA
    ema_decay: float = 0.995

    # I/O
    output_dir: str = "runs/stage1"
    save_every: int = 1000
    keep_last: int = 3

    # Hub sync
    push_to_hub: bool = False
    hf_repo_id: str = "WeirdRunner/Ouroboros"
    hf_token: Optional[str] = None

    # Monitoring
    log_every: int = 50
    val_tokens: int = 2_000_000
    val_every: int = 500
    gen_every: int = 500
    gen_max_tokens: int = 120
    spike_threshold: float = 0.5

    # Reproducibility
    seed: int = 42

    # wandb
    wandb_project: str = "ouroboros-stage1"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"


def resolve_hf_token(cfg: PretrainConfig) -> Optional[str]:
    """Return the Hugging Face write token from config or environment only."""
    if cfg.hf_token:
        return cfg.hf_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


class TokenStream:
    """Stream FineWeb-Edu into fixed-size packed chunks with epoch-varying offsets."""

    def __init__(self, cfg: PretrainConfig, tokenizer, eos_id: int) -> None:
        """Store configuration, tokenizer, and validation-buffer state."""
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self._val_ids: Optional[List[int]] = None
        self._val_doc_hashes: set[str] = set()

    def epoch_offset(self, epoch: int) -> int:
        """Return the deterministic random chunk offset for a given epoch."""
        rng = random.Random(self.cfg.seed + epoch * 104_729)
        return rng.randint(0, self.cfg.chunk_size - 1)

    def _shuffled_stream(self, epoch: int):
        """Return a buffer-shuffled streaming dataset for one epoch."""
        ds = load_dataset(
            self.cfg.dataset_name,
            name=self.cfg.dataset_config,
            split="train",
            streaming=True,
        )
        # reason: reshuffle document order every epoch to break shard-local periodicity.
        return ds.shuffle(buffer_size=self.cfg.shuffle_buffer, seed=self.cfg.seed + epoch)

    def _tokenize(self, text: str) -> List[int]:
        """Encode one document and append the EOS separator."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids.append(self.eos_id)
        return ids

    def _doc_hash(self, text: str) -> str:
        """Return a stable document hash for train/val separation."""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def build_val_buffer(self, verbose: bool = True) -> None:
        """Materialize a held-out validation token buffer before training starts."""
        if verbose:
            print(f"  Building val buffer ({self.cfg.val_tokens:,} tokens) ...")
        stream = self._shuffled_stream(epoch=-1)
        buf: List[int] = []

        for ex in stream:
            text = (ex.get("text") or "").strip()
            if not text:
                continue
            digest = self._doc_hash(text)
            if digest in self._val_doc_hashes:
                continue
            self._val_doc_hashes.add(digest)
            buf.extend(self._tokenize(text))
            if len(buf) >= self.cfg.val_tokens:
                break

        self._val_ids = buf[: self.cfg.val_tokens]
        if verbose:
            print(f"  Val buffer: {len(self._val_ids):,} tokens from {len(self._val_doc_hashes):,} docs")

    def iter_train_chunks(self, epoch: int) -> Iterator[torch.Tensor]:
        """Yield Tensor[chunk_size] training chunks for one deterministic epoch pass."""
        offset = self.epoch_offset(epoch)
        stream = self._shuffled_stream(epoch)
        buf: List[int] = []
        skipped = 0

        for ex in stream:
            text = (ex.get("text") or "").strip()
            if not text:
                continue
            if self._val_doc_hashes and self._doc_hash(text) in self._val_doc_hashes:
                # reason: keep the validation buffer document-disjoint from training.
                continue

            buf.extend(self._tokenize(text))

            if skipped < offset:
                discard = min(offset - skipped, len(buf))
                buf = buf[discard:]
                skipped += discard

            while len(buf) >= self.cfg.chunk_size:
                chunk = buf[: self.cfg.chunk_size]
                buf = buf[self.cfg.chunk_size :]
                yield torch.tensor(chunk, dtype=torch.long)
        # reason: short tail fragments are dropped so every training position is real text.

    def val_chunks(self) -> Iterator[torch.Tensor]:
        """Yield non-overlapping validation chunks from the held-out token buffer."""
        if self._val_ids is None:
            raise RuntimeError("Call build_val_buffer() before val_chunks().")
        ids = self._val_ids
        size = self.cfg.chunk_size
        for start in range(0, len(ids) - size + 1, size):
            yield torch.tensor(ids[start : start + size], dtype=torch.long)


class SpikeMonitor:
    """Track a smoothed training-loss EMA and warn on sudden spikes."""

    def __init__(self, beta: float = 0.99, threshold: float = 0.5) -> None:
        """Initialize the spike detector with a smoothing factor and threshold."""
        self.beta = beta
        self.threshold = threshold
        self._ema: Optional[float] = None
        self.spikes: List[Tuple[int, float, float]] = []

    def update(self, step: int, loss: float) -> bool:
        """Update the EMA and record whether the latest loss is a spike."""
        if self._ema is None:
            self._ema = loss
            return False
        self._ema = self.beta * self._ema + (1.0 - self.beta) * loss
        # reason: early EMA values are biased low without correction.
        bias_corrected = self._ema / (1.0 - self.beta ** (step + 1))
        is_spike = (loss - bias_corrected) > self.threshold
        if is_spike:
            self.spikes.append((step, loss, bias_corrected))
        return is_spike

    @property
    def smoothed(self) -> float:
        """Return the current raw EMA value for display."""
        return self._ema if self._ema is not None else float("nan")



def pretrain_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Compute standard next-token cross-entropy over a packed token batch."""
    shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1)).float()
    shift_labels = tokens[:, 1:].contiguous().view(-1)
    return F.cross_entropy(shift_logits, shift_labels)


class ModelEMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        one_minus = 1.0 - self.decay
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=one_minus)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.decay = float(sd.get("decay", self.decay))
        for name, tensor in sd.get("shadow", {}).items():
            if name in self.shadow and self.shadow[name].shape == tensor.shape:
                self.shadow[name].copy_(tensor)



def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    """Linear warmup followed by cosine decay to min_lr_ratio times base LR."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        if total_steps <= warmup_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)



def dist_is_initialized() -> bool:
    """Return True when torch.distributed collectives are ready to use."""
    return dist.is_available() and dist.is_initialized()


def is_main_process(rank: int) -> bool:
    """Return True for rank zero, the only process that writes logs or checkpoints."""
    return rank == 0


def distributed_mean(value: float, device: torch.device) -> float:
    """Average one scalar across all ranks when DDP is active."""
    if not dist_is_initialized():
        return float(value)
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def distributed_sum_int(value: int, device: torch.device) -> int:
    """Sum one integer across all ranks when DDP is active."""
    if not dist_is_initialized():
        return int(value)
    tensor = torch.tensor([value], device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def distributed_min_int(value: int, device: torch.device) -> int:
    """Compute the minimum integer across ranks when DDP is active."""
    if not dist_is_initialized():
        return int(value)
    tensor = torch.tensor([value], device=device, dtype=torch.int32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def find_free_port() -> int:
    """Return a free localhost TCP port for single-node DDP spawning."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def shard_train_chunk_iterator(
    chunk_iter: Iterator[torch.Tensor],
    rank: int,
    world_size: int,
    global_skip_chunks: int,
) -> Iterator[torch.Tensor]:
    """Shard the deterministic global chunk stream across ranks without overlap."""
    if world_size <= 1:
        return itertools.islice(chunk_iter, global_skip_chunks, None)
    # reason: every rank must consume a disjoint slice of the same global chunk order.
    return itertools.islice(chunk_iter, global_skip_chunks + rank, None, world_size)


def take_local_micro_batch(
    chunk_iter: Iterator[torch.Tensor],
    local_batch_size: int,
) -> List[torch.Tensor]:
    """Collect exactly one per-rank micro-batch from a sharded chunk iterator."""
    chunks: List[torch.Tensor] = []
    for _ in range(local_batch_size):
        try:
            chunks.append(next(chunk_iter))
        except StopIteration:
            break
    return chunks


def autocast_context(device: torch.device, dtype: torch.dtype):
    """Return an autocast context that only activates on CUDA."""
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda"))



def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs for deterministic control flow."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def vram_gb(device: torch.device) -> float:
    """Return allocated VRAM in GiB or 0.0 on CPU."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / (1024 ** 3)



def checkpoint_step_from_name(name: str) -> int:
    """Extract the numeric step from a checkpoint directory name."""
    prefix = "checkpoint-"
    if prefix not in name:
        return -1
    tail = name.split(prefix, 1)[1]
    digits = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else -1



def effective_tokens_per_step(cfg: PretrainConfig) -> int:
    """Return the global token count consumed by one optimizer step."""
    return cfg.batch_size * cfg.grad_accum * cfg.chunk_size



def stage1_success_banner(step: int, preset: str) -> None:
    """Print the Stage 1 completion banner with the Phase 2 resume command."""
    print(
        "  * Stage 1 criterion met:\n"
        "    val_ce < 3.0  AND  mean UWR > 0.05 (non-degenerate generation)\n"
        "    Load this checkpoint into Stage 2:\n"
        "      python train_sft.py \\\n"
        f"        --preset {preset} \\\n"
        f"        --resume_from runs/stage1/checkpoint-{step:07d} \\\n"
        "        --ema_decay 0.995"
    )


@torch.no_grad()
def compute_val_ce(
    model: BaselineTRMMamba,
    ema: ModelEMA,
    token_stream: TokenStream,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int,
    batch_size: int,
) -> float:
    """Compute held-out per-token CE with EMA weights in batched chunks."""
    live_backup: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name in ema.shadow:
            live_backup[name] = param.data.clone()
            param.data.copy_(ema.shadow[name].to(device=param.device, dtype=param.dtype))

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_buf: List[torch.Tensor] = []

    def _run_batch(chunks: List[torch.Tensor]) -> None:
        nonlocal total_loss, total_tokens
        tokens = torch.stack(chunks).to(device)
        with autocast_context(device, dtype):
            logits = model(tokens)
        loss = pretrain_loss(logits, tokens)
        n_tokens = tokens.size(0) * (tokens.size(1) - 1)
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    for chunk in token_stream.val_chunks():
        batch_buf.append(chunk)
        if len(batch_buf) == batch_size:
            _run_batch(batch_buf)
            batch_buf = []
    if batch_buf:
        _run_batch(batch_buf)

    model.train()
    for name, param in model.named_parameters():
        if name in live_backup:
            param.data.copy_(live_backup[name])

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def run_generation_callback(
    model: BaselineTRMMamba,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    step: int,
    max_new_tokens: int,
    max_seq_len: int,
    wandb_run=None,
) -> float:
    """Greedy-decode open-ended text completions from live training weights."""
    model.eval()
    print(f"\n  -- Generation @ step {step} (live weights) --")
    mean_uwr = 0.0
    wandb_payload: Dict[str, Any] = {}

    for idx, prompt in enumerate(GEN_PROMPTS_STAGE1):
        ids = torch.tensor(
            tokenizer.encode(prompt, add_special_tokens=False),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        eos_id = tokenizer.eos_token_id
        generated: List[int] = []

        for _ in range(max_new_tokens):
            if ids.size(1) > max_seq_len:
                ids = ids[:, -max_seq_len:]
            with autocast_context(device, dtype):
                logits = model(ids)
            next_id = int(logits[:, -1, :].argmax(dim=-1).item())
            if eos_id is not None and next_id == eos_id:
                break
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

        text = tokenizer.decode(generated, skip_special_tokens=True)
        words = text.split()
        uwr = len(set(words)) / max(len(words), 1)
        mean_uwr += uwr
        print(f"  P: {prompt}")
        print(f"  C: {text[:160]}")
        print(f"     uwr={uwr:.3f}")
        wandb_payload[f"gen/prompt_{idx:02d}"] = text

    mean_uwr /= max(len(GEN_PROMPTS_STAGE1), 1)
    print(f"  Mean UWR: {mean_uwr:.3f}\n")

    if wandb_run is not None:
        import wandb

        wandb.log({"gen/mean_uwr_stage1": mean_uwr, **wandb_payload}, step=step)

    model.train()
    return mean_uwr



def sync_checkpoint_to_hub(
    checkpoint_dir: Path,
    repo_id: str,
    token: str,
) -> bool:
    """Upload one checkpoint directory to the Hugging Face Hub without raising."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[hub] huggingface_hub not installed - skipping Hub sync.")
        return False

    remote_name = checkpoint_dir.name[:-4] if checkpoint_dir.name.endswith(".tmp") else checkpoint_dir.name
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, token=token, private=True, exist_ok=True)
        print(f"  [hub] uploading {remote_name} -> {repo_id} ...")
        future = api.upload_folder(
            repo_id=repo_id,
            folder_path=str(checkpoint_dir),
            path_in_repo=remote_name,
            token=token,
            commit_message=f"Upload {remote_name}",
            run_as_future=True,
        )
        commit_info = future.result(timeout=HF_UPLOAD_TIMEOUT_SECONDS)
        oid = getattr(commit_info, "oid", "?")
        oid_short = oid[:8] if isinstance(oid, str) and oid != "?" else "?"
        print(f"  [hub] uploaded  {remote_name} (commit={oid_short})")
        return True
    except FutureTimeoutError:
        print(f"  [hub] upload timed out for {remote_name}; local copy retained.")
        return False
    except Exception as exc:
        print(f"  [hub] upload failed for {remote_name}: {exc}")
        return False



def cleanup_temporary_checkpoints(output_dir: Path) -> None:
    """Remove stale .tmp checkpoint directories left by previously interrupted runs."""
    if not output_dir.exists():
        return
    for p in output_dir.iterdir():
        if p.is_dir() and p.name.endswith(".tmp"):
            print(f"  [cleanup] removing stale tmp checkpoint: {p.name}")
            shutil.rmtree(p, ignore_errors=True)



def save_checkpoint(
    output_dir: Path,
    step: int,
    epoch: int,
    chunks_in_epoch: int,
    tokens_processed: int,
    model: BaselineTRMMamba,
    model_config: BaselineConfig,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler,
    ema: ModelEMA,
    val_ce: Optional[float],
    cfg: PretrainConfig,
    hf_token: Optional[str],
) -> Optional[Path]:
    """
    Write a checkpoint to local disk first (always), then attempt Hub push (fire-and-forget).

    Local finalization is never blocked by Hub failures.
    Returns the finalized local Path on success, None if the local write itself fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name  = f"checkpoint-{step:07d}"
    final_dir  = output_dir / ckpt_name
    tmp_dir    = output_dir / f"{ckpt_name}.tmp"

    # Remove any stale .tmp from a previous interrupted run at this step
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Build EMA shadow dict with lm_head.weight alias for tied embeddings
    shadow = dict(ema.shadow)
    if model_config.tie_embeddings:
        if "lm_head.weight" not in shadow and "token_embedding.weight" in shadow:
            shadow["lm_head.weight"] = shadow["token_embedding.weight"]

    cfg_dict = {k: v for k, v in asdict(cfg).items() if k != "hf_token"}

    state = {
        "step": step,
        "epoch": epoch,
        "chunks_in_epoch": chunks_in_epoch,
        "tokens_processed": tokens_processed,
        "model_state_dict": model.state_dict(),
        "ema_backbone_state_dict": shadow,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "ema": ema.state_dict(),
        "backbone_config": asdict(model_config),
        "pretrain_config": cfg_dict,
        "val_ce": val_ce,
    }

    try:
        torch.save(state, tmp_dir / "training_state.pt")
        with (tmp_dir / "resolved_backbone_config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(model_config), f, indent=2)
    except Exception as exc:
        print(f"  [ckpt] ERROR: could not write checkpoint to {tmp_dir}: {exc}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    # ── Step 1: always finalize locally first ────────────────────────────────
    if final_dir.exists():
        shutil.rmtree(final_dir, ignore_errors=True)
    tmp_dir.replace(final_dir)
    print(f"  [ckpt] saved  -> {final_dir}")

    # ── Step 2: prune old local checkpoints ──────────────────────────────────
    retain = max(cfg.keep_last, 1)
    existing = sorted(
        [
            p for p in output_dir.iterdir()
            if p.is_dir()
            and p.name.startswith("checkpoint-")
            and not p.name.endswith(".tmp")
        ],
        key=lambda p: checkpoint_step_from_name(p.name),
    )
    for old in existing[:-retain]:
        shutil.rmtree(old, ignore_errors=True)
        print(f"  [ckpt] pruned -> {old.name}")

    # ── Step 3: attempt Hub push (fire-and-forget, never blocks) ─────────────
    if cfg.push_to_hub and hf_token:
        uploaded = sync_checkpoint_to_hub(final_dir, cfg.hf_repo_id, hf_token)
        if not uploaded:
            print(
                f"  [hub]  warn: step {step} Hub sync failed; "
                f"local checkpoint retained at {final_dir}"
            )

    return final_dir



def _list_local_checkpoints(output_dir: Path) -> List[Path]:
    """
    Return finalized local checkpoint directories sorted newest-first.
    Skips .tmp dirs and any directory whose name doesn't parse as a step number.
    """
    if not output_dir.exists() or not output_dir.is_dir():
        return []
    candidates = []
    for p in output_dir.iterdir():
        if p.is_dir() and not p.name.endswith(".tmp"):
            step = checkpoint_step_from_name(p.name)
            if step >= 0:
                candidates.append(p)
    return sorted(candidates, key=lambda p: checkpoint_step_from_name(p.name), reverse=True)



def _try_load_state(path: Path, device: torch.device) -> Optional[Dict[str, Any]]:
    """
    Load training_state.pt from a checkpoint directory.
    Returns None (instead of raising) when the checkpoint is missing, incomplete, or corrupted.
    Required keys: model_state_dict, optimizer, scheduler.
    """
    state_path = path / "training_state.pt"
    if not state_path.exists():
        return None
    try:
        state = torch.load(state_path, map_location=device)
    except Exception as exc:
        print(f"  [resume] corrupt checkpoint {path.name}: {exc} — skipping")
        return None
    if any(k not in state for k in ("model_state_dict", "optimizer", "scheduler")):
        print(f"  [resume] incomplete checkpoint {path.name} — skipping")
        return None
    return state



def _list_remote_checkpoint_names(repo_id: str, token: str) -> List[str]:
    """
    Return Hub checkpoint directory names sorted newest-first.
    Returns [] silently on any Hub error (repo not found, bad token, network, etc.).
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        return []

    import re as _re
    _CKPT_RE = _re.compile(r"^checkpoint-(\d+)$")

    try:
        api = HfApi(token=token)
        repo_files = list(api.list_repo_files(repo_id=repo_id, token=token))
    except Exception:
        return []

    names = set()
    for f in repo_files:
        top = Path(f).parts[0] if Path(f).parts else ""
        if _CKPT_RE.match(top):
            names.add(top)

    return sorted(names, key=lambda n: int(_CKPT_RE.match(n).group(1)), reverse=True)



def _download_checkpoint_from_hub(
    checkpoint_name: str,
    output_dir: Path,
    repo_id: str,
    token: str,
) -> Optional[Path]:
    """
    Download a single checkpoint directory from the Hub into output_dir.
    Returns the local path on success, None on any failure.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None

    local_dir = output_dir / checkpoint_name
    if local_dir.exists():
        shutil.rmtree(local_dir, ignore_errors=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_dir),
            token=token,
            force_download=True,
            allow_patterns=[f"{checkpoint_name}/*"],
        )
    except Exception as exc:
        print(f"  [hub]  download failed for {checkpoint_name}: {exc}")
        return None

    return local_dir if local_dir.exists() else None



def load_latest_checkpoint(
    output_dir: Path,
    model: BaselineTRMMamba,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler,
    ema: ModelEMA,
    device: torch.device,
    cfg: PretrainConfig,
    hf_token: Optional[str],
    verbose: bool = True,
) -> Tuple[int, int, int, int]:
    """
    Resume from the newest valid checkpoint using this priority order:
      1. output_dir is itself a checkpoint dir (direct path given via --resume_from)
      2. Newest finalized local checkpoint under output_dir (skip corrupted, try next)
      3. Newest checkpoint on the Hub (download, then load; only if no local found)

    Returns (step, epoch, chunks_in_epoch, tokens_processed).
    Returns (0, 0, 0, 0) and prints a message when no checkpoint is found.
    """

    def _restore(state: Dict[str, Any], label: str) -> Tuple[int, int, int, int]:
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        if state.get("ema"):
            ema.load_state_dict(state["ema"])
        step  = int(state.get("step", 0))
        epoch = int(state.get("epoch", 0))
        chunks = int(state.get("chunks_in_epoch", 0))
        tokens = int(state.get("tokens_processed", 0))
        if verbose:
            print(
                f"  [resume] {label}  "
                f"step={step}  epoch={epoch}  tokens={tokens:,}  val_ce={state.get('val_ce')}"
            )
        return step, epoch, chunks, tokens

    # ── Case 1: output_dir IS the checkpoint directory ────────────────────────
    direct_state = _try_load_state(output_dir, device)
    if direct_state is not None:
        return _restore(direct_state, f"direct  {output_dir.name}")

    # ── Case 2: scan local checkpoints newest-first ───────────────────────────
    local_ckpts = _list_local_checkpoints(output_dir)
    for ckpt_dir in local_ckpts:
        state = _try_load_state(ckpt_dir, device)
        if state is not None:
            return _restore(state, f"local   {ckpt_dir.name}")

    # ── Case 3: Hub fallback (only if local is completely empty) ─────────────
    if cfg.push_to_hub and hf_token:
        if verbose:
            print(f"  [resume] no local checkpoints found; checking Hub ({cfg.hf_repo_id}) ...")
        for ckpt_name in _list_remote_checkpoint_names(cfg.hf_repo_id, hf_token):
            if verbose:
                print(f"  [hub]  downloading {ckpt_name} ...")
            downloaded = _download_checkpoint_from_hub(ckpt_name, output_dir, cfg.hf_repo_id, hf_token)
            if downloaded is None:
                continue
            state = _try_load_state(downloaded, device)
            if state is not None:
                return _restore(state, f"hub     {ckpt_name}")
            if verbose:
                print(f"  [hub]  {ckpt_name} was downloaded but could not be loaded — skipping")

    if verbose:
        print("  [resume] No checkpoint found — starting from scratch.")
    return 0, 0, 0, 0



def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser with one flag for every PretrainConfig field."""
    defaults = PretrainConfig()
    parser = argparse.ArgumentParser(
        description="Stage 1 FineWeb-Edu pre-training for BaselineTRMMamba",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset_name", default=defaults.dataset_name)
    parser.add_argument("--dataset_config", default=defaults.dataset_config)
    parser.add_argument("--tokenizer_name", default=defaults.tokenizer_name)
    parser.add_argument("--token_budget", type=int, default=defaults.token_budget)
    parser.add_argument("--shuffle_buffer", type=int, default=defaults.shuffle_buffer)
    parser.add_argument("--num_epochs", type=int, default=defaults.num_epochs)

    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default=defaults.preset)
    parser.add_argument("--chunk_size", type=int, default=defaults.chunk_size)

    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--min_lr_ratio", type=float, default=defaults.min_lr_ratio)
    parser.add_argument("--warmup_steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--beta1", type=float, default=defaults.beta1)
    parser.add_argument("--beta2", type=float, default=defaults.beta2)
    parser.add_argument("--adam_eps", type=float, default=defaults.adam_eps)
    parser.add_argument("--max_grad_norm", type=float, default=defaults.max_grad_norm)

    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--grad_accum", type=int, default=defaults.grad_accum)

    parser.add_argument("--ema_decay", type=float, default=defaults.ema_decay)

    parser.add_argument("--output_dir", default=defaults.output_dir)
    parser.add_argument("--save_every", type=int, default=defaults.save_every)
    parser.add_argument("--keep_last", type=int, default=defaults.keep_last)

    parser.add_argument("--push_to_hub", action="store_true", default=defaults.push_to_hub)
    parser.add_argument("--hf_repo_id", default=defaults.hf_repo_id)
    parser.add_argument("--hf_token", default=defaults.hf_token)

    parser.add_argument("--log_every", type=int, default=defaults.log_every)
    parser.add_argument("--val_tokens", type=int, default=defaults.val_tokens)
    parser.add_argument("--val_every", type=int, default=defaults.val_every)
    parser.add_argument("--gen_every", type=int, default=defaults.gen_every)
    parser.add_argument("--gen_max_tokens", type=int, default=defaults.gen_max_tokens)
    parser.add_argument("--spike_threshold", type=float, default=defaults.spike_threshold)

    parser.add_argument("--seed", type=int, default=defaults.seed)

    parser.add_argument("--wandb_project", default=defaults.wandb_project)
    parser.add_argument("--wandb_run_name", default=defaults.wandb_run_name)
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default=defaults.wandb_mode,
    )

    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser



def args_to_config(args: argparse.Namespace) -> PretrainConfig:
    """Convert parsed CLI arguments into a PretrainConfig instance."""
    cfg_kwargs = {dc_field.name: getattr(args, dc_field.name) for dc_field in fields(PretrainConfig)}
    cfg = PretrainConfig(**cfg_kwargs)
    if args.dry_run:
        cfg.token_budget = 10_000_000
        cfg.val_every = 50
        cfg.gen_every = 50
        cfg.save_every = 100
        cfg.wandb_mode = "disabled"
    return cfg



def build_optimizer(model: BaselineTRMMamba, cfg: PretrainConfig) -> AdamW:
    """Build AdamW with separate decay and no-decay parameter groups."""
    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    base_kwargs = dict(lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps)
    try:
        return AdamW(
            [
                {"params": decay_params, "weight_decay": cfg.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            fused=torch.cuda.is_available(),
            **base_kwargs,
        )
    except TypeError:
        return AdamW(
            [
                {"params": decay_params, "weight_decay": cfg.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            **base_kwargs,
        )



def init_wandb(cfg: PretrainConfig):
    """Initialize wandb with a sanitized config dictionary when enabled."""
    if cfg.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("[warn] wandb not installed - logging to stdout only.")
        return None

    wandb_cfg = {k: v for k, v in asdict(cfg).items() if k != "hf_token"}
    return wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        mode=cfg.wandb_mode,
        config=wandb_cfg,
    )



def maybe_log_wandb(step: int, payload: Dict[str, Any], wandb_run) -> None:
    """Write one metrics payload to wandb if a run is active."""
    if wandb_run is None:
        return
    import wandb

    wandb.log(payload, step=step)



def print_header(
    cfg: PretrainConfig,
    model_config: BaselineConfig,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    total_steps: int,
    world_size: int,
    local_batch_size: int,
) -> None:
    """Print a compact startup summary for the pre-training run."""
    print()
    print("=" * 72)
    print("  Stage 1 Pre-training - Project Ouroboros")
    print("=" * 72)
    print(f"  dataset          : {cfg.dataset_name} / {cfg.dataset_config}")
    print(f"  tokenizer        : {cfg.tokenizer_name}  vocab={len(tokenizer):,}")
    print(f"  preset           : {cfg.preset}")
    print(
        f"  model            : d_model={model_config.d_model}  groups={model_config.n_groups}  "
        f"heads={model_config.n_heads}/{model_config.n_kv_heads}"
    )
    print(f"  chunk_size       : {cfg.chunk_size}")
    print(f"  batch x accum    : {cfg.batch_size} global x {cfg.grad_accum}")
    if world_size > 1:
        print(f"  world_size       : {world_size}  (DDP auto-enabled)")
        print(f"  per_gpu_batch    : {local_batch_size}")
    print(f"  tokens / step    : {effective_tokens_per_step(cfg):,}")
    print(f"  token_budget     : {cfg.token_budget:,}")
    print(f"  total_steps      : {total_steps:,}")
    print(f"  dtype            : {dtype}")
    print(f"  device           : {device}")
    print(f"  output_dir       : {cfg.output_dir}")
    print(f"  push_to_hub      : {cfg.push_to_hub}")
    print("=" * 72)
    print()



def smoke_test_20_steps() -> None:
    """Run a 20-step offline smoke test that exercises training, val, and checkpointing."""
    import tempfile

    class ToyTokenizer:
        """Tiny whitespace tokenizer used to test the training pipeline offline."""

        def __init__(self) -> None:
            self.eos_token = "<eos>"
            self.eos_token_id = 4_095
            self.pad_token = self.eos_token
            self.pad_token_id = self.eos_token_id
            self._vocab: Dict[str, int] = {}
            self._next_id = 10

        def __len__(self) -> int:
            return 4_096

        def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
            _ = add_special_tokens
            ids: List[int] = []
            for token in text.replace("\n", " \n ").split():
                if token not in self._vocab:
                    self._vocab[token] = self._next_id
                    self._next_id += 1
                ids.append(self._vocab[token])
            return ids

        def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
            inverse = {v: k for k, v in self._vocab.items()}
            pieces: List[str] = []
            for idx in ids:
                if skip_special_tokens and idx == self.eos_token_id:
                    continue
                pieces.append(inverse.get(idx, f"tok{idx}"))
            return " ".join(pieces)

    class FakeIterableDataset:
        """Simple iterable dataset with a deterministic shuffle method for smoke tests."""

        def __init__(self, items: List[Dict[str, str]]) -> None:
            self.items = list(items)

        def shuffle(self, buffer_size: int, seed: int):
            _ = buffer_size
            rng = random.Random(seed)
            shuffled = list(self.items)
            rng.shuffle(shuffled)
            return FakeIterableDataset(shuffled)

        def __iter__(self):
            return iter(self.items)

    class FakeMamba(torch.nn.Module):
        """Small feed-forward stand-in for mamba_ssm.Mamba during offline smoke tests."""

        def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
            super().__init__()
            hidden = d_model * max(int(expand), 1)
            _ = d_state, d_conv
            self.in_proj = torch.nn.Linear(d_model, hidden, bias=False)
            self.out_proj = torch.nn.Linear(hidden, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.out_proj(F.silu(self.in_proj(x)))

    toy_tokenizer = ToyTokenizer()
    smoke_cfg = PretrainConfig(
        token_budget=20 * 2 * 2 * 32,
        shuffle_buffer=16,
        num_epochs=1,
        chunk_size=32,
        batch_size=2,
        grad_accum=2,
        lr=5e-3,
        warmup_steps=2,
        output_dir="unused",
        save_every=10,
        keep_last=2,
        val_tokens=512,
        val_every=10,
        gen_every=10,
        gen_max_tokens=16,
        wandb_mode="disabled",
    )
    padded_vocab = math.ceil(len(toy_tokenizer) / 128) * 128
    model_config = BaselineConfig(
        vocab_size=padded_vocab,
        max_seq_len=smoke_cfg.chunk_size,
        d_model=32,
        n_groups=1,
        n_heads=4,
        n_kv_heads=2,
    )

    docs = []
    base_patterns = [
        "alpha beta gamma delta alpha beta gamma delta",
        "one two three four one two three four",
        "red blue green yellow red blue green yellow",
        "left right up down left right up down",
    ]
    for idx in range(80):
        body = " ".join([base_patterns[idx % len(base_patterns)]] * 40)
        docs.append({"text": f"doc {idx} {body}"})

    original_load_dataset = globals()["load_dataset"]
    globals()["load_dataset"] = lambda *args, **kwargs: FakeIterableDataset(docs)

    import baseline_trm_mamba as baseline_module

    original_mamba_available = getattr(baseline_module, "MAMBA_AVAILABLE", None)
    original_mamba = getattr(baseline_module, "Mamba", None)
    baseline_module.MAMBA_AVAILABLE = True
    baseline_module.Mamba = FakeMamba

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    set_seed(smoke_cfg.seed)

    try:
        token_stream = TokenStream(smoke_cfg, toy_tokenizer, toy_tokenizer.eos_token_id)
        token_stream.build_val_buffer()
        offset0 = token_stream.epoch_offset(0)
        if offset0 == 0:
            raise AssertionError("epoch 0 offset should be non-zero in the smoke test")
        print(f"[smoke] epoch_offset={offset0}")

        model = BaselineTRMMamba(model_config).to(device=device, dtype=dtype)
        model.train()
        optimizer = build_optimizer(model, smoke_cfg)
        scheduler = cosine_with_warmup(
            optimizer=optimizer,
            warmup_steps=smoke_cfg.warmup_steps,
            total_steps=20,
            min_lr_ratio=smoke_cfg.min_lr_ratio,
        )
        ema = ModelEMA(model, decay=smoke_cfg.ema_decay)
        scaler = None

        losses: List[float] = []
        train_iter = iter(token_stream.iter_train_chunks(epoch=0))
        optimizer.zero_grad(set_to_none=True)

        for step_idx in range(20):
            step_loss = 0.0
            for _ in range(smoke_cfg.grad_accum):
                micro_chunks = [next(train_iter) for _ in range(smoke_cfg.batch_size)]
                tokens = torch.stack(micro_chunks).to(device)
                with autocast_context(device, dtype):
                    logits = model(tokens)
                if torch.isnan(logits).any():
                    raise AssertionError("NaN detected in logits during smoke test")
                loss = pretrain_loss(logits, tokens)
                if torch.isnan(loss).any():
                    raise AssertionError("NaN detected in loss during smoke test")
                (loss / smoke_cfg.grad_accum).backward()
                step_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), smoke_cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)

            mean_loss = step_loss / smoke_cfg.grad_accum
            losses.append(mean_loss)
            if step_idx + 1 in (1, 10, 20):
                print(f"[smoke] step {step_idx + 1:2d}  loss={mean_loss:.4f}")

        expected = math.log(model_config.vocab_size)
        if not (expected - 1.0 <= losses[0] <= expected + 1.0):
            raise AssertionError(
                f"initial smoke loss {losses[0]:.4f} not close to ln(vocab)={expected:.4f}"
            )
        if not losses[-1] < losses[0]:
            raise AssertionError(
                f"smoke loss failed to decrease: step1={losses[0]:.4f} step20={losses[-1]:.4f}"
            )

        val_ce = compute_val_ce(
            model=model,
            ema=ema,
            token_stream=token_stream,
            device=device,
            dtype=dtype,
            vocab_size=model_config.vocab_size,
            batch_size=smoke_cfg.batch_size,
        )
        print(f"[smoke] val_ce computed: {val_ce:.4f}")

        with tempfile.TemporaryDirectory(prefix="stage1_smoke_") as tmp_root:
            ckpt_dir = save_checkpoint(
                output_dir=Path(tmp_root),
                step=20,
                epoch=0,
                chunks_in_epoch=20 * smoke_cfg.batch_size * smoke_cfg.grad_accum,
                tokens_processed=20 * effective_tokens_per_step(smoke_cfg),
                model=model,
                model_config=model_config,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                val_ce=val_ce,
                cfg=smoke_cfg,
                hf_token=None,
            )
            if ckpt_dir is None:
                raise AssertionError("checkpoint save unexpectedly returned None")

            state = torch.load(ckpt_dir / "training_state.pt", map_location="cpu")
            if "ema_backbone_state_dict" not in state:
                raise AssertionError("checkpoint missing ema_backbone_state_dict")
            if "lm_head.weight" not in state["ema_backbone_state_dict"]:
                raise AssertionError("checkpoint EMA shadow missing lm_head.weight alias")

            model2 = BaselineTRMMamba(model_config).to(device=device, dtype=dtype)
            optimizer2 = build_optimizer(model2, smoke_cfg)
            scheduler2 = cosine_with_warmup(
                optimizer=optimizer2,
                warmup_steps=smoke_cfg.warmup_steps,
                total_steps=20,
                min_lr_ratio=smoke_cfg.min_lr_ratio,
            )
            ema2 = ModelEMA(model2, decay=smoke_cfg.ema_decay)
            restored = load_latest_checkpoint(
                output_dir=Path(tmp_root),
                model=model2,
                optimizer=optimizer2,
                scheduler=scheduler2,
                scaler=None,
                ema=ema2,
                device=device,
                cfg=smoke_cfg,
                hf_token=None,
            )
            if restored[0] != 20:
                raise AssertionError(f"checkpoint round-trip restored wrong step: {restored[0]}")
            print("[smoke] checkpoint saved and reloaded cleanly")
    finally:
        globals()["load_dataset"] = original_load_dataset
        baseline_module.MAMBA_AVAILABLE = original_mamba_available
        baseline_module.Mamba = original_mamba

    print("[smoke] All checks passed - launching main training loop")



def _distributed_worker(local_rank: int, world_size: int, master_port: int, argv: List[str]) -> None:
    """Spawned DDP worker entrypoint for single-node multi-GPU training."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    raise SystemExit(run_training(argv))


def maybe_launch_multi_gpu(cfg: PretrainConfig, argv: List[str]) -> bool:
    """Auto-launch one DDP worker per visible CUDA device when that is usable."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return False
    if not torch.cuda.is_available():
        return False
    world_size = torch.cuda.device_count()
    if world_size < 2:
        return False
    if cfg.batch_size < world_size or (cfg.batch_size % world_size) != 0:
        print(
            f"[ddp] detected {world_size} CUDA devices, but batch_size={cfg.batch_size} "
            "cannot be split evenly across ranks; using the original single-process path."
        )
        return False

    per_gpu = cfg.batch_size // world_size
    print(
        f"[ddp] detected {world_size} CUDA devices; launching single-node DDP "
        f"with global batch_size={cfg.batch_size} ({per_gpu} per GPU)."
    )
    mp.spawn(
        _distributed_worker,
        nprocs=world_size,
        args=(world_size, find_free_port(), list(argv)),
        join=True,
    )
    return True


def run_training(argv: Optional[List[str]] = None) -> int:
    """Run the actual training job in either single-process or DDP mode."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = args_to_config(args)
    hf_token = resolve_hf_token(cfg)
    if cfg.push_to_hub and not hf_token:
        raise SystemExit("--push_to_hub requires --hf_token or HF_TOKEN/HUGGINGFACE_HUB_TOKEN")

    if cfg.preset not in PRESETS:
        raise SystemExit(f"Unknown preset {cfg.preset!r}. Expected one of {sorted(PRESETS)}")
    if cfg.grad_accum <= 0 or cfg.batch_size <= 0:
        raise SystemExit("batch_size and grad_accum must both be positive")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    distributed = world_size > 1
    process_group_owner = False

    if distributed and (cfg.batch_size < world_size or (cfg.batch_size % world_size) != 0):
        raise SystemExit(
            f"For DDP, batch_size must be divisible by world_size. "
            f"Got batch_size={cfg.batch_size}, world_size={world_size}."
        )

    if distributed and not dist_is_initialized():
        if not torch.cuda.is_available():
            raise SystemExit("DDP auto-launch expects CUDA GPUs, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        process_group_owner = True

    set_seed(cfg.seed)

    if distributed:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
    else:
        dtype = torch.float32

    local_batch_size = cfg.batch_size // world_size if distributed else cfg.batch_size
    wandb_run = init_wandb(cfg) if is_main_process(rank) else None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_name,
            use_fast=True,
        )
        if tokenizer.eos_token_id is None:
            if tokenizer.eos_token is None:
                raise SystemExit("Tokenizer has no EOS token; Stage 1 packing requires a document separator.")
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_config = BaselineConfig(
            vocab_size=math.ceil(len(tokenizer) / 128) * 128,
            max_seq_len=cfg.chunk_size,
            **PRESETS[cfg.preset],
        )
        assert model_config.max_seq_len == cfg.chunk_size, (
            f"chunk_size={cfg.chunk_size} must equal model max_seq_len={model_config.max_seq_len}"
        )

        total_steps = max(1, math.ceil(cfg.token_budget / max(effective_tokens_per_step(cfg), 1)))
        if is_main_process(rank):
            print_header(
                cfg=cfg,
                model_config=model_config,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                total_steps=total_steps,
                world_size=world_size,
                local_batch_size=local_batch_size,
            )

        try:
            raw_model = BaselineTRMMamba(model_config).to(device=device, dtype=dtype)
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc
        raw_model.train()

        n_params = count_parameters(raw_model)
        if is_main_process(rank):
            print(f"Model parameters : {n_params:,} ({n_params / 1e6:.1f} M)")
            print()

        token_stream = TokenStream(cfg, tokenizer, tokenizer.eos_token_id)
        token_stream.build_val_buffer(verbose=is_main_process(rank))

        optimizer = build_optimizer(raw_model, cfg)
        scheduler = cosine_with_warmup(
            optimizer=optimizer,
            warmup_steps=cfg.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=cfg.min_lr_ratio,
        )
        scaler = None
        if device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
        ema = ModelEMA(raw_model, decay=cfg.ema_decay)
        spike_monitor = SpikeMonitor(threshold=cfg.spike_threshold)

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if is_main_process(rank):
            cleanup_temporary_checkpoints(output_dir)

        resume_search = Path(args.resume_from) if args.resume_from else output_dir
        if distributed:
            if is_main_process(rank):
                step, start_epoch, chunks_in_epoch, tokens_processed = load_latest_checkpoint(
                    output_dir=resume_search,
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    device=device,
                    cfg=cfg,
                    hf_token=hf_token,
                    verbose=True,
                )
                dist.barrier()
            else:
                dist.barrier()
                step, start_epoch, chunks_in_epoch, tokens_processed = load_latest_checkpoint(
                    output_dir=resume_search,
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    device=device,
                    cfg=cfg,
                    hf_token=hf_token,
                    verbose=False,
                )
        else:
            step, start_epoch, chunks_in_epoch, tokens_processed = load_latest_checkpoint(
                output_dir=resume_search,
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                device=device,
                cfg=cfg,
                hf_token=hf_token,
                verbose=True,
            )

        train_model = raw_model
        if distributed:
            train_model = DDP(
                raw_model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )

        if tokens_processed >= cfg.token_budget:
            if is_main_process(rank):
                print(
                    f"[done] Resumed checkpoint already met the token budget: "
                    f"{tokens_processed:,} >= {cfg.token_budget:,}"
                )
            if distributed:
                dist.barrier()
            return 0

        col_hdr = (
            f"{'step':>7}  {'train_ce':>9}  {'val_ce':>9}  {'smth':>9}  "
            f"{'gnorm':>7}  {'lr':>9}  {'VRAM':>7}  {'tok/s':>9}"
        )
        if is_main_process(rank):
            print(col_hdr)
            print("-" * len(col_hdr))

        last_val_ce: Optional[float] = None
        last_mean_uwr = 0.0
        success_announced = False
        last_saved_step = step
        train_start = time.perf_counter()
        tokens_since_log = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = None
        if is_main_process(rank):
            pbar = tqdm(total=total_steps, initial=step, desc="Stage1", dynamic_ncols=True)
        current_epoch = start_epoch

        for epoch in range(start_epoch, cfg.num_epochs):
            current_epoch = epoch
            skip_chunks = chunks_in_epoch if epoch == start_epoch else 0
            epoch_offset = token_stream.epoch_offset(epoch)
            if is_main_process(rank):
                print(f"  epoch {epoch}  offset={epoch_offset}  skipping={skip_chunks} chunks")

            global_chunk_iter = token_stream.iter_train_chunks(epoch)
            chunk_iter = shard_train_chunk_iterator(global_chunk_iter, rank, world_size, skip_chunks)

            while True:
                accum_loss = 0.0
                step_ready = True

                for _ in range(cfg.grad_accum):
                    micro_chunks = take_local_micro_batch(chunk_iter, local_batch_size)
                    has_full_batch = int(len(micro_chunks) == local_batch_size)
                    if distributed:
                        # reason: all ranks must either advance one full micro-step together or stop together.
                        has_full_batch = distributed_min_int(has_full_batch, device)
                    if not has_full_batch:
                        step_ready = False
                        break

                    tokens = torch.stack(micro_chunks).to(device)
                    tokens_since_log += tokens.numel()

                    with autocast_context(device, dtype):
                        logits = train_model(tokens)
                    loss = pretrain_loss(logits, tokens)
                    if scaler is not None:
                        scaler.scale(loss / cfg.grad_accum).backward()
                    else:
                        (loss / cfg.grad_accum).backward()
                    accum_loss += loss.detach().item()

                if not step_ready:
                    optimizer.zero_grad(set_to_none=True)
                    break

                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.max_grad_norm))
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(raw_model)

                step += 1
                tokens_processed += effective_tokens_per_step(cfg)
                chunks_in_epoch += cfg.batch_size * cfg.grad_accum
                mean_loss = distributed_mean(accum_loss / cfg.grad_accum, device)
                grad_norm_log = distributed_mean(grad_norm, device)

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(ce=f"{mean_loss:.3f}", gnorm=f"{grad_norm_log:.3f}")

                if is_main_process(rank) and spike_monitor.update(step, mean_loss):
                    tqdm.write(
                        f"  [spike] step={step}  raw={mean_loss:.4f}  ema={spike_monitor.smoothed:.4f}"
                    )

                if step % cfg.log_every == 0 or step == 1:
                    global_tokens_since_log = distributed_sum_int(tokens_since_log, device)
                    cur_lr = scheduler.get_last_lr()[0]
                    if is_main_process(rank):
                        elapsed = max(time.perf_counter() - train_start, 1e-6)
                        tok_s = global_tokens_since_log / elapsed
                        val_str = f"{last_val_ce:.4f}" if last_val_ce is not None else "-"
                        tqdm.write(
                            f"{step:7d}  {mean_loss:9.4f}  {val_str:>9}  {spike_monitor.smoothed:9.4f}  "
                            f"{grad_norm_log:7.4f}  {cur_lr:9.2e}  {vram_gb(device):7.3f}  {tok_s:9.0f}"
                        )
                        maybe_log_wandb(
                            step,
                            {
                                "train/ce": mean_loss,
                                "train/ce_smooth": spike_monitor.smoothed,
                                "train/grad_norm": grad_norm_log,
                                "train/lr": cur_lr,
                                "train/tok_s": tok_s,
                                "train/vram_gb": vram_gb(device),
                                "train/tokens_processed": tokens_processed,
                                "train/world_size": world_size,
                            },
                            wandb_run,
                        )
                    train_start = time.perf_counter()
                    tokens_since_log = 0

                need_val = (step % cfg.val_every == 0) or (tokens_processed >= cfg.token_budget)
                if need_val:
                    if is_main_process(rank):
                        last_val_ce = compute_val_ce(
                            model=raw_model,
                            ema=ema,
                            token_stream=token_stream,
                            device=device,
                            dtype=dtype,
                            vocab_size=model_config.vocab_size,
                            batch_size=local_batch_size,
                        )
                        tqdm.write(f"  [val] step={step}  val_ce={last_val_ce:.4f}")
                        maybe_log_wandb(step, {"val/ce": last_val_ce}, wandb_run)
                        if last_val_ce < 3.0 and last_mean_uwr > 0.05 and not success_announced:
                            stage1_success_banner(step, cfg.preset)
                            success_announced = True
                    if distributed:
                        dist.barrier()

                need_gen = (step % cfg.gen_every == 0) or (tokens_processed >= cfg.token_budget)
                if need_gen:
                    if is_main_process(rank):
                        last_mean_uwr = run_generation_callback(
                            model=raw_model,
                            tokenizer=tokenizer,
                            device=device,
                            dtype=dtype,
                            step=step,
                            max_new_tokens=cfg.gen_max_tokens,
                            max_seq_len=model_config.max_seq_len,
                            wandb_run=wandb_run,
                        )
                        if last_val_ce is not None and last_val_ce < 3.0 and last_mean_uwr > 0.05 and not success_announced:
                            stage1_success_banner(step, cfg.preset)
                            success_announced = True
                    if distributed:
                        dist.barrier()

                need_save = step % cfg.save_every == 0
                if need_save:
                    if is_main_process(rank):
                        ckpt_path = save_checkpoint(
                            output_dir=output_dir,
                            step=step,
                            epoch=epoch,
                            chunks_in_epoch=chunks_in_epoch,
                            tokens_processed=tokens_processed,
                            model=raw_model,
                            model_config=model_config,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            ema=ema,
                            val_ce=last_val_ce,
                            cfg=cfg,
                            hf_token=hf_token,
                        )
                        if ckpt_path is not None:
                            last_saved_step = step
                    if distributed:
                        dist.barrier()

                if tokens_processed >= cfg.token_budget or step >= total_steps:
                    break

            if tokens_processed >= cfg.token_budget or step >= total_steps:
                break
            chunks_in_epoch = 0

        if pbar is not None:
            pbar.close()

        need_final_save = step != last_saved_step
        if need_final_save:
            if is_main_process(rank):
                ckpt_path = save_checkpoint(
                    output_dir=output_dir,
                    step=step,
                    epoch=current_epoch if cfg.num_epochs > 0 else 0,
                    chunks_in_epoch=chunks_in_epoch,
                    tokens_processed=tokens_processed,
                    model=raw_model,
                    model_config=model_config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    val_ce=last_val_ce,
                    cfg=cfg,
                    hf_token=hf_token,
                )
                if ckpt_path is not None:
                    last_saved_step = step
            if distributed:
                dist.barrier()

        if is_main_process(rank):
            spike_rate = len(spike_monitor.spikes) / max(step, 1)
            print()
            print("=" * 72)
            print("  Stage 1 complete")
            print("=" * 72)
            print(f"  optimizer steps : {step:,}")
            print(f"  tokens processed: {tokens_processed:,} / {cfg.token_budget:,}")
            print(f"  final val_ce    : {last_val_ce if last_val_ce is not None else 'n/a'}")
            print(f"  last mean UWR   : {last_mean_uwr:.4f}")
            print(f"  spike count     : {len(spike_monitor.spikes)} ({spike_rate:.2%} of steps)")
            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(f"  peak VRAM       : {peak:.2f} GiB")
            if distributed:
                print(f"  world_size      : {world_size}")
            if spike_rate > 0.10:
                print("  advice          : spike rate > 10% - try --shuffle_buffer 20000 and restart")
            elif tokens_processed < cfg.token_budget:
                print("  advice          : increase --num_epochs or lower --token_budget for this pass")
            elif last_val_ce is not None and last_val_ce >= 3.0:
                print("  advice          : extend to --token_budget 3000000000 if spikes are clean")
            print("=" * 72)

        if wandb_run is not None:
            import wandb

            wandb.finish()
        if distributed:
            dist.barrier()
        return 0
    finally:
        if process_group_owner and dist_is_initialized():
            dist.destroy_process_group()


def main(argv: Optional[List[str]] = None) -> int:
    """Parse CLI arguments and auto-launch DDP on multi-GPU Kaggle-style hosts."""
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    args = parser.parse_args(argv_list)
    cfg = args_to_config(args)
    if maybe_launch_multi_gpu(cfg, argv_list):
        return 0
    return run_training(argv_list)


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        smoke_test_20_steps()
    raise SystemExit(main())
