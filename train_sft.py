#!/usr/bin/env python3
"""
Stage 2 SFT Training Script - Project Ouroboros / SERF Framework
================================================================
Trains BaselineTRMMamba on Bespoke-Stratos-17k, or on a mixed Stage 2
instruction corpus, using standard next-token cross-entropy loss.

Key Stage 2 features:
  - Proper <think>...</think> target formatting.
  - Cosine LR schedule with linear warmup.
  - Exponential Moving Average (EMA) of weights for stable inference.
  - Periodic validation CE and generation callbacks.
  - Checkpoint saves resolved_backbone_config.json for inference.
  - Optional local-first Hugging Face Hub checkpoint sync.
  - Resume from direct checkpoint paths, local output dirs, or Hub fallback.

Stage 2 success criterion:
  Validation CE < 1.5 on answer tokens only AND generated text is semantically coherent
  (not just grammatical) on the test prompts at --gen_every.

Hardware presets (choose via --preset):
  nano   d_model=512   n_groups=1  heads=8/4   ~92M  -> T4 / quick sanity runs
  small  d_model=1024  n_groups=2  heads=16/8  ~270M -> T4 / Kaggle single GPU
  medium d_model=2048  n_groups=2  heads=16/8  ~760M -> dual T4 / larger runs

Install:
  pip install "causal-conv1d>=1.4.0" mamba-ssm --no-build-isolation
  pip install transformers datasets wandb tqdm huggingface_hub

Run:
  python train_sft.py --preset small --output_dir ./runs/stage2_small

Dry-run style sanity check:
  python train_sft.py --preset nano --max_samples 300 --max_steps 100 \
    --val_every 50 --gen_every 50 --wandb_mode disabled
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
except ImportError:
    sys.exit("PyTorch not found. Install: pip install torch")


from training_utils import (
    ModelEMA,
    autocast_context,
    build_adamw_optimizer,
    checkpoint_step_from_name,
    cleanup_temporary_checkpoints,
    cosine_with_warmup,
    download_checkpoint_from_hub,
    ema_scope,
    list_local_checkpoints,
    list_remote_checkpoint_names,
    pad_vocab_size,
    resolve_hf_token,
    set_seed,
    sync_checkpoint_to_hub,
    try_load_state,
    vram_gb,
)

if not torch.cuda.is_available():
    sys.exit(
        "No CUDA GPU found. mamba-ssm requires CUDA kernels.\n"
        "Run on Kaggle (T4/Dual-T4) or Google Colab."
    )

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
        "baseline_trm_mamba.py must be in the current working directory."
    )

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 SFT trainer for BaselineTRMMamba",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--preset",
        choices=["nano", "small", "medium"],
        default="nano",
        help="Model size preset. nano=92M, small=270M, medium=760M.",
    )
    parser.add_argument("--max_seq_len", type=int, default=1024)

    # Dataset
    parser.add_argument("--dataset_name", default="bespokelabs/Bespoke-Stratos-17k")
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap samples for dry-runs. None = use full dataset or full per-source cap.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.05,
        help="Fraction of data held out for validation CE.",
    )
    parser.add_argument(
        "--dataset_mix",
        default="stratos",
        choices=["stratos", "full"],
        help="'stratos' = Bespoke-Stratos-17k only. 'full' = Stratos + MetaMathQA + OpenHermes-2.5 + OpenR1-Math-220k + OpenR1-Code.",
    )

    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override total steps. -1 = derived from epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Micro-batch size per gradient-accumulation step.",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
        help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.1,
        help="LR at end of cosine decay = lr * min_lr_ratio.",
    )
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)

    # I/O
    parser.add_argument("--output_dir", default="runs/stage2")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--keep_last", type=int, default=3)
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Path to a checkpoint directory or parent directory to resume from.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push finalized checkpoints to the Hugging Face Hub.",
    )
    parser.add_argument("--hf_repo_id", default="WeirdRunner/Ouroboros")
    parser.add_argument(
        "--hf_token",
        default=None,
        help="HF token for Hub sync. Falls back to HF_TOKEN/HUGGINGFACE_HUB_TOKEN.",
    )

    # Monitoring
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=250)
    parser.add_argument("--gen_every", type=int, default=250)
    parser.add_argument("--gen_max_tokens", type=int, default=120)
    parser.add_argument("--spike_threshold", type=float, default=0.5)
    parser.add_argument(
        "--session_timeout_hours",
        type=float,
        default=12.0,
        help="Total wall-clock budget in hours. The trainer saves an emergency checkpoint before this expires.",
    )
    parser.add_argument(
        "--graceful_exit_buffer_minutes",
        type=float,
        default=15.0,
        help="Save checkpoint and exit this many minutes before session_timeout_hours.",
    )

    # wandb
    parser.add_argument("--wandb_project", default="ouroboros-serf-phase2")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )

    return parser.parse_args(argv)


PRESETS: Dict[str, Dict[str, Any]] = {
    "nano": dict(d_model=512, n_groups=1, n_heads=8, n_kv_heads=4),
    "small": dict(d_model=1024, n_groups=2, n_heads=16, n_kv_heads=8),
    "medium": dict(d_model=2048, n_groups=2, n_heads=16, n_kv_heads=8),
}

GEN_PROMPTS = [
    "What is 15 + 27?",
    "Write a Python function that returns the factorial of n.",
    "What is the capital of Japan?",
    "Explain what a neural network is in simple terms.",
    "Solve for x: 3x + 6 = 21.",
]


class SpikeMonitor:
    """Track a smoothed training-loss EMA and warn on sudden spikes."""

    def __init__(self, beta: float = 0.99, threshold: float = 0.5) -> None:
        self.beta = beta
        self.threshold = threshold
        self._ema: Optional[float] = None
        self.spikes: List[Tuple[int, float, float]] = []

    def update(self, step: int, loss: float) -> bool:
        if self._ema is None:
            self._ema = loss
            return False
        self._ema = self.beta * self._ema + (1.0 - self.beta) * loss
        bias_corrected = self._ema / (1.0 - self.beta ** (step + 1))
        is_spike = (loss - bias_corrected) > self.threshold
        if is_spike:
            self.spikes.append((step, loss, bias_corrected))
        return is_spike

    @property
    def smoothed(self) -> float:
        return self._ema if self._ema is not None else float("nan")


def dist_is_initialized() -> bool:
    """Return True when torch.distributed collectives are ready to use."""
    return dist.is_available() and dist.is_initialized()


def is_main_process(rank: int) -> bool:
    return rank == 0


def distributed_mean(value: float, device: torch.device) -> float:
    if not dist_is_initialized():
        return float(value)
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def distributed_sum_int(value: int, device: torch.device) -> int:
    if not dist_is_initialized():
        return int(value)
    tensor = torch.tensor([value], device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _distributed_worker(local_rank: int, world_size: int, master_port: int, argv: List[str]) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    run_training(argv)


def maybe_launch_multi_gpu(args: argparse.Namespace, argv: List[str]) -> bool:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return False
    if not torch.cuda.is_available():
        return False
    world_size = torch.cuda.device_count()
    if world_size < 2:
        return False
    if args.batch_size < world_size or (args.batch_size % world_size) != 0:
        print(
            f"[ddp] detected {world_size} CUDA devices, but batch_size={args.batch_size} "
            "cannot be split evenly across ranks; using the original single-process path."
        )
        return False

    per_gpu = args.batch_size // world_size
    print(
        f"[ddp] detected {world_size} CUDA devices; launching single-node DDP "
        f"with global batch_size={args.batch_size} ({per_gpu} per GPU)."
    )
    mp.spawn(
        _distributed_worker,
        nprocs=world_size,
        args=(world_size, find_free_port(), list(argv)),
        join=True,
    )
    return True


def pad_vocab(actual: int, multiple: int = 128) -> int:
    return math.ceil(actual / multiple) * multiple


def unique_word_ratio(text: str) -> float:
    words = text.split()
    return len(set(words)) / max(len(words), 1)


def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    """Return (n_correct, n_valid) for next-token predictions on supervised labels only."""
    with torch.no_grad():
        valid = labels != -100
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            return 0, 0
        preds = logits.argmax(dim=-1)
        n_correct = int((preds[valid] == labels[valid]).sum().item())
        return n_correct, n_valid


def sanitize_args_for_serialization(args: argparse.Namespace) -> Dict[str, Any]:
    """Return a JSON-safe args dict with secrets removed."""
    cfg = dict(vars(args))
    cfg.pop("hf_token", None)
    return cfg


def build_data_fingerprint(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the subset of config fields that define the Stage 2 data stream."""
    return {
        "dataset_name": cfg.get("dataset_name"),
        "dataset_mix": cfg.get("dataset_mix"),
        "tokenizer_name": cfg.get("tokenizer_name"),
        "max_samples": cfg.get("max_samples"),
        "max_seq_len": cfg.get("max_seq_len"),
        "val_fraction": cfg.get("val_fraction"),
        "seed": cfg.get("seed"),
    }


def data_fingerprint_mismatch(saved_cfg: Dict[str, Any], current_cfg: Dict[str, Any]) -> bool:
    """Return True when a resumed Stage 2 checkpoint targets a different dataset stream."""
    return build_data_fingerprint(saved_cfg) != build_data_fingerprint(current_cfg)


def _normalize_hf_stage_subdir(subdir: Optional[str]) -> str:
    return str(subdir or "").strip().strip("/")


def _stage_repo_checkpoint_path(checkpoint_name: str, subdir: Optional[str]) -> str:
    clean_subdir = _normalize_hf_stage_subdir(subdir)
    return f"{clean_subdir}/{checkpoint_name}" if clean_subdir else checkpoint_name


def _try_load_state_cpu(path: Path) -> Optional[Dict[str, Any]]:
    state_path = path / "training_state.pt"
    if not state_path.exists():
        return None
    try:
        return torch.load(state_path, map_location="cpu")
    except Exception as exc:
        print(f"  [resume] corrupt checkpoint {path.name}: {exc} - skipping")
        return None


def list_remote_stage_checkpoint_names(repo_id: str, token: str, subdir: Optional[str]) -> List[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []

    clean_subdir = _normalize_hf_stage_subdir(subdir)
    prefix_parts = PurePosixPath(clean_subdir).parts if clean_subdir else ()

    try:
        api = HfApi(token=token)
        repo_files = list(api.list_repo_files(repo_id=repo_id, token=token))
    except Exception:
        return []

    names: set[str] = set()
    for file_name in repo_files:
        parts = PurePosixPath(file_name).parts
        if clean_subdir:
            if parts[: len(prefix_parts)] != prefix_parts:
                continue
            rel_parts = parts[len(prefix_parts) :]
        else:
            rel_parts = parts
        if not rel_parts:
            continue
        top = rel_parts[0]
        if checkpoint_step_from_name(top) >= 0:
            names.add(top)
    return sorted(names, key=checkpoint_step_from_name, reverse=True)


def download_stage_checkpoint_from_hub(
    checkpoint_name: str,
    output_dir: Path,
    repo_id: str,
    token: str,
    subdir: Optional[str],
) -> Optional[Path]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None

    clean_subdir = _normalize_hf_stage_subdir(subdir)
    remote_path = _stage_repo_checkpoint_path(checkpoint_name, clean_subdir)
    local_dir = output_dir / clean_subdir / checkpoint_name if clean_subdir else output_dir / checkpoint_name
    if local_dir.exists():
        shutil.rmtree(local_dir, ignore_errors=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_dir),
            token=token,
            force_download=True,
            allow_patterns=[f"{remote_path}/*"],
        )
    except Exception as exc:
        print(f"  [hub]  download failed for {remote_path}: {exc}")
        return None

    return local_dir if local_dir.exists() else None


def sync_stage_checkpoint_to_hub(
    checkpoint_dir: Path,
    repo_id: str,
    token: str,
    subdir: Optional[str],
) -> bool:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[hub] huggingface_hub not installed - skipping Hub sync.")
        return False

    remote_name = checkpoint_dir.name[:-4] if checkpoint_dir.name.endswith(".tmp") else checkpoint_dir.name
    remote_path = _stage_repo_checkpoint_path(remote_name, subdir)
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, token=token, private=True, exist_ok=True)
        print(f"  [hub] uploading {remote_path} -> {repo_id} ...")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(checkpoint_dir),
            path_in_repo=remote_path,
            token=token,
            commit_message=f"Upload {remote_path}",
            run_as_future=False,
        )
        print(f"  [hub] uploaded  {remote_path}")
        return True
    except Exception as exc:
        print(f"  [hub] upload failed for {remote_path}: {exc}")
        return False


_THINK_RE = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL)
_SOLUTION_RE = re.compile(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL)
_CLEAN_RE = re.compile(
    r"<\|begin_of_thought\|>|<\|end_of_thought\|>|"
    r"<\|begin_of_solution\|>|<\|end_of_solution\|>|"
    r"<think>|</think>"
)


def _parse_assistant_blob(assistant_blob: str) -> Tuple[str, str]:
    """Parse assistant text into (reasoning, answer), preserving existing <think> traces when present."""
    assistant_blob = str(assistant_blob or "").strip()
    if not assistant_blob:
        return "", ""

    think_match = _THINK_RE.search(assistant_blob)
    solution_match = _SOLUTION_RE.search(assistant_blob)

    if think_match:
        reasoning = think_match.group(1).strip()
        answer = solution_match.group(1).strip() if solution_match else assistant_blob[think_match.end():].strip()
        return reasoning.strip(), answer.strip()

    if "<think>" in assistant_blob and "</think>" in assistant_blob:
        parts = assistant_blob.split("</think>", 1)
        reasoning = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
        return reasoning.strip(), answer.strip()

    return "", _CLEAN_RE.sub("", assistant_blob).strip()


def _extract_chat_pair(turns: Any) -> Tuple[str, str]:
    """Extract the first user/human prompt and first assistant/gpt reply from a chat-style list."""
    question = ""
    assistant_blob = ""
    for turn in (turns or []):
        role = str(turn.get("role") or turn.get("from") or "").lower().strip()
        value = str(turn.get("content") or turn.get("value") or "").strip()
        if role in {"user", "human"} and value and not question:
            question = value
        elif role in {"assistant", "gpt"} and value and not assistant_blob:
            assistant_blob = value
    return question, assistant_blob


def _pick_openr1_math_generation(example: Dict[str, Any]) -> str:
    """Choose one high-quality OpenR1 math generation, preferring verified traces when available."""
    generations = example.get("generations") or []
    if not generations:
        return ""

    math_verify = list(example.get("correctness_math_verify") or [])
    llama_verify = list(example.get("correctness_llama") or [])
    for idx, generation in enumerate(generations):
        is_math_ok = idx < len(math_verify) and bool(math_verify[idx])
        is_llama_ok = idx < len(llama_verify) and bool(llama_verify[idx])
        if is_math_ok or is_llama_ok:
            return str(generation).strip()
    return str(generations[0]).strip()


def _extract_bespoke(example: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract (question, reasoning_chain, final_answer) from a Bespoke-Stratos row."""
    question, assistant_blob = _extract_chat_pair(example.get("conversations"))
    if not question or not assistant_blob:
        return "", "", ""
    reasoning, answer = _parse_assistant_blob(assistant_blob)
    return question.strip(), reasoning.strip(), answer.strip()


def _extract_openr1_math(example: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract (question, reasoning, answer) from OpenR1-Math-220k."""
    question, assistant_blob = _extract_chat_pair(example.get("messages"))
    if not question:
        question = str(example.get("problem") or example.get("question") or "").strip()

    generation = _pick_openr1_math_generation(example)
    if generation:
        assistant_blob = generation

    reasoning = ""
    answer = ""
    if assistant_blob:
        reasoning, answer = _parse_assistant_blob(assistant_blob)

    if not reasoning and example.get("solution"):
        reasoning = str(example.get("solution") or "").strip()
    if not answer and example.get("answer"):
        answer = str(example.get("answer") or "").strip()
    elif not answer and not reasoning and example.get("solution"):
        answer = str(example.get("solution") or "").strip()

    return question.strip(), reasoning.strip(), answer.strip()


def _extract_openr1_code(example: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract (question, reasoning, answer) from OpenR1 code SFT data."""
    question, assistant_blob = _extract_chat_pair(example.get("messages"))
    if not question:
        question = str(example.get("prompt") or "").strip()

    if not question:
        title = str(example.get("title") or "").strip()
        description = str(example.get("description") or "").strip()
        if title or description:
            question = f"{title}\n\n{description}".strip()

    if not assistant_blob:
        assistant_blob = str(example.get("generation") or example.get("solution") or example.get("editorial") or "").strip()

    if not question or not assistant_blob:
        return "", "", ""

    reasoning, answer = _parse_assistant_blob(assistant_blob)
    return question.strip(), reasoning.strip(), answer.strip()


def _format_training_text(question: str, reasoning: str, answer: str, eos: str) -> str:
    """Build the exact Stage 2 target format taught to the model."""
    if reasoning:
        return (
            f"User: {question}\n\n"
            f"Assistant: <think>\n{reasoning}\n</think>\n"
            f"{answer}{eos}"
        )
    return f"User: {question}\n\nAssistant: {answer}{eos}"

def _build_prompt_prefix(question: str, reasoning: str) -> str:
    """Build the non-supervised prompt prefix for one SFT sample.

    Stage 2 masks only the user prompt and the bare assistant header so the model
    is explicitly trained to emit the opening <think> tag when reasoning is present.
    """
    _ = reasoning
    return f"User: {question}\n\nAssistant: "


def _build_sft_sample(
    tokenizer,
    question: str,
    reasoning: str,
    answer: str,
    eos: str,
    max_seq_len: int,
) -> Optional[Dict[str, Any]]:
    """Tokenize one SFT sample, skipping rows that overflow max_seq_len or have no supervised target."""
    text = _format_training_text(question, reasoning, answer, eos)
    prefix_text = _build_prompt_prefix(question, reasoning)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 4 or len(ids) > max_seq_len:
        return None

    prompt_len = len(tokenizer.encode(prefix_text, add_special_tokens=False))
    if prompt_len >= len(ids):
        return None

    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "prompt_len": prompt_len,
    }


def _build_sft_sample_truncated(
    tokenizer,
    question: str,
    reasoning: str,
    answer: str,
    eos: str,
    max_seq_len: int,
) -> Optional[Dict[str, Any]]:
    """Like _build_sft_sample but truncates reasoning instead of skipping long examples."""
    if not reasoning:
        return _build_sft_sample(tokenizer, question, reasoning, answer, eos, max_seq_len)

    prefix_text = f"User: {question}\n\nAssistant: "
    answer_text = f"{answer}{eos}"
    think_open_text = "<think>\n"
    think_close_text = "\n</think>\n"

    q_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    a_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    open_ids = tokenizer.encode(think_open_text, add_special_tokens=False)
    close_ids = tokenizer.encode(think_close_text, add_special_tokens=False)
    r_ids = tokenizer.encode(reasoning, add_special_tokens=False)

    overhead = len(open_ids) + len(close_ids)
    base_len = len(q_ids) + len(a_ids)

    if base_len > max_seq_len:
        return None

    budget = max_seq_len - base_len - overhead
    if budget >= len(r_ids):
        return _build_sft_sample(tokenizer, question, reasoning, answer, eos, max_seq_len)
    if budget <= 0:
        return _build_sft_sample(tokenizer, question, "", answer, eos, max_seq_len)

    truncated_r_ids = r_ids[:budget] if random.random() < 0.5 else r_ids[-budget:]
    truncated_reasoning = tokenizer.decode(truncated_r_ids, skip_special_tokens=False)
    return _build_sft_sample(tokenizer, question, truncated_reasoning, answer, eos, max_seq_len)


def load_and_tokenize(
    dataset_name: str,
    tokenizer,
    max_samples: Optional[int],
    max_seq_len: int,
) -> List[Dict[str, Any]]:
    """Load one dataset, format rows, tokenize, and filter invalid or overlength samples."""
    print(f"Loading {dataset_name} ...")
    raw = load_dataset(dataset_name, split="train")
    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    eos = tokenizer.eos_token or "<|endoftext|>"
    samples: List[Dict[str, Any]] = []
    skipped_invalid = 0

    for ex in tqdm(raw, desc="Formatting + tokenizing", leave=False):
        q, r, a = _extract_bespoke(ex)
        if not q or not a:
            skipped_invalid += 1
            continue

        sample = _build_sft_sample_truncated(tokenizer, q, r, a, eos, max_seq_len)
        if sample is None:
            skipped_invalid += 1
            continue

        samples.append(sample)

    print(
        f"  {len(samples)} samples kept ({skipped_invalid} skipped as invalid/too-short). "
        f"Reasoning truncated where needed to fit max_seq_len={max_seq_len}."
    )
    return samples


def _extract_metamath(example: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract (question, reasoning, answer) from MetaMathQA."""
    question = str(example.get("original_question") or example.get("query") or "").strip()
    answer = str(example.get("response") or example.get("output") or "").strip()
    return question, "", answer


def _extract_openhermes(example: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract (question, reasoning, answer) from OpenHermes-2.5 conversations."""
    question = ""
    answer = ""
    for turn in (example.get("conversations") or []):
        role = str(turn.get("from", "")).lower().strip()
        value = str(turn.get("value", "")).strip()
        if role == "human" and not question:
            question = value
        elif role == "gpt" and not answer:
            answer = value
    return question, "", answer


def _load_first_available_dataset(
    candidates: List[Tuple[str, Optional[str]]],
    split: str,
) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    """Load the first available dataset/config pair and report the resolved source label."""
    for ds_name, config_name in candidates:
        try:
            if config_name:
                dataset = load_dataset(ds_name, config_name, split=split)
                label = f"{ds_name}[{config_name}]"
            else:
                dataset = load_dataset(ds_name, split=split)
                label = ds_name
            return dataset, ds_name, label
        except Exception:
            continue
    return None, None, None


def load_mixed_dataset(
    tokenizer,
    max_samples_per_source: Optional[int],
    max_seq_len: int,
) -> List[Dict[str, Any]]:
    """Load the full Stage 2 mix with Stratos, MetaMath, OpenHermes, OpenR1-Math, and OpenR1-Code."""
    sources = [
        {
            "name": "bespokelabs/Bespoke-Stratos-17k",
            "candidates": [("bespokelabs/Bespoke-Stratos-17k", None)],
            "split": "train",
            "extractor": _extract_bespoke,
            "ratio": 0.30,
        },
        {
            "name": "meta-math/MetaMathQA",
            "candidates": [("meta-math/MetaMathQA", None)],
            "split": "train",
            "extractor": _extract_metamath,
            "ratio": 0.20,
        },
        {
            "name": "teknium/OpenHermes-2.5",
            "candidates": [("teknium/OpenHermes-2.5", None)],
            "split": "train",
            "extractor": _extract_openhermes,
            "ratio": 0.15,
        },
        {
            "name": "open-r1/OpenR1-Math-220k",
            "candidates": [
                ("open-r1/OpenR1-Math-220k", "default"),
                ("open-r1/OpenR1-Math-220k", None),
            ],
            "split": "train",
            "extractor": _extract_openr1_math,
            "ratio": 0.20,
        },
        {
            "name": "open-r1/OpenR1-Code",
            "candidates": [
                ("open-r1/OpenR1-Code", None),
                ("open-r1/codeforces-cots", "solutions_w_editorials_py"),
                ("open-r1/codeforces-cots", "solutions_py"),
                ("open-r1/codeforces-cots", "solutions"),
            ],
            "split": "train",
            "extractor": _extract_openr1_code,
            "ratio": 0.15,
        },
    ]
    eos = tokenizer.eos_token or "<|endoftext|>"
    source_specs: List[Dict[str, Any]] = []

    for source in sources:
        print(f"  Loading {source['name']} ...")
        raw, resolved_name, resolved_label = _load_first_available_dataset(source["candidates"], source["split"])
        if raw is None or resolved_name is None or resolved_label is None:
            print(f"  [warn] Could not load {source['name']} from any configured candidate - skipping.")
            continue

        available = len(raw)
        if max_samples_per_source is not None:
            available = min(available, max_samples_per_source)
            raw = raw.select(range(available))

        print(f"    resolved -> {resolved_label}")
        source_specs.append(
            {
                "name": source["name"],
                "resolved_name": resolved_name,
                "resolved_label": resolved_label,
                "dataset": raw,
                "extractor": source["extractor"],
                "ratio": source["ratio"],
                "available": available,
            }
        )

    if not source_specs:
        return []

    weight_sum = sum(spec["ratio"] for spec in source_specs)
    for spec in source_specs:
        spec["weight"] = spec["ratio"] / weight_sum

    # reason: anchor the actual mix to the scarcest available source after any per-source cap.
    target_total = max(1, int(min(spec["available"] / spec["weight"] for spec in source_specs)))
    raw_counts = {spec["name"]: target_total * spec["weight"] for spec in source_specs}
    target_counts = {
        spec["name"]: min(spec["available"], int(math.floor(raw_counts[spec["name"]])))
        for spec in source_specs
    }

    remaining = target_total - sum(target_counts.values())
    if remaining > 0:
        by_fraction = sorted(
            source_specs,
            key=lambda spec: raw_counts[spec["name"]] - target_counts[spec["name"]],
            reverse=True,
        )
        while remaining > 0:
            progressed = False
            for spec in by_fraction:
                name = spec["name"]
                if target_counts[name] < spec["available"]:
                    target_counts[name] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                break

    all_samples: List[Dict[str, Any]] = []
    actual_counts: Dict[str, int] = {}

    for spec in source_specs:
        ds_name = spec["name"]
        extractor = spec["extractor"]
        target = target_counts[ds_name]
        kept = 0
        skipped_invalid = 0

        for ex in tqdm(spec["dataset"], desc=f"  {spec['resolved_label'].split('/')[-1]}", leave=False):
            q, r, a = extractor(ex)
            if not q or not a:
                skipped_invalid += 1
                continue

            sample = _build_sft_sample_truncated(tokenizer, q, r, a, eos, max_seq_len)
            if sample is None:
                skipped_invalid += 1
                continue

            sample["source"] = ds_name
            sample["resolved_source"] = spec["resolved_label"]
            all_samples.append(sample)
            kept += 1
            if kept >= target:
                break

        actual_counts[ds_name] = kept
        print(
            f"    kept {kept} / target {target} samples "
            f"(invalid={skipped_invalid}; reasoning truncated where needed)."
        )

    random.shuffle(all_samples)
    print(f"  Total mixed samples: {len(all_samples)}")
    if actual_counts:
        pretty_counts = ", ".join(
            f"{name.split('/')[-1]}={count}" for name, count in actual_counts.items()
        )
        print(f"  Mix counts: {pretty_counts}")
    return all_samples


def split_train_val_samples(
    all_samples: List[Dict[str, Any]],
    seed: int,
    val_fraction: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically split tokenized SFT samples into train and validation sets."""
    if len(all_samples) < 2:
        raise ValueError("Need at least 2 samples to create non-empty train and validation splits.")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(all_samples), generator=generator).tolist()
    n_val = min(len(all_samples) - 1, max(1, int(len(all_samples) * val_fraction)))
    val_idx = set(perm[:n_val])
    train_samples = [sample for idx, sample in enumerate(all_samples) if idx not in val_idx]
    val_samples = [sample for idx, sample in enumerate(all_samples) if idx in val_idx]
    return train_samples, val_samples


def collate(samples: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad a micro-batch to the longest example and mask padding in labels."""
    max_len = max(sample["input_ids"].size(0) for sample in samples)
    batch_size = len(samples)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for idx, sample in enumerate(samples):
        ids = sample["input_ids"]
        length = ids.size(0)
        input_ids[idx, :length] = ids
        pl = min(sample.get("prompt_len", 0), length)
        labels[idx, pl:length] = ids[pl:]
        mask[idx, :length] = True

    return {"input_ids": input_ids, "attention_mask": mask, "labels": labels}


@torch.no_grad()
def run_generation_callback(
    model: BaselineTRMMamba,
    ema: ModelEMA,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    step: int,
    max_new_tokens: int,
    max_seq_len: int,
    wandb_run=None,
) -> None:
    """Run greedy decoding on the fixed prompts using EMA weights."""
    model.eval()
    print(f"\n  -- Generation @ step {step} (EMA weights) --")
    wandb_log: Dict[str, str] = {}
    mean_uwr = 0.0

    with ema_scope(model, ema):
        for prompt in GEN_PROMPTS:
            prefix = f"User: {prompt}\n\nAssistant: "
            ids = torch.tensor(
                tokenizer.encode(prefix, add_special_tokens=False),
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

            output_text = tokenizer.decode(generated, skip_special_tokens=True)
            uwr = unique_word_ratio(output_text)
            mean_uwr += uwr
            display = output_text[:180].replace("\n", " ")
            print(f"  Q: {prompt}")
            print(f"  A: {display}")
            print(f"     uwr={uwr:.3f}")
            wandb_log[f"gen/{prompt[:40]}"] = output_text

    mean_uwr /= max(len(GEN_PROMPTS), 1)
    print(f"  Mean UWR: {mean_uwr:.3f}\n")

    if wandb_run is not None:
        import wandb

        wandb.log({"gen/mean_uwr": mean_uwr, **wandb_log}, step=step)

    model.train()


@torch.no_grad()
def compute_val_metrics(
    model: BaselineTRMMamba,
    ema: ModelEMA,
    val_samples: List[Dict[str, Any]],
    pad_id: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    vocab_size: int,
) -> Tuple[float, float]:
    """Compute answer-only validation CE and token accuracy using EMA weights."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with ema_scope(model, ema):
        for start in range(0, len(val_samples), batch_size):
            batch_samples = val_samples[start : start + batch_size]
            batch = collate(batch_samples, pad_id)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast_context(device, dtype):
                logits = model(input_ids, attention_mask=attn_mask)

            shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size).float()
            shift_labels = labels[:, 1:].contiguous().view(-1)
            valid = shift_labels != -100
            if valid.any():
                total_loss += F.cross_entropy(
                    shift_logits[valid], shift_labels[valid], reduction="sum"
                ).item()
                correct, count = masked_token_accuracy(shift_logits, shift_labels)
                total_correct += correct
                total_tokens += count

    model.train()
    val_ce = total_loss / max(total_tokens, 1)
    val_acc = total_correct / max(total_tokens, 1)
    return val_ce, val_acc


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: BaselineTRMMamba,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    config: BaselineConfig,
    val_ce: Optional[float],
    keep_last: int,
    epoch: int,
    samples_seen: int,
    sft_config: Dict[str, Any],
    push_to_hub: bool = False,
    hf_repo_id: str = "WeirdRunner/Ouroboros",
    hf_token: Optional[str] = None,
    hf_stage_subdir: str = "runs/stage2",
) -> Optional[Path]:
    """Write a Stage 2 checkpoint locally first, then try a Hub push."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"checkpoint-{step:07d}"
    final_dir = output_dir / ckpt_name
    tmp_dir = output_dir / f"{ckpt_name}.tmp"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    shadow = dict(ema.shadow)
    if config.tie_embeddings and "lm_head.weight" not in shadow and "token_embedding.weight" in shadow:
        shadow["lm_head.weight"] = shadow["token_embedding.weight"]

    cfg_dict = {k: v for k, v in (sft_config or {}).items() if k != "hf_token"}
    state = {
        "stage": "sft",
        "step": step,
        "epoch": epoch,
        "samples_seen": samples_seen,
        "val_ce": val_ce,
        "model_state_dict": model.state_dict(),
        "ema_backbone_state_dict": shadow,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "ema": ema.state_dict(),
        "backbone_config": asdict(config),
        "sft_config": cfg_dict,
        "data_fingerprint": build_data_fingerprint(cfg_dict),
    }

    try:
        torch.save(state, tmp_dir / "training_state.pt")
        with (tmp_dir / "resolved_backbone_config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2)
    except Exception as exc:
        print(f"  [ckpt] ERROR: could not write checkpoint to {tmp_dir}: {exc}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    if final_dir.exists():
        shutil.rmtree(final_dir, ignore_errors=True)
    tmp_dir.replace(final_dir)
    print(f"  [ckpt] saved  -> {final_dir}")

    retain = max(int(keep_last), 1)
    existing = sorted(
        [
            entry
            for entry in output_dir.iterdir()
            if entry.is_dir()
            and entry.name.startswith("checkpoint-")
            and not entry.name.endswith(".tmp")
            and not entry.name.startswith(".hub_")
            and checkpoint_step_from_name(entry.name) >= 0
        ],
        key=lambda entry: checkpoint_step_from_name(entry.name),
    )
    for old in existing[:-retain]:
        shutil.rmtree(old, ignore_errors=True)
        print(f"  [ckpt] pruned -> {old.name}")

    if push_to_hub and hf_token:
        uploaded = sync_stage_checkpoint_to_hub(final_dir, hf_repo_id, hf_token, hf_stage_subdir)
        if not uploaded:
            print(
                f"  [hub]  warn: step {step} Hub sync failed; "
                f"local checkpoint retained at {final_dir}"
            )

    return final_dir


def _looks_like_pretrain_checkpoint(state: Dict[str, Any]) -> bool:
    """Return True when a checkpoint looks like a Stage 1 pre-training save."""
    return ("pretrain_config" in state) or (
        "tokens_processed" in state and "chunks_in_epoch" in state
    )


def _load_ema_shadow_from_alias(ema: ModelEMA, shadow_state: Dict[str, torch.Tensor]) -> None:
    """Best-effort load of EMA shadow tensors keyed by parameter name."""
    for name, tensor in shadow_state.items():
        if name in ema.shadow and ema.shadow[name].shape == tensor.shape:
            ema.shadow[name].copy_(tensor)


def load_latest_checkpoint(
    output_dir: Path,
    model: BaselineTRMMamba,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    device: torch.device,
    push_to_hub: bool = False,
    hf_repo_id: str = "WeirdRunner/Ouroboros",
    hf_token: Optional[str] = None,
    verbose: bool = True,
    current_sft_config: Optional[Dict[str, Any]] = None,
    hf_stage_subdir: str = "runs/stage2",
) -> Tuple[int, int, int, bool]:
    """Load the newest valid Stage 2 checkpoint.

    Returns (step, epoch, samples_seen, reset_optimizer).
    reset_optimizer=True when data stream changed or Stage 1 was loaded.
    """

    def _classify_state(state: Dict[str, Any]) -> str:
        if _looks_like_pretrain_checkpoint(state) and "sft_config" not in state:
            return "stage1"
        if "sft_config" in state:
            return "stage2"
        return "unknown"

    def _restore_stage1(state: Dict[str, Any], label: str) -> Tuple[int, int, int, bool]:
        model.load_state_dict(state["model_state_dict"])
        if state.get("ema"):
            ema.load_state_dict(state["ema"])
        elif state.get("ema_backbone_state_dict"):
            _load_ema_shadow_from_alias(ema, state["ema_backbone_state_dict"])
        if verbose:
            print(
                f"  [init]   {label}  loaded Stage 1 weights; "
                "resetting optimizer/scheduler for Stage 2."
            )
        return 0, 0, 0, True

    def _restore_stage2(state: Dict[str, Any], label: str) -> Optional[Tuple[int, int, int, bool]]:
        if any(key not in state for key in ("model_state_dict", "optimizer", "scheduler")):
            if verbose:
                print(f"  [resume] incomplete Stage 2 checkpoint {label} - skipping")
            return None

        epoch = int(state.get("epoch", 0))
        samples_seen = int(state.get("samples_seen", 0))
        model.load_state_dict(state["model_state_dict"])
        if state.get("ema"):
            ema.load_state_dict(state["ema"])
        elif state.get("ema_backbone_state_dict"):
            _load_ema_shadow_from_alias(ema, state["ema_backbone_state_dict"])

        saved_cfg = dict(state.get("sft_config") or {})
        saved_fingerprint = dict(state.get("data_fingerprint") or build_data_fingerprint(saved_cfg))
        current_fingerprint = build_data_fingerprint(current_sft_config or {})
        data_changed = bool(current_sft_config) and (saved_fingerprint != current_fingerprint)

        if data_changed:
            if verbose:
                print(
                    f"  [resume] {label}  loaded model weights (step={state.get('step', 0)}), "
                    "but data stream changed — resetting step/optimizer/scheduler for new data."
                )
                print(f"  [resume] saved_data={saved_fingerprint}")
                print(f"  [resume] new_data  ={current_fingerprint}")
            return 0, 0, 0, True

        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        step = int(state.get("step", 0))
        if verbose:
            print(
                f"  [resume] {label}  step={step}  epoch={epoch}  "
                f"samples_seen={samples_seen}  val_ce={state.get('val_ce')}"
            )
        return step, epoch, samples_seen, False

    search_root = Path(output_dir)
    direct_candidates: List[Path] = []

    if search_root.is_file() and search_root.name == "training_state.pt":
        direct_candidates.append(search_root.parent)
        search_root = search_root.parent.parent if search_root.parent.name.startswith("checkpoint-") else search_root.parent
    elif (search_root / "training_state.pt").exists():
        direct_candidates.append(search_root)
        if search_root.name.startswith("checkpoint-"):
            search_root = search_root.parent
    elif search_root.name.startswith("checkpoint-"):
        search_root = search_root.parent

    local_root = search_root if search_root.exists() else search_root.parent

    local_stage2: List[Dict[str, Any]] = []
    local_stage1_fallback: Optional[Dict[str, Any]] = None
    local_stage1_label: Optional[str] = None
    seen_paths: set[str] = set()

    for candidate in direct_candidates + list_local_checkpoints(local_root):
        candidate_str = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if candidate_str in seen_paths:
            continue
        seen_paths.add(candidate_str)
        state = _try_load_state_cpu(candidate)
        if state is None or "model_state_dict" not in state:
            continue
        kind = _classify_state(state)
        step = int(state.get("step", checkpoint_step_from_name(candidate.name)))
        if kind == "stage2":
            local_stage2.append({"step": step, "label": candidate.name, "state": state})
        elif kind == "stage1" and local_stage1_fallback is None:
            local_stage1_fallback = state
            local_stage1_label = candidate.name

    local_stage2.sort(key=lambda r: r["step"], reverse=True)

    if verbose:
        local_s2 = local_stage2[0]["step"] if local_stage2 else None
        print(
            f"  [resume] local Stage 2 candidates: {len(local_stage2)} "
            f"(newest step={local_s2 if local_s2 is not None else 'none'})"
        )

    for record in local_stage2:
        try:
            result = _restore_stage2(record["state"], record["label"])
            if result is not None:
                return result
        except Exception as exc:
            if verbose:
                print(f"  [resume] failed to restore local {record['label']}: {exc} - skipping")

    hub_resume_dir = local_root / ".hub_resume"
    if hf_token:
        if verbose:
            print("  [resume] no local Stage 2 found; checking Hub for Stage 2 checkpoints...")

        remote_stage2 = list_remote_stage_checkpoint_names(hf_repo_id, hf_token, hf_stage_subdir)
        remote_root = list_remote_checkpoint_names(hf_repo_id, hf_token)

        candidates_tried = 0
        stage_seen: set[str] = set()
        for ckpt_name in remote_stage2 + remote_root:
            source_key = ("stage" if ckpt_name in remote_stage2 and ckpt_name not in stage_seen else "root")
            if ckpt_name in remote_stage2:
                stage_seen.add(ckpt_name)
            if candidates_tried >= 3:
                if verbose:
                    print("  [resume] reached Hub download limit (3); stopping Hub search.")
                break
            if verbose:
                repo_hint = _stage_repo_checkpoint_path(ckpt_name, hf_stage_subdir) if source_key == "stage" else ckpt_name
                print(f"  [hub]  checking {repo_hint} ...")
            hub_resume_dir.mkdir(parents=True, exist_ok=True)
            if source_key == "stage":
                downloaded = download_stage_checkpoint_from_hub(ckpt_name, hub_resume_dir, hf_repo_id, hf_token, hf_stage_subdir)
            else:
                downloaded = download_checkpoint_from_hub(ckpt_name, hub_resume_dir, hf_repo_id, hf_token)
            if downloaded is None:
                continue
            candidates_tried += 1
            state = _try_load_state_cpu(downloaded)
            if state is None or "model_state_dict" not in state:
                continue
            kind = _classify_state(state)
            if kind == "stage2":
                try:
                    result = _restore_stage2(state, ckpt_name)
                    if result is not None:
                        shutil.rmtree(hub_resume_dir, ignore_errors=True)
                        return result
                except Exception as exc:
                    if verbose:
                        print(f"  [resume] failed to restore Hub {ckpt_name}: {exc} - skipping")
            elif kind == "stage1" and local_stage1_fallback is None:
                local_stage1_fallback = state
                local_stage1_label = ckpt_name

    if hub_resume_dir.exists():
        shutil.rmtree(hub_resume_dir, ignore_errors=True)

    if local_stage1_fallback is not None:
        if verbose:
            print(f"  [resume] no Stage 2 checkpoint found; using Stage 1 fallback from {local_stage1_label}.")
        return _restore_stage1(local_stage1_fallback, local_stage1_label or "stage1")

    if verbose:
        print("  [resume] No checkpoint found - starting from scratch.")
    return 0, 0, 0, True


def run_training(argv: Optional[List[str]] = None) -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("WANDB__SERVICE_WAIT", "30")
    args = parse_args(argv)
    hf_token = resolve_hf_token(args.hf_token)
    if args.push_to_hub and not hf_token:
        sys.exit("--push_to_hub requires --hf_token or HF_TOKEN/HUGGINGFACE_HUB_TOKEN")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    distributed = world_size > 1
    process_group_owner = False

    if distributed and (args.batch_size < world_size or (args.batch_size % world_size) != 0):
        raise SystemExit(
            f"For DDP, batch_size must be divisible by world_size. "
            f"Got batch_size={args.batch_size}, world_size={world_size}."
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

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        if is_main_process(rank):
            cleanup_temporary_checkpoints(output_dir)
        dist.barrier()
    else:
        cleanup_temporary_checkpoints(output_dir)

    if distributed:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats(device)

    wandb_run = None
    if is_main_process(rank) and args.wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config=sanitize_args_for_serialization(args),
            )
        except ImportError:
            print("[warn] wandb not installed - logging to stdout only.")

    try:
        if is_main_process(rank):
            print()
            print("=" * 64)
            print("  Stage 2 SFT - Project Ouroboros")
            print("=" * 64)
            print(f"  preset        : {args.preset}")
            print(f"  seq_len       : {args.max_seq_len}")
            print(f"  batch x accum : {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
            if distributed:
                print(f"  world_size    : {world_size}  (DDP auto-enabled)")
                print(f"  per_gpu_batch : {args.batch_size // world_size}")
            print(f"  lr            : {args.lr}  warmup={args.warmup_steps}")
            print(f"  dtype         : {dtype}")
            print(f"  output_dir    : {output_dir}")
            print(f"  dataset_mix   : {args.dataset_mix}")
            print(f"  answer_only_ce: True")
            print(f"  token_acc     : True  (answer tokens only)")
            print(f"  push_to_hub   : {args.push_to_hub}")
            print(f"  timeout       : {args.session_timeout_hours}h  (buffer={args.graceful_exit_buffer_minutes:.0f} min)")
            print("=" * 64)
            print()

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        if is_main_process(rank):
            print(f"Loading tokenizer: {args.tokenizer_name}")
            print(f"  vocab: {len(tokenizer):,}  pad_token: '{tokenizer.pad_token}'")
            print()

        if args.dataset_mix == "full":
            all_samples = load_mixed_dataset(tokenizer, args.max_samples, args.max_seq_len)
        else:
            all_samples = load_and_tokenize(args.dataset_name, tokenizer, args.max_samples, args.max_seq_len)
        if not all_samples:
            sys.exit("No samples loaded. Check dataset connectivity and --max_samples.")

        train_samples, val_samples = split_train_val_samples(
            all_samples=all_samples,
            seed=args.seed,
            val_fraction=args.val_fraction,
        )
        pad_id = tokenizer.pad_token_id or 0

        if is_main_process(rank):
            print(f"  train: {len(train_samples)}  val: {len(val_samples)}")
            print()

        preset_kwargs = PRESETS[args.preset]
        padded_vocab = pad_vocab_size(len(tokenizer), 128)
        config = BaselineConfig(vocab_size=padded_vocab, max_seq_len=args.max_seq_len, dropout=0.0, **preset_kwargs)
        raw_model = BaselineTRMMamba(config).to(device=device, dtype=dtype)
        raw_model.train()

        n_params = count_parameters(raw_model)
        if is_main_process(rank):
            print(f"Model  : {n_params / 1e6:.1f}M parameters  (preset={args.preset})")
            print(
                f"Config : d_model={config.d_model}  n_groups={config.n_groups} "
                f" heads={config.n_heads}/{config.n_kv_heads}  mlp_hidden={config.mlp_hidden}"
            )
            print(f"Vocab  : {padded_vocab:,} (padded from {len(tokenizer):,})")
            print()

        optimizer, fused_enabled = build_adamw_optimizer(
            model=raw_model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
            prefer_fused=True,
        )
        if is_main_process(rank):
            print(f"Optimizer : AdamW ({'fused CUDA kernel' if fused_enabled else 'standard'})")

        steps_per_epoch = max(1, math.ceil(len(train_samples) / max(args.batch_size * args.grad_accum, 1)))
        total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.num_epochs
        scheduler = cosine_with_warmup(optimizer, args.warmup_steps, total_steps, args.min_lr_ratio)
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
        ema = ModelEMA(raw_model, decay=args.ema_decay)
        spike_monitor = SpikeMonitor(threshold=args.spike_threshold)

        if is_main_process(rank):
            print(
                f"Schedule  : cosine warmup={args.warmup_steps} total={total_steps} "
                f"({args.num_epochs} epochs x {steps_per_epoch} steps/epoch)"
            )
            print()

        resume_search = Path(args.resume_from) if args.resume_from else output_dir
        if args.resume_from and list_local_checkpoints(output_dir):
            resume_path = Path(args.resume_from).resolve()
            output_root = output_dir.resolve()
            resume_inside_output = (resume_path == output_root) or (output_root in resume_path.parents)
            if not resume_inside_output and is_main_process(rank):
                print("  [resume] found local Stage 2 checkpoints in output_dir; preferring them over external --resume_from")
                resume_search = output_dir
        if resume_search.is_dir():
            if distributed:
                if is_main_process(rank):
                    cleanup_temporary_checkpoints(resume_search)
                dist.barrier()
            else:
                cleanup_temporary_checkpoints(resume_search)

        if distributed:
            if is_main_process(rank):
                start_step, start_epoch, samples_seen, need_opt_reset = load_latest_checkpoint(
                    resume_search,
                    raw_model,
                    ema,
                    optimizer,
                    scheduler,
                    scaler,
                    device,
                    push_to_hub=args.push_to_hub,
                    hf_repo_id=args.hf_repo_id,
                    hf_token=hf_token,
                    verbose=True,
                    current_sft_config=sanitize_args_for_serialization(args),
                    hf_stage_subdir=args.hf_stage_subdir,
                )
                if need_opt_reset:
                    optimizer, fused_enabled = build_adamw_optimizer(
                        model=raw_model,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        betas=(0.9, 0.95),
                        eps=1e-8,
                        prefer_fused=True,
                    )
                    scheduler = cosine_with_warmup(optimizer, args.warmup_steps, total_steps, args.min_lr_ratio)
                    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
                    print("  [resume] optimizer/scheduler reset; training starts fresh from step 0.")
                dist.barrier()
            else:
                dist.barrier()
                start_step, start_epoch, samples_seen, _ = load_latest_checkpoint(
                    resume_search,
                    raw_model,
                    ema,
                    optimizer,
                    scheduler,
                    scaler,
                    device,
                    push_to_hub=args.push_to_hub,
                    hf_repo_id=args.hf_repo_id,
                    hf_token=None,
                    verbose=False,
                    current_sft_config=sanitize_args_for_serialization(args),
                    hf_stage_subdir=args.hf_stage_subdir,
                )
        else:
            start_step, start_epoch, samples_seen, need_opt_reset = load_latest_checkpoint(
                resume_search,
                raw_model,
                ema,
                optimizer,
                scheduler,
                scaler,
                device,
                push_to_hub=args.push_to_hub,
                hf_repo_id=args.hf_repo_id,
                hf_token=hf_token,
                verbose=True,
                current_sft_config=sanitize_args_for_serialization(args),
                hf_stage_subdir=args.hf_stage_subdir,
            )
            if need_opt_reset:
                optimizer, fused_enabled = build_adamw_optimizer(
                    model=raw_model,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    prefer_fused=True,
                )
                scheduler = cosine_with_warmup(optimizer, args.warmup_steps, total_steps, args.min_lr_ratio)
                scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
                if is_main_process(rank):
                    print("  [resume] optimizer/scheduler reset; training starts fresh from step 0.")

        train_model = raw_model
        if distributed:
            train_model = DDP(
                raw_model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )

        local_batch_size = args.batch_size // world_size if distributed else args.batch_size
        if samples_seen < 0:
            if is_main_process(rank):
                print("  [resume] data stream changed - restarting sample cursor from the beginning of the new dataset.")
            samples_seen = 0
        elif samples_seen == 0 and start_step > 0:
            samples_seen = start_step * args.batch_size * args.grad_accum

        perm_cache: Dict[int, List[int]] = {}

        def epoch_permutation(epoch: int) -> List[int]:
            if epoch not in perm_cache:
                generator = torch.Generator().manual_seed(args.seed + epoch)
                perm_cache[epoch] = torch.randperm(len(train_samples), generator=generator).tolist()
            return perm_cache[epoch]

        def fetch_micro_batch(global_sample_start: int) -> List[Dict[str, Any]]:
            batch: List[Dict[str, Any]] = []
            rank_start = global_sample_start + (rank * local_batch_size)
            for pos in range(rank_start, rank_start + local_batch_size):
                epoch = pos // len(train_samples)
                offset = pos % len(train_samples)
                perm = epoch_permutation(epoch)
                batch.append(train_samples[perm[offset]])
            return batch

        col_hdr = (
            f"{'Step':>7}  {'Train CE':>9}  {'Train Acc':>10}  {'Val CE':>9}  {'Val Acc':>8}  "
            f"{'GNorm':>7}  {'LR':>9}  {'VRAM':>7}  {'Tok/s':>7}"
        )
        if is_main_process(rank):
            print(col_hdr)
            print("-" * len(col_hdr))

        pbar = tqdm(total=total_steps, initial=start_step, desc="Training", dynamic_ncols=True) if is_main_process(rank) else None
        step = start_step
        last_saved_step = start_step
        last_val_ce: Optional[float] = None
        last_val_acc: Optional[float] = None
        tokens_since_log = 0
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        session_start = time.perf_counter()
        timeout_limit_s = args.session_timeout_hours * 3600.0
        timeout_buffer_s = args.graceful_exit_buffer_minutes * 60.0
        timeout_triggered = False

        def check_timeout() -> bool:
            nonlocal timeout_triggered
            if timeout_triggered or not is_main_process(rank):
                return timeout_triggered
            elapsed = time.perf_counter() - session_start
            if elapsed + timeout_buffer_s >= timeout_limit_s:
                remaining_min = (timeout_limit_s - elapsed) / 60.0
                print(
                    f"\n  [timeout] {elapsed / 3600:.2f}h elapsed — "
                    f"{remaining_min:.1f} min remaining (< {args.graceful_exit_buffer_minutes:.0f} min buffer)."
                )
                timeout_triggered = True
            return timeout_triggered

        def broadcast_timeout() -> bool:
            nonlocal timeout_triggered
            if not distributed:
                return timeout_triggered
            tensor = torch.tensor([int(timeout_triggered)], device=device, dtype=torch.int32)
            dist.broadcast(tensor, src=0)
            timeout_triggered = bool(tensor.item())
            return timeout_triggered

        while step < total_steps:
            accum_loss = 0.0
            accum_correct = 0
            accum_tokens = 0

            for _ in range(args.grad_accum):
                micro_batch = fetch_micro_batch(samples_seen)
                batch = collate(micro_batch, pad_id)
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                tokens_since_log += int(attn_mask.sum().item())
                samples_seen += args.batch_size

                with autocast_context(device, dtype):
                    logits = train_model(input_ids, attention_mask=attn_mask)

                shift_logits = logits[:, :-1, :].contiguous().view(-1, config.vocab_size).float()
                shift_labels = labels[:, 1:].contiguous().view(-1)
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
                n_correct, n_valid = masked_token_accuracy(shift_logits, shift_labels)

                if dtype == torch.float16:
                    scaler.scale(loss / args.grad_accum).backward()
                else:
                    (loss / args.grad_accum).backward()

                accum_loss += loss.detach().item()
                accum_correct += n_correct
                accum_tokens += n_valid

            if dtype == torch.float16:
                scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm))

            if dtype == torch.float16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(raw_model)

            step += 1
            mean_ce = distributed_mean(accum_loss / args.grad_accum, device)
            global_correct = distributed_sum_int(accum_correct, device)
            global_valid = distributed_sum_int(accum_tokens, device)
            mean_acc = global_correct / max(global_valid, 1)
            grad_norm_log = distributed_mean(grad_norm, device)

            if is_main_process(rank) and spike_monitor.update(step, mean_ce):
                tqdm.write(f"  [spike] step={step}  raw={mean_ce:.4f}  ema={spike_monitor.smoothed:.4f}")

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(ce=f"{mean_ce:.3f}", acc=f"{mean_acc:.3f}", gn=f"{grad_norm_log:.3f}")

            if step % args.log_every == 0 or step == 1:
                global_tokens_since_log = distributed_sum_int(tokens_since_log, device)
                cur_lr = scheduler.get_last_lr()[0]
                if is_main_process(rank):
                    elapsed = max(time.perf_counter() - t0, 1e-6)
                    tok_s = global_tokens_since_log / elapsed
                    vram = vram_gb(device)
                    val_ce_str = f"{last_val_ce:.4f}" if last_val_ce is not None else "        -"
                    val_acc_str = f"{last_val_acc:.4f}" if last_val_acc is not None else "       -"
                    tqdm.write(
                        f"{step:>7}  {mean_ce:>9.4f}  {mean_acc:>10.4f}  {val_ce_str:>9}  {val_acc_str:>8}  "
                        f"{grad_norm_log:>7.4f}  {cur_lr:>9.2e}  {vram:>7.3f}  {tok_s:>7.0f}"
                    )

                    if wandb_run is not None:
                        import wandb

                        wandb.log(
                            {
                                "train/ce": mean_ce,
                                "train/ce_smooth": spike_monitor.smoothed,
                                "train/accuracy": mean_acc,
                                "train/grad_norm": grad_norm_log,
                                "train/lr": cur_lr,
                                "train/tok_s": tok_s,
                                "train/vram_gb": vram,
                                "train/world_size": world_size,
                            },
                            step=step,
                        )
                t0 = time.perf_counter()
                tokens_since_log = 0

            check_timeout()
            if broadcast_timeout():
                current_epoch = samples_seen // max(len(train_samples), 1)
                if is_main_process(rank):
                    print(f"  [timeout] Saving emergency checkpoint at step {step} ...")
                    save_checkpoint(
                        output_dir=output_dir,
                        step=step,
                        model=raw_model,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler if dtype == torch.float16 else None,
                        config=config,
                        val_ce=last_val_ce,
                        keep_last=args.keep_last,
                        epoch=current_epoch,
                        samples_seen=samples_seen,
                        sft_config=sanitize_args_for_serialization(args),
                        push_to_hub=args.push_to_hub,
                        hf_repo_id=args.hf_repo_id,
                        hf_token=hf_token,
                        hf_stage_subdir=args.hf_stage_subdir,
                    )
                    last_saved_step = step
                if distributed:
                    dist.barrier()
                break

            if step % args.val_every == 0 or step == total_steps:
                if is_main_process(rank):
                    last_val_ce, last_val_acc = compute_val_metrics(
                        raw_model,
                        ema,
                        val_samples,
                        pad_id,
                        device,
                        dtype,
                        args.batch_size,
                        config.vocab_size,
                    )
                    tqdm.write(
                        f"  [val] step={step}  val_ce={last_val_ce:.4f}  val_acc={last_val_acc:.4f}"
                    )

                    if wandb_run is not None:
                        import wandb

                        wandb.log({"val/ce": last_val_ce, "val/accuracy": last_val_acc}, step=step)

                    if last_val_ce < 1.5:
                        tqdm.write(
                            "  * val_ce < 1.5 - Stage 2 success criterion met. "
                            "Consider proceeding to Phase 3."
                        )
                if distributed:
                    dist.barrier()

            if step % args.gen_every == 0 or step == total_steps:
                if is_main_process(rank):
                    run_generation_callback(
                        raw_model,
                        ema,
                        tokenizer,
                        device,
                        dtype,
                        step=step,
                        max_new_tokens=args.gen_max_tokens,
                        max_seq_len=args.max_seq_len,
                        wandb_run=wandb_run,
                    )
                if distributed:
                    dist.barrier()

            if step % args.save_every == 0 or step == total_steps:
                current_epoch = samples_seen // max(len(train_samples), 1)
                if is_main_process(rank):
                    save_checkpoint(
                        output_dir=output_dir,
                        step=step,
                        model=raw_model,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler if dtype == torch.float16 else None,
                        config=config,
                        val_ce=last_val_ce,
                        keep_last=args.keep_last,
                        epoch=current_epoch,
                        samples_seen=samples_seen,
                        sft_config=sanitize_args_for_serialization(args),
                        push_to_hub=args.push_to_hub,
                        hf_repo_id=args.hf_repo_id,
                        hf_token=hf_token,
                        hf_stage_subdir=args.hf_stage_subdir,
                    )
                    last_saved_step = step
                if distributed:
                    dist.barrier()

        if pbar is not None:
            pbar.close()

        if not timeout_triggered and step != last_saved_step:
            current_epoch = samples_seen // max(len(train_samples), 1)
            if is_main_process(rank):
                save_checkpoint(
                    output_dir=output_dir,
                    step=step,
                    model=raw_model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if dtype == torch.float16 else None,
                    config=config,
                    val_ce=last_val_ce,
                    keep_last=args.keep_last,
                    epoch=current_epoch,
                    samples_seen=samples_seen,
                    sft_config=sanitize_args_for_serialization(args),
                    push_to_hub=args.push_to_hub,
                    hf_repo_id=args.hf_repo_id,
                    hf_token=hf_token,
                )
                last_saved_step = step
            if distributed:
                dist.barrier()

        if is_main_process(rank):
            total_time = time.perf_counter() - session_start
            peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print()
            print("=" * 64)
            print("  Stage 2 Training Complete")
            print("=" * 64)
            print(f"  Total steps  : {step}")
            print(f"  Total time   : {total_time / 60:.1f} min")
            print(f"  Peak VRAM    : {peak_vram_gb:.2f} GB")
            if distributed:
                print(f"  World size   : {world_size}")
            print(f"  Spike count  : {len(spike_monitor.spikes)}")
            if last_val_ce is not None:
                print(f"  Final val CE : {last_val_ce:.4f}  (answer tokens only)")
                if last_val_acc is not None:
                    print(f"  Final val acc: {last_val_acc:.4f}  (answer-token accuracy)")
                if last_val_ce < 1.5:
                    print("  Status       : SUCCESS - proceed to Phase 3 (incremental recursion)")
                else:
                    print("  Status       : val_ce >= 1.5 - extend training or check data quality")
            if timeout_triggered:
                print(f"  Status       : TIMEOUT - checkpoint saved at step {step}")
            print("=" * 64)

        if wandb_run is not None and is_main_process(rank):
            try:
                import wandb

                wandb.finish(quiet=True)
            except Exception as exc:
                print(f"[warn] wandb.finish() failed during shutdown: {exc}")
        sys.stdout.flush()
        sys.stderr.flush()
        return 0
    finally:
        if process_group_owner and dist_is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as exc:
                print(f"[warn] dist.destroy_process_group() failed during shutdown: {exc}")


def main(argv: Optional[List[str]] = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv_list)
    if maybe_launch_multi_gpu(args, argv_list):
        return 0
    return run_training(argv_list)


if __name__ == "__main__":
    raise SystemExit(main())
