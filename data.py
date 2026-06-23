"""
data.py
=======
The data pipeline for Ouroboros training: a Coconut curriculum dataset, the
DGAC hyperparameter bundle, and the canonical-dataset loader.

One file because the three concerns are one mechanism — turning raw
question/steps/answer rows into the {input_ids, labels, attention_mask}
tensors model.Ouroboros.forward consumes, plus the few numbers the trainer
needs to size the curriculum. Nothing here reimplements what torch/transformers
already give: CoconutDataset extends torch.utils.data.Dataset, collate is a
plain pad-to-max function, the loader is jsonl + an optional Hub fallback.

The output contract is deliberately minimal (B9): only input_ids / labels /
attention_mask per item, no q_lens / n_latents / pad_id. model.Ouroboros
derives each row's latent depth from how many <|lat|> tokens its input_ids
actually contains — input_ids is the single source of truth, so collate-time
copies of those fields were redundant and a footgun (a caller could rely on a
stale collate value instead of the ground-truth token positions).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median_high
from typing import Any, Optional

from torch.utils.data import Dataset

import torch

_LAT_TOKEN = "<|lat|>"


# ── DGAC config ───────────────────────────────────────────────────────────────

@dataclass
class DGACConfig:
    """
    All DGAC / HaltGate training hyperparameters in one place. Pass to
    CurriculumTrainer; None (don't pass it) disables DGAC entirely.

    The CE-tolerance halt supervision is the primary signal — stronger and
    lower-variance than vanilla ACT or PonderNet (it picks the smallest depth
    that preserves full-depth CE). The ACT ponder + cosine-diversity terms are
    the regularizers from the original DGAC formulation.

    P1c: when lambda_ponder_kl > 0, a PonderNet KL-to-geometric-prior term
    REPLACES the ACT ponder term (and its hand-tuned warmup/ramp schedule) with
    a single principled prior (desired mean halt depth). The CE-tolerance
    supervision and diversity terms stay active in both modes.
    """
    halt_supervision_weight: float = 0.1
    halt_ce_tolerance: float = 0.02
    halt_probe_steps: str = "1,2,4,stage_k"

    lambda_ponder_max: float = 0.01      # ACT ponder ramp ceiling
    lambda_diversity: float = 0.1
    tau: float = 0.9                     # cosine-sim floor before repulsion
    warmup_steps: int = 200
    ramp_steps: int = 300

    # P1c — PonderNet KL alternative to ACT ponder (0.0 = current ACT behaviour)
    lambda_ponder_kl: float = 0.0
    pondernet_prior_mean: float = 2.0    # geometric-prior mean halt steps; q = 1/mean


# ── chat template ─────────────────────────────────────────────────────────────

_WARNED_ABOUT_TEMPLATE = False


def apply_chat_template(tokenizer, question: str) -> str:
    """
    Wrap a question in the tokenizer's chat template so the <|lat|> tokens that
    follow land inside the assistant turn — matching the training distribution.
    Falls back to a plain prompt if the tokenizer has no template or it errors,
    so a tokenizer-less smoke can still run. The training/inference contract is
    documented in plan/refactor.md: lat tokens must sit inside the assistant
    block, never over raw question text.
    """
    global _WARNED_ABOUT_TEMPLATE
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        if not _WARNED_ABOUT_TEMPLATE:
            print("  [warn] tokenizer.apply_chat_template failed; using plain prompt fallback.")
            _WARNED_ABOUT_TEMPLATE = True
        return f"User: {question}\nAssistant: "


# ── dataset ───────────────────────────────────────────────────────────────────

class CoconutDataset(Dataset):
    """
    Produces correctly-formatted input_ids + labels for one Coconut curriculum
    stage.

    Per __getitem__:
        input_ids       [L] long  — question + [lat*n] + supervised
        labels          [L] long  — -100 at question+lat positions, supervised ids elsewhere
        attention_mask  [L] bool

    Stage 0 is CoT warmup (n_latent=0, no lat tokens). Stage k inserts k lat
    tokens between the question and the supervised remainder. stochastic_depth
    (P1a, Huginn §4.2) samples n_latent ~ Uniform(1, stage_k) per sample instead
    of always using the max, training depth extrapolation and collapse
    resistance; default off matches the original fixed-depth behaviour.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: Any,
        lat_token_id: int,
        stage_k: int,
        max_seq_len: int,
        stochastic_depth: bool = False,
        seed: int = 0,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.lat_token_id = int(lat_token_id)
        self.stage_k = int(stage_k)
        self.max_seq_len = int(max_seq_len)
        self.stochastic_depth = bool(stochastic_depth)
        self._rng = random.Random(int(seed))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # A sample can be unbuildable (empty question, nothing to supervise,
        # truncation too aggressive); skip to the next rather than hand the
        # collator a None. Wraparound keeps the batch full.
        for offset in range(len(self.samples)):
            item = self._build(self.samples[(idx + offset) % len(self.samples)])
            if item is not None:
                return item
        raise RuntimeError("no buildable sample found in the dataset")

    def _build(self, sample: dict[str, Any]) -> Optional[dict[str, torch.Tensor]]:
        question = str(sample.get("question", "")).strip()
        if not question:
            return None

        q_ids = self.tokenizer.encode(
            apply_chat_template(self.tokenizer, question), add_special_tokens=False
        )

        steps = sample.get("steps") or []
        if isinstance(steps, str):
            try:
                steps = json.loads(steps)
            except json.JSONDecodeError:
                steps = [steps]
        steps = [str(s) for s in steps if str(s).strip()]

        upper = min(self.stage_k, len(steps))
        if self.stage_k == 0:
            n_latent = 0                                     # CoT warmup
        elif self.stochastic_depth and upper >= 1:
            # P1a: per-sample depth Uniform(1, upper). Lower bound 1, not 0 —
            # every stage-k>0 sample gets at least one latent pass. The RNG is
            # one instance per dataset (seeded in __init__); DataLoader workers
            # reseed it via worker_init_fn so draws stay independent across
            # workers and reproducible per seed.
            n_latent = self._rng.randint(1, upper)
        else:
            n_latent = upper

        supervised: list[int] = []
        for step_text in steps[n_latent:]:
            supervised.extend(self.tokenizer.encode(step_text + "\n", add_special_tokens=False))

        answer_ids = self.tokenizer.encode(str(sample.get("answer_full", "")), add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            answer_ids.append(int(self.tokenizer.eos_token_id))
        supervised.extend(answer_ids)

        if not supervised:
            return None

        total = len(q_ids) + n_latent + len(supervised)
        if total > self.max_seq_len:
            allowed = self.max_seq_len - len(q_ids) - n_latent
            if allowed < 4:
                return None
            supervised = supervised[:allowed]

        full_ids = q_ids + [self.lat_token_id] * n_latent + supervised
        labels = [-100] * len(q_ids) + [-100] * n_latent + supervised
        assert len(full_ids) == len(labels)

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.bool),
        }

    @staticmethod
    def collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
        """Pad to the max length in the batch. Returns the three fields only."""
        max_len = max(item["input_ids"].size(0) for item in batch)
        b = len(batch)
        input_ids = torch.full((b, max_len), int(pad_id), dtype=torch.long)
        labels = torch.full((b, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((b, max_len), dtype=torch.bool)
        for i, item in enumerate(batch):
            n = item["input_ids"].size(0)
            input_ids[i, :n] = item["input_ids"]
            labels[i, :n] = item["labels"]
            attention_mask[i, :n] = item["attention_mask"]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    @staticmethod
    def load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                steps = row.get("steps")
                if isinstance(steps, str):
                    try:
                        row["steps"] = json.loads(steps)
                    except json.JSONDecodeError:
                        row["steps"] = [steps]
                rows.append(row)
        return rows


# ── canonical dataset load ────────────────────────────────────────────────────

def _download_dataset_from_hub(
    data_dir: Path,
    hf_repo_id: str = "WeirdRunner/Ouroboros",
    hf_config: str = "coconut-v1",
) -> None:
    """Fall back to a Hub dataset when local jsonl is absent."""
    from datasets import load_dataset as hf_load_dataset

    print(f"  [data] Local files missing. Downloading {hf_repo_id}[{hf_config}] from Hub...")
    data_dir.mkdir(parents=True, exist_ok=True)

    def _write_split(out_path: Path, hf_split: str) -> list[dict[str, Any]]:
        try:
            ds = hf_load_dataset(hf_repo_id, hf_config, split=hf_split, token=True)
        except Exception as exc:
            print(f"  [data] Could not load split '{hf_split}': {exc}")
            return []
        rows: list[dict[str, Any]] = []
        with out_path.open("w", encoding="utf-8") as fh:
            for row in ds:
                steps = row.get("steps", [])
                if isinstance(steps, str):
                    try:
                        steps = json.loads(steps)
                    except json.JSONDecodeError:
                        steps = [steps]
                sample = {
                    "id": row.get("id", ""),
                    "source": row.get("source", ""),
                    "question": row.get("question", ""),
                    "steps": steps,
                    "answer_full": row.get("answer_full", ""),
                    "answer_norm": row.get("answer_norm", ""),
                    "n_steps": int(row.get("n_steps", len(steps))),
                }
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                rows.append(sample)
        print(f"  [data] {hf_split}: {len(rows)} samples -> {out_path}")
        return rows

    train_rows = _write_split(data_dir / "train.jsonl", "train")
    val_rows = _write_split(data_dir / "val.jsonl", "validation")

    def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        n_steps = [r["n_steps"] for r in rows]
        return {
            "n_samples": len(rows),
            "n_steps_mean": round(fmean(n_steps), 2),
            "n_steps_min": min(n_steps),
            "n_steps_max": max(n_steps),
            "n_steps_median": int(median_high(n_steps)),
            "by_source": dict(Counter(r["source"] for r in rows)),
        }

    stats = {"train": _stats(train_rows), "val": _stats(val_rows)}
    with (data_dir / "stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    t = stats.get("train", {})
    print(f"  [data] stats.json written. median_steps={t.get('n_steps_median')} "
          f"recommended --max_stage={t.get('n_steps_median')}")


def load_canonical_dataset(
    data_dir: Path,
    max_samples: Optional[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load train/val jsonl + stats, downloading from Hub if local files are absent."""
    data_dir = Path(data_dir)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    stats_path = data_dir / "stats.json"

    if not train_path.exists():
        _download_dataset_from_hub(data_dir)
    if not train_path.exists():
        raise FileNotFoundError(
            f"train.jsonl not found at {train_path} and Hub download failed."
        )

    train = CoconutDataset.load_jsonl(train_path)
    val = CoconutDataset.load_jsonl(val_path) if val_path.exists() else []
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

    if max_samples is not None:
        n_val = max(1, max_samples // 20) if val else 0
        n_train = max(max_samples - n_val, 0)
        train = train[:n_train]
        val = val[:n_val] if n_val else []

    print(f"  Loaded {len(train)} train / {len(val)} val from {data_dir}")
    if stats:
        t = stats.get("train", {})
        print(f"  Step stats: median={t.get('n_steps_median')} "
              f"mean={t.get('n_steps_mean')} max={t.get('n_steps_max')}")
    return train, val, stats


def get_max_stage(max_stage_override: Optional[int], stats: dict[str, Any]) -> int:
    """
    Resolve the curriculum depth: explicit override, else the dataset's median
    step count, else a conservative default. Decoupled from argparse so the
    smoke test and any caller can use it without a Namespace.
    """
    if max_stage_override is not None:
        return int(max_stage_override)
    median = stats.get("train", {}).get("n_steps_median")
    if median is not None:
        print(f"  --max_stage not set; using n_steps_median={median} from stats.json")
        return int(median)
    print("  [warn] --max_stage not set and stats.json absent; defaulting to 10")
    return 10
