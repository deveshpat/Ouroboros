"""Dataset loading and Coconut-style latent sample construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class PromptFeature:
    input_ids: list[int]
    labels: list[int]
    id: str = ""
    answer_norm: str = ""


class JsonlCoconutDataset(Dataset):
    def __init__(self, features: Sequence[PromptFeature]):
        self.features = list(features)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        feature = self.features[index]
        return {
            "input_ids": feature.input_ids,
            "labels": feature.labels,
            "id": feature.id,
            "answer_norm": feature.answer_norm,
        }


class CoconutCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.full((len(rows), max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(rows), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        for idx, row in enumerate(rows):
            ids = torch.tensor(row["input_ids"], dtype=torch.long)
            labs = torch.tensor(row["labels"], dtype=torch.long)
            input_ids[idx, : ids.numel()] = ids
            labels[idx, : labs.numel()] = labs
            attention_mask[idx, : ids.numel()] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_rows(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else raw.get("data", [])
    else:
        rows = list(iter_jsonl(path))
    if limit is not None:
        rows = rows[: max(int(limit), 0)]
    return [dict(row) for row in rows]


def normalize_answer(text: Any) -> str:
    return str(text or "").strip().replace(",", "")


def row_answer(row: dict[str, Any]) -> str:
    return str(row.get("answer_full") or row.get("answer") or row.get("answer_norm") or "").strip()


def row_steps(row: dict[str, Any]) -> list[str]:
    steps = row.get("steps") or []
    if isinstance(steps, str):
        try:
            parsed = json.loads(steps)
            steps = parsed if isinstance(parsed, list) else [steps]
        except json.JSONDecodeError:
            steps = [steps]
    return [str(step).strip() for step in steps if str(step).strip()]


def format_question(tokenizer, question: str, *, use_chat_template: bool) -> str:
    question = str(question or "").strip()
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return question + "\n"


def make_feature(
    row: dict[str, Any],
    tokenizer,
    *,
    latent_token_id: int,
    stage: int,
    max_seq_len: int,
    use_chat_template: bool = True,
) -> PromptFeature | None:
    question = str(row.get("question") or row.get("prompt") or "").strip()
    if not question:
        return None

    q_ids = tokenizer.encode(
        format_question(tokenizer, question, use_chat_template=use_chat_template),
        add_special_tokens=False,
    )
    steps = row_steps(row)
    n_latents = min(max(int(stage), 0), len(steps))
    supervised: list[int] = []
    for step in steps[n_latents:]:
        supervised.extend(tokenizer.encode(step + "\n", add_special_tokens=False))
    answer = row_answer(row)
    if answer:
        supervised.extend(tokenizer.encode(answer, add_special_tokens=False))
    if tokenizer.eos_token_id is not None:
        supervised.append(int(tokenizer.eos_token_id))
    if not supervised:
        return None

    allowed_supervised = max(int(max_seq_len) - len(q_ids) - n_latents, 0)
    if allowed_supervised < 2:
        return None
    supervised = supervised[:allowed_supervised]
    input_ids = q_ids + [int(latent_token_id)] * n_latents + supervised
    labels = [-100] * (len(q_ids) + n_latents) + supervised
    return PromptFeature(
        input_ids=input_ids,
        labels=labels,
        id=str(row.get("id") or row.get("idx") or ""),
        answer_norm=normalize_answer(row.get("answer_norm") or row.get("answer")),
    )


def build_features(
    rows: Iterable[dict[str, Any]],
    tokenizer,
    *,
    latent_token_id: int,
    stage: int,
    max_seq_len: int,
    use_chat_template: bool = True,
) -> list[PromptFeature]:
    features: list[PromptFeature] = []
    for row in rows:
        feature = make_feature(
            row,
            tokenizer,
            latent_token_id=latent_token_id,
            stage=stage,
            max_seq_len=max_seq_len,
            use_chat_template=use_chat_template,
        )
        if feature is not None:
            features.append(feature)
    return features


def make_loader(
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
