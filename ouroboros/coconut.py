"""Ouroboros Coconut runtime.

This file is intentionally plain: load rows, build latent-token examples, wrap a
CausalLM with Coconut-style hidden-state feedback, save a release bundle, and
push that bundle to the Hub with the Hugging Face API.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers.modeling_outputs import CausalLMOutputWithPast

DEFAULT_BASE_MODEL = "ai21labs/AI21-Jamba-Reasoning-3B"
DEFAULT_LATENT_TOKEN = "<|lat|>"
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "x_proj",
    "dt_proj",
    "out_proj",
)


@dataclass(frozen=True)
class PromptFeature:
    input_ids: list[int]
    labels: list[int]
    id: str = ""
    answer_norm: str = ""


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]


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


class OuroborosCoconutForCausalLM(nn.Module):
    """A small, model-agnostic Coconut wrapper.

    Every ``latent_token_id`` placeholder receives the previous token's last
    hidden state. This avoids custom cache machinery and keeps the runtime easy
    to inspect. It is slower than the old optimized path, but it is the shortest
    correct path for training, eval, inference smoke tests, and release bundles.
    """

    def __init__(self, base_causallm: nn.Module, latent_token_id: int):
        super().__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = int(latent_token_id)

    @property
    def config(self):
        return self.base_causallm.config

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def get_input_embeddings(self):
        return self.base_causallm.get_input_embeddings()

    def set_input_embeddings(self, value) -> None:
        self.base_causallm.set_input_embeddings(value)

    def save_pretrained(self, *args, **kwargs):
        return self.base_causallm.save_pretrained(*args, **kwargs)

    def push_to_hub(self, *args, **kwargs):
        return self.base_causallm.push_to_hub(*args, **kwargs)

    def _fill_latents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        embeds = self.get_input_embeddings()(input_ids)
        latent_positions = (input_ids == self.latent_token_id).nonzero(as_tuple=False)
        if latent_positions.numel() == 0:
            return embeds

        latent_lists: list[list[int]] = [
            [int(pos.item()) for row, pos in latent_positions if int(row.item()) == batch_idx]
            for batch_idx in range(input_ids.size(0))
        ]
        max_latents = max((len(items) for items in latent_lists), default=0)

        for latent_index in range(max_latents):
            outputs = self.base_causallm(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = _last_hidden(outputs)
            patched = embeds.clone()
            for batch_idx, positions in enumerate(latent_lists):
                if latent_index >= len(positions):
                    continue
                pos = positions[latent_index]
                if pos <= 0:
                    continue
                patched[batch_idx, pos, :] = hidden[batch_idx, pos - 1, :].to(patched.dtype)
            embeds = patched
        return embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: Any,
    ) -> CausalLMOutputWithPast:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        inputs_embeds = self._fill_latents(input_ids, attention_mask)
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        loss = None
        if labels is not None:
            logits = outputs.logits
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=getattr(outputs, "attentions", None),
        )


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


def _last_hidden(outputs: Any) -> torch.Tensor:
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    raise RuntimeError("Model did not return hidden states; pass output_hidden_states=True.")


def load_rows(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else raw.get("data", [])
    else:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if limit is not None:
        rows = rows[: max(int(limit), 0)]
    return [dict(row) for row in rows]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_answer(text: Any) -> str:
    value = str(text or "").strip()
    return value.replace(",", "")


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
    remaining = steps[n_latents:]

    supervised: list[int] = []
    for step in remaining:
        supervised.extend(tokenizer.encode(step + "\n", add_special_tokens=False))
    answer = row_answer(row)
    if answer:
        supervised.extend(tokenizer.encode(answer, add_special_tokens=False))
    if tokenizer.eos_token_id is not None:
        supervised.append(int(tokenizer.eos_token_id))
    if not supervised:
        return None

    reserve = len(q_ids) + n_latents
    allowed_supervised = max(int(max_seq_len) - reserve, 0)
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


def resolve_device(requested: str = "auto") -> torch.device:
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    requested = (requested or "auto").lower()
    if requested == "auto":
        if device.type == "cuda":
            major, _minor = torch.cuda.get_device_capability(device)
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float32
    choices = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if requested not in choices:
        raise ValueError(f"Unsupported dtype {requested!r}")
    return choices[requested]


def ensure_latent_token(tokenizer, model, latent_token: str) -> int:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    token_id = tokenizer.convert_tokens_to_ids(latent_token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [latent_token]})
        token_id = tokenizer.convert_tokens_to_ids(latent_token)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    return int(token_id)


def load_lora_coconut(
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    latent_token: str = DEFAULT_LATENT_TOKEN,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: Sequence[str] = DEFAULT_LORA_TARGETS,
    load_in_4bit: bool = False,
    trust_remote_code: bool = True,
) -> tuple[OuroborosCoconutForCausalLM, Any, int]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = resolve_device("auto") if device is None else device
    dtype = resolve_dtype("auto", device) if dtype is None else dtype
    if load_in_4bit and device.type != "cuda":
        raise ValueError("load_in_4bit requires a CUDA device.")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=trust_remote_code)

    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "low_cpu_mem_usage": True}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0} if device.type == "cuda" else None
    else:
        kwargs["torch_dtype"] = dtype

    base = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if not load_in_4bit:
        base = base.to(device)
    base.config.use_cache = False
    latent_id = ensure_latent_token(tokenizer, base, latent_token)
    if load_in_4bit:
        base = prepare_model_for_kbit_training(base)
    lora = get_peft_model(
        base,
        LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            target_modules=list(target_modules),
            lora_dropout=float(lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    return OuroborosCoconutForCausalLM(lora, latent_id), tokenizer, latent_id


def load_published_coconut(
    *,
    base_model: str,
    adapter_id_or_path: str,
    latent_token: str = DEFAULT_LATENT_TOKEN,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = True,
) -> tuple[OuroborosCoconutForCausalLM, Any, int]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device("auto") if device is None else device
    dtype = resolve_dtype("auto", device) if dtype is None else dtype
    tokenizer = AutoTokenizer.from_pretrained(adapter_id_or_path, use_fast=True, trust_remote_code=trust_remote_code)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    ).to(device)
    latent_id = ensure_latent_token(tokenizer, base, latent_token)
    model = PeftModel.from_pretrained(base, adapter_id_or_path, is_trainable=False).to(device)
    model.eval()
    return OuroborosCoconutForCausalLM(model, latent_id), tokenizer, latent_id


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
    embeds = model._fill_latents(input_ids, attention_mask)

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


def write_model_card(
    bundle_dir: Path,
    *,
    base_model: str,
    stage: int,
    latent_token: str,
) -> None:
    card = f"""---
library_name: peft
base_model: {base_model}
tags:
- peft
- lora
- coconut
- latent-reasoning
---

# Ouroboros

This repo is a PEFT adapter plus tokenizer for the Ouroboros Coconut runtime.
Load it with `ouroboros.coconut.load_published_coconut` and run
`ouroboros.coconut.generate`.

Runtime facts:

- Base model: `{base_model}`
- Latent token: `{latent_token}`
- Default latent stage used at export: `{stage}`

This is the simplified release bundle: adapter weights, tokenizer files, and a
small Coconut runtime helper.
"""
    (bundle_dir / "README.md").write_text(card, encoding="utf-8")


def save_release_bundle(
    model: OuroborosCoconutForCausalLM,
    tokenizer,
    bundle_dir: str | Path,
    *,
    base_model: str,
    stage: int,
    latent_token: str = DEFAULT_LATENT_TOKEN,
) -> Path:
    bundle = Path(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(bundle, safe_serialization=True)
    tokenizer.save_pretrained(bundle)
    write_model_card(bundle, base_model=base_model, stage=stage, latent_token=latent_token)
    shutil.copyfile(Path(__file__), bundle / "coconut.py")
    return bundle


def push_release_bundle(
    bundle_dir: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = True,
    commit_message: str = "Release Ouroboros Coconut bundle",
) -> str:
    from huggingface_hub import HfApi

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, token=token)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(bundle_dir),
        token=token,
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"
