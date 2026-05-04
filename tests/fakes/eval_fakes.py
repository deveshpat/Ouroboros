from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch
from torch import nn


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        # Keep ids inside the fake model vocabulary.
        self._vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self._next_id = 3

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True):
        assert tokenize is False
        prefix = "\n".join(str(message.get("content", "")) for message in messages)
        return prefix + ("\nAssistant: " if add_generation_prompt else "")

    def encode(self, text: str, add_special_tokens: bool = False):
        ids = []
        for token in text.replace("\n", " ").split():
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[token] % 31)
        return ids or [1]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True):
        ids = list(ids)
        if skip_special_tokens:
            ids = [idx for idx in ids if idx not in {self.pad_token_id, self.eos_token_id}]
        return " ".join(str(idx) for idx in ids)


class FakeBackbone(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 8) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.grad_enabled_observations: list[bool] = []
        self.device_type_observations: list[str] = []

    def forward(self, inputs_embeds, attention_mask=None, use_cache: bool = False):
        self.grad_enabled_observations.append(torch.is_grad_enabled())
        self.device_type_observations.append(inputs_embeds.device.type)
        # Make every step deterministic while still depending on the input shape.
        return SimpleNamespace(last_hidden_state=inputs_embeds + 0.125)


class FakeCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 8) -> None:
        super().__init__()
        self.model = FakeBackbone(vocab_size=vocab_size, hidden_size=hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            # Always emit eos for generation so tests stay tiny and deterministic.
            self.lm_head.weight[2].fill_(1.0)
        self.train_calls = 0
        self.eval_calls = 0

    def train(self, mode: bool = True):
        if mode:
            self.train_calls += 1
        return super().train(mode)

    def eval(self):
        self.eval_calls += 1
        return super().eval()

    def save_pretrained(self, path: str) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"lm_head.weight": self.lm_head.weight.detach().cpu()}, out / "adapter_model.safetensors")
        (out / "adapter_config.json").write_text("{}", encoding="utf-8")


class FakeHaltGate(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.train_calls = 0
        self.eval_calls = 0

    def forward(self, h_curr, h_prev):
        return torch.zeros(h_curr.size(0), device=h_curr.device)

    def train(self, mode: bool = True):
        if mode:
            self.train_calls += 1
        return super().train(mode)

    def eval(self):
        self.eval_calls += 1
        return super().eval()
