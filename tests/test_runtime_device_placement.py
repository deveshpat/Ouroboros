from types import SimpleNamespace

import torch
import torch.nn as nn

from ouroboros.coconut.latent import prepare_latent_runtime
from ouroboros.models import runtime as model_runtime
from ouroboros.models.loading import _clear_model_handle_cache


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)

    def forward(self, *, input_ids=None, attention_mask=None, use_cache=False, **_kwargs):
        assert input_ids is not None
        assert attention_mask is not None
        assert input_ids.device.type == "cpu"
        assert attention_mask.device.type == "cpu"
        hidden = self.embed_tokens(input_ids)
        return SimpleNamespace(last_hidden_state=hidden)


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.config = SimpleNamespace(hidden_size=4)


def test_jamba_probe_uses_embedding_device_not_requested_fallback_device():
    model = TinyCausalLM()

    # ``meta`` is deliberately not where the embedding lives.  The regression
    # caught here was creating probe tensors on the requested/default CUDA device
    # even when a sharded model had placed embed_tokens elsewhere.
    model_runtime._run_jamba_probe_once(model, torch.device("meta"), torch.float32)


def test_prepare_latent_runtime_resolves_embedding_device():
    model = TinyCausalLM()

    runtime = prepare_latent_runtime(model, torch.device("meta"), torch.float32)

    assert runtime.device == torch.device("cpu")
    assert runtime.embed_tokens is model.model.embed_tokens


def test_clear_model_handle_cache_removes_stale_resize_handles():
    model = TinyCausalLM()
    model._ouro_cache_backbone = object()
    model._ouro_cache_embed_tokens = object()
    model._ouro_cache_lm_head = object()
    model.model._ouro_cache_embed_tokens = object()

    _clear_model_handle_cache(model)

    assert not hasattr(model, "_ouro_cache_backbone")
    assert not hasattr(model, "_ouro_cache_embed_tokens")
    assert not hasattr(model, "_ouro_cache_lm_head")
    assert not hasattr(model.model, "_ouro_cache_embed_tokens")
