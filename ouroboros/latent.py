"""Latent reasoning model wrapper and DGAC halt-gate utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from ouroboros.config import DEFAULT_BASE_MODEL, DEFAULT_LATENT_TOKEN, DEFAULT_LORA_TARGETS, DgacConfig
from ouroboros.utils import resolve_device, resolve_dtype


class HaltGate(nn.Module):
    """Small DGAC halt policy over consecutive latent states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Linear(2 * int(hidden_size), 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h_curr: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(torch.cat([h_curr, h_prev], dim=-1))).squeeze(-1)


@dataclass(frozen=True)
class LatentFill:
    embeds: torch.Tensor
    hidden_sequences: list[list[torch.Tensor]]


def compute_dgac_lambda1(step: int, warmup: int, ramp: int, lmax: float) -> float:
    if step < warmup:
        return 0.0
    return float(lmax) * min((int(step) - int(warmup)) / max(int(ramp), 1), 1.0)


def _last_hidden(outputs: Any) -> torch.Tensor:
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    raise RuntimeError("Model did not return hidden states; pass output_hidden_states=True.")


def _hidden_size(config: Any) -> int:
    for name in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer hidden size from model config.")


class OuroborosCoconutForCausalLM(nn.Module):
    """Coconut-style hidden-state feedback around any HuggingFace CausalLM."""

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

    @property
    def hidden_size(self) -> int:
        return _hidden_size(self.config)

    def get_input_embeddings(self):
        return self.base_causallm.get_input_embeddings()

    def set_input_embeddings(self, value) -> None:
        self.base_causallm.set_input_embeddings(value)

    def save_pretrained(self, *args, **kwargs):
        return self.base_causallm.save_pretrained(*args, **kwargs)

    def _fill_latents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        collect_hidden: bool = False,
    ) -> LatentFill:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        embeds = self.get_input_embeddings()(input_ids)
        latent_positions = (input_ids == self.latent_token_id).nonzero(as_tuple=False)
        sequences: list[list[torch.Tensor]] = [[] for _ in range(input_ids.size(0))]
        if latent_positions.numel() == 0:
            return LatentFill(embeds=embeds, hidden_sequences=sequences)

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
                latent_hidden = hidden[batch_idx, pos - 1, :]
                patched[batch_idx, pos, :] = latent_hidden.to(patched.dtype)
                if collect_hidden:
                    sequences[batch_idx].append(latent_hidden.unsqueeze(0))
            embeds = patched
        return LatentFill(embeds=embeds, hidden_sequences=sequences)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        *,
        halt_gate: HaltGate | None = None,
        dgac: DgacConfig | None = None,
        global_step: int = 0,
        **_: Any,
    ) -> CausalLMOutputWithPast:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        fill = self._fill_latents(input_ids, attention_mask, collect_hidden=halt_gate is not None)
        outputs = self.base_causallm(
            inputs_embeds=fill.embeds,
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
            if halt_gate is not None and dgac is not None and dgac.enabled:
                loss = loss + dgac_regularization_loss(
                    fill.hidden_sequences,
                    halt_gate=halt_gate,
                    config=dgac,
                    global_step=global_step,
                    device=self.device,
                )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=getattr(outputs, "attentions", None),
        )


def dgac_regularization_loss(
    hidden_sequences: list[list[torch.Tensor]],
    *,
    halt_gate: HaltGate,
    config: DgacConfig,
    global_step: int,
    device: torch.device,
) -> torch.Tensor:
    lambda1 = compute_dgac_lambda1(
        global_step,
        config.warmup_steps,
        config.ramp_steps,
        config.lambda_ponder_max,
    )
    terms: list[torch.Tensor] = []
    for sequence in hidden_sequences:
        if len(sequence) < 2:
            continue
        remainder = torch.ones(1, device=device, dtype=torch.float32)
        ponder = torch.zeros(1, device=device, dtype=torch.float32)
        diversity = torch.zeros(1, device=device, dtype=torch.float32)
        for idx in range(1, len(sequence)):
            h_curr = sequence[idx].to(device=device, dtype=torch.float32)
            h_prev = sequence[idx - 1].to(device=device, dtype=torch.float32)
            halt_prob = halt_gate(h_curr, h_prev)
            ponder = ponder + remainder
            if idx < len(sequence) - 1:
                remainder = remainder * (1.0 - halt_prob)
            diversity = diversity + F.relu(F.cosine_similarity(h_curr, h_prev, dim=-1) - config.tau)
        terms.append(float(lambda1) * ponder.mean() + config.lambda_diversity * diversity.mean())
    if not terms:
        return torch.zeros((), device=device, dtype=torch.float32)
    return torch.stack(terms).mean()


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
    base.config.use_cache = False
    latent_id = ensure_latent_token(tokenizer, base, latent_token)
    model = PeftModel.from_pretrained(base, adapter_id_or_path, is_trainable=False).to(device)
    model.eval()
    return OuroborosCoconutForCausalLM(model, latent_id), tokenizer, latent_id
