"""
model.py
========
Ouroboros: Coconut latent-reasoning Jamba. A real JambaForCausalLM subclass —
not a wrapper around one — so every standard causal-LM entry point
(forward, generate, gradient_checkpointing_enable, save_pretrained, .device,
.dtype, device_map/dtype-aware loading, ...) is inherited rather than
reimplemented.

The one structural override is forward(): latent passes run there,
unconditionally, whenever <|lat|> tokens are present in input_ids. There is
no separate method to remember to call and no flag that defaults to "off" —
that data-driven trigger is what makes latent reasoning mandatory rather
than optional.

Deliberately NOT here: tokenization, chat templating, text-in/text-out
generation helpers, or anything else that's solely an lm-eval-compatibility
concern. Those belong to a separate bridge class once that's needed —
this file's job is to be a correct, minimal, standard-shaped model class.
`.generate()` already works with raw token ids via the inherited
GenerationMixin; callers tokenize and append <|lat|> ids themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.jamba.configuration_jamba import JambaConfig
from transformers.models.jamba.modeling_jamba import JambaForCausalLM

__all__ = ["Ouroboros", "OuroborosConfig", "OuroborosCausalLMOutputWithPast", "HaltGate"]

_DEFAULT_ADAPTER_REPO = "WeirdRunner/Ouroboros"
_DEFAULT_BASE_MODEL = "ai21labs/AI21-Jamba-Reasoning-3B"
_LAT_TOKEN = "<|lat|>"
# B4: 0.9, not 0.5. A barely-trained gate oscillates around 0.5, so a strict
# prob > 0.5 there is maximally sensitive to noise. 0.9 is the value that
# works once the gate has had any DGAC supervision at all; 0.5 is only
# appropriate for a fully-trained gate, which is never the case mid-curriculum.
# Calibrate after DGAC training if a different threshold fits a given checkpoint.
_DEFAULT_HALT_THRESHOLD = 0.9

# LoRA target modules for Jamba's hybrid attention+Mamba+MoE stack: the four
# attention projections plus the three Mamba SSM projections plus the MoE
# expert output projection. Kept here (not imported from a deleted module) so
# model.py stays self-contained.
_DEFAULT_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj", "x_proj", "dt_proj", "out_proj",
]


# ── config ───────────────────────────────────────────────────────────────────

class OuroborosConfig(JambaConfig):
    """
    JambaConfig + exactly the fields forward() actually reads. Round-trips
    through config.json on save_pretrained / push_to_hub like every other
    field below — no separate bookkeeping needed at call sites.

    stage_k is deliberately not a field here: forward() derives each row's
    latent depth from how many <|lat|> tokens that row's input_ids actually
    contains, so the model itself never needs a "how many passes" default.
    Curriculum-stage bookkeeping is a training-harness concern; if a
    recorded default ever turns out to be useful for callers, it's a
    one-line add, just not adding it speculatively.
    """

    model_type = "ouroboros"

    lat_token_id: int = 65536       # overwritten with the real id in from_pretrained
    halt_threshold: float = _DEFAULT_HALT_THRESHOLD
    use_halt_gate: bool = True
    # B7: never tie embed_tokens to lm_head. After resize_token_embeddings adds
    # the <|lat|> row, tying would overwrite lm_head's new row with embed's (or
    # vice versa) and pollute every load report with "will NOT tie them".
    # Set explicitly in the load classmethods too (instance attr before resize)
    # because JambaConfig's __init__ may not forward this kwarg.
    tie_word_embeddings: bool = False
    # P0: opt-in Mamba/KV cache for the latent-pass loop. Default False keeps
    # _run_latent_passes byte-identical to the use_cache=False recompute path;
    # the cache path is inference-only (gated on not torch.is_grad_enabled()).
    use_latent_cache: bool = False


# ── output ───────────────────────────────────────────────────────────────────

@dataclass
class OuroborosCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    """
    MoeCausalLMOutputWithPast + latent-pass bookkeeping for DGAC.

    actual_n_latents / hidden_sequences populate only when
    output_hidden_sequences=True is passed to forward() — same opt-in shape
    output_attentions / output_router_logits already use. Every call site
    that only wants .loss (every training/eval call) is unaffected.
    """

    actual_n_latents: Optional[torch.LongTensor] = None
    hidden_sequences: Optional[List[List[torch.Tensor]]] = None


# ── halt gate ────────────────────────────────────────────────────────────────

class HaltGate(nn.Module):
    """Zero-initialized DGAC gate — halt_prob ≈ 0.5 at the start of training.

    Input: h_curr [B, D], h_prev [B, D] — consecutive latent hidden states.
    Output: halt_prob [B].
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h_curr: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(torch.cat([h_curr, h_prev], dim=-1))).squeeze(-1)


# ── model ────────────────────────────────────────────────────────────────────

class Ouroboros(JambaForCausalLM):
    """
    Coconut latent-reasoning over AI21-Jamba-Reasoning-3B.

    IS a JambaForCausalLM. forward() is the only structural override:
    whenever input_ids contains <|lat|> tokens and there's no KV cache yet
    (a prefill step — the first generate() call, or a full teacher-forced
    training/eval batch), it runs latent passes over each row's
    question-only prefix and splices the resulting hidden states into the
    <|lat|> positions before delegating to the parent forward for the real
    backbone pass, loss, and lm_head.

    Training/eval: call exactly like any HF causal LM —
        out = model(input_ids=..., attention_mask=..., labels=...)
        out.loss
    Generation: tokenize, append config.lat_token_id stage_k times, then
        model.generate(input_ids=..., **generate_kwargs)   # inherited, untouched
    """

    config_class = OuroborosConfig

    def __init__(self, config: OuroborosConfig) -> None:
        super().__init__(config)
        self.halt_gate = HaltGate(config.hidden_size) if config.use_halt_gate else None
        # Deliberately not calling self.post_init() again: HaltGate's explicit
        # zero-init above must not be overwritten by the generic config-driven
        # weight initializer, which already ran once inside super().__init__()
        # before this module existed.

    # ── construction ─────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        adapter_repo: str = _DEFAULT_ADAPTER_REPO,
        *,
        base_model_id: str = _DEFAULT_BASE_MODEL,
        torch_dtype: Union[str, torch.dtype] = "auto",
        device_map: Optional[Union[str, dict]] = None,
        load_in_4bit: bool = False,
        halt_threshold: float = _DEFAULT_HALT_THRESHOLD,
        use_halt_gate: bool = True,
        halt_gate_subfolder: str = "",
        token: Optional[str] = None,
        **kwargs: Any,
    ) -> "Ouroboros":
        """
        Two-repo load: base Jamba weights (`base_model_id`) + the Ouroboros
        LoRA adapter and halt_gate.pt (`adapter_repo`). Neither repo alone is
        a complete checkpoint (adapter_repo has no full-size weights at all),
        so this can't be a single call to the inherited
        PreTrainedModel.from_pretrained the way a normal single-repo HF model
        would load — named adapter_repo rather than the conventional
        pretrained_model_name_or_path specifically so that two-repo reality
        is visible at the call site instead of implied.

        halt_gate_subfolder defaults to the adapter repo root. Pass it
        explicitly (e.g. "diloco_state/anchor") when the gate checkpoint
        you want lives in a subfolder instead — deliberately not auto-
        discovered, so loading a specific (possibly experimental/undertrained)
        gate checkpoint is always something the caller opted into, not
        something that happened to be found.
        """
        from huggingface_hub import hf_hub_download
        from peft import LoraConfig, inject_adapter_in_model
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, BitsAndBytesConfig

        # Tokenizer is loaded transiently, purely to size the vocab and find
        # <|lat|>'s id — not kept as a model attribute. Tokenization is a
        # caller concern, not this class's.
        tokenizer = AutoTokenizer.from_pretrained(adapter_repo, token=token)
        lat_id = tokenizer.convert_tokens_to_ids(_LAT_TOKEN)
        if lat_id is None or lat_id == tokenizer.unk_token_id:
            raise ValueError(f"{_LAT_TOKEN!r} not found in the {adapter_repo} tokenizer vocab.")
        target_vocab_size = len(tokenizer)

        config = OuroborosConfig.from_pretrained(base_model_id, token=token)
        config.use_mamba_kernels = torch.cuda.is_available()
        config.lat_token_id = int(lat_id)
        config.halt_threshold = float(halt_threshold)
        config.use_halt_gate = bool(use_halt_gate)
        config.tie_word_embeddings = False   # B7: set on the instance before resize
        config.use_latent_cache = False      # P0: off by default at load time

        resolved_dtype = cls._resolve_dtype(torch_dtype)
        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
            "token": token,
            **kwargs,
        }

        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=resolved_dtype if resolved_dtype != torch.float32 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["torch_dtype"] = resolved_dtype
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        # Inherited PreTrainedModel.from_pretrained does the heavy lifting:
        # sharded-checkpoint loading, device_map dispatch, dtype casting,
        # quantization. cls(config) constructs an Ouroboros (not a plain
        # JambaForCausalLM) because `cls` is bound to whatever class
        # from_pretrained was actually called on.
        model = super().from_pretrained(base_model_id, config=config, **load_kwargs)

        # The adapter checkpoint's embed_tokens/lm_head are modules_to_save at
        # target_vocab_size (the base checkpoint is one token short — <|lat|>).
        # Shapes must match before set_peft_model_state_dict can load them.
        if target_vocab_size != model.config.vocab_size:
            model.resize_token_embeddings(target_vocab_size)

        # In-place LoRA injection, NOT get_peft_model/PeftModel.from_pretrained
        # — those wrap the model, so the returned object's type would be
        # PeftModel, not Ouroboros. inject_adapter_in_model mutates in place;
        # model.forward stays OUR forward.
        lora_config = LoraConfig.from_pretrained(adapter_repo, token=token)
        inject_adapter_in_model(lora_config, model)
        adapter_weights_path = hf_hub_download(adapter_repo, "adapter_model.safetensors", token=token)
        raw_state = load_file(adapter_weights_path)
        # The pre-rewrite (May-15) anchor saves keys under
        # base_model.model._ouro_cache_backbone...; the flat rewrite renamed that
        # namespace to model.layers... (same weights, same latent-pass math — see
        # remap_adapter.py). Remap old-format checkpoints into the current
        # namespace before load; post-rewrite saves pass through unchanged.
        from remap_adapter import load_remapped, load_lora_into_model
        remapped = load_remapped(raw_state)
        # Load the adapter directly. peft.set_peft_model_state_dict's
        # transformers-v5 conversion path is broken on peft 0.19.1 +
        # transformers 5.10.2 (WeightConverter kwarg mismatch in the MoE branch,
        # hit even for num_experts=1), and the conversion is unnecessary here —
        # load_remapped already yields the canonical key format. load_lora_into_model
        # matches each saved tensor to its model parameter and returns counts so a
        # wrong remap (silently-zero LoRA / untrained base) is caught, not hidden.
        loaded, skipped = load_lora_into_model(model, remapped)
        n_saved = len(remapped)
        if loaded != n_saved:
            raise RuntimeError(
                f"Adapter load mismatch: {loaded}/{n_saved} trained tensors loaded "
                f"({skipped} had no matching parameter). The remap is likely wrong for this checkpoint."
            )

        if model.halt_gate is not None:
            gate_filename = f"{halt_gate_subfolder.strip('/')}/halt_gate.pt".lstrip("/")
            try:
                gate_path = hf_hub_download(adapter_repo, gate_filename, token=token)
            except Exception as exc:
                print(
                    f"  [warn] {adapter_repo}/{gate_filename} not found ({type(exc).__name__}); "
                    "running with halt_gate=None (fixed-depth latent inference — full requested "
                    "depth every call, no early stop)."
                )
                model.halt_gate = None  # absent -> fixed-depth fallback
            else:
                state = torch.load(gate_path, map_location="cpu", weights_only=True)
                model.halt_gate.load_state_dict(state)
                # Pin fp32 explicitly rather than relying on a from_pretrained
                # dtype-cast exemption mechanism — simple, and impossible to
                # get subtly wrong regardless of internal cast ordering.
                model.halt_gate = model.halt_gate.float().to(model.device)
                print(f"  [info] loaded halt_gate.pt from {adapter_repo}/{gate_filename}")

        return model.eval()

    # ── construction: training ──────────────────────────────────────────

    @classmethod
    def for_training(
        cls,
        base_model_id: str = _DEFAULT_BASE_MODEL,
        tokenizer: Any = None,
        *,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list[str]] = None,
        use_halt_gate: bool = True,
        halt_threshold: float = _DEFAULT_HALT_THRESHOLD,
        device: Optional[torch.device] = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        **kwargs: Any,
    ) -> "Ouroboros":
        """
        Training constructor: a real Ouroboros (so Ouroboros.forward runs the
        latent passes) over freshly-loaded base Jamba weights with a FRESH,
        randomly-initialized LoRA adapter injected in place and a zero-init
        HaltGate. Nothing is loaded from an adapter repo — this is the start of
        a curriculum, not a resume.

        Mirrors from_pretrained's two-step shape (base weights + in-place LoRA)
        but skips every "load existing adapter/gate" step and instead freezes
        the base so only LoRA + halt_gate + the resized embed/lm_head rows train.

        tokenizer must already have <|lat|> added; it is read transiently to
        size the vocab and find lat's id (not stored on the model — tokenization
        is a caller concern, same as from_pretrained).
        """
        from peft import LoraConfig, inject_adapter_in_model

        if tokenizer is None:
            raise ValueError("for_training requires a tokenizer with <|lat|> already added.")
        lat_id = tokenizer.convert_tokens_to_ids(_LAT_TOKEN)
        if lat_id is None or lat_id == tokenizer.unk_token_id:
            raise ValueError(f"{_LAT_TOKEN!r} not found in the tokenizer vocab.")
        target_vocab_size = len(tokenizer)

        config = OuroborosConfig.from_pretrained(base_model_id)
        config.use_mamba_kernels = torch.cuda.is_available()
        config.lat_token_id = int(lat_id)
        config.halt_threshold = float(halt_threshold)
        config.use_halt_gate = bool(use_halt_gate)
        config.tie_word_embeddings = False   # B7
        config.use_latent_cache = False      # P0: training always uses the no-cache path

        resolved_dtype = cls._resolve_dtype(torch_dtype)
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
            "torch_dtype": resolved_dtype,
            **kwargs,
        }
        if device is not None and device.type == "cuda":
            load_kwargs["device_map"] = {"": device.index if device.index is not None else 0}

        model = super().from_pretrained(base_model_id, config=config, **load_kwargs)
        model.config.use_cache = False

        if target_vocab_size != model.config.vocab_size:
            model.resize_token_embeddings(target_vocab_size)

        # In-place LoRA injection (NOT get_peft_model) keeps the object an
        # Ouroboros, so model.forward stays OUR forward. Fresh A/B weights — no
        # set_peft_model_state_dict call — the adapter starts at zero delta.
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules or _DEFAULT_LORA_TARGET_MODULES,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        inject_adapter_in_model(lora_config, model)

        # inject_adapter_in_model does NOT freeze the base the way get_peft_model
        # does, so freeze everything then unfreeze exactly what should train:
        # the LoRA deltas, the HaltGate, and the resized embedding/lm_head rows
        # (the <|lat|> row must be learned; the rest of embed/lm_head stays
        # trainable too, matching how the existing checkpoint treats them as
        # modules_to_save).
        for param in model.parameters():
            param.requires_grad_(False)
        for name, param in model.named_parameters():
            if "lora_" in name or "halt_gate" in name:
                param.requires_grad_(True)
        for embed_name in ("model.embed_tokens.weight", "lm_head.weight"):
            try:
                model.get_parameter(embed_name).requires_grad_(True)
            except Exception:
                pass  # sharded/renamed path — embed/lm_head freezing is best-effort

        if model.halt_gate is not None:
            model.halt_gate = model.halt_gate.float()
        if device is not None and device.type != "cuda":
            model = model.to(device)

        return model.train()

    # ── checkpointing: adapter-only save ────────────────────────────────

    def save_adapter(self, output_dir: str) -> None:
        """
        Save ONLY the trained delta — LoRA weights, the HaltGate, and the
        resized embed_tokens/lm_head — not the ~6GB base model. The base
        Jamba weights are frozen and reloaded from the base repo, so writing
        them would be pure disk waste (the disk-overflow trap on Kaggle).

        Format mirrors from_pretrained's load contract: adapter_model.safetensors
        (lora_* keys + embed_tokens/lm_head) + halt_gate.pt + adapter_config.json,
        reloadable by set_peft_model_state_dict + a strict=False embed/lm_head load.
        """
        import json
        import os

        from peft import get_peft_model_state_dict

        os.makedirs(output_dir, exist_ok=True)

        # LoRA delta (+ resized embed/lm_head, which carry the trained <|lat|> row).
        try:
            adapter_sd = get_peft_model_state_dict(self)
        except Exception:
            # Fallback for an in-place-injected (non-PeftModel) adapter: the
            # lora_ keys + the modules_to_save embeddings. Guaranteed-correct
            # even if get_peft_model_state_dict expects a peft_config attribute.
            keep = ("lora_",)
            adapter_sd = {
                k: v for k, v in self.state_dict().items()
                if any(kk in k for kk in keep)
            }
            for embed_name in ("model.embed_tokens.weight", "lm_head.weight"):
                if embed_name in self.state_dict():
                    adapter_sd[embed_name] = self.state_dict()[embed_name]

        try:
            from safetensors.torch import save_file
            save_file(adapter_sd, os.path.join(output_dir, "adapter_model.safetensors"))
        except ImportError:
            torch.save(adapter_sd, os.path.join(output_dir, "adapter_model.bin"))

        if self.halt_gate is not None:
            torch.save(self.halt_gate.state_dict(), os.path.join(output_dir, "halt_gate.pt"))

        # Minimal adapter_config.json so the adapter is self-describing and
        # PeftModel.from_pretrained / set_peft_model_state_dict can reload it.
        # r/alpha aren't recoverable from an in-place-injected model (no
        # peft_config), so best-effort: the loader mainly needs target_modules
        # + base_model_name_or_path + modules_to_save.
        adapter_config = {
            "peft_type": "LORA",
            "base_model_name_or_path": getattr(self.config, "_name_or_path", _DEFAULT_BASE_MODEL),
            "target_modules": _DEFAULT_LORA_TARGET_MODULES,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "modules_to_save": ["embed_tokens", "lm_head"],
        }
        with open(os.path.join(output_dir, "adapter_config.json"), "w", encoding="utf-8") as fh:
            json.dump(adapter_config, fh, indent=2)

    # ── forward: the one structural override ────────────────────────────

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_sequences: bool = False,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[OuroborosCausalLMOutputWithPast, MoeCausalLMOutputWithPast, tuple]:
        """
        Same call signature any training loop / GenerationMixin already
        expects from a causal LM. Training/eval code calls this exactly like
        model(input_ids=..., attention_mask=..., labels=...) and reads
        .loss — there's no forward_batch()-style wrapper to remember, and no
        path through this method that can silently skip latent reasoning
        when <|lat|> tokens are present.

        has_cache distinguishes prefill (run latent passes) from a
        mid-generation decode step (don't — the latents from the prefill
        step are already baked into the KV cache; re-injecting would be
        both wrong and wasted work).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        has_cache = past_key_values is not None and bool(past_key_values.get_seq_length())

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        actual_n_latents: Optional[torch.LongTensor] = None
        hidden_sequences: Optional[List[List[torch.Tensor]]] = None
        if not has_cache and input_ids is not None:
            inputs_embeds, actual_n_latents, hidden_sequences = self._inject_latents(
                input_ids,
                inputs_embeds,
                attention_mask,
                output_hidden_sequences=output_hidden_sequences,
            )

        # return_dict is forced True here regardless of the caller's wish so
        # base_out is always the structured ModelOutput we can attach fields
        # to (or pass straight through) — tuple conversion for a caller that
        # asked for one happens once, at the very end, on whichever object
        # is actually being returned.
        base_out = super().forward(
            input_ids=None,  # JambaModel.forward requires exactly one of input_ids/inputs_embeds
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            logits_to_keep=logits_to_keep,
            return_dict=True,
            **kwargs,
        )

        if actual_n_latents is None:
            return base_out if return_dict else base_out.to_tuple()

        wrapped = OuroborosCausalLMOutputWithPast(
            **base_out, actual_n_latents=actual_n_latents, hidden_sequences=hidden_sequences,
        )
        return wrapped if return_dict else wrapped.to_tuple()

    # ── latent passes (private) ──────────────────────────────────────────

    def _inject_latents(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor],
        *,
        output_hidden_sequences: bool,
    ) -> tuple[torch.FloatTensor, torch.LongTensor, Optional[List[List[torch.Tensor]]]]:
        """
        Derive q_lens / n_latents directly from <|lat|> token positions in
        input_ids (self.config.lat_token_id) — no separate collate-time
        fields needed; input_ids is the single source of truth. A row's
        <|lat|> tokens are assumed contiguous (true by construction wherever
        Coconut samples are built: question_ids + [lat_id]*n_latent +
        supervised_ids). Rows with zero <|lat|> tokens get n_latent=0 and
        pass through unchanged.

        Runs latent passes over each row's question-only prefix, then
        splices the resulting hidden states back into inputs_embeds at the
        original <|lat|> positions.
        """
        B, L = input_ids.shape
        device = input_ids.device

        is_lat = input_ids == self.config.lat_token_id          # [B, L] bool
        n_latents = is_lat.sum(dim=1)                            # [B] long

        # First <|lat|> position per row, or L (past every real index) for
        # rows with none — unambiguous regardless of argmax tie-breaking,
        # since L is guaranteed larger than any real position.
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        sentinel = torch.where(is_lat, positions, torch.full_like(positions, L))
        q_lens = sentinel.min(dim=1).values                      # [B] long

        if attention_mask is None:
            attn_bool = torch.ones((B, L), dtype=torch.bool, device=device)
        else:
            attn_bool = attention_mask.bool()
        ctx_mask = (positions < q_lens.unsqueeze(1)) & attn_bool

        latent_ctx, _, actual_k = self._run_latent_passes(inputs_embeds, ctx_mask, n_latents)

        patched = inputs_embeds.clone()
        max_k = int(actual_k.max().item()) if actual_k.numel() else 0
        for step in range(max_k):
            active = (actual_k > step).nonzero(as_tuple=False).flatten()
            if active.numel() == 0:
                break
            inject_pos = q_lens[active] + step
            in_bounds = inject_pos < L
            if not bool(in_bounds.all()):
                active, inject_pos = active[in_bounds], inject_pos[in_bounds]
            if active.numel() == 0:
                continue
            h = latent_ctx[active, L + step, :].to(patched.dtype)
            patched[active, inject_pos, :] = h

        hidden_sequences = (
            self._collect_hidden_sequences(latent_ctx, L, actual_k) if output_hidden_sequences else None
        )
        return patched, actual_k, hidden_sequences

    def _run_latent_passes(
        self,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor,
        target_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dispatch: P0 cache path (inference-only + opt-in) or the proven no-cache
        recompute path. The no-cache path is what training always uses (and the
        cache path's correctness fallback), so it lives in its own method —
        _run_latent_passes_nocache — to let the cache path fall back WITHOUT
        re-entering this dispatch (which would recurse under use_latent_cache +
        no_grad).
        """
        device = ctx.device
        B = ctx.size(0)
        target_k = target_k.to(device=device, dtype=torch.long)
        max_k = int(target_k.max().item()) if target_k.numel() else 0
        if max_k <= 0:
            return ctx, ctx_mask, torch.zeros(B, dtype=torch.long, device=device)

        # P0: the cache path is INFERENCE-ONLY (no grad) AND opt-in. Under
        # training (grad enabled) or when the flag is off, the no-cache path
        # runs unchanged — byte-identical to pre-P0 behaviour, so training math
        # is provably unaffected. The O(stage_k^2)->O(stage_k) win is realized
        # at inference/generation, where stage_k is largest; threading cache
        # under grad would reintroduce the in-place-cache-mutation autograd
        # problem (the reason the training path uses use_cache=False in the
        # first place).
        halt_gate = self.halt_gate if self.config.use_halt_gate else None
        if self.config.use_latent_cache and not torch.is_grad_enabled():
            return self._run_latent_passes_cached(ctx, ctx_mask, target_k, halt_gate)
        return self._run_latent_passes_nocache(ctx, ctx_mask, target_k, halt_gate)

    def _run_latent_passes_nocache(
        self,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor,
        target_k: torch.Tensor,
        halt_gate: Optional["HaltGate"],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The proven use_cache=False recompute path: run up to target_k.max()
        Coconut latent passes in embedding space.

        Each pass: forward self.model over each active row's current
        question+latents prefix, take the last valid hidden state, optionally
        query self.halt_gate, append as a new column. ctx_mask marks valid
        positions; ctx itself is passed at full width (only masked positions
        are read), avoiding an extra crop/clone of the embeddings.

        Calls self.model(...) directly, never self(...)/self.forward(...),
        which would recurse back into latent injection.

        No grad/autocast context is applied here — both are the caller's
        ambient context (no_grad / inference_mode / full grad for training;
        torch.autocast or none), exactly like calling self.model(...) from
        anywhere else. autocast, being a context manager active for the
        whole call stack, already covers every nested call here for free
        when a caller has one active around the outer forward() call.
        """
        device = ctx.device
        B = ctx.size(0)
        max_k = int(target_k.max().item()) if target_k.numel() else 0

        actual_k = torch.zeros(B, dtype=torch.long, device=device)
        halted = torch.zeros(B, dtype=torch.bool, device=device)
        prev_h = ctx.new_zeros(B, ctx.size(-1))

        for step in range(max_k):
            active = ((target_k > step) & ~halted).nonzero(as_tuple=False).flatten()
            if active.numel() == 0:
                break

            plen = ctx_mask[active].sum(dim=1)
            max_pl = int(plen.max().item())
            pfx = ctx[active, :max_pl]
            pmask = torch.arange(max_pl, device=device).unsqueeze(0) < plen.unsqueeze(1)
            pfx = torch.where(pmask.unsqueeze(-1), pfx, pfx.new_zeros(1, 1, pfx.size(-1)))

            out = self.model(inputs_embeds=pfx, attention_mask=pmask, use_cache=False)
            h = out.last_hidden_state
            last = (plen - 1).clamp(min=0)
            h_step = h[torch.arange(active.numel(), device=h.device), last.to(h.device)].to(device)

            append = torch.ones(active.numel(), dtype=torch.bool, device=device)
            if halt_gate is not None:
                has_prev = actual_k[active] > 0
                if bool(has_prev.any()):
                    prob = halt_gate(h_step[has_prev].float(), prev_h[active[has_prev]].float())
                    stop = prob > self.config.halt_threshold
                    if bool(stop.any()):
                        local_stop = has_prev.nonzero(as_tuple=False).flatten()[stop]
                        append[local_stop] = False
                        halted[active[local_stop]] = True

            new_col = ctx.new_zeros(B, 1, ctx.size(-1))
            new_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            if bool(append.any()):
                ai = active[append]
                new_col[ai, 0] = h_step[append].to(ctx.dtype)
                new_mask[ai, 0] = True
                actual_k[ai] += 1
                prev_h[ai] = h_step[append]

            ctx = torch.cat([ctx, new_col], dim=1)
            ctx_mask = torch.cat([ctx_mask, new_mask], dim=1)

        return ctx, ctx_mask, actual_k

    def _run_latent_passes_cached(
        self,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor,
        target_k: torch.Tensor,
        halt_gate: Optional["HaltGate"],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        P0: inference-only cache-backed latent passes. Forward the question
        prefix ONCE with use_cache=True, then forward only the single new
        latent column each step with past_key_values=<live cache> — attention
        sublayers go O(stage_k^2)->O(stage_k), and Mamba SSM state goes to
        genuinely O(1) per step IF Jamba's hybrid cache supports single-token
        incremental update (runtime-verify; if not, the attention win alone
        stands and correctness is unaffected).

        Reached only when config.use_latent_cache and not torch.is_grad_enabled()
        (see _run_latent_passes), so no cache-cloning-for-autograd is needed.

        HaltGate + shared batch cache: forward the new column for the FULL
        batch, gate AFTER. The cache advances by 1 position for every row
        uniformly, so per-row sequence positions stay aligned. Halted rows get
        a zero column forwarded (their cache position advances over a masked
        position) but the column is discarded — never appended to ctx, never
        read again (their actual_k is frozen). The dirty halted-row Mamba state
        is harmless because this cache is local and dropped after the loop; only
        ctx/ctx_mask/actual_k escape, matching the no-cache path's contract.
        If the backbone returns no cache, fall back to the no-cache path.
        """
        device = ctx.device
        B = ctx.size(0)

        # The cache path is best-effort: some configs can't drive use_cache=True
        # through Jamba's hybrid cache. The concrete case is an all-attention
        # config (no Mamba/LinearAttention layers): JambaModel._update_mamba_mask
        # calls past_key_values.has_previous_state(), which raises ValueError
        # "can only be called on LinearAttention layers" when the cache holds
        # only attention layers. Rather than special-case that, fall back to the
        # proven no-cache path on ANY backbone error here — correctness always
        # wins over the O(stage_k^2)->O(stage_k) speedup, and the fallback is
        # the exact path training uses (so outputs match). The real Jamba-3B
        # (which has Mamba layers) does NOT hit this and enjoys the cache win.
        try:
            return self._run_latent_passes_cached_impl(ctx, ctx_mask, target_k, halt_gate)
        except Exception:
            # Fall back to the no-cache path DIRECTLY (not via _run_latent_passes,
            # which would re-enter the dispatch and recurse under
            # use_latent_cache + no_grad). Correctness over the speedup.
            return self._run_latent_passes_nocache(ctx, ctx_mask, target_k, halt_gate)

    def _run_latent_passes_cached_impl(
        self,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor,
        target_k: torch.Tensor,
        halt_gate: Optional["HaltGate"],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cache-path body. Raises if the backbone rejects use_cache=True; the
        caller (_run_latent_passes_cached) catches and falls back to no-cache."""
        device = ctx.device
        B = ctx.size(0)

        initial_embeds = torch.where(
            ctx_mask.unsqueeze(-1), ctx, ctx.new_zeros((1, 1, ctx.size(-1)))
        )
        out0 = self.model(inputs_embeds=initial_embeds, attention_mask=ctx_mask, use_cache=True)
        cache = getattr(out0, "past_key_values", None)
        if cache is None:
            # Cache unsupported here — re-run the proven no-cache path.
            return self._run_latent_passes(ctx, ctx_mask, target_k)

        h = out0.last_hidden_state
        prefix_lens = ctx_mask.sum(dim=1).to(dtype=torch.long).clamp_min(1)
        last_pos = (prefix_lens - 1).to(h.device)
        h_step = h[torch.arange(B, device=h.device), last_pos, :].to(device)

        actual_k = torch.zeros(B, dtype=torch.long, device=device)
        halted = torch.zeros(B, dtype=torch.bool, device=device)
        prev_h = ctx.new_zeros(B, ctx.size(-1))
        max_k = int(target_k.max().item()) if target_k.numel() else 0

        for step in range(max_k):
            active = ((target_k > step) & ~halted).nonzero(as_tuple=False).flatten()
            if active.numel() == 0:
                break

            append = torch.ones(B, dtype=torch.bool, device=device)
            if halt_gate is not None and step > 0:
                has_prev = actual_k > 0
                gateable = has_prev & ~halted
                if bool(gateable.any()):
                    prob = halt_gate(h_step.float(), prev_h.float())
                    stop = prob > self.config.halt_threshold
                    append = append & ~(gateable & stop)
                    halted = halted | (gateable & stop)

            append_active = append & ~halted
            new_col = ctx.new_zeros(B, 1, ctx.size(-1))
            new_mask = append_active.view(B, 1)
            if bool(append_active.any()):
                new_col[append_active, 0] = h_step[append_active].to(ctx.dtype)
                actual_k[append_active] += 1
                prev_h[append_active] = h_step[append_active]

            ctx = torch.cat([ctx, new_col], dim=1)
            ctx_mask = torch.cat([ctx_mask, new_mask], dim=1)

            if step + 1 >= max_k:
                break
            # Forward the single new column through the cache for the FULL batch
            # (halted rows forward a zero column at a masked position — advances
            # their cache position uniformly, keeping all rows aligned).
            out = self.model(
                inputs_embeds=new_col, attention_mask=ctx_mask,
                past_key_values=cache, use_cache=True,
            )
            cache = out.past_key_values
            h_step = out.last_hidden_state[:, -1, :].to(device)

        return ctx, ctx_mask, actual_k

    @staticmethod
    def _collect_hidden_sequences(
        latent_ctx: torch.Tensor,
        original_len: int,
        actual_k: torch.Tensor,
    ) -> List[List[torch.Tensor]]:
        """Per-row list of latent hidden states, for DGAC's halt-gate loss."""
        B = latent_ctx.size(0)
        seqs: List[List[torch.Tensor]] = [[] for _ in range(B)]
        max_k = int(actual_k.max().item()) if actual_k.numel() else 0
        for step in range(max_k):
            active = (actual_k > step).nonzero(as_tuple=False).flatten()
            if active.numel() == 0:
                break
            h = latent_ctx[active, original_len + step, :]
            for local, sample in enumerate(active.tolist()):
                seqs[sample].append(h[local : local + 1])
        return seqs

    # ── static helpers ───────────────────────────────────────────────────

    @staticmethod
    def _resolve_dtype(torch_dtype: Union[str, torch.dtype]) -> torch.dtype:
        """
        "auto" picks FP16 on T4/V100 (sm<80) and BF16 on Ampere+ (sm>=80) —
        BF16 on pre-Ampere is software-emulated at FP32 throughput, not a
        real speed win, so "auto" deliberately isn't "match the checkpoint
        dtype" the way base PreTrainedModel.from_pretrained's torch_dtype
        ="auto" means; it's "pick the right dtype for this GPU."
        """
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        s = str(torch_dtype).strip().lower()
        if s != "auto":
            mapping = {
                "float32": torch.float32, "fp32": torch.float32,
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            }
            if s not in mapping:
                raise ValueError(f"Unsupported torch_dtype {torch_dtype!r}.")
            return mapping[s]
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.get_device_capability(0) >= (8, 0) else torch.float16
        return torch.float32


# ── Auto registration (B5) ───────────────────────────────────────────────────
# Without this, a saved config.json saying "model_type": "ouroboros" can only
# be reloaded via AutoModel.from_pretrained(..., trust_remote_code=True), and HF
# warns "using a model of type jamba to instantiate a model of type ouroboros"
# on our own loads. Registering the config/model_type pair makes the Ouroboros
# class the canonical mapping. Idempotent: register is dict-keyed, so re-import
# is a no-op; the try/except guards any version quirk where double-register
# raises. Runs at import time — safe because model.py already imports
# transformers unconditionally at module top.
try:
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("ouroboros", OuroborosConfig)
    AutoModelForCausalLM.register(OuroborosConfig, Ouroboros)
except Exception:
    # Already registered, or a transformers version that rejects it. Either way
    # not fatal: callers can still construct Ouroboros directly.
    pass
