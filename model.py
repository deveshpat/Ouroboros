"""
ouroboros/ouroboros.py
======================
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
_DEFAULT_HALT_THRESHOLD = 0.5


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
        """
        from huggingface_hub import hf_hub_download
        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
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
        config.lat_token_id = int(lat_id)
        config.halt_threshold = float(halt_threshold)
        config.use_halt_gate = bool(use_halt_gate)

        resolved_dtype = cls._resolve_dtype(torch_dtype)
        load_kwargs: dict[str, Any] = dict(token=token, **kwargs)

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": 'eager',
        }

        load_kwargs["use_mamba_kernels"] = torch.cuda.is_available()
        
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
        set_peft_model_state_dict(model, load_file(adapter_weights_path))

        if model.halt_gate is not None:
            try:
                gate_path = hf_hub_download(adapter_repo, "halt_gate.pt", token=token)
            except Exception:
                model.halt_gate = None  # absent on the Hub -> fixed-depth fallback
            else:
                state = torch.load(gate_path, map_location="cpu", weights_only=True)
                model.halt_gate.load_state_dict(state)
                # Pin fp32 explicitly rather than relying on a from_pretrained
                # dtype-cast exemption mechanism — simple, and impossible to
                # get subtly wrong regardless of internal cast ordering.
                model.halt_gate = model.halt_gate.float().to(model.device)

        return model.eval()

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
        Run up to target_k.max() Coconut latent passes in embedding space.

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
        target_k = target_k.to(device=device, dtype=torch.long)
        halt_gate = self.halt_gate if self.config.use_halt_gate else None

        max_k = int(target_k.max().item()) if target_k.numel() else 0
        if max_k <= 0:
            return ctx, ctx_mask, torch.zeros(B, dtype=torch.long, device=device)

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
