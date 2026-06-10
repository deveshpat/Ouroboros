"""Custom lm-evaluation-harness model class for Ouroboros.

Fixes both README boundaries:
  Boundary 1 — uses load_components + run_single_prompt instead of stock HF/PEFT
               backend; correct vocab size, <|lat|> token, latent passes, HaltGate.
  Boundary 2 — calls ensure_environment() inside __init__, which runs in each
               accelerate worker subprocess, not just the parent process.

Registration: @register_model("ouroboros") allows --model ouroboros in the CLI.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

# lm-eval imports — guarded so the module is importable even when lm-eval is absent.
try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
except ImportError:  # pragma: no cover
    LM = object  # type: ignore[assignment,misc]

    def register_model(*args, **kwargs):  # type: ignore[misc]
        def decorator(cls):
            return cls
        return decorator

from ouroboros.coconut.latent import prepare_latent_runtime, run_latent_passes
from ouroboros.inference.generation import (
    DEFAULT_HALT_THRESHOLD,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_STAGE_K,
    load_components,
    run_single_prompt,
)
from ouroboros.models.loading import module_first_device
from ouroboros.models.runtime import extract_last_hidden_state


@register_model("ouroboros")
class OuroborosLM(LM):
    """lm-eval model wrapper around the faithful Ouroboros inference stack."""

    # lm-eval checks this; we are text-only.
    MULTIMODAL = False

    def __init__(
        self,
        # ── model loading ──────────────────────────────────────────────────
        base_model: str = "ai21labs/AI21-Jamba-Reasoning-3B",
        adapter_repo: str = "WeirdRunner/Ouroboros",
        adapter_subfolder: str = "diloco_state/anchor",
        adapter_dir: str = "",
        adapter_cache_dir: str = "/kaggle/working/ouroboros_inference_adapter",
        # ── runtime ────────────────────────────────────────────────────────
        device: str = "auto",
        dtype: str = "auto",
        model_device_map: str = "single",
        use_4bit: bool = False,
        disable_mamba_kernels: bool = False,
        # ── latent ─────────────────────────────────────────────────────────
        stage_k: int = DEFAULT_STAGE_K,
        halt_threshold: float = DEFAULT_HALT_THRESHOLD,
        use_halt_gate: bool = True,
        # ── context budget ─────────────────────────────────────────────────
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        # ── Boundary 2: bootstrap inside this worker process ───────────────
        bootstrap: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Boundary 2 fix: ensure_environment() runs here, inside each accelerate
        # worker subprocess. The parent process bootstrap does NOT propagate into
        # worker processes; this call is the correct place.
        if bootstrap:
            from ouroboros.bootstrap import ensure_environment
            ensure_environment()

        args = SimpleNamespace(
            base_model=base_model,
            adapter_repo=adapter_repo,
            adapter_subfolder=adapter_subfolder,
            adapter_dir=adapter_dir or None,
            adapter_cache_dir=adapter_cache_dir,
            device=device,
            dtype=dtype,
            model_device_map=model_device_map,
            use_4bit=bool(use_4bit),
            disable_mamba_kernels=bool(disable_mamba_kernels),
            stage_k=int(stage_k),
            halt_threshold=float(halt_threshold),
            use_halt_gate=bool(use_halt_gate),
            require_halt_gate=False,
        )

        # Boundary 1 fix: load via faithful Ouroboros path — adds <|lat|>,
        # resizes embeddings, loads adapter + optional HaltGate.
        self._model, self._tokenizer, self._halt_gate, self._device = load_components(args)
        self._runtime = prepare_latent_runtime(self._model, self._device)
        self._stage_k = int(stage_k)
        self._halt_threshold = float(halt_threshold)
        self._dtype = dtype
        self._max_seq_len = int(max_seq_len)

    # ── lm-eval required properties ────────────────────────────────────────

    @property
    def eot_token_id(self) -> int:
        return self._tokenizer.eos_token_id  # type: ignore[return-value]

    @property
    def max_length(self) -> int:
        return self._max_seq_len

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        # Latent passes are inherently sequential per sample.
        # Batching across samples through the latent loop requires collation work
        # not yet implemented; stay at 1 for correctness.
        return 1

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def tokenizer_name(self) -> str:
        return getattr(
            self._tokenizer,
            "name_or_path",
            self._model.config._name_or_path,
        )

    # ── tokenizer helpers ──────────────────────────────────────────────────

    def tok_encode(self, string: str, **kwargs: Any) -> list[int]:
        return self._tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: list[int], **kwargs: Any) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    # ── latent context builder ─────────────────────────────────────────────

    def _build_latent_ctx(
        self, context_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed context and run K latent passes. Returns (ctx, ctx_mask)."""
        # Reserve tokens for latent passes so total stays within max_seq_len.
        budget = max(1, self._max_seq_len - self._stage_k)
        if len(context_ids) > budget:
            context_ids = context_ids[-budget:]

        q_tensor = torch.tensor(
            context_ids, device=self._device, dtype=torch.long
        ).unsqueeze(0)

        lat_args = SimpleNamespace(
            halt_threshold=self._halt_threshold,
            latent_cache=False,
            mac_mps_latent_cache=False,
        )

        with torch.inference_mode():
            ctx = self._runtime.embed_tokens(q_tensor)
            ctx_mask = torch.ones(
                (1, ctx.size(1)), dtype=torch.bool, device=self._device
            )
            ctx, ctx_mask, _ = run_latent_passes(
                runtime=self._runtime,
                ctx=ctx,
                ctx_mask=ctx_mask,
                n_latent=self._stage_k,
                halt_gate=self._halt_gate,
                args=lat_args,
            )
        return ctx, ctx_mask

    # ── lm-eval interface: generation ─────────────────────────────────────

    def generate_until(
        self, requests: list[Any], disable_tqdm: bool = False
    ) -> list[str]:
        results: list[str] = []
        for request in tqdm(requests, disable=disable_tqdm, desc="generate_until"):
            context, gen_kwargs = request.args
            until: list[str] = gen_kwargs.get("until", [self._tokenizer.eos_token or ""])
            max_gen_toks: int = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))

            gen_args = SimpleNamespace(
                gen_max_tokens=max_gen_toks,
                max_new_tokens=max_gen_toks,
                max_seq_len=self._max_seq_len,
                halt_threshold=self._halt_threshold,
                dtype=self._dtype,
                latent_cache=False,
                mac_mps_latent_cache=False,
            )

            with torch.inference_mode():
                result = run_single_prompt(
                    model=self._model,
                    tokenizer=self._tokenizer,
                    halt_gate=self._halt_gate,
                    prompt=context,
                    stage_k=self._stage_k,
                    device=self._device,
                    args=gen_args,
                    # lm-eval has already formatted the context; do not re-apply.
                    use_chat_template=False,
                )

            text = result.text
            for stop in until:
                if stop and stop in text:
                    text = text[: text.index(stop)]

            results.append(text)
        return results

    # ── lm-eval interface: loglikelihood ───────────────────────────────────

    def loglikelihood(
        self, requests: list[Any], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for request in tqdm(requests, disable=disable_tqdm, desc="loglikelihood"):
            context, continuation = request.args
            results.append(self._score_continuation(context, continuation))
        return results

    def _score_continuation(
        self, context: str, continuation: str
    ) -> tuple[float, bool]:
        """Score P(continuation | latent_context) under the Ouroboros model."""
        context_ids = self.tok_encode(context)
        cont_ids = self.tok_encode(continuation)
        if not cont_ids:
            return 0.0, True

        ctx, ctx_mask = self._build_latent_ctx(context_ids)
        latent_len = ctx.size(1)  # L + K (question tokens + K latent vectors)

        cont_tensor = torch.tensor(
            cont_ids, device=self._device, dtype=torch.long
        )
        with torch.inference_mode():
            with self._runtime.autocast():
                cont_embeds = self._runtime.embed_tokens(cont_tensor.unsqueeze(0))
                cont_mask = torch.ones(
                    (1, len(cont_ids)), dtype=torch.bool, device=self._device
                )
                full_ctx = torch.cat([ctx, cont_embeds], dim=1)
                full_mask = torch.cat([ctx_mask, cont_mask], dim=1)

                outputs = self._runtime.backbone(
                    inputs_embeds=full_ctx,
                    attention_mask=full_mask,
                    use_cache=False,
                )
                hidden = extract_last_hidden_state(outputs, "loglikelihood score")
                lm_head_device = module_first_device(
                    self._runtime.lm_head, self._device
                )
                logits = self._runtime.lm_head(
                    hidden.to(lm_head_device)
                )  # [1, latent_len + T, vocab]

        T = len(cont_ids)
        # Position latent_len-1 predicts cont_ids[0], ..., latent_len-1+T-1 predicts cont_ids[T-1]
        cont_logits = logits[0, latent_len - 1 : latent_len - 1 + T, :].float()
        cont_labels = torch.tensor(
            cont_ids, device=cont_logits.device, dtype=torch.long
        )

        log_probs = F.log_softmax(cont_logits, dim=-1)
        token_log_probs = log_probs[
            torch.arange(T, device=cont_logits.device), cont_labels
        ]
        total_ll = float(token_log_probs.sum().item())
        is_greedy = bool((cont_logits.argmax(dim=-1) == cont_labels).all().item())

        return total_ll, is_greedy

    # ── lm-eval interface: rolling loglikelihood ───────────────────────────

    def loglikelihood_rolling(
        self, requests: list[Any], disable_tqdm: bool = False
    ) -> list[float]:
        """Compute total log-likelihood over a string (used for perplexity tasks).

        Splits the token sequence into non-overlapping chunks of max_seq_len,
        running latent passes on the preceding chunk as context for each window.
        """
        results: list[float] = []
        for request in tqdm(requests, disable=disable_tqdm, desc="loglikelihood_rolling"):
            (string,) = request.args
            token_ids = self.tok_encode(string)
            if not token_ids:
                results.append(0.0)
                continue

            total_ll = 0.0
            chunk = self._max_seq_len - self._stage_k  # tokens scored per window
            chunk = max(chunk, 1)

            for start in range(0, len(token_ids), chunk):
                window = token_ids[start : start + chunk]
                context_ids = token_ids[max(0, start - chunk) : start]

                if context_ids:
                    ctx, ctx_mask = self._build_latent_ctx(context_ids)
                else:
                    # Empty context: single padding token
                    ctx, ctx_mask = self._build_latent_ctx(
                        [self._tokenizer.bos_token_id or 0]
                    )

                cont_tensor = torch.tensor(
                    window, device=self._device, dtype=torch.long
                )
                with torch.inference_mode():
                    with self._runtime.autocast():
                        cont_embeds = self._runtime.embed_tokens(cont_tensor.unsqueeze(0))
                        cont_mask = torch.ones(
                            (1, len(window)), dtype=torch.bool, device=self._device
                        )
                        full_ctx = torch.cat([ctx, cont_embeds], dim=1)
                        full_mask = torch.cat([ctx_mask, cont_mask], dim=1)

                        outputs = self._runtime.backbone(
                            inputs_embeds=full_ctx,
                            attention_mask=full_mask,
                            use_cache=False,
                        )
                        hidden = extract_last_hidden_state(outputs, "rolling loglikelihood")
                        lm_head_device = module_first_device(
                            self._runtime.lm_head, self._device
                        )
                        logits = self._runtime.lm_head(hidden.to(lm_head_device)).float()

                latent_len = ctx.size(1)
                T = len(window)
                cont_logits = logits[0, latent_len - 1 : latent_len - 1 + T, :]
                cont_labels = torch.tensor(
                    window, device=cont_logits.device, dtype=torch.long
                )
                log_probs = F.log_softmax(cont_logits, dim=-1)
                total_ll += float(
                    log_probs[torch.arange(T, device=cont_logits.device), cont_labels]
                    .sum()
                    .item()
                )

            results.append(total_ll)
        return results

    def apply_chat_template(self, chat_history, **kwargs):
      """
      Convert lm-eval chat turns into a formatted prompt using
      the tokenizer's native chat template.

      Accepts **kwargs so newer lm-eval versions can pass add_generation_prompt
      (and any future arguments) without raising a TypeError.
      """
      
      add_generation_prompt = kwargs.get("add_generation_prompt", True)
  
      if hasattr(self._tokenizer, "apply_chat_template"):
          return self._tokenizer.apply_chat_template(
              chat_history,
              tokenize=False,
              add_generation_prompt=True,
          )
  
      # Fallback for tokenizers without native templates
      prompt = ""
  
      for turn in chat_history:
          role = turn.get("role", "user")
          content = turn.get("content", "")
  
          prompt += f"<|{role}|>\n{content}\n"
  
      prompt += "<|assistant|>\n"
  
      return prompt
