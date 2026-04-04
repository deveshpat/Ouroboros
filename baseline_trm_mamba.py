"""Baseline Transformer-Mamba hybrid language model.

This module implements the non-recursive Phase 0 backbone for Project Ouroboros.
The design goal is a clean, convergence-safe hybrid stack that can be dropped into an
external training script without modification.

Architecture
    TokenEmbedding
    -> n_groups x TRMMambaGroup
       - each group = 1 x Transformer block + 7 x Mamba layers
    -> final RMSNorm
    -> tied LM head

Install requirements
    pip install "causal-conv1d>=1.4.0" mamba-ssm --no-build-isolation

Scaling presets (documented only; defaults stay on nano for smoke tests)
    +--------+---------+----------+---------+------------+---------------+
    | preset | d_model | n_groups | n_heads | n_kv_heads | approx params |
    +--------+---------+----------+---------+------------+---------------+
    | nano   |     512 |        1 |       8 |          4 | ~90M          |
    | small  |    1024 |        2 |      16 |          8 | ~270M         |
    | medium |    2048 |        2 |      16 |          8 | ~760M         |
    +--------+---------+----------+---------+------------+---------------+

Notes
    * RoPE follows the LLaMA-3 / Qwen2.5 convention: non-interleaved with
      rope_theta = 1_000_000.0.
    * SwiGLU uses the canonical hidden-size formula ceil((8/3) * d_model / 64) * 64.
    * Residual-output projections receive GPT-2 / LLaMA-style scaled init.
    * Importing this file must work even when mamba_ssm is unavailable. The failure is
      deferred until a MambaLayer is instantiated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_MAMBA_IMPORT_ERROR: Optional[ImportError] = None

try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError as _mamba_err:
    Mamba = None  # type: ignore[assignment]
    MAMBA_AVAILABLE = False
    _MAMBA_IMPORT_ERROR = _mamba_err


@dataclass
class BaselineConfig:
    """Configuration for the non-recursive Transformer-Mamba baseline."""

    # ── Vocabulary ──────────────────────────────────────────────────────────
    # 151 680 = ceil(151 665 / 128) * 128
    # 151 665 is the Qwen2.5-0.5B tokenizer vocabulary size.
    # Padding to a multiple of 128 maximises CUDA Tensor Core efficiency.
    vocab_size: int = 151_680

    # ── Model geometry ──────────────────────────────────────────────────────
    # nano preset (default) — fits on any T4 for smoke testing
    # small:  d_model=1024, n_groups=2, n_heads=16, n_kv_heads=8
    # medium: d_model=2048, n_groups=2, n_heads=16, n_kv_heads=8
    d_model: int = 512
    n_groups: int = 1
    mamba_per_group: int = 7  # DO NOT change — enforces the 1:7 ratio

    # ── Attention ───────────────────────────────────────────────────────────
    n_heads: int = 8
    n_kv_heads: int = 4

    # ── RoPE ────────────────────────────────────────────────────────────────
    rope_theta: float = 1_000_000.0

    # ── Mamba SSM ───────────────────────────────────────────────────────────
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # ── Training helpers ────────────────────────────────────────────────────
    max_seq_len: int = 2048
    dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True

    # ── Derived (populated by __post_init__) ────────────────────────────────
    head_dim: int = field(init=False)
    mlp_hidden: int = field(init=False)

    def __post_init__(self) -> None:
        """Validate divisibility and derive dependent dimensions."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads "
                f"({self.n_kv_heads}) for GQA."
            )
        if self.mamba_per_group != 7:
            raise ValueError("mamba_per_group must be 7 to maintain the 1:7 TRM-Mamba ratio.")

        self.head_dim = self.d_model // self.n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim. Got head_dim={self.head_dim}.")

        # reason: LLaMA-2/3 use a narrower SwiGLU hidden size than the old 4x MLP rule.
        raw_hidden = (8.0 / 3.0) * self.d_model
        self.mlp_hidden = int(math.ceil(raw_hidden / 64.0) * 64)

    @property
    def total_residual_layers(self) -> int:
        """Return the number of projections that write into the residual stream."""
        return self.n_groups * (2 + self.mamba_per_group)


class RMSNorm(nn.Module):
    """Root-mean-square normalization without centering."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Create a learnable RMSNorm over the last dimension."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor over its last dimension."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Precompute and lazily cache RoPE cosine and sine tables."""

    def __init__(self, head_dim: int, base: float) -> None:
        """Create the inverse-frequency buffer for a specific head dimension."""
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cache", torch.empty(0), persistent=False)
        self.register_buffer("sin_cache", torch.empty(0), persistent=False)
        self._cached_seq_len = 0
        self._cache_device: Optional[torch.device] = None
        self._cache_dtype: Optional[torch.dtype] = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Build or rebuild the RoPE cache for a target sequence length."""
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # reason: compute trig functions in float32, then cast once for the attention dtype.
        cos = emb.cos()[None, None, :, :].to(dtype=dtype)
        sin = emb.sin()[None, None, :, :].to(dtype=dtype)

        self.cos_cache = cos
        self.sin_cache = sin
        self._cached_seq_len = seq_len
        self._cache_device = device
        self._cache_dtype = dtype

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached RoPE cos/sin tensors sliced to the requested length."""
        needs_rebuild = (
            seq_len > self._cached_seq_len
            or self._cache_device != device
            or self._cache_dtype != dtype
        )
        if needs_rebuild:
            self._build_cache(seq_len=seq_len, device=device, dtype=dtype)

        return (
            self.cos_cache[:, :, :seq_len, :],
            self.sin_cache[:, :, :seq_len, :],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dimension in half and rotate it as [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)



def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply non-interleaved rotary position embeddings to query and key tensors."""
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def _is_residual_writer_module(module_name: str) -> bool:
    """Return True for projections that write back into the residual stream."""
    return (
        module_name.endswith(".attn.o_proj")
        or module_name.endswith(".mlp.down_proj")
        or module_name.endswith(".mamba.out_proj")
    )


class CausalSelfAttentionGQA(nn.Module):
    """Grouped-query causal self-attention with RoPE and SDPA."""

    def __init__(self, config: BaselineConfig) -> None:
        """Create the attention projections and rotary cache."""
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_repeats = config.n_heads // config.n_kv_heads
        self.dropout_p = config.dropout
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_theta)

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def _validate_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        input_shape: Tuple[int, int, int],
    ) -> None:
        """Validate the optional [batch, seq] attention mask shape."""
        if attention_mask is None:
            return
        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                "attention_mask must have shape [batch, seq]. "
                f"Got {tuple(attention_mask.shape)} for input {input_shape}."
            )

    def _build_attention_bias(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the additive bias that combines causal and padding masking."""
        attn_bias = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=dtype, device=device)
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).triu(1)
        attn_bias.masked_fill_(causal_mask[None, None], float("-inf"))
        pad_mask = ~attention_mask[:, None, None, :]
        attn_bias.masked_fill_(pad_mask, float("-inf"))
        return attn_bias

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply causal grouped-query attention to a [B, T, D] hidden-state tensor."""
        batch_size, seq_len, model_dim = x.shape

        self._validate_attention_mask(attention_mask, batch_size, seq_len, tuple(x.shape))

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(seq_len, x.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        if self.n_repeats > 1:
            # reason: rotate once on the compact KV representation, then broadcast for GQA.
            k = k.repeat_interleave(self.n_repeats, dim=1)
            v = v.repeat_interleave(self.n_repeats, dim=1)

        if attention_mask is None:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # reason: SDPA forbids combining is_causal=True with a custom padding mask.
            attn_bias = self._build_attention_bias(
                attention_mask=attention_mask,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=q.dtype,
                device=q.device,
            )
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
        return self.o_proj(attn_out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network with a LLaMA-style hidden width."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        """Create the gate, up, and down projections for SwiGLU."""
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU transformation to a hidden-state tensor."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TRMBlock(nn.Module):
    """Pre-norm Transformer block with attention and SwiGLU sublayers."""

    def __init__(self, config: BaselineConfig) -> None:
        """Create the norms and sublayers for one Transformer block."""
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = CausalSelfAttentionGQA(config)
        self.mlp_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp = SwiGLU(config.d_model, config.mlp_hidden)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention and MLP residual updates to a hidden-state tensor."""
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MambaLayer(nn.Module):
    """Pre-norm residual wrapper around mamba_ssm.Mamba."""

    def __init__(self, config: BaselineConfig) -> None:
        """Instantiate the wrapped Mamba block or raise a clear install error."""
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is required but not installed.\n"
                'Install with:  pip install "causal-conv1d>=1.4.0" mamba-ssm '
                "--no-build-isolation"
            ) from _MAMBA_IMPORT_ERROR

        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mamba = Mamba(
            d_model=config.d_model,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply a pre-norm Mamba update and add it back to the residual stream."""
        residual = x
        h = self.norm(x)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            # reason: Mamba is content-mixing over time, so padded timesteps must be nulled.
            h = h * mask
        h = self.mamba(h)
        if attention_mask is not None:
            h = h * mask
        return residual + h


class TRMMambaGroup(nn.Module):
    """One Transformer block followed by exactly seven Mamba residual layers."""

    def __init__(self, config: BaselineConfig) -> None:
        """Create one TRM block and the fixed-ratio stack of Mamba layers."""
        super().__init__()
        self.trm_block = TRMBlock(config)
        self.mamba_blocks = nn.ModuleList([MambaLayer(config) for _ in range(config.mamba_per_group)])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the Transformer block and then every Mamba layer in sequence."""
        x = self.trm_block(x, attention_mask)
        for mamba_layer in self.mamba_blocks:
            x = mamba_layer(x, attention_mask)
        return x


class BaselineTRMMamba(nn.Module):
    """Top-level non-recursive Transformer-Mamba language model."""

    def __init__(self, config: BaselineConfig) -> None:
        """Create embeddings, hybrid groups, final norm, and the tied LM head."""
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.groups = nn.ModuleList([TRMMambaGroup(config) for _ in range(config.n_groups)])
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return next-token logits for an input token-id tensor."""
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2D [batch, seq]. Got shape {tuple(input_ids.shape)}."
            )

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}."
            )

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    "attention_mask must have shape [batch, seq]. "
                    f"Got {tuple(attention_mask.shape)} for input {tuple(input_ids.shape)}."
                )
            # reason: every submodule expects the keep-mask convention True = valid token.
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

        x = self.token_embedding(input_ids)
        for group in self.groups:
            x = group(x, attention_mask)
        x = self.final_norm(x)
        return self.lm_head(x)

    def _init_weights(self) -> None:
        """Apply base init first, then residual-output scaling in a second pass."""
        residual_std = 0.02 / math.sqrt(self.config.total_residual_layers)

        initialized: set[int] = set()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if id(module.weight) not in initialized:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    initialized.add(id(module.weight))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                if id(module.weight) not in initialized:
                    # reason: tied embeddings must be initialized once no matter how many modules reference them.
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    initialized.add(id(module.weight))

        for name, module in self.named_modules():
            if not _is_residual_writer_module(name):
                continue

            weight = getattr(module, "weight", None)
            if isinstance(weight, nn.Parameter):
                # reason: only projections that write back into the residual stream get scaled.
                nn.init.normal_(weight, mean=0.0, std=residual_std)



def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters, counting tied tensors only once."""
    seen: set[int] = set()
    total = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) not in seen:
            seen.add(id(param))
            total += param.numel()
    return total


if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float32
    )
    torch.manual_seed(42)

    # ── Build model ─────────────────────────────────────────────────────────
    # Default config is the nano preset: d_model=512, n_groups=1.
    # Change to BaselineConfig(d_model=1024, n_groups=2, n_heads=16, n_kv_heads=8)
    # for the small preset (~270M), or
    # BaselineConfig(d_model=2048, n_groups=2, n_heads=16, n_kv_heads=8)
    # for the medium preset (~760M).
    config = BaselineConfig()
    model = BaselineTRMMamba(config).to(device=device, dtype=dtype)
    model.train()

    total_params = count_parameters(model)
    print(f"device     : {device}")
    print(f"dtype      : {dtype}")
    print(f"parameters : {total_params:,}  ({total_params / 1e6:.1f} M)")
    print(f"vocab_size : {config.vocab_size}")
    print(f"d_model    : {config.d_model}")
    print(f"n_groups   : {config.n_groups}")
    print(f"n_heads    : {config.n_heads}  n_kv_heads: {config.n_kv_heads}")
    print(f"mlp_hidden : {config.mlp_hidden}")
    print(f"total_residual_layers: {config.total_residual_layers}")
    print()

    # ── Dummy batch ─────────────────────────────────────────────────────────
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    attn_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    attn_mask[1, 120:] = False

    # ── Forward pass ────────────────────────────────────────────────────────
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        logits = model(input_ids, attention_mask=attn_mask)

    print(
        f"logits shape : {tuple(logits.shape)}  "
        f"(expected [{batch_size}, {seq_len}, {config.vocab_size}])"
    )
    assert tuple(logits.shape) == (batch_size, seq_len, config.vocab_size), "Shape mismatch!"

    # ── Sanity: no NaN/Inf in logits ───────────────────────────────────────
    if torch.isnan(logits).any():
        print("FAIL: NaN detected in logits", file=sys.stderr)
        sys.exit(1)
    if torch.isinf(logits).any():
        print("FAIL: Inf detected in logits", file=sys.stderr)
        sys.exit(1)
    print("logits health : OK (no NaN, no Inf)")

    # ── Loss and backward pass ─────────────────────────────────────────────
    shift_logits = logits[:, :-1, :].contiguous().view(-1, config.vocab_size).float()
    shift_targets = targets[:, 1:].contiguous().view(-1)
    loss = F.cross_entropy(shift_logits, shift_targets)

    print(
        f"initial loss : {loss.item():.4f}  "
        f"(theoretical random-init ≈ {math.log(config.vocab_size):.4f})"
    )

    if loss.item() > math.log(config.vocab_size) * 1.2:
        print(
            "WARN: loss is unexpectedly high (>1.2× theoretical). "
            "Check initialization."
        )

    loss.backward()
    print("backward     : OK")

    # ── Gradient health check ──────────────────────────────────────────────
    missing_grad = []
    nan_grad = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing_grad.append(name)
        elif torch.isnan(param.grad).any():
            nan_grad.append(name)

    if missing_grad:
        print(f"FAIL: {len(missing_grad)} parameters have no gradient:", file=sys.stderr)
        for name in missing_grad[:10]:
            print(f"  {name}", file=sys.stderr)
        sys.exit(1)
    if nan_grad:
        print(f"FAIL: {len(nan_grad)} parameters have NaN gradients:", file=sys.stderr)
        for name in nan_grad[:10]:
            print(f"  {name}", file=sys.stderr)
        sys.exit(1)

    grad_norms = [
        (name, param.grad.norm().item())
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    ]
    total_grad_norm = sum(norm_value ** 2 for _, norm_value in grad_norms) ** 0.5
    print(f"grad norms   : total={total_grad_norm:.4f}")
    print()
    print("All checks passed. Baseline architecture is healthy.")
