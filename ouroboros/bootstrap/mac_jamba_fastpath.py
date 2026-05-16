"""Mac-only Jamba Mamba scan patching.

Transformers only uses its Jamba fast path on CUDA. On Apple Silicon, the
default path falls back to a Python token loop inside ``JambaMambaMixer``. This
module installs a guarded MPS replacement that keeps Transformers' projections
and depthwise convolution, then routes the SSM recurrence through custom Metal
kernels when the tensor shape is supported.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_PATCH_ATTR = "_ouroboros_mac_mps_scan_patch"
_ORIGINAL_ATTR = "_ouroboros_original_slow_forward"


def _resolve_selective_scan_fn() -> Optional[Callable[..., Any]]:
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except Exception:
        return None
    return selective_scan_fn


def _zero_bias_dt_projection(dt_proj: nn.Module, time_step: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    bias = getattr(dt_proj, "bias", None)
    if bias is None:
        return dt_proj(time_step).transpose(1, 2), None

    saved_bias = bias.data
    try:
        with torch.no_grad():
            bias.data = torch.zeros_like(saved_bias)
        projected = dt_proj(time_step).transpose(1, 2)
    finally:
        with torch.no_grad():
            bias.data = saved_bias
    return projected, saved_bias.float()


class _MacSelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, a_matrix, b_state, c_state, d_skip, gate, delta_bias):
        input_dtypes = (
            u.dtype,
            delta.dtype,
            a_matrix.dtype,
            b_state.dtype,
            c_state.dtype,
            d_skip.dtype,
            gate.dtype,
            delta_bias.dtype,
        )
        u_f = u.float()
        delta_pre = delta.float() + delta_bias.float()[None, :, None]
        delta_f = F.softplus(delta_pre)
        a_f = a_matrix.float()
        b_f = b_state.float()
        c_f = c_state.float()
        d_f = d_skip.float()
        gate_f = gate.float()

        batch_size, intermediate_size, seq_len = u.shape
        state_size = int(a_f.shape[1])
        state = torch.zeros(
            (batch_size, intermediate_size, state_size),
            device=u.device,
            dtype=torch.float32,
        )
        states = []
        outputs = []
        for token_idx in range(seq_len):
            delta_t = delta_f[:, :, token_idx]
            transition = torch.exp(delta_t[:, :, None] * a_f[None, :, :])
            state = (
                transition * state
                + delta_t[:, :, None]
                * b_f[:, None, :, token_idx]
                * u_f[:, :, token_idx, None]
            )
            states.append(state)
            y_t = (state * c_f[:, None, :, token_idx]).sum(dim=-1)
            pre_gate = y_t + d_f[None, :] * u_f[:, :, token_idx]
            outputs.append(pre_gate * F.silu(gate_f[:, :, token_idx]))

        ctx.input_dtypes = input_dtypes
        ctx.save_for_backward(
            u_f,
            delta_pre,
            delta_f,
            a_f,
            b_f,
            c_f,
            d_f,
            gate_f,
            torch.stack(states, dim=2),
        )
        return torch.stack(outputs, dim=2).to(dtype=u.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (
            u_f,
            delta_pre,
            delta_f,
            a_f,
            b_f,
            c_f,
            d_f,
            gate_f,
            states,
        ) = ctx.saved_tensors
        grad = grad_output.float()
        batch_size, intermediate_size, seq_len = u_f.shape
        state_size = int(a_f.shape[1])

        grad_u = torch.zeros_like(u_f)
        grad_delta = torch.zeros_like(delta_f)
        grad_a = torch.zeros_like(a_f)
        grad_b = torch.zeros_like(b_f)
        grad_c = torch.zeros_like(c_f)
        grad_d = torch.zeros_like(d_f)
        grad_gate = torch.zeros_like(gate_f)
        grad_bias = torch.zeros_like(d_f)

        grad_next_state = torch.zeros(
            (batch_size, intermediate_size, state_size),
            device=u_f.device,
            dtype=torch.float32,
        )
        zero_state = torch.zeros_like(grad_next_state)
        gate_sigmoid = torch.sigmoid(gate_f)
        gate_silu = gate_f * gate_sigmoid
        gate_silu_grad = gate_sigmoid * (1.0 + gate_f * (1.0 - gate_sigmoid))

        for token_idx in range(seq_len - 1, -1, -1):
            state_t = states[:, :, token_idx, :]
            prev_state = states[:, :, token_idx - 1, :] if token_idx > 0 else zero_state
            delta_t = delta_f[:, :, token_idx]
            u_t = u_f[:, :, token_idx]
            b_t = b_f[:, :, token_idx]
            c_t = c_f[:, :, token_idx]
            transition = torch.exp(delta_t[:, :, None] * a_f[None, :, :])

            pre_gate = (state_t * c_t[:, None, :]).sum(dim=-1) + d_f[None, :] * u_t
            grad_out_t = grad[:, :, token_idx]
            grad_gate[:, :, token_idx] = grad_out_t * pre_gate * gate_silu_grad[:, :, token_idx]

            grad_pre_gate = grad_out_t * gate_silu[:, :, token_idx]
            grad_d += (grad_pre_gate * u_t).sum(dim=0)
            grad_u[:, :, token_idx] += grad_pre_gate * d_f[None, :]

            grad_state = grad_next_state + grad_pre_gate[:, :, None] * c_t[:, None, :]
            grad_c[:, :, token_idx] = (grad_pre_gate[:, :, None] * state_t).sum(dim=1)

            grad_transition = grad_state * prev_state
            grad_next_state = grad_state * transition

            grad_delta_t = (
                grad_transition * (transition * a_f[None, :, :])
            ).sum(dim=-1) + (
                grad_state * (b_t[:, None, :] * u_t[:, :, None])
            ).sum(dim=-1)
            grad_a += (grad_transition * (transition * delta_t[:, :, None])).sum(dim=0)
            grad_b[:, :, token_idx] = (
                grad_state * (delta_t[:, :, None] * u_t[:, :, None])
            ).sum(dim=1)
            grad_u[:, :, token_idx] += (
                grad_state * (delta_t[:, :, None] * b_t[:, None, :])
            ).sum(dim=-1)

            grad_delta_pre = grad_delta_t * torch.sigmoid(delta_pre[:, :, token_idx])
            grad_delta[:, :, token_idx] = grad_delta_pre
            grad_bias += grad_delta_pre.sum(dim=0)

        dtypes = ctx.input_dtypes
        return (
            grad_u.to(dtype=dtypes[0]),
            grad_delta.to(dtype=dtypes[1]),
            grad_a.to(dtype=dtypes[2]),
            grad_b.to(dtype=dtypes[3]),
            grad_c.to(dtype=dtypes[4]),
            grad_d.to(dtype=dtypes[5]),
            grad_gate.to(dtype=dtypes[6]),
            grad_bias.to(dtype=dtypes[7]),
        )


def _mac_selective_scan(u, delta, a_matrix, b_state, c_state, d_skip, gate, delta_bias):
    if (
        u.device.type == "mps"
        and int(u.shape[0]) == 1
        and int(a_matrix.shape[1]) <= 64
    ):
        try:
            return _MacMetalSelectiveScanFn.apply(
                u.contiguous(),
                delta.contiguous(),
                a_matrix.contiguous(),
                b_state.contiguous(),
                c_state.contiguous(),
                d_skip.contiguous(),
                gate.contiguous(),
                delta_bias.contiguous(),
            )
        except Exception:
            pass
    return _MacSelectiveScanFn.apply(u, delta, a_matrix, b_state, c_state, d_skip, gate, delta_bias)


@functools.lru_cache(maxsize=1)
def _scan_shader_library():
    source = r"""
#include <metal_stdlib>
using namespace metal;

static inline float softplus_f(float x) {
    return x > 20.0f ? x : log(1.0f + exp(x));
}

static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + exp(-x));
}

static inline float silu_f(float x) {
    return x * sigmoid_f(x);
}

kernel void scan_forward(
    device const float* u,
    device const float* delta,
    device const float* a_matrix,
    device const float* b_state,
    device const float* c_state,
    device const float* d_skip,
    device const float* gate,
    device const float* delta_bias,
    device float* output,
    device float* states,
    constant uint& dim,
    constant uint& seq_len,
    constant uint& state_size,
    uint idx [[thread_position_in_grid]]
) {
    uint d = idx;
    if (d >= dim) return;

    float state[64];
    for (uint n = 0; n < state_size; ++n) {
        state[n] = 0.0f;
    }

    for (uint t = 0; t < seq_len; ++t) {
        uint udx = d * seq_len + t;
        float dt = softplus_f(delta[udx] + delta_bias[d]);
        float y = 0.0f;
        for (uint n = 0; n < state_size; ++n) {
            float transition = exp(dt * a_matrix[d * state_size + n]);
            float next_state = (
                transition * state[n]
                + dt * b_state[n * seq_len + t] * u[udx]
            );
            state[n] = next_state;
            states[(d * seq_len + t) * state_size + n] = next_state;
            y += next_state * c_state[n * seq_len + t];
        }
        float pre_gate = y + d_skip[d] * u[udx];
        output[udx] = pre_gate * silu_f(gate[udx]);
    }
}

kernel void scan_backward_main(
    device const float* grad_output,
    device const float* u,
    device const float* delta_pre,
    device const float* delta,
    device const float* a_matrix,
    device const float* b_state,
    device const float* c_state,
    device const float* d_skip,
    device const float* gate,
    device const float* states,
    device float* grad_u,
    device float* grad_delta,
    device float* grad_a,
    device float* grad_d,
    device float* grad_gate,
    device float* grad_bias,
    device float* grad_state_all,
    constant uint& dim,
    constant uint& seq_len,
    constant uint& state_size,
    uint idx [[thread_position_in_grid]]
) {
    uint d = idx;
    if (d >= dim) return;

    float grad_next_state[64];
    for (uint n = 0; n < state_size; ++n) {
        grad_next_state[n] = 0.0f;
        grad_a[d * state_size + n] = 0.0f;
    }

    float grad_d_acc = 0.0f;
    float grad_bias_acc = 0.0f;
    for (int ti = int(seq_len) - 1; ti >= 0; --ti) {
        uint t = uint(ti);
        uint udx = d * seq_len + t;
        float gate_sigmoid = sigmoid_f(gate[udx]);
        float gate_silu = gate[udx] * gate_sigmoid;
        float gate_silu_grad = gate_sigmoid * (1.0f + gate[udx] * (1.0f - gate_sigmoid));

        float y = 0.0f;
        for (uint n = 0; n < state_size; ++n) {
            y += states[(d * seq_len + t) * state_size + n] * c_state[n * seq_len + t];
        }
        float pre_gate = y + d_skip[d] * u[udx];
        float grad_out_t = grad_output[udx];
        grad_gate[udx] = grad_out_t * pre_gate * gate_silu_grad;

        float grad_pre_gate = grad_out_t * gate_silu;
        grad_d_acc += grad_pre_gate * u[udx];
        float grad_u_t = grad_pre_gate * d_skip[d];
        float grad_delta_t = 0.0f;
        float dt = delta[udx];

        for (uint n = 0; n < state_size; ++n) {
            float prev_state = t == 0 ? 0.0f : states[(d * seq_len + (t - 1)) * state_size + n];
            float transition = exp(dt * a_matrix[d * state_size + n]);
            float grad_state = grad_next_state[n] + grad_pre_gate * c_state[n * seq_len + t];
            grad_state_all[(d * seq_len + t) * state_size + n] = grad_state;

            float grad_transition = grad_state * prev_state;
            grad_next_state[n] = grad_state * transition;
            grad_delta_t += (
                grad_transition * transition * a_matrix[d * state_size + n]
                + grad_state * b_state[n * seq_len + t] * u[udx]
            );
            grad_a[d * state_size + n] += grad_transition * transition * dt;
            grad_u_t += grad_state * dt * b_state[n * seq_len + t];
        }

        float grad_delta_pre = grad_delta_t * sigmoid_f(delta_pre[udx]);
        grad_delta[udx] = grad_delta_pre;
        grad_bias_acc += grad_delta_pre;
        grad_u[udx] = grad_u_t;
    }
    grad_d[d] = grad_d_acc;
    grad_bias[d] = grad_bias_acc;
}

kernel void scan_backward_bc(
    device const float* grad_output,
    device const float* u,
    device const float* delta,
    device const float* gate,
    device const float* states,
    device const float* grad_state_all,
    device float* grad_b,
    device float* grad_c,
    constant uint& dim,
    constant uint& seq_len,
    constant uint& state_size,
    uint idx [[thread_position_in_grid]]
) {
    uint total = state_size * seq_len;
    if (idx >= total) return;
    uint n = idx / seq_len;
    uint t = idx - n * seq_len;

    float grad_b_acc = 0.0f;
    float grad_c_acc = 0.0f;
    for (uint d = 0; d < dim; ++d) {
        uint udx = d * seq_len + t;
        float gate_silu = silu_f(gate[udx]);
        float grad_pre_gate = grad_output[udx] * gate_silu;
        grad_c_acc += grad_pre_gate * states[(d * seq_len + t) * state_size + n];
        grad_b_acc += grad_state_all[(d * seq_len + t) * state_size + n] * delta[udx] * u[udx];
    }
    grad_b[n * seq_len + t] = grad_b_acc;
    grad_c[n * seq_len + t] = grad_c_acc;
}
"""
    return torch.mps.compile_shader(source)


class _MacMetalSelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, a_matrix, b_state, c_state, d_skip, gate, delta_bias):
        input_dtypes = (
            u.dtype,
            delta.dtype,
            a_matrix.dtype,
            b_state.dtype,
            c_state.dtype,
            d_skip.dtype,
            gate.dtype,
            delta_bias.dtype,
        )
        u_f = u.float().contiguous()
        delta_f = delta.float().contiguous()
        a_f = a_matrix.float().contiguous()
        b_f = b_state.float().contiguous()
        c_f = c_state.float().contiguous()
        d_f = d_skip.float().contiguous()
        gate_f = gate.float().contiguous()
        bias_f = delta_bias.float().contiguous()
        delta_pre = (delta_f + bias_f[None, :, None]).contiguous()
        delta_softplus = F.softplus(delta_pre).contiguous()

        batch_size, dim, seq_len = u_f.shape
        if int(batch_size) != 1:
            raise RuntimeError("Mac Metal selective scan currently supports batch_size=1")
        state_size = int(a_f.shape[1])
        output = torch.empty_like(u_f)
        states = torch.empty(
            (batch_size, dim, seq_len, state_size),
            device=u.device,
            dtype=torch.float32,
        )
        shader = _scan_shader_library()
        shader.scan_forward(
            u_f.view(-1),
            delta_f.view(-1),
            a_f.view(-1),
            b_f.view(-1),
            c_f.view(-1),
            d_f,
            gate_f.view(-1),
            bias_f,
            output.view(-1),
            states.view(-1),
            int(dim),
            int(seq_len),
            int(state_size),
        )
        ctx.input_dtypes = input_dtypes
        ctx.save_for_backward(
            u_f,
            delta_pre,
            delta_softplus,
            a_f,
            b_f,
            c_f,
            d_f,
            gate_f,
            states,
        )
        return output.to(dtype=u.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (
            u_f,
            delta_pre,
            delta_f,
            a_f,
            b_f,
            c_f,
            d_f,
            gate_f,
            states,
        ) = ctx.saved_tensors
        grad_f = grad_output.float().contiguous()
        batch_size, dim, seq_len = u_f.shape
        state_size = int(a_f.shape[1])

        grad_u = torch.empty_like(u_f)
        grad_delta = torch.empty_like(delta_f)
        grad_a = torch.empty_like(a_f)
        grad_b = torch.empty_like(b_f)
        grad_c = torch.empty_like(c_f)
        grad_d = torch.empty_like(d_f)
        grad_gate = torch.empty_like(gate_f)
        grad_bias = torch.empty_like(d_f)
        grad_state_all = torch.empty_like(states)

        shader = _scan_shader_library()
        shader.scan_backward_main(
            grad_f.view(-1),
            u_f.view(-1),
            delta_pre.contiguous().view(-1),
            delta_f.view(-1),
            a_f.view(-1),
            b_f.view(-1),
            c_f.view(-1),
            d_f,
            gate_f.view(-1),
            states.contiguous().view(-1),
            grad_u.view(-1),
            grad_delta.view(-1),
            grad_a.view(-1),
            grad_d,
            grad_gate.view(-1),
            grad_bias,
            grad_state_all.view(-1),
            int(dim),
            int(seq_len),
            int(state_size),
        )
        shader.scan_backward_bc(
            grad_f.view(-1),
            u_f.view(-1),
            delta_f.view(-1),
            gate_f.view(-1),
            states.contiguous().view(-1),
            grad_state_all.view(-1),
            grad_b.view(-1),
            grad_c.view(-1),
            int(dim),
            int(seq_len),
            int(state_size),
        )

        dtypes = ctx.input_dtypes
        return (
            grad_u.to(dtype=dtypes[0]),
            grad_delta.to(dtype=dtypes[1]),
            grad_a.to(dtype=dtypes[2]),
            grad_b.to(dtype=dtypes[3]),
            grad_c.to(dtype=dtypes[4]),
            grad_d.to(dtype=dtypes[5]),
            grad_gate.to(dtype=dtypes[6]),
            grad_bias.to(dtype=dtypes[7]),
        )


def _mac_mps_selective_scan_forward(
    mixer: nn.Module,
    input_states: torch.Tensor,
    *,
    cache_params: Any = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    selective_scan_fn = _resolve_selective_scan_fn()
    if selective_scan_fn is None:
        original = getattr(type(mixer), _ORIGINAL_ATTR)
        return original(mixer, input_states, cache_params, attention_mask)

    batch_size, seq_len, _ = input_states.shape
    del batch_size

    projected_states = mixer.in_proj(input_states).transpose(1, 2)
    hidden_states, gate = projected_states.chunk(2, dim=1)

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    if cache_params is not None:
        original = getattr(type(mixer), _ORIGINAL_ATTR)
        return original(mixer, input_states, cache_params, attention_mask)

    hidden_states = mixer.act(mixer.conv1d(hidden_states)[..., :seq_len])

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    ssm_parameters = mixer.x_proj(hidden_states.transpose(1, 2))
    time_step, b_state, c_state = torch.split(
        ssm_parameters,
        [mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size],
        dim=-1,
    )

    time_step = mixer.dt_layernorm(time_step)
    b_state = mixer.b_layernorm(b_state)
    c_state = mixer.c_layernorm(c_state)

    discrete_time_step, time_proj_bias = _zero_bias_dt_projection(mixer.dt_proj, time_step)
    a_matrix = -torch.exp(mixer.A_log.float())
    del selective_scan_fn
    scan_outputs = _mac_selective_scan(
        hidden_states,
        discrete_time_step,
        a_matrix,
        b_state.transpose(1, 2),
        c_state.transpose(1, 2),
        mixer.D.float(),
        gate,
        time_proj_bias,
    )
    return mixer.out_proj(scan_outputs.transpose(1, 2))


def install_mac_jamba_mps_fastpath(*, verbose: bool = False) -> bool:
    """Patch Transformers' Jamba mixer to use the Mac selective-scan path on MPS.

    Returns ``True`` when the patch is installed or already present. The patched
    method delegates back to Transformers for non-MPS tensors and cache-backed
    generation, so the training path is the only changed surface.
    """
    if _resolve_selective_scan_fn() is None:
        return False

    try:
        from transformers.models.jamba import modeling_jamba as jamba_modeling
    except Exception:
        return False

    mixer_cls = getattr(jamba_modeling, "JambaMambaMixer", None)
    if mixer_cls is None:
        return False
    if bool(getattr(mixer_cls, _PATCH_ATTR, False)):
        return True

    original_slow_forward = mixer_cls.slow_forward
    setattr(mixer_cls, _ORIGINAL_ATTR, original_slow_forward)

    def patched_slow_forward(self, input_states, cache_params=None, attention_mask=None):
        if getattr(input_states, "device", None) is None or input_states.device.type != "mps":
            return original_slow_forward(self, input_states, cache_params, attention_mask)
        return _mac_mps_selective_scan_forward(
            self,
            input_states,
            cache_params=cache_params,
            attention_mask=attention_mask,
        )

    mixer_cls.slow_forward = patched_slow_forward
    setattr(mixer_cls, _PATCH_ATTR, True)
    if verbose:
        print("  Mac Jamba MPS selective-scan patch: ACTIVE")
    return True


__all__ = [
    "install_mac_jamba_mps_fastpath",
]
