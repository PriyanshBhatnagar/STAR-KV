import torch
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaConfig, repeat_kv, rotate_half,
    LlamaRotaryEmbedding, apply_rotary_pos_emb, eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache

from abx_rope_batched_quant import abx as recompute_k_qk
from bx_quant import bx as recompute_p_v
from quant_utils import quantize_r_split_int8_int4_packed, unpack_uint8_to_int4_signed

import math

# Set flags to control which Triton kernels are used
reorder = True
triton_kernel_k = True
triton_kernel_v = True

# Top-r ranks stored in int8; remaining ranks packed as int4
UPPER_RANKS_K = 16
UPPER_RANKS_V = 256


def apply_rotary_pos_emb_custom(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if x.shape[-2] != cos.shape[-2]:
        cos = cos[:, :, -1, :].unsqueeze(2)
        sin = sin[:, :, -1, :].unsqueeze(2)
    embed = (x * cos) + (rotate_half(x) * sin)
    return embed


class LlamaCustomAttention(nn.Module):
    """Llama attention with quantized low-rank K/V cache.

    Extends the non-quantized variant by splitting the low-rank KV cache into
    an int8 upper block (top UPPER_RANKS_K/V ranks) and a packed int4 lower
    block. The Triton ABX-quant and BX-quant kernels handle dequantization
    and projection in a single fused pass.

    Requires: quant_utils.py, abx_rope_batched_quant.py (not included in this
    release — implement quantize_r_split_int8_int4_packed and
    unpack_uint8_to_int4_signed per your hardware target).
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: Optional[int] = None,
        decomp_goup_num: Optional[int] = 1,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.is_causal = True
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.decomp_goup_num = decomp_goup_num
        self.decomp_goup_num_kv = decomp_goup_num // self.num_key_value_groups

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values_upper: Optional[Cache] = None,
        past_key_values_lower: Optional[Cache] = None,
        scales_upper: Optional[Cache] = None,
        scales_lower: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        k_u = self.k_proj.U
        k_vs = self.k_proj.VS
        k_intermediate = (
            torch.matmul(hidden_states, k_vs.weight.T)
            .view(*input_shape, self.decomp_goup_num_kv, -1)
            .transpose(1, 2)
        )

        k_intermediate_upper_q, scale_k_upper, k_intermediate_lower_q, scale_k_lower, r4_k = (
            quantize_r_split_int8_int4_packed(k_intermediate, split_r=UPPER_RANKS_K)
        )

        v_u = self.v_proj.U
        v_vs = self.v_proj.VS
        v_intermediate = torch.matmul(hidden_states, v_vs.weight.T)

        v_intermediate_upper_q, scale_v_upper, v_intermediate_lower_q, scale_v_lower, r4_v = (
            quantize_r_split_int8_int4_packed(v_intermediate, split_r=UPPER_RANKS_V)
        )

        if past_key_values_upper is not None and past_key_values_lower is not None:
            key_states_upper_q, value_states_upper_q = past_key_values_upper.update(
                k_intermediate_upper_q, v_intermediate_upper_q, self.layer_idx
            )
            key_states_lower_q, value_states_lower_q = past_key_values_lower.update(
                k_intermediate_lower_q, v_intermediate_lower_q, self.layer_idx
            )
            scale_k_upper, scale_v_upper = scales_upper.update(scale_k_upper, scale_v_upper, self.layer_idx)
            scale_k_lower, scale_v_lower = scales_lower.update(scale_k_lower, scale_v_lower, self.layer_idx)
        else:
            key_states_upper_q = k_intermediate_upper_q
            key_states_lower_q = k_intermediate_lower_q
            value_states_upper_q = v_intermediate_upper_q
            value_states_lower_q = v_intermediate_lower_q

        cos, sin = position_embeddings
        query_states = apply_rotary_pos_emb_custom(query_states, cos, sin)

        if not triton_kernel_k:
            key_states_upper = (key_states_upper_q * scale_k_upper).to(query_states.dtype)
            key_states_lower = (unpack_uint8_to_int4_signed(key_states_lower_q, r4_k) * scale_k_lower).to(query_states.dtype)
            key_states = torch.cat([key_states_upper, key_states_lower], dim=-1)
            key_states = torch.matmul(key_states, k_u.transpose(-2, -1).to(key_states.dtype))
            key_states = apply_rotary_pos_emb_custom(key_states, cos, sin)
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        else:
            key_states_upper_q = key_states_upper_q.repeat_interleave(self.num_key_value_groups, dim=1)
            key_states_lower_q = key_states_lower_q.repeat_interleave(self.num_key_value_groups, dim=1)
            A = query_states
            B = k_u.transpose(-2, -1)
            X1 = key_states_upper_q
            X2 = key_states_lower_q
            attn_weights = recompute_k_qk(A, B, X1, X2, scale_k_upper, scale_k_lower, r4_k) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training
        )

        if not triton_kernel_v:
            value_states_upper = (value_states_upper_q * scale_v_upper).to(query_states.dtype)
            value_states_lower = (unpack_uint8_to_int4_signed(value_states_lower_q, r4_v) * scale_v_lower).to(query_states.dtype)
            value_states = torch.cat([value_states_upper, value_states_lower], dim=-1)

        if reorder:
            if triton_kernel_v:
                B = attn_weights.squeeze(-2)
                X1 = value_states_upper_q
                X2 = value_states_lower_q
                attn_prob_inter = recompute_p_v(
                    B, X1, X2, scale_v_upper, scale_v_lower,
                    original_rank=r4_v + UPPER_RANKS_V, upper_rank=UPPER_RANKS_V,
                )
            else:
                attn_prob_inter = torch.matmul(attn_weights.squeeze(-2), value_states).unsqueeze(2)

            up_proj_head = v_u.weight.T.view(-1, self.num_key_value_heads, self.head_dim).permute(1, 0, 2)
            up_proj_head = up_proj_head.repeat_interleave(self.num_key_value_groups, dim=0)
            attn_output = (
                torch.matmul(attn_prob_inter.squeeze(-2).transpose(0, 1), up_proj_head)
                .transpose(0, 1)
                .unsqueeze(-2)
            )
        else:
            value_states = (
                torch.matmul(value_states, v_u.weight.T)
                .contiguous()
                .view(*value_states.shape[:-1], -1, self.head_dim)
                .contiguous()
                .transpose(1, 2)
            )
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None
