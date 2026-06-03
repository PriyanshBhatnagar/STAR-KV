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

from abx_rope_batched import abx as recompute_k_qk

import math

# Set reorder=True to compute attention output in low-rank space before projecting back.
# Set triton_kernel=True to use the fused Triton ABX+RoPE kernel for decode.
reorder = True
triton_kernel = True


def apply_rotary_pos_emb_custom(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if x.shape[-2] != cos.shape[-2]:
        cos = cos[:, :, -1, :].unsqueeze(2)
        sin = sin[:, :, -1, :].unsqueeze(2)
    embed = (x * cos) + (rotate_half(x) * sin)
    return embed


class LlamaCustomAttention(nn.Module):
    """Llama attention with low-rank decomposed K/V projections.

    K and V projections are factored as W = U @ VS, where VS is stored in the
    KV cache in compressed form and U expands back to full head_dim at decode
    time. The Triton ABX kernel fuses the VS expansion, RoPE, and query-key dot
    product into a single GPU kernel.
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
        past_key_values: Optional[Cache] = None,
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

        v_u = self.v_proj.U
        v_vs = self.v_proj.VS
        v_intermediate = torch.matmul(hidden_states, v_vs.weight.T)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(k_intermediate, v_intermediate, self.layer_idx)
        else:
            key_states = k_intermediate
            value_states = v_intermediate

        cos, sin = position_embeddings
        query_states = apply_rotary_pos_emb_custom(query_states, cos, sin)

        if not triton_kernel:
            key_states = torch.matmul(key_states, k_u.transpose(-2, -1).to(key_states.dtype))
            key_states = apply_rotary_pos_emb_custom(key_states, cos, sin)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        else:
            A = query_states
            B = k_u.transpose(-2, -1)
            X = key_states
            attn_weights = recompute_k_qk(A, B, X, dtype=torch.float16) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training
        )

        if reorder:
            attn_prob_inter = torch.matmul(attn_weights.squeeze(-2), value_states).unsqueeze(2)
            up_proj_head = v_u.weight.T.view(-1, self.num_key_value_heads, self.head_dim).permute(1, 0, 2)
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

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None
