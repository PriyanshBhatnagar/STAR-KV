from typing import Optional

import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaConfig, rotate_half

# True: decode through the fused Triton ABX+RoPE kernel. False: the pure-PyTorch
# reference. Switched by model.set_model_mode.
triton_kernel = True


def apply_rotary_pos_emb_custom(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if x.shape[-2] != cos.shape[-2]:
        cos = cos[:, :, -1, :].unsqueeze(2)
        sin = sin[:, :, -1, :].unsqueeze(2)
    return (x * cos) + (rotate_half(x) * sin)


class LlamaCustomAttention(nn.Module):
    """Llama attention with low-rank K/V projections: the module's state only.

    K and V are factored as W = U @ VS; the cache holds VS @ x and U expands it at
    decode. model.replace_attn_with_triton fills in k_proj / v_proj with the
    exported factors and installs the forward (model._patched_attn_forward, or
    model._kv_quant_forward for the 4-bit cache), so this class has none of its own.
    """

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.k_proj = self.v_proj = None       # set by replace_attn_with_triton
