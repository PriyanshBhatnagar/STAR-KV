"""Shared model components for adaptive low-rank KV cache compression.

Provides:
  - DiagonalLinear / MaskedLinear: building blocks for the factored projection
  - DecomposeLinear_headwise / DecomposeLinear: training-time SVD modules with soft-threshold
  - replace_linear_layer: inject decomposed modules into a LlamaForCausalLM
  - get_rank / collect_*_parameter_size: rank tracking utilities
  - export_kproj_for_triton / export_vproj_for_triton: fold Sigma, prune dead ranks
  - KProjInferenceWrapper / VProjInferenceWrapper: minimal wrappers for LlamaCustomAttention
  - replace_attn_with_triton: full attention-module replacement for inference
  - set_model_mode: switch between triton / no_triton / bf16_sdpa at runtime
  - fold_kv_hadamard / set_kv_fake_quant / PackedKVLayer: the optional 4-bit KV
    quantization add-on (format and kernels in kv_quant.py)
"""

import math
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

import LlamaLoRaAttention_headwise as attn_module
from LlamaLoRaAttention_headwise import LlamaCustomAttention
from LlamaLoRaAttention_headwise import apply_rotary_pos_emb_custom as _rope
from transformers.models.llama.modeling_llama import rotate_half
from soft_thres_layer import soft_thres_layer
import kv_quant

# Width granularity of the K cache's rank axis. The decode kernel walks that axis
# in BLOCK_SIZE_R tiles, so every head is stored at a multiple of this: it keeps
# each head's segment an exact number of 32-byte memory sectors (16 bf16 values),
# which is what makes the per-head widths free to read. Must stay equal to
# abx_rope_batched._abx_fwd's BLOCK_SIZE_R.
RANK_TILE = 16


class RankMajorKLayer(DynamicLayer):
    """Cache layer for a compressed attention layer: K is stored rank-major.

    K is [B, W, seq] -- the token axis LAST -- so a head's lanes stay contiguous
    across tokens. The obvious [B, seq, W] instead puts each head's slice in a
    different row every token, and the decode kernel then issues one scattered
    32-byte read per token rather than long contiguous runs. Same bytes, but
    measured 4-12% slower on this kernel, worst exactly where heads are narrow.

    V is untouched: [B, seq, rank_v], token axis at -2 like any other cache.
    Only the concatenation axis differs, which is why this is a cache LAYER
    rather than a whole cache -- uncompressed (skipped) layers keep DynamicLayer.

    Not wired for beam search: DynamicLayer's reorder/crop helpers assume the
    token axis is at -2.
    """

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.keys = torch.cat([self.keys, key_states], dim=-1)      # tokens last
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def get_seq_length(self) -> int:
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-1]


class PackedKVLayer(DynamicLayer):
    """Cache layer holding a compressed layer's latents packed to 4 bits.

    The kv_quant add-on's format (see kv_quant.py), all token-major:
      keys [B, L, W/2] uint8     key_scales   [B, L, H, 2] bf16
      values [B, L, row] uint8   value_scales [B, L, 2]    bf16
    update() takes the codes the attention computes -- K rank-major [B, W, T],
    V [B, T, rank_v] -- quantizes and packs them, and appends. It returns them
    unchanged: prefill runs on its exact codes, and only decode reads the cache.
    """

    def __init__(self, meta):
        super().__init__()
        self.meta = meta
        self.key_scales = self.value_scales = None

    def update(self, key_states, value_states, *args, **kwargs):
        m = self.meta
        B, W, T = key_states.shape
        kq, ks = kv_quant.pack_k_step(key_states.transpose(1, 2).reshape(B * T, 1, W),
                                      m.ranks, m.offsets, m.nouts, W)
        vq, vs = kv_quant.pack_v_step(value_states.reshape(B * T, 1, -1), m.nout_v)
        new = (kq.view(B, T, -1), ks.view(B, T, *ks.shape[2:]), vq.view(B, T, -1), vs.view(B, T, 2))
        if not self.is_initialized:
            self.dtype, self.device, self.is_initialized = key_states.dtype, key_states.device, True
            self.keys, self.key_scales, self.values, self.value_scales = new
        else:
            old = (self.keys, self.key_scales, self.values, self.value_scales)
            self.keys, self.key_scales, self.values, self.value_scales = (
                kv_quant.append(o, n) for o, n in zip(old, new))
        return key_states, value_states

    def get_seq_length(self) -> int:
        return self.keys.shape[1] if self.is_initialized else 0


class StarKVCache(DynamicCache):
    """DynamicCache whose compressed layers hold K rank-major.

    Skipped layers run stock attention and keep a stock DynamicLayer, so the
    two kinds of layer coexist in one cache. `replace_attn_with_triton` installs
    this automatically whenever a caller lets the model build its own cache.
    `packed` maps layer index -> KVQuantMeta for layers using the 4-bit cache.
    """

    def __init__(self, num_hidden_layers: int, skip_layers: tuple = (), packed=None):
        skip = set(skip_layers)
        packed = packed or {}
        Cache.__init__(self, layers=[
            DynamicLayer() if i in skip
            else PackedKVLayer(packed[i]) if i in packed
            else RankMajorKLayer()
            for i in range(num_hidden_layers)
        ])


def _rope_tables(inv_freq, seq_len, device, dtype, attn_scaling=1.0):
    """cos/sin covering absolute positions 0..seq_len-1, shaped [seq_len, head_dim].

    Needed wherever RoPE is applied to the whole cached K at once: the model's
    position_embeddings only cover the current query position, and reusing that
    single rotation for every cached key erases relative position entirely.
    """
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq.to(device=device, dtype=torch.float32)[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return (emb.cos() * attn_scaling).to(dtype), (emb.sin() * attn_scaling).to(dtype)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DiagonalLinear(nn.Module):
    """Learnable diagonal matrix, soft-thresholded in phase 1, hard-masked after.

    Phase 1: the rank is whatever survives `diag > alpha`, and the surviving
    singular values are scaled by tanh(s*(diag - alpha)). That scaling is what
    makes the threshold differentiable, and it is fine while alpha is still
    searching.

    Phase 2 onward: the threshold has done its job and is removed. `freeze_rank`
    picks the top-k directions outright and stores a binary keep mask; the
    forward pass then uses `diag * keep_mask`, so

      * the rank is PINNED -- it can no longer drift as diag trains, which it
        otherwise can, since a surviving value that falls back under alpha
        silently drops a direction mid-recovery; and
      * kept singular values are used at FULL magnitude. Under the soft
        threshold a direction just above alpha is multiplied by
        tanh(s*(diag-alpha)) ~ 0, so rounding the rank up to a tile boundary
        would pay full cache for directions that contribute almost nothing --
        the newly admitted ones are by construction the ones nearest alpha.
    """

    def __init__(self, feature_size: int, thres: float):
        super().__init__()
        self.diag = nn.Parameter(torch.ones(feature_size), requires_grad=True)
        self.soft_thres_layer = soft_thres_layer(50, 0.0, float(thres))
        # Persistent so a mid-training checkpoint reloads with its rank intact.
        self.register_buffer("keep_mask", torch.ones(feature_size))
        self.register_buffer("rank_frozen", torch.zeros((), dtype=torch.bool))

    def forward(self, x):
        if bool(self.rank_frozen):
            return x @ torch.diag(self.diag * self.keep_mask)
        return x @ self.soft_thres_layer(torch.diag(self.diag))

    @torch.no_grad()
    def freeze_rank(self, rank: int):
        """Pin the top-`rank` directions and retire the soft threshold.

        Returns the rank actually pinned. Selection is by singular-value
        magnitude, which is the same set the threshold was selecting -- the
        threshold is a magnitude cutoff -- so this changes which directions
        survive only when `rank` differs from the current one.
        """
        rank = max(1, min(int(rank), self.diag.numel()))
        idx = torch.topk(self.diag.detach().float(), rank).indices
        mask = torch.zeros_like(self.keep_mask)
        mask[idx] = 1.0
        self.keep_mask.copy_(mask)
        self.rank_frozen.fill_(True)
        return rank

    def effective_diag(self) -> torch.Tensor:
        """Singular values as the forward pass uses them, in either mode."""
        if bool(self.rank_frozen):
            return self.diag.detach().float() * self.keep_mask.float()
        return self.soft_thres_layer(self.diag.detach().float())

    def keep_indices(self) -> torch.Tensor:
        """Indices of the surviving directions, in either mode."""
        if bool(self.rank_frozen):
            return self.keep_mask.nonzero(as_tuple=False).view(-1)
        eff = self.soft_thres_layer(self.diag.detach().float())
        return (eff > 0).nonzero(as_tuple=False).view(-1)

    def current_rank(self) -> int:
        if bool(self.rank_frozen):
            return int(self.keep_mask.sum())
        return int((self.diag > self.soft_thres_layer.alpha).sum())

    def set_value(self, value: torch.Tensor):
        n = value.shape[0]
        self.diag.data[:n] = value
        self.diag.data.requires_grad_(True)


class MaskedLinear(nn.Linear):
    """nn.Linear with a non-trainable binary mask applied to weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.mask = nn.Parameter(torch.ones(out_features, in_features), requires_grad=False)

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask)

    def set_mask(self, mask: torch.Tensor):
        h, w = mask.shape
        self.mask.data[:h, :w] = mask
        self.mask.data.requires_grad_(False)

    def set_value(self, weight: torch.Tensor):
        h, w = weight.shape
        self.weight.data[:h, :w] = weight
        self.weight.data.requires_grad_(True)


# ---------------------------------------------------------------------------
# Decomposed projection modules
# ---------------------------------------------------------------------------

class DecomposeLinear_headwise(nn.Module):
    """Per-head SVD factorisation of a K projection: W ≈ U · diag(S) · V.

    Each attention head gets its own (U_h, S_h, V_h) block so thresholds can
    be learned independently per head.
    """

    def __init__(self, linear_layer: nn.Linear, device=None):
        super().__init__()
        cur_device = linear_layer.weight.device
        linear_weight = linear_layer.weight.detach()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.device = cur_device
        self._num_heads = getattr(linear_layer, "_num_heads", None)
        self._head_dim = getattr(linear_layer, "_head_dim", None)
        self._head_thresholds = getattr(linear_layer, "_head_thresholds", None)

        U, Sigma, V = self._svd(linear_weight)
        self.rank = Sigma.shape[0]

        self.U = MaskedLinear(self.rank, self.out_features, False).to(cur_device)
        self.V = MaskedLinear(self.in_features, self.rank, False).to(cur_device)

        if getattr(self, "_r_list", None) is not None:
            H = len(self._r_list)
            th = self._head_thresholds
            if th is None:
                th_list = [0.0] * H
            elif isinstance(th, (float, int)):
                th_list = [float(th)] * H
            else:
                th_list = [float(x) for x in th]

            self.Sigma_blocks = nn.ModuleList(
                DiagonalLinear(r_h, thres=th_list[i]).to(cur_device)
                for i, r_h in enumerate(self._r_list)
            )
            offset = 0
            for r_h, mod in zip(self._r_list, self.Sigma_blocks):
                mod.set_value(Sigma[offset:offset + r_h].to(cur_device))
                offset += r_h

            # Constrain U to be block-diagonal: head h's output rows may only
            # read head h's rank block. The SVD initialises it that way, but
            # without this mask training fills the off-block entries with
            # cross-head mixing that a headwise decomposition cannot express --
            # export_kproj_for_triton slices per head and silently drops it,
            # so the deployed model diverges from the trained one.
            head_mask = torch.zeros(self.out_features, self.rank, device=cur_device)
            row = col = 0
            for r_h in self._r_list:
                head_mask[row:row + self._head_dim, col:col + r_h] = 1.0
                row += self._head_dim
                col += r_h
            self.U.set_mask(head_mask)
        else:
            self.Sigma = DiagonalLinear(self.rank, 0.0).to(cur_device)
            self.Sigma.set_value(Sigma.to(cur_device))

        self.U.set_value(U.to(cur_device))
        self.V.set_value(V.to(cur_device))
        self.bias = None

    def forward(self, x):
        h = self.V(x)
        if hasattr(self, "Sigma_blocks"):
            chunks = torch.split(h, self._r_list, dim=-1)
            h = torch.cat([mod(c) for mod, c in zip(self.Sigma_blocks, chunks)], dim=-1)
        else:
            h = self.Sigma(h)
        return self.U(h)

    def _svd(self, W: torch.Tensor):
        Out, In = W.shape
        Hd = self._head_dim
        if Hd is None:
            if self._num_heads is None:
                self._r_list = None
                return torch.linalg.svd(W, full_matrices=False)
            Hd = In // self._num_heads
        if Out % Hd != 0:
            self._r_list = None
            return torch.linalg.svd(W, full_matrices=False)

        # Persist the resolved head_dim: it may have been derived from
        # _num_heads above, and the U block mask / Triton export both need it.
        self._head_dim = Hd

        H = Out // Hd
        W_heads = W.view(H, Hd, In)

        U_blocks, S_blocks, V_blocks = [], [], []
        for h in range(H):
            Uh, Sh, Vh = torch.linalg.svd(W_heads[h], full_matrices=False)
            U_blocks.append(Uh)
            S_blocks.append(Sh)
            V_blocks.append(Vh)

        r_list = [u.shape[1] for u in U_blocks]
        r_total = sum(r_list)
        U_big = W.new_zeros(Out, r_total)
        S_big = W.new_zeros(r_total)
        V_big = W.new_zeros(r_total, In)

        row, col = 0, 0
        for h in range(H):
            r_h = r_list[h]
            U_big[row:row + Hd, col:col + r_h] = U_blocks[h]
            S_big[col:col + r_h] = S_blocks[h]
            V_big[col:col + r_h, :] = V_blocks[h]
            row += Hd
            col += r_h

        self._r_list = r_list
        return U_big, S_big, V_big


class DecomposeLinear(nn.Module):
    """Global SVD factorisation of a V projection: W ≈ U · diag(S) · V."""

    def __init__(self, linear_layer: nn.Linear, device=None):
        super().__init__()
        cur_device = linear_layer.weight.device
        linear_weight = linear_layer.weight.detach()

        self.rank = min(linear_weight.shape)
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.device = cur_device

        self.U = MaskedLinear(self.rank, self.out_features, False).to(cur_device)
        self.Sigma = DiagonalLinear(self.rank, 0.0).to(cur_device)
        self.V = MaskedLinear(self.in_features, self.rank, False).to(cur_device)

        U, Sigma, V = torch.linalg.svd(linear_weight, full_matrices=False)
        self.U.set_value(U.to(cur_device))
        self.Sigma.set_value(Sigma.to(cur_device))
        self.V.set_value(V.to(cur_device))
        self.bias = None

    def forward(self, x):
        return self.U(self.Sigma(self.V(x)))


# ---------------------------------------------------------------------------
# Fused (post-training) projections: only U and VS survive
# ---------------------------------------------------------------------------

class FusedDecomposeLinear(nn.Module):
    """Joint low-rank projection after fusion: W ~= U @ VS.

    Sigma is gone -- its surviving entries are multiplied into VS and its dead
    directions are physically removed, so `rank` is the real kept rank rather
    than the full-rank width with zeroed rows. Nothing here has a soft
    threshold, a mask, or anything else that only made sense during training.
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.VS = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.U = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.bias = None

    kv_quant_blocks = None   # set by fold_kv_hadamard
    kv_fake_quant = False    # set by set_kv_fake_quant

    def forward(self, x):
        h = self.VS(x)
        if self.kv_fake_quant:
            h = kv_quant.fake_quant(h, self.kv_quant_blocks)
        return self.U(h)


class FusedDecomposeLinear_headwise(nn.Module):
    """Per-head low-rank K projection after fusion: W ~= U @ VS, U block-diagonal.

    Heads keep different numbers of directions, so VS is their concatenation and
    `head_ranks` records the split. It is a persistent buffer rather than a
    Python attribute precisely so the block structure survives in the checkpoint
    -- without it, a pruned U/VS pair is just two matrices with no way to
    recover which rows belong to which head, and the Triton export could not
    slice them.
    """

    def __init__(self, in_features: int, out_features: int, head_ranks,
                 head_dim: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._head_dim = head_dim
        ranks = [int(r) for r in head_ranks]
        self.rank = int(sum(ranks))
        self.VS = nn.Linear(in_features, self.rank, bias=False, device=device, dtype=dtype)
        self.U = nn.Linear(self.rank, out_features, bias=False, device=device, dtype=dtype)
        self.register_buffer(
            "head_ranks", torch.tensor(ranks, dtype=torch.int32, device=device)
        )
        self.bias = None

        # Hold U block-diagonal under any later fine-tune (e.g. phase 3). Fusion
        # emits it with off-block entries at zero, but U is a plain nn.Linear
        # now, so a single optimizer step would train them into cross-head
        # mixing -- which export_kproj_for_triton slices away, silently
        # reopening the gap between the trained and the deployed model.
        # Masking the gradient enforces this at zero inference cost, and the
        # mask is non-persistent so the checkpoint still holds only U, VS and
        # head_ranks (it is rebuilt from head_ranks on construction anyway).
        block_mask = torch.zeros(out_features, self.rank, device=device, dtype=dtype)
        row = col = 0
        for r in ranks:
            block_mask[row:row + head_dim, col:col + r] = 1.0
            row += head_dim
            col += r
        self.register_buffer("block_mask", block_mask, persistent=False)
        self.U.weight.register_hook(lambda g: g * self.block_mask)

    @property
    def _r_list(self):
        return [int(r) for r in self.head_ranks.tolist()]

    kv_quant_blocks = None   # set by fold_kv_hadamard
    kv_fake_quant = False    # set by set_kv_fake_quant

    def forward(self, x):
        h = self.VS(x)
        if self.kv_fake_quant:
            h = kv_quant.fake_quant(h, self.kv_quant_blocks)
        return self.U(h)


@torch.no_grad()
def _fuse_joint(dec: DecomposeLinear) -> FusedDecomposeLinear:
    """DecomposeLinear -> FusedDecomposeLinear, dead directions removed."""
    device = dec.U.weight.device
    dtype = dec.U.weight.dtype
    s_eff = dec.Sigma.effective_diag()
    keep = dec.Sigma.keep_indices()
    if keep.numel() == 0:                      # never emit a rank-0 layer
        keep = torch.zeros(1, dtype=torch.long, device=device)

    # MaskedLinear's forward is linear(x, weight * mask), so the mask is part of
    # the effective weight and must be applied before pruning.
    V_eff = (dec.V.weight * dec.V.mask).detach().float()
    U_eff = (dec.U.weight * dec.U.mask).detach().float()

    out = FusedDecomposeLinear(dec.in_features, dec.out_features, keep.numel(),
                               device=device, dtype=dtype)
    out.VS.weight.data.copy_((V_eff[keep] * s_eff[keep, None]).to(dtype))
    out.U.weight.data.copy_(U_eff[:, keep].to(dtype))
    return out


@torch.no_grad()
def _fuse_headwise(dec: DecomposeLinear_headwise) -> FusedDecomposeLinear_headwise:
    """DecomposeLinear_headwise -> FusedDecomposeLinear_headwise, per head."""
    device = dec.U.weight.device
    dtype = dec.U.weight.dtype
    Hd = dec._head_dim
    V_eff = (dec.V.weight * dec.V.mask).detach().float()
    U_eff = (dec.U.weight * dec.U.mask).detach().float()

    VS_blocks, U_blocks, ranks = [], [], []
    col = 0
    for h, sb in enumerate(dec.Sigma_blocks):
        r_h = sb.diag.numel()
        s_eff = sb.effective_diag()
        keep = sb.keep_indices()
        if keep.numel() == 0:
            keep = torch.zeros(1, dtype=torch.long, device=device)
        cols = col + keep
        VS_blocks.append(V_eff[cols] * s_eff[keep, None])
        U_blocks.append(U_eff[h * Hd:(h + 1) * Hd][:, cols])
        ranks.append(int(keep.numel()))
        col += r_h

    out = FusedDecomposeLinear_headwise(dec.in_features, dec.out_features, ranks,
                                        Hd, device=device, dtype=dtype)
    out.VS.weight.data.copy_(torch.cat(VS_blocks, dim=0).to(dtype))
    # Reassemble U block-diagonally at the pruned widths.
    U_new = U_eff.new_zeros(dec.out_features, sum(ranks))
    off = 0
    for h, (blk, r) in enumerate(zip(U_blocks, ranks)):
        U_new[h * Hd:(h + 1) * Hd, off:off + r] = blk
        off += r
    out.U.weight.data.copy_(U_new.to(dtype))
    return out


@torch.no_grad()
def enforce_rank_floor(model, min_rank: int, skip_layers: tuple = (0, 1, 31)):
    """Clamp every soft-threshold so at least `min_rank` directions survive it.

    Nothing stops alpha from climbing past a head's entire spectrum, and heads
    that collapse to rank 1-4 stop carrying signal while still occupying a slot.
    The floor has to be applied during training, not at fusion: a direction
    pruned by the threshold has s_eff = 0, so its VS row is zero, and
    "restoring" it afterwards would just re-insert a zero row the model was
    never trained to use.

    Since `keep` is `diag > alpha`, guaranteeing k survivors means holding alpha
    below the k-th largest singular value. The margin is relative because diag
    is bf16 during training, where subtracting a tiny absolute epsilon rounds
    straight back to the original value.

    The floor is usually free in cache terms: the K cache pads every head up to
    its layer's max rank, so raising a short head costs nothing unless the floor
    exceeds that max.
    """
    if min_rank <= 0:
        return
    skip = set(skip_layers)
    for i, block in enumerate(model.model.layers):
        if i in skip:
            continue
        attn = block.self_attn

        sigmas = []
        kp = attn.k_proj
        if isinstance(kp, DecomposeLinear_headwise):
            sigmas.extend(kp.Sigma_blocks if hasattr(kp, "Sigma_blocks") else [kp.Sigma])
        vp = attn.v_proj
        if isinstance(vp, DecomposeLinear):
            sigmas.append(vp.Sigma)

        for sig in sigmas:
            if bool(sig.rank_frozen):
                continue          # rank is pinned; alpha no longer gates anything
            diag = sig.diag.detach().float()
            k = min(min_rank, diag.numel())
            kth = float(torch.topk(diag, k).values[-1])
            margin = max(abs(kth) * 1e-2, 1e-4)
            sig.soft_thres_layer.alpha.data.clamp_(max=kth - margin)


def align_up(r: int, multiple: int) -> int:
    """Smallest multiple of `multiple` that is >= r. multiple <= 1 is a no-op."""
    if multiple <= 1:
        return int(r)
    return int(-(-int(r) // multiple) * multiple)


@torch.no_grad()
@torch.no_grad()
def freeze_ranks_at_multiple(model, multiple: int = 16,
                             skip_layers: tuple = (0, 1, 31),
                             verbose: bool = True,
                             v_multiple: int = 1):
    """Phase-2 entry: pin every rank and retire the soft threshold.

    Phase 1 searched for the rank with a learnable threshold alpha. Once that
    search is over the threshold has done its job, so each projection's rank is
    fixed outright:

        r_new = align_up(r_phase1, multiple)      # K
        r_new = align_up(r_phase1, v_multiple)    # V, default 1 = pin as-is

    and `DiagonalLinear.freeze_rank` stores a binary keep mask over the top-r_new
    singular directions. From here the forward pass is `diag * keep_mask` --
    no tanh, no alpha.

    Two things this fixes versus lowering alpha to admit more directions:

      * Magnitude. Under the soft threshold a direction just above alpha is
        scaled by tanh(s*(diag - alpha)) ~ 0, and the directions a rank increase
        admits are BY CONSTRUCTION the ones nearest alpha. Measured on a rank
        42 -> 48 round-up, the six new directions came back at 0.72, 0.66, 0.56,
        0.42, 0.30 and 0.05 of their true singular values -- full cache cost,
        a fraction of the signal. With a keep mask they are restored at 1.00.

      * Stability. Phase 2 keeps training `diag`, so a surviving value that
        drifts back under alpha silently drops a direction mid-recovery, and
        the fused checkpoint no longer matches the rank the budget was checked
        against. A pinned mask cannot drift.

    K rounds to `multiple` because the decode kernel walks the rank dimension in
    BLOCK_SIZE_R=16 tiles and masks the tail, so rounding up to a tile boundary
    adds directions inside tiles that are already being issued. V is a single
    global factorisation consumed by a plain GEMM with no such masking, so it
    defaults to v_multiple=1: pinned, not widened.

    Phase 3 then merges Sigma into V and truncates U/VS to these ranks
    (`fuse_and_prune`), which reads the same keep masks.
    """
    skip = set(skip_layers)
    k_changed, v_changed = {}, {}

    for i, block in enumerate(model.model.layers):
        if i in skip:
            continue

        kp = block.self_attn.k_proj
        if isinstance(kp, DecomposeLinear_headwise):
            sigmas = kp.Sigma_blocks if hasattr(kp, "Sigma_blocks") else [kp.Sigma]
            before, after = [], []
            for sig in sigmas:
                r = sig.current_rank()
                before.append(r)
                after.append(sig.freeze_rank(align_up(r, max(multiple, 1))))
            k_changed[i] = (before, after)

        vp = block.self_attn.v_proj
        if isinstance(vp, DecomposeLinear):
            r = vp.Sigma.current_rank()
            v_changed[i] = (r, vp.Sigma.freeze_rank(align_up(r, max(v_multiple, 1))))

    if verbose:
        if k_changed:
            b = sum(sum(v[0]) for v in k_changed.values())
            a = sum(sum(v[1]) for v in k_changed.values())
            print(f"  [rank freeze] K pinned, rounded to multiples of {multiple}: "
                  f"{b} -> {a} directions ({a / b - 1:+.1%}) across {len(k_changed)} layers")
        if v_changed:
            b = sum(v[0] for v in v_changed.values())
            a = sum(v[1] for v in v_changed.values())
            tag = (f"rounded to multiples of {v_multiple}" if v_multiple > 1
                   else "pinned as-is")
            print(f"  [rank freeze] V {tag}: {b} -> {a} directions "
                  f"({a / b - 1:+.1%}) across {len(v_changed)} layers")
        print("  [rank freeze] soft threshold retired; singular values now used at "
              "full magnitude and ranks can no longer drift during recovery")
    return {"k": k_changed, "v": v_changed}


# ---------------------------------------------------------------------------
# KV cache accounting (elements per token -- what actually ships)
# ---------------------------------------------------------------------------
# collect_K/V_parameter_size below count PROJECTION WEIGHT parameters,
# (in+out)*rank against in*out. That is not the KV cache size: the cache holds
# `rank` elements per token against `out_features` uncompressed, so the two
# differ by a model-dependent factor (in+out)/in -- 2.00x for MHA
# (longchat-7b, 4096->4096) and 1.33x for GQA (Llama-3.2-3B, 3072->1024) -- and
# the weight metric additionally saturates at 0% until rank drops below
# in*out/(in+out). The functions here count the cache instead.


def _k_head_ranks(module) -> list:
    if hasattr(module, "Sigma_blocks"):
        return [sb.current_rank() for sb in module.Sigma_blocks]
    return [module.Sigma.current_rank()]


def collect_K_cache_size(model, multiple: int = 1) -> int:
    """K cache elements per token, summed over compressed layers: sum_h(rank_h),
    since every head is stored at its own width. Pair it with
    multiple=RANK_TILE to get the shipped size exactly, or call
    shipped_K_cache_size().

    `multiple` applies the same rounding freeze_ranks_at_multiple does, so the
    two stay consistent.
    """
    total = 0
    for name, m in model.named_modules():
        if not (isinstance(m, DecomposeLinear_headwise) and "k_proj" in name):
            continue
        ranks = [align_up(r, multiple) for r in _k_head_ranks(m)]
        total += sum(ranks)
    return total


def collect_V_cache_size(model, multiple: int = 1) -> int:
    """V cache elements per token, summed over compressed layers.

    `multiple` applies the same rounding freeze_ranks_at_multiple(v_multiple=...)
    does, so the phase-1 budget prices leveling in rather than being surprised
    by it afterwards.
    """
    return sum(
        align_up(get_rank(m), multiple)
        for name, m in model.named_modules()
        if isinstance(m, DecomposeLinear) and "v_proj" in name
    )


def shipped_K_cache_size(model) -> int:
    """K cache elements per token that the exported model actually allocates.

    Every head is stored at its own rank rounded up to RANK_TILE, so this is
    sum_h(align_up(rank_h, RANK_TILE)) -- independent of --rank-multiple, which
    only decides what phase 2 rounds the *trained* ranks to. Quote this against
    full_K_cache_size() for the honest K compression of a checkpoint.
    """
    return collect_K_cache_size(model, RANK_TILE)


def full_K_cache_size(model) -> int:
    return sum(
        m.out_features
        for name, m in model.named_modules()
        if isinstance(m, DecomposeLinear_headwise) and "k_proj" in name
    )


def full_V_cache_size(model) -> int:
    return sum(
        m.out_features
        for name, m in model.named_modules()
        if isinstance(m, DecomposeLinear) and "v_proj" in name
    )


@torch.no_grad()
def fuse_and_prune(model, skip_layers: tuple = (0, 1, 31)):
    """Swap every training-time decomposition for a pruned U/VS pair, in place.

    This is the real fusion: afterwards the model holds two low-rank factors and
    nothing else -- no Sigma, no soft-threshold, no mask, and no zeroed rows for
    directions the threshold killed. The rank is carried by the tensor shapes
    (plus head_ranks for the head-wise split), so a fused checkpoint no longer
    depends on `diag > alpha` to know what survived.
    """
    skip = set(skip_layers)
    for i, block in enumerate(model.model.layers):
        if i in skip:
            continue
        attn = block.self_attn
        if isinstance(attn.v_proj, DecomposeLinear):
            attn.v_proj = _fuse_joint(attn.v_proj)
        if isinstance(attn.k_proj, DecomposeLinear_headwise):
            attn.k_proj = _fuse_headwise(attn.k_proj)
    return model


@torch.no_grad()
def build_fused_from_state_dict(model, config, state_dict,
                                skip_layers: tuple = (0, 1, 31)):
    """Install fused modules shaped from a fused checkpoint's own tensors.

    Pruned factors have per-layer (and per-head) widths, so the modules cannot
    be built from the base model's Linear layers the way replace_linear_layer
    does -- the shapes have to come from the checkpoint before load_state_dict
    runs. VS/U tensor shapes plus head_ranks carry everything needed.
    """
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    skip = set(skip_layers)
    for i, block in enumerate(model.model.layers):
        if i in skip:
            continue
        attn = block.self_attn
        pre = f"model.layers.{i}.self_attn."

        k_vs = state_dict.get(pre + "k_proj.VS.weight")
        if k_vs is not None:
            ranks = state_dict[pre + "k_proj.head_ranks"].tolist()
            out_f = state_dict[pre + "k_proj.U.weight"].shape[0]
            ref = attn.k_proj.weight
            attn.k_proj = FusedDecomposeLinear_headwise(
                k_vs.shape[1], out_f, ranks, head_dim,
                device=ref.device, dtype=ref.dtype,
            )

        v_vs = state_dict.get(pre + "v_proj.VS.weight")
        if v_vs is not None:
            out_f = state_dict[pre + "v_proj.U.weight"].shape[0]
            ref = attn.v_proj.weight
            attn.v_proj = FusedDecomposeLinear(
                v_vs.shape[1], out_f, v_vs.shape[0],
                device=ref.device, dtype=ref.dtype,
            )
    return model


def is_fused_state_dict(state_dict) -> bool:
    """True if the checkpoint holds pruned U/VS factors rather than U/Sigma/V."""
    return any(k.endswith("k_proj.VS.weight") or k.endswith("v_proj.VS.weight")
               for k in state_dict)


def infer_skip_layers(state_dict, num_layers: int) -> tuple:
    """Which attention layers a checkpoint left uncompressed.

    The checkpoint is the authority here: a layer is compressed iff it carries
    decomposition tensors. Deriving this beats trusting a --skip-layers flag
    that has to be repeated identically at train, eval and benchmark time --
    getting it wrong leaves a plain nn.Linear where the export expects factors,
    which used to surface as an AttributeError deep inside the export.
    """
    compressed = set()
    for key in state_dict:
        marker = None
        for tag in (".self_attn.k_proj.", ".self_attn.v_proj."):
            if tag in key:
                marker = key.split(tag)[1]
                break
        if marker is None or marker == "weight":     # plain Linear -> untouched
            continue
        try:
            compressed.add(int(key.split(".layers.")[1].split(".")[0]))
        except (IndexError, ValueError):
            continue
    return tuple(sorted(set(range(num_layers)) - compressed))


def load_compressed_checkpoint(model, config, weights_path,
                               skip_layers: tuple = (0, 1, 31),
                               init_svd: bool = False):
    """Load either checkpoint format, installing whichever modules it needs.

    Fused checkpoints (the format train.py now writes) get pruned U/VS modules
    sized from the file; legacy U/Sigma/V checkpoints fall back to
    replace_linear_layer. Raises if any decomposition tensor fails to load,
    since with init_svd=False an unloaded factor stays zero and would silently
    produce a dead layer rather than a merely inaccurate one.

    Returns (model, fused, skip_layers) -- skip_layers as read off the
    checkpoint, which callers must use for replace_attn_with_triton and
    set_model_mode so every stage agrees on which layers are compressed.
    """
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    fused = is_fused_state_dict(state_dict)

    # Trust the checkpoint over the flag: it records exactly which layers were
    # compressed, and a stale --skip-layers otherwise leaves an uncompressed
    # layer for the export to choke on.
    actual_skip = infer_skip_layers(state_dict, len(model.model.layers))
    if set(actual_skip) != set(skip_layers):
        print(f"  note: checkpoint leaves layers {actual_skip} uncompressed, "
              f"but --skip-layers said {tuple(sorted(skip_layers))}; "
              f"using the checkpoint's.")
    skip_layers = actual_skip

    if fused:
        build_fused_from_state_dict(model, config, state_dict, skip_layers)
    else:
        replace_linear_layer(model, config, skip_layers=skip_layers,
                             init_svd=init_svd)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    critical = [k for k in missing
                if any(t in k for t in (".U.", ".V.", ".VS.", "Sigma"))]
    if critical:
        raise RuntimeError(
            f"{len(critical)} decomposition tensors missing from {weights_path}; "
            f"they would stay zero. First few: {critical[:5]}"
        )
    return model, fused, skip_layers


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------

def replace_linear_layer(model, config, skip_layers: tuple = (0, 1, 31),
                         init_svd: bool = True):
    """Replace k_proj with DecomposeLinear_headwise and v_proj with DecomposeLinear
    for all transformer layers not in skip_layers.

    init_svd=False skips the SVD factorisation and leaves U/Sigma/V zeroed,
    keeping only the shapes (which is all that sets the per-head rank list).
    Use it when a checkpoint is loaded immediately afterwards -- load_state_dict
    overwrites every factor anyway, and the SVD costs ~15 minutes on CPU for a
    3B model. Training must keep the default, since it starts from the SVD.
    """
    if not init_svd:
        _real_svd = torch.linalg.svd

        def _shape_only_svd(W, full_matrices=False):
            out_f, in_f = W.shape
            k = min(out_f, in_f)
            return W.new_zeros(out_f, k), W.new_zeros(k), W.new_zeros(k, in_f)

        torch.linalg.svd = _shape_only_svd

    def _helper(module):
        for name, child in module.named_children():
            parent_name = type(module).__name__
            if isinstance(child, nn.Linear) and parent_name != "DecomposeLinear":
                if "k_proj" in name:
                    l = module.__getattr__(name)
                    setattr(l, "_num_heads", int(config.num_attention_heads))
                    if config.head_dim is not None:
                        setattr(l, "_head_dim", int(config.head_dim))
                    module.__setattr__(name, DecomposeLinear_headwise(l))
                elif "v_proj" in name:
                    l = module.__getattr__(name)
                    module.__setattr__(name, DecomposeLinear(l))
                else:
                    _helper(child)
            else:
                _helper(child)

    try:
        for i, block in enumerate(model.model.layers):
            if i not in set(skip_layers):
                _helper(block)
    finally:
        if not init_svd:
            torch.linalg.svd = _real_svd


# ---------------------------------------------------------------------------
# Rank / parameter counting
# ---------------------------------------------------------------------------

def get_rank(module) -> int:
    """Effective rank: the frozen keep mask if set, else `diag > alpha`."""
    if hasattr(module, "Sigma_blocks"):
        return sum(sb.current_rank() for sb in module.Sigma_blocks)
    return module.Sigma.current_rank()


def _param_size(module, equal: bool) -> int:
    rank = get_rank(module)
    n = module.U.weight.shape[0] * rank + module.V.weight.shape[1] * rank
    if equal:
        return min(n, module.in_features * module.out_features)
    return n


def collect_K_parameter_size(model, equal: bool = False) -> int:
    return sum(
        _param_size(m, equal)
        for name, m in model.named_modules()
        if isinstance(m, DecomposeLinear_headwise) and "k_proj" in name
    )


def collect_V_parameter_size(model, equal: bool = False) -> int:
    return sum(
        _param_size(m, equal)
        for name, m in model.named_modules()
        if isinstance(m, DecomposeLinear) and "v_proj" in name
    )


# ---------------------------------------------------------------------------
# Export: fold Sigma into V, prune dead ranks
# ---------------------------------------------------------------------------

@torch.no_grad()
def export_kproj_for_triton(
    decomp: DecomposeLinear_headwise,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[nn.Linear, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold soft-thresholded S into V per head.

    Returns (VS_linear, U_tensor, dense_weight, ranks, offsets).

    Heads keep different numbers of singular directions, and each is stored at
    its OWN width -- its rank rounded up to RANK_TILE -- laid out side by side
    along one flat rank axis. `offsets` says where each head's segment starts,
    and a layer stores sum(widths) per token.

    dense_weight is U @ VS reconstructed back to the original [out, in] shape,
    computed once here so prefill can do a single full-size matmul (matching
    the uncompressed baseline's cost) instead of reconstructing per token.

    VS_linear   : nn.Linear  [W, in_features]      W = sum of the per-head widths
    U_tensor    : Tensor     [W, head_dim]         rank-major, as the kernel reads it
    dense_weight: Tensor     [H * head_dim, in_features]
    ranks       : Tensor     [H]  int32, real rank kept per head
    offsets     : Tensor     [H]  int32, column offset of each head's segment
    """
    device = decomp.U.weight.device
    H = len(decomp._r_list)
    Hd = decomp._head_dim

    if isinstance(decomp, FusedDecomposeLinear_headwise):
        # Already fused and pruned: VS holds each head's surviving rows back to
        # back, U is block-diagonal at those widths. Nothing to threshold.
        VS_w = decomp.VS.weight.detach().float()
        U_w = decomp.U.weight.detach().float()
        VS_list, U_list = [], []
        col = 0
        for h, r_h in enumerate(decomp._r_list):
            VS_list.append(VS_w[col:col + r_h])
            U_list.append(U_w[h * Hd:(h + 1) * Hd, col:col + r_h])
            col += r_h
        return _pack_kproj_ragged(VS_list, U_list, decomp.in_features, H, Hd,
                                  device, dtype)

    U_w = decomp.U.weight.detach().float()
    V_w = decomp.V.weight.detach().float()

    VS_list, U_list = [], []
    col_off = 0
    for h in range(H):
        r_h = decomp._r_list[h]
        sig = decomp.Sigma_blocks[h]
        keep = sig.keep_indices()
        if keep.numel() == 0:
            keep = torch.zeros(1, dtype=torch.long, device=device)
        s_vals = sig.effective_diag()[keep]
        cols = col_off + keep
        VS_list.append(V_w[cols] * s_vals[:, None])
        U_list.append(U_w[h * Hd:(h + 1) * Hd][:, cols])
        col_off += r_h

    return _pack_kproj_ragged(VS_list, U_list, decomp.in_features, H, Hd,
                              device, dtype)


def _pack_kproj_ragged(VS_list, U_list, in_features, H, Hd, device, dtype):
    """Pack per-head factors side by side, each at its own width.

    Head h keeps rank_h directions and occupies width_h = align_up(rank_h,
    RANK_TILE) columns of one flat rank axis, starting at offsets[h]. The tile
    round-up is what the decode kernel walks in, and it keeps every segment
    32-byte aligned; beyond rank_h the rows stay zero, so the kernel's tail mask
    and a plain full-width read agree. A layer stores sum(widths) per token.

    Shared by both export paths (fused and legacy) so the widths, the offsets,
    the rank vector and the dense reconstruction can only ever be built one way.
    """
    ranks = [int(v.shape[0]) for v in VS_list]
    widths = [align_up(r, RANK_TILE) for r in ranks]
    offsets = [sum(widths[:h]) for h in range(H)]
    W = sum(widths)

    # float32 while assembling: dense_weight below is a product of these, and
    # rounding the factors to bf16 first would bake that error into prefill.
    VS_cat = torch.zeros(W, in_features, device=device, dtype=torch.float32)
    U_cat = torch.zeros(W, Hd, device=device, dtype=torch.float32)
    dense_weight = torch.empty(H * Hd, in_features, device=device, dtype=torch.float32)

    for h, (off, r) in enumerate(zip(offsets, ranks)):
        vs_h = VS_list[h].float()                 # [r, in_features]
        u_h = U_list[h].float()                   # [head_dim, r]
        VS_cat[off:off + r] = vs_h
        U_cat[off:off + r] = u_h.T                # rank-major, as the kernel reads it
        # dense_weight[h] = U_h @ VS_h reconstructs the original [head_dim, in] block
        dense_weight[h * Hd:(h + 1) * Hd] = u_h @ vs_h

    VS_lin = nn.Linear(in_features, W, bias=False).to(device=device, dtype=dtype)
    VS_lin.weight = nn.Parameter(VS_cat.to(dtype))

    return (
        VS_lin,
        U_cat.to(dtype),
        dense_weight.to(dtype),
        torch.tensor(ranks, dtype=torch.int32, device=device),
        torch.tensor(offsets, dtype=torch.int32, device=device),
    )


@torch.no_grad()
def export_vproj_for_triton(
    decomp: DecomposeLinear,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[nn.Linear, nn.Linear, torch.Tensor]:
    """Fold soft-thresholded S into V; return (VS_linear, U_linear, dense_weight).

    dense_weight = U @ VS reconstructed back to [out_features, in_features],
    computed once here so prefill can do a single full-size matmul instead of
    reconstructing per token.

    VS_linear   : nn.Linear  [rank, in_features]
    U_linear    : nn.Linear  [out_features, rank]
    dense_weight: Tensor     [out_features, in_features]
    """
    device = decomp.U.weight.device

    if isinstance(decomp, FusedDecomposeLinear):
        # Already fused and pruned: VS/U are the exported factors verbatim.
        VS = decomp.VS.weight.detach().to(dtype)
        U_pruned = decomp.U.weight.detach().to(dtype)
    else:
        U_w = decomp.U.weight.detach().float()
        V_w = decomp.V.weight.detach().float()

        sig = decomp.Sigma
        keep = sig.keep_indices()
        if keep.numel() == 0:
            keep = torch.zeros(1, dtype=torch.long, device=device)
        s_vals = sig.effective_diag()[keep]

        VS = (V_w[keep] * s_vals[:, None]).to(dtype)
        U_pruned = U_w[:, keep].to(dtype)

    rk = VS.shape[0]
    VS_lin = nn.Linear(decomp.in_features, rk, bias=False).to(device=device, dtype=dtype)
    U_lin = nn.Linear(rk, decomp.out_features, bias=False).to(device=device, dtype=dtype)
    VS_lin.weight = nn.Parameter(VS.to(device))
    U_lin.weight = nn.Parameter(U_pruned.to(device))

    dense_weight = (U_pruned.float() @ VS.float()).to(dtype).to(device)
    return VS_lin, U_lin, dense_weight


# ---------------------------------------------------------------------------
# Inference wrappers (satisfy LlamaCustomAttention's .U and .VS interface)
# ---------------------------------------------------------------------------

class KProjInferenceWrapper(nn.Module):
    """Wraps exported k_proj so LlamaCustomAttention can access .VS, .U,
    .dense_weight, .ranks and .offsets.

    The K cache is ragged: head h lives at columns [offsets[h], offsets[h] +
    width_h) of a flat rank axis, where width_h is ranks[h] rounded up to
    RANK_TILE. `segments` is the same information as plain Python ints, so the
    reference paths can slice per head without a device sync on every step.
    """

    def __init__(self, VS_linear: nn.Linear, U_tensor: torch.Tensor,
                 dense_weight: torch.Tensor, ranks: torch.Tensor,
                 offsets: torch.Tensor):
        super().__init__()
        self.VS = VS_linear
        self.register_buffer("U", U_tensor)
        self.register_buffer("dense_weight", dense_weight)
        self.register_buffer("ranks", ranks)
        self.register_buffer("offsets", offsets)
        self.segments = list(zip(offsets.tolist(), ranks.tolist()))


class VProjInferenceWrapper(nn.Module):
    """Wraps exported v_proj so LlamaCustomAttention can access .VS, .U, .dense_weight.

    `U_by_query_head` is U reshaped to [num_query_heads, rank_v, head_dim]: the
    per-KV-head basis already broadcast across its GQA group and transposed into
    the layout the decode matmul consumes. It is pure weight data, identical on
    every step, so building it here once replaces a repeat_interleave +
    transpose that otherwise ran per token per layer.
    """

    def __init__(self, VS_linear: nn.Linear, U_linear: nn.Linear,
                 dense_weight: torch.Tensor, U_by_query_head: torch.Tensor):
        super().__init__()
        self.VS = VS_linear
        self.U = U_linear
        self.register_buffer("dense_weight", dense_weight)
        self.register_buffer("U_by_query_head", U_by_query_head)


# ---------------------------------------------------------------------------
# Unified prefill+decode attention forward
# ---------------------------------------------------------------------------

def _require_rank_major_layer(past_key_values, layer_idx: int) -> None:
    """Fail clearly when a compressed layer is handed a stock cache layer.

    K is rank-major here, so a DynamicLayer would concatenate it along the rank
    axis and raise a shape error from inside torch.cat that says nothing about
    the cause. `replace_attn_with_triton` installs the right cache automatically;
    this only triggers when a caller passes one of its own.
    """
    layers = getattr(past_key_values, "layers", None)
    if layers is None:
        return
    if layer_idx < len(layers):
        layer_cls = type(layers[layer_idx])
    else:
        # The cache grows lazily and will append this class for the new index.
        layer_cls = getattr(past_key_values, "layer_class_to_replicate", None)
        if layer_cls is None:
            return
    if not issubclass(layer_cls, (RankMajorKLayer, PackedKVLayer)):
        raise TypeError(
            f"layer {layer_idx} is compressed and stores K rank-major, but the "
            f"cache supplied a {layer_cls.__name__}. Pass "
            f"StarKVCache(num_hidden_layers, skip_layers), or pass no cache at "
            f"all and let the model build one."
        )


def _expand_ragged_k(key_states: torch.Tensor, k_u: torch.Tensor, segments) -> torch.Tensor:
    """Rebuild full [B, H, seq, head_dim] keys from the ragged latent cache.

    key_states is rank-major [B, W, seq] with head h at rows [off, off + width_h);
    only the first `rank_h` of those carry signal, the rest of the tile is zero, so
    each head is expanded from its real rank.

    Heads have different widths, so this is a loop of small matmuls instead of
    one batched GEMM. That is fine here because every caller is a reference or
    baseline path -- the fused decode kernel never materialises full keys.
    """
    u = k_u.to(key_states.dtype)
    return torch.stack(
        [key_states[:, off:off + r, :].transpose(1, 2) @ u[off:off + r]
         for off, r in segments],
        dim=1,
    )

def _patched_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Prefill uses SDPA; decode uses fused Triton ABX+RoPE or standard QK^T."""
    from abx_rope_batched import abx as _abx

    input_shape = hidden_states.shape[:-1]
    query_len = input_shape[-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    k_u = self.k_proj.U
    k_vs = self.k_proj.VS
    v_vs = self.v_proj.VS

    # Ragged K latent, rank-major: [B, W, seq]. Heads sit side by side along W at
    # their own widths -- no per-head view exists (the widths differ) and none is
    # needed, the decode kernel locates each head through k_proj.offsets. Tokens
    # are the LAST axis so each head's lanes stay contiguous across them; see
    # RankMajorKLayer. Producing it as weight @ hidden^T writes that layout
    # directly instead of transposing a [B, seq, W] result afterwards.
    k_inter = torch.matmul(k_vs.weight, hidden_states.transpose(1, 2))
    v_inter = torch.matmul(hidden_states, v_vs.weight.T)

    if past_key_values is not None:
        _require_rank_major_layer(past_key_values, self.layer_idx)
        key_states, value_states = past_key_values.update(k_inter, v_inter, self.layer_idx)
    else:
        key_states, value_states = k_inter, v_inter

    cos, sin = position_embeddings
    query_states = _rope(query_states, cos, sin)

    H = self.num_key_value_heads
    Hd = self.head_dim

    if query_len > 1:
        # Prefill: same cost as the uncompressed baseline. Use the dense
        # U@VS reconstruction (computed once at export time, not per token)
        # directly on hidden_states, instead of round-tripping through the
        # compact cache representation. The compact k_inter/v_inter computed
        # above still get written to past_key_values regardless, so decode
        # continues from that low-rank latent cache as usual. Only valid for
        # single-shot (non-chunked) prefill, i.e. hidden_states covers the
        # whole new segment with no pre-existing cache mixed in.
        key_full = (
            F.linear(hidden_states, self.k_proj.dense_weight)
            .view(*input_shape, H, Hd)
            .transpose(1, 2)
        )
        key_full = _rope(key_full, cos, sin)
        value_full = (
            F.linear(hidden_states, self.v_proj.dense_weight)
            .view(*input_shape, H, Hd)
            .transpose(1, 2)
        )
        causal_mask = (
            # key_states is rank-major [B, W, seq]: the cache length is the LAST
            # axis, not -2.
            attention_mask[:, :, :, : key_states.shape[-1]]
            if attention_mask is not None else None
        )
        if causal_mask is None:
            # No padding: let SDPA broadcast the 8 KV heads against 24 query
            # heads internally (flash/cuDNN native GQA), matching the
            # uncompressed baseline instead of materialising a 3x-larger
            # repeated K/V via repeat_interleave.
            attn_output = F.scaled_dot_product_attention(
                query_states, key_full, value_full, scale=self.scaling,
                enable_gqa=True, is_causal=True,
            )
        else:
            key_full = key_full.repeat_interleave(self.num_key_value_groups, dim=1)
            value_full = value_full.repeat_interleave(self.num_key_value_groups, dim=1)
            attn_output = F.scaled_dot_product_attention(
                query_states, key_full, value_full, attn_mask=causal_mask, scale=self.scaling
            )
    else:
        # Decode: fused Triton kernel or standard QK^T
        if attn_module.triton_kernel:
            # inv_freq/attn_scaling come from the model's own rotary embedding,
            # so the RoPE the kernel applies to K matches the one applied to Q
            # above for every rope type (llama3, yarn, linear, ...).
            attn_weights = _abx(
                query_states,
                k_u,
                key_states.contiguous(),
                ranks=self.k_proj.ranks,
                offsets=self.k_proj.offsets,
                inv_freq=self.rope_inv_freq,
                attn_scaling=self.rope_attn_scaling,
                # Compute in the cache's own dtype. This was pinned to float16,
                # which on a bf16 model rounds the reconstructed K, the cos/sin
                # and the Q.K product through a format nothing else here uses:
                # measured 1.0-3.4% slower than bf16 (the fp32->fp16 conversion
                # costs more than the truncation to bf16), and fp16 tops out at
                # 65504, which a K summed over up to 80 rank directions can
                # approach where bf16 -- same exponent range as fp32 -- cannot.
                # The accumulation is fp32 either way and the logits are stored
                # in the cache dtype regardless, so the extra fp16 mantissa was
                # discarded immediately.
                dtype=key_states.dtype,
            ) / math.sqrt(self.head_dim)
        else:
            key_full = _expand_ragged_k(key_states, k_u, self.k_proj.segments)
            # The cached keys span absolute positions 0..kv_len-1, but cos/sin
            # from position_embeddings cover only the current query position.
            # Rebuild the full tables rather than letting _rope fall back to
            # broadcasting the last position's rotation over every cached key.
            k_cos, k_sin = _rope_tables(
                self.rope_inv_freq, key_full.shape[-2],
                key_full.device, key_full.dtype, self.rope_attn_scaling,
            )
            key_full = key_full * k_cos + rotate_half(key_full) * k_sin
            # GQA: expand K from num_key_value_heads to num_attention_heads so it
            # matches query_states. No-op for MHA (num_key_value_groups == 1).
            key_full = key_full.repeat_interleave(self.num_key_value_groups, dim=1)
            attn_weights = torch.matmul(query_states, key_full.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            # key_states is rank-major [B, W, seq]; cache length is the last axis.
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-1]]

        # `dtype=torch.float32` would materialise the whole attention row in fp32
        # and immediately cast it back: at 64k x batch 8 that is a 65 MB tensor
        # written and re-read for nothing, and it costs more than the softmax
        # itself (396us, against 121us without it). Asking for a bf16 result runs
        # the same math -- torch reduces and exponentiates in fp32 for a bf16
        # input -- and rounds once at the end instead of twice.
        #
        # Not bitwise identical to the fp32 round trip on long rows: the two
        # disagree on ~0.3% of outputs by at most one bf16 ulp (3.1e-3 relative,
        # against bf16's own 3.9e-3 resolution).
        attn_weights = F.softmax(attn_weights, dim=-1).to(query_states.dtype)
        attn_weights = F.dropout(
            attn_weights,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )

        # Aggregate compact V then expand with U (avoids full V materialisation).
        # attn_weights spans all query heads; value_states is the shared global
        # compact V, so this is one GEMM with the heads stacked rather than one
        # GEMV per head. The U basis, already broadcast across each GQA group,
        # is precomputed at export time (see VProjInferenceWrapper).
        prob_v = attn_weights.squeeze(2) @ value_states                  # [B, Hq, rank_v]
        # Expand with U as a batched GEMM over heads, NOT as [B,Hq,1,rank_v] @
        # [Hq,rank_v,Hd]. That broadcast form is B*Hq separate problems with
        # M=1 -- a pile of GEMVs the card cannot fill. Putting the batch in M
        # instead leaves Hq problems of shape [B,rank_v] @ [rank_v,Hd], which is
        # bit-for-bit the same arithmetic and measured 7x faster at batch 8,
        # 17x at batch 16 (456us -> 27us).
        attn_output = (
            torch.bmm(prob_v.transpose(0, 1), self.v_proj.U_by_query_head)
            .transpose(0, 1)                                              # [B, Hq, Hd]
            .unsqueeze(2)                                                 # [B, Hq, 1, Hd]
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), None


def _bf16_sdpa_fwd(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Expand K/V to full head_dim and use SDPA — baseline for latency comparison."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    Q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    k_vs, k_u = self.k_proj.VS, self.k_proj.U
    v_vs, v_u = self.v_proj.VS, self.v_proj.U
    H, Hd = self.num_key_value_heads, self.head_dim

    k_inter = torch.matmul(k_vs.weight, hidden_states.transpose(1, 2))   # [B, W, seq]
    v_inter = torch.matmul(hidden_states, v_vs.weight.T)

    if past_key_values is not None:
        k_inter, v_inter = past_key_values.update(k_inter, v_inter, self.layer_idx)

    cos, sin = position_embeddings
    Q = _rope(Q, cos, sin)

    K_full = _expand_ragged_k(k_inter, k_u, self.k_proj.segments)
    if K_full.shape[-2] == Q.shape[-2]:
        # Prefill: position_embeddings already covers every position.
        K_full = _rope(K_full, cos, sin)
    else:
        # Decode: the cached keys span absolute positions 0..kv_len-1, but cos/sin
        # covers only the current query position, so broadcasting it would rotate
        # every cached key by the newest position and erase relative position.
        # Same correction _patched_attn_forward's no-Triton branch makes.
        k_cos, k_sin = _rope_tables(
            self.rope_inv_freq, K_full.shape[-2],
            K_full.device, K_full.dtype, self.rope_attn_scaling,
        )
        K_full = K_full * k_cos + rotate_half(K_full) * k_sin
    V_full = (
        torch.matmul(v_inter, v_u.weight.T)
        .view(*v_inter.shape[:-1], H, Hd)
        .transpose(-3, -2)
        .contiguous()
    )

    mask = (
        # k_inter is rank-major [B, W, seq], so the cache length is the last axis.
        attention_mask[:, :, :, : k_inter.shape[-1]]
        if attention_mask is not None else None
    )
    # Causal only when this call is a prefill over several queries. At decode the
    # single query legitimately attends to every cached key, and is_causal there
    # would align the mask top-left and leave it attending to key 0 alone.
    # Without this the baseline reads the whole sequence bidirectionally, which
    # is neither what the model computes nor what it should be timed against.
    is_causal = mask is None and Q.shape[-2] > 1
    out = F.scaled_dot_product_attention(
        Q, K_full, V_full, attn_mask=mask, scale=self.scaling,
        is_causal=is_causal,
        # K/V are per KV head; let SDPA broadcast them across each GQA group
        # rather than silently mismatching the head counts.
        enable_gqa=self.num_key_value_groups > 1,
    )
    out = out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return self.o_proj(out), None


# ---------------------------------------------------------------------------
# Replace attention modules for inference
# ---------------------------------------------------------------------------

def replace_attn_with_triton(
    model,
    config,
    skip_layers: tuple = (0, 1, 31),
    dtype: torch.dtype = torch.bfloat16,
    kv_quant: bool = False,
):
    """Export decomposed K/V projections and install LlamaCustomAttention with
    the unified prefill+decode forward in all non-skip layers.

    kv_quant=True stores the cache packed to 4 bits and decodes through
    kv_quant.py's kernels (fold_kv_hadamard must have run on the fused model).
    """
    num_heads = config.num_attention_heads
    if kv_quant and config.num_key_value_heads != num_heads:
        raise NotImplementedError("the kv_quant decode kernels are MHA-only")

    # The decode kernel re-derives RoPE for the cached K internally, so it needs
    # the exact frequencies the model uses for Q. 
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None or not hasattr(rotary, "inv_freq"):
        raise RuntimeError(
            "model.model.rotary_emb.inv_freq not found; the Triton decode kernel "
            "needs it to apply RoPE consistently with the query path."
        )
    rope_inv_freq = rotary.inv_freq.detach().float()
    rope_attn_scaling = float(getattr(rotary, "attention_scaling", 1.0))
    compressed_any = False

    for i, block in enumerate(model.model.layers):
        if i in set(skip_layers):
            continue

        orig = block.self_attn
        device = next(orig.parameters()).device

        ratio = getattr(orig.k_proj, "kv_outlier_ratio", None)
        if kv_quant and ratio is None:
            raise RuntimeError(f"layer {i}: run fold_kv_hadamard(model) before kv_quant export")

        new_attn = LlamaCustomAttention(config, layer_idx=i).to(device=device, dtype=dtype)

        new_attn.q_proj.weight.data.copy_(orig.q_proj.weight.data.to(dtype))
        new_attn.o_proj.weight.data.copy_(orig.o_proj.weight.data.to(dtype))

        k_VS, k_U, k_dense, k_ranks, k_offsets = export_kproj_for_triton(
            orig.k_proj, dtype=dtype
        )
        new_attn.k_proj = KProjInferenceWrapper(k_VS, k_U, k_dense, k_ranks, k_offsets)

        v_VS, v_U, v_dense = export_vproj_for_triton(orig.v_proj, dtype=dtype)
        # Broadcast the per-KV-head V basis across its GQA group and transpose
        # into the decode matmul's layout, once, instead of per token.
        n_rep = num_heads // config.num_key_value_heads
        rank_v = v_U.weight.shape[1]
        v_U_by_q = (
            v_U.weight.detach()
            .view(config.num_key_value_heads, config.head_dim, rank_v)
            .repeat_interleave(n_rep, dim=0)
            .transpose(-1, -2)
            .contiguous()
        )
        new_attn.v_proj = VProjInferenceWrapper(v_VS, v_U, v_dense, v_U_by_q)

        new_attn.register_buffer("rope_inv_freq", rope_inv_freq.to(device), persistent=False)
        new_attn.rope_attn_scaling = rope_attn_scaling
        if kv_quant:
            new_attn.kvq = KVQuantMeta(k_ranks, k_offsets, rank_v, ratio).to(device)

        new_attn.forward = types.MethodType(
            _kv_quant_forward if kv_quant else _patched_attn_forward, new_attn)
        block.self_attn = new_attn
        compressed_any = True

    if compressed_any:
        _install_rank_major_cache(model, config, skip_layers)
    return model


def _needs_rank_major_cache(cache) -> bool:
    """Should this call get a StarKVCache built for it?

    Yes when the caller passed nothing, and also when it passed a stock cache
    that is still empty: `generate` builds its own DynamicCache before the first
    forward, and swapping it there is both safe and necessary, since generate
    carries whatever the first step returns. A cache with tokens already in it is
    left alone -- replacing it would silently drop them, and
    _require_rank_major_layer will reject it with an explanation instead.
    """
    if cache is None:
        return True
    if isinstance(cache, StarKVCache):
        return False
    return isinstance(cache, DynamicCache) and cache.get_seq_length() == 0


def _install_rank_major_cache(model, config, skip_layers: tuple):
    """Make the model build a StarKVCache when a caller does not pass one.

    Compressed layers store K rank-major ([B, W, seq]), which the stock
    DynamicCache cannot hold: it concatenates every layer on dim -2, which for
    that shape is the rank axis, not the tokens. LlamaModel.forward builds its
    own DynamicCache whenever use_cache is on and past_key_values is None, so
    without this every plain `model(..., use_cache=True)` would raise on the
    second step. Wrapping the inner model's forward keeps that invisible to
    callers instead of making each one remember to pass a cache.
    """
    inner = model.model
    if getattr(inner, "_starkv_cache_installed", False):
        return
    _inner_forward = inner.forward
    n_layers = config.num_hidden_layers
    skip = tuple(skip_layers)
    packed = {i: b.self_attn.kvq for i, b in enumerate(inner.layers) if hasattr(b.self_attn, "kvq")}

    def forward_with_starkv_cache(*args, past_key_values=None, use_cache=None, **kwargs):
        wants_cache = use_cache if use_cache is not None else bool(model.config.use_cache)
        if wants_cache and _needs_rank_major_cache(past_key_values):
            past_key_values = StarKVCache(n_layers, skip, packed)
        return _inner_forward(*args, past_key_values=past_key_values,
                              use_cache=use_cache, **kwargs)

    inner.forward = forward_with_starkv_cache
    inner._starkv_cache_installed = True


def set_model_mode(model, mode: str, skip_layers: tuple = (0, 1, 31)):
    """Switch all non-skip attention layers between triton / no_triton / bf16_sdpa."""
    # The packed cache is readable only by kv_quant's kernels. Refuse before
    # touching any state, so a rejected call leaves the model as it was.
    if mode != "triton" and any(hasattr(b.self_attn, "kvq") for b in model.model.layers):
        raise ValueError("this model uses the 4-bit packed KV cache; only mode='triton' can read it")
    attn_module.triton_kernel = (mode == "triton")
    for i, block in enumerate(model.model.layers):
        if i in set(skip_layers):
            continue
        attn = block.self_attn
        if hasattr(attn, "kvq"):
            attn.forward = types.MethodType(_kv_quant_forward, attn)
        elif mode in ("triton", "no_triton"):
            attn.forward = types.MethodType(_patched_attn_forward, attn)
        elif mode == "bf16_sdpa":
            attn.forward = types.MethodType(_bf16_sdpa_fwd, attn)


# ---------------------------------------------------------------------------
# 4-bit KV quantization: an add-on to the low-rank cache (format in kv_quant.py)
# ---------------------------------------------------------------------------

def _fused_kv_modules(model):
    for mod in model.modules():
        if isinstance(mod, (FusedDecomposeLinear, FusedDecomposeLinear_headwise)):
            yield mod


@torch.no_grad()
def fold_kv_hadamard(model, outlier_ratio: float = kv_quant.OUTLIER_RATIO, seed: int = 1234):
    """Rotate every fused K/V latent by a block Hadamard, folded into VS and U.

    VS <- T^T VS and U <- U T per head, so the codes are rotated and U undoes it:
    T is orthonormal, so U T T^T VS == U VS and the model computes the same
    function. Only the quantizer sees the
    rotated basis, which spreads a block's range across its channels. Records each
    head's (start, n_outlier, rank) for fake_quant and the packed cache. Run once,
    on the fused model, before replace_attn_with_triton.
    """
    found = False
    for mod in _fused_kv_modules(model):
        if mod.kv_quant_blocks is not None:
            raise RuntimeError("fold_kv_hadamard has already been applied")
        ranks = mod._r_list if isinstance(mod, FusedDecomposeLinear_headwise) else [mod.rank]
        gen = torch.Generator().manual_seed(seed)
        VS, U = mod.VS.weight.data, mod.U.weight.data
        blocks, a = [], 0
        for r in ranks:
            no = kv_quant.n_outlier(r, outlier_ratio)
            T = kv_quant.block_hadamard(r, no, gen).to(VS.device)
            VS[a:a + r] = (T.T @ VS[a:a + r].double()).to(VS.dtype)
            U[:, a:a + r] = (U[:, a:a + r].double() @ T).to(U.dtype)
            blocks.append((a, no, r))
            a += r
        mod.kv_quant_blocks = blocks
        mod.kv_outlier_ratio = outlier_ratio
        found = True
    if not found:
        raise RuntimeError("no fused K/V projections found; load a fused checkpoint first")


def set_kv_fake_quant(model, enabled: bool = True):
    """Fake-quantize every fused K/V latent in fp32 -- for measuring accuracy.

    Acts on the plain PyTorch path (the fused modules' forward), so it quantizes
    every forward, prefill included: a slightly pessimistic bound on the deployed
    packed cache, whose prefill is exact.
    """
    for mod in _fused_kv_modules(model):
        if mod.kv_quant_blocks is None:
            raise RuntimeError("run fold_kv_hadamard(model) first")
        mod.kv_fake_quant = enabled


class KVQuantMeta(nn.Module):
    """Per-layer constants of the 4-bit packed cache, shared by the cache layer
    (to pack) and the decode forward (to read)."""

    def __init__(self, k_ranks, k_offsets, rank_v: int, outlier_ratio: float):
        super().__init__()
        ranks = [int(r) for r in k_ranks.tolist()]
        widths = [align_up(r, RANK_TILE) for r in ranks]
        nouts = [kv_quant.n_outlier(r, outlier_ratio) for r in ranks]
        t = lambda v: torch.tensor(v, dtype=torch.int32)
        self.register_buffer("ranks", t(ranks), persistent=False)
        self.register_buffer("offsets", k_offsets.detach().to(torch.int32).cpu(), persistent=False)
        self.register_buffer("nouts", t(nouts), persistent=False)
        self.register_buffer("head_of_lane", t([h for h, w in enumerate(widths) for _ in range(w)]),
                             persistent=False)
        tile_head, tile_nout, tile_last = kv_quant.k_tile_meta(ranks, nouts, "cpu")
        self.register_buffer("tile_head", tile_head, persistent=False)
        self.register_buffer("tile_nout", tile_nout, persistent=False)
        self.register_buffer("tile_last", tile_last, persistent=False)
        self.rank_v = int(rank_v)
        self.nout_v = kv_quant.n_outlier(rank_v, outlier_ratio)

    @property
    def tiles(self):
        return self.tile_head, self.tile_nout, self.tile_last


def _kv_quant_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Attention over the 4-bit packed cache.

    Prefill is the bf16 path's own -- exact, through dense_weight -- and the
    PackedKVLayer quantizes the codes it writes. Decode reads only the packed
    cache: fold_q + abx_z for the logits, pv_fold for P @ V.
    """
    if hidden_states.shape[1] > 1 or past_key_values is None:
        return _patched_attn_forward(self, hidden_states, position_embeddings, attention_mask,
                                     position_ids, past_key_values, cache_position, **kwargs)
    B = hidden_states.shape[0]
    q = self.q_proj(hidden_states).view(B, 1, -1, self.head_dim).transpose(1, 2)
    q = _rope(q, *position_embeddings)
    k_inter = torch.matmul(self.k_proj.VS.weight, hidden_states.transpose(1, 2))    # [B, W, 1]
    v_inter = torch.matmul(hidden_states, self.v_proj.VS.weight.T)                  # [B, 1, rank_v]
    past_key_values.update(k_inter, v_inter, self.layer_idx)
    lay = past_key_values.layers[self.layer_idx]
    if not isinstance(lay, PackedKVLayer):
        raise TypeError(f"layer {self.layer_idx} decodes from the packed cache but got a "
                        f"{type(lay).__name__}; pass no cache and let the model build one")
    m = self.kvq
    ab = kv_quant.fold_q(q, self.k_proj.U, m.head_of_lane, self.scaling, self.rope_attn_scaling)
    logits = kv_quant.abx_z(ab, lay.keys, lay.key_scales, m.tiles, self.rope_inv_freq)
    if attention_mask is not None:
        logits = logits + attention_mask[:, :, :, : logits.shape[-1]]
    p = F.softmax(logits, dim=-1).to(q.dtype)
    pv = kv_quant.pv_fold(p.squeeze(2), lay.values, lay.value_scales, m.nout_v, m.rank_v)
    out = torch.bmm(pv.to(q.dtype).transpose(0, 1), self.v_proj.U_by_query_head).transpose(0, 1)
    return self.o_proj(out.reshape(B, 1, -1)), None
