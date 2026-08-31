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
"""

import math
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache

import LlamaLoRaAttention_headwise as attn_module
from LlamaLoRaAttention_headwise import LlamaCustomAttention
from LlamaLoRaAttention_headwise import apply_rotary_pos_emb_custom as _rope
from transformers.models.llama.modeling_llama import rotate_half
from soft_thres_layer import soft_thres_layer


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
    """Learnable diagonal matrix gated by a soft threshold (training time)."""

    def __init__(self, feature_size: int, thres: float):
        super().__init__()
        self.diag = nn.Parameter(torch.ones(feature_size), requires_grad=True)
        self.soft_thres_layer = soft_thres_layer(50, 0.0, float(thres))

    def forward(self, x):
        return x @ self.soft_thres_layer(torch.diag(self.diag))

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

    def forward(self, x):
        return self.U(self.VS(x))


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

    def forward(self, x):
        return self.U(self.VS(x))


@torch.no_grad()
def _fuse_joint(dec: DecomposeLinear) -> FusedDecomposeLinear:
    """DecomposeLinear -> FusedDecomposeLinear, dead directions removed."""
    device = dec.U.weight.device
    dtype = dec.U.weight.dtype
    diag = dec.Sigma.diag.detach().float()
    s_eff = dec.Sigma.soft_thres_layer(diag)
    keep = (s_eff > 0).nonzero(as_tuple=False).view(-1)
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
        s_eff = sb.soft_thres_layer(sb.diag.detach().float())
        keep = (s_eff > 0).nonzero(as_tuple=False).view(-1)
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
            diag = sig.diag.detach().float()
            k = min(min_rank, diag.numel())
            kth = float(torch.topk(diag, k).values[-1])
            margin = max(abs(kth) * 1e-2, 1e-4)
            sig.soft_thres_layer.alpha.data.clamp_(max=kth - margin)


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
    """Return effective rank of a decomposed module after soft-thresholding."""
    if hasattr(module, "Sigma_blocks"):
        return sum(
            int(torch.sum(sb.diag > sb.soft_thres_layer.alpha).item())
            for sb in module.Sigma_blocks
        )
    return int(torch.sum(module.Sigma.diag > module.Sigma.soft_thres_layer.alpha).item())


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


def collect_KV_parameter_size(model, equal: bool = False) -> int:
    return sum(
        _param_size(m, equal)
        for name, m in model.named_modules()
        if isinstance(m, (DecomposeLinear, DecomposeLinear_headwise))
        and ("k_proj" in name or "v_proj" in name)
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

    Returns (VS_linear, U_tensor, dense_weight, ranks).

    Heads keep different numbers of singular directions. Storage is padded up
    to the longest head's rank so strides stay uniform (padded rows/cols are
    zero, so this is exact), and `ranks` carries each head's real rank so the
    decode kernel can stop early instead of grinding through the zero padding.

    dense_weight is U @ VS reconstructed back to the original [out, in] shape,
    computed once here so prefill can do a single full-size matmul (matching
    the uncompressed baseline's cost) instead of reconstructing per token.

    VS_linear   : nn.Linear  [H * max_rank, in_features]
    U_tensor    : Tensor     [H, head_dim, max_rank]
    dense_weight: Tensor     [H * head_dim, in_features]
    ranks       : Tensor     [H]  int32, real rank kept per head
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
        return _pad_and_pack_kproj(VS_list, U_list, decomp.in_features, H, Hd,
                                   device, dtype)

    U_w = decomp.U.weight.detach().float()
    V_w = decomp.V.weight.detach().float()

    VS_list, U_list = [], []
    col_off = 0
    for h in range(H):
        r_h = decomp._r_list[h]
        sig = decomp.Sigma_blocks[h]
        diag = sig.diag.detach().float()
        alpha = float(sig.soft_thres_layer.alpha)
        keep = (diag > alpha).nonzero(as_tuple=False).view(-1)
        if keep.numel() == 0:
            keep = torch.zeros(1, dtype=torch.long, device=device)
        s_vals = sig.soft_thres_layer(diag[keep])
        cols = col_off + keep
        VS_list.append(V_w[cols] * s_vals[:, None])
        U_list.append(U_w[h * Hd:(h + 1) * Hd][:, cols])
        col_off += r_h

    return _pad_and_pack_kproj(VS_list, U_list, decomp.in_features, H, Hd,
                               device, dtype)


def _pad_and_pack_kproj(VS_list, U_list, in_features, H, Hd, device, dtype):
    """Pad per-head factors to the layer's max rank and pack them for the kernel.

    Shared by both export paths (fused and legacy) so the padding, the rank
    vector and the dense reconstruction can only ever be built one way.
    """
    # Real rank per head, captured before padding — this is what lets the decode
    # kernel skip the zero padding instead of computing through it.
    ranks = torch.tensor([v.shape[0] for v in VS_list], dtype=torch.int32, device=device)

    max_r = max(v.shape[0] for v in VS_list)
    VS_list = [F.pad(v, (0, 0, 0, max_r - v.shape[0])) for v in VS_list]
    U_list = [F.pad(u, (0, max_r - u.shape[1])) for u in U_list]
    VS_cat = torch.cat(VS_list, dim=0).to(dtype)
    U_ten = torch.stack(U_list, dim=0).to(dtype)

    VS_lin = nn.Linear(in_features, H * max_r, bias=False).to(device=device, dtype=dtype)
    VS_lin.weight = nn.Parameter(VS_cat.to(device))

    # dense_weight[h] = U_ten[h] @ VS_cat[h] reconstructs the original [head_dim, in] block
    dense_weight = torch.einsum(
        "hdr,hri->hdi", U_ten.float(), VS_cat.view(H, max_r, -1).float()
    ).reshape(H * Hd, -1).to(dtype).to(device)

    return VS_lin, U_ten.to(device), dense_weight, ranks


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
        diag = sig.diag.detach().float()
        alpha = float(sig.soft_thres_layer.alpha)
        keep = (diag > alpha).nonzero(as_tuple=False).view(-1)
        if keep.numel() == 0:
            keep = torch.zeros(1, dtype=torch.long, device=device)
        s_vals = sig.soft_thres_layer(diag[keep])

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
    """Wraps exported k_proj so LlamaCustomAttention can access .VS, .U, .dense_weight, .ranks."""

    def __init__(self, VS_linear: nn.Linear, U_tensor: torch.Tensor,
                 dense_weight: torch.Tensor, ranks: torch.Tensor):
        super().__init__()
        self.VS = VS_linear
        self.register_buffer("U", U_tensor)
        self.register_buffer("dense_weight", dense_weight)
        self.register_buffer("ranks", ranks)


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

    k_inter = (
        torch.matmul(hidden_states, k_vs.weight.T)
        .view(*input_shape, self.decomp_goup_num_kv, -1)
        .transpose(1, 2)
    )
    v_inter = torch.matmul(hidden_states, v_vs.weight.T)

    if past_key_values is not None:
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
            attention_mask[:, :, :, : key_states.shape[-2]]
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
                k_u.transpose(-2, -1).contiguous(),
                key_states.contiguous(),
                ranks=self.k_proj.ranks,
                inv_freq=self.rope_inv_freq,
                attn_scaling=self.rope_attn_scaling,
                dtype=torch.float16,
            ) / math.sqrt(self.head_dim)
        else:
            key_full = torch.matmul(key_states, k_u.transpose(-2, -1).to(key_states.dtype))
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
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
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
        attn_output = prob_v.unsqueeze(-2) @ self.v_proj.U_by_query_head  # [B, Hq, 1, Hd]

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

    k_inter = (
        torch.matmul(hidden_states, k_vs.weight.T)
        .view(*input_shape, H, -1).transpose(1, 2)
    )
    v_inter = torch.matmul(hidden_states, v_vs.weight.T)

    if past_key_values is not None:
        k_inter, v_inter = past_key_values.update(k_inter, v_inter, self.layer_idx)

    cos, sin = position_embeddings
    Q = _rope(Q, cos, sin)

    K_full = torch.matmul(k_inter, k_u.transpose(-2, -1).contiguous())
    K_full = _rope(K_full, cos, sin)
    V_full = (
        torch.matmul(v_inter, v_u.weight.T)
        .view(*v_inter.shape[:-1], H, Hd)
        .transpose(-3, -2)
        .contiguous()
    )

    mask = (
        attention_mask[:, :, :, : k_inter.shape[2]]
        if attention_mask is not None else None
    )
    out = F.scaled_dot_product_attention(Q, K_full, V_full, attn_mask=mask, scale=self.scaling)
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
):
    """Export decomposed K/V projections and install LlamaCustomAttention with
    the unified prefill+decode forward in all non-skip layers."""
    num_heads = config.num_attention_heads

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

    for i, block in enumerate(model.model.layers):
        if i in set(skip_layers):
            continue

        orig = block.self_attn
        device = next(orig.parameters()).device

        new_attn = LlamaCustomAttention(
            config, layer_idx=i, decomp_goup_num=num_heads
        ).to(device=device, dtype=dtype)

        new_attn.q_proj.weight.data.copy_(orig.q_proj.weight.data.to(dtype))
        new_attn.o_proj.weight.data.copy_(orig.o_proj.weight.data.to(dtype))

        k_VS, k_U, k_dense, k_ranks = export_kproj_for_triton(orig.k_proj, dtype=dtype)
        new_attn.k_proj = KProjInferenceWrapper(k_VS, k_U, k_dense, k_ranks)

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

        new_attn.forward = types.MethodType(_patched_attn_forward, new_attn)
        block.self_attn = new_attn

        if i == 2:
            # Ranks are per KV head, not per query head -- dividing by
            # num_attention_heads under-reports by the GQA group size.
            rk = k_VS.weight.shape[0] // config.num_key_value_heads
            print(f"  Layer {i}: k rank/kv-head max={rk} "
                  f"(real per head: {k_ranks.tolist()}), v rank={v_VS.weight.shape[0]}")

    return model


def set_model_mode(model, mode: str, skip_layers: tuple = (0, 1, 31)):
    """Switch all non-skip attention layers between triton / no_triton / bf16_sdpa."""
    attn_module.triton_kernel = (mode == "triton")
    attn_module.reorder = True
    for i, block in enumerate(model.model.layers):
        if i in set(skip_layers):
            continue
        attn = block.self_attn
        if mode in ("triton", "no_triton"):
            attn.forward = types.MethodType(_patched_attn_forward, attn)
        elif mode == "bf16_sdpa":
            attn.forward = types.MethodType(_bf16_sdpa_fwd, attn)
