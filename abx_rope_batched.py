"""Fused low-rank KV + RoPE Triton kernel, with per-head dynamic rank.

Implements the fused ABX operation: out = A @ (B @ X^T + RoPE(B @ X^T))
where:
    A: query states       (batch, num_heads, 1, head_dim)
    B: U^T of K SVD       (total_width, head_dim)
    X: compressed K cache (batch, total_width, seq_len)

Heads keep different numbers of singular directions. Each KV head is stored at
its own width (its rank rounded up to BLOCK_SIZE_R), side by side along one
flat rank axis, and found through a per-head offset. The rank is passed as a
per-KV-head vector, so the GEMM trip count varies per head too:

    R = tl.load(r_ptr + kv_head)          # per-head trip count

A head that kept 30 directions stops after 2 tiles, and the tail of its last
tile is masked, so it is never read; --check asserts that garbage there leaves
the result bit-identical. Differing trip counts across program ids is
block-level divergence, not warp divergence, so there is no intra-warp penalty.

Run `python abx_rope_batched.py --check` to verify correctness.
Run `python abx_rope_batched.py` to benchmark across sequence lengths.
"""

import torch
import triton
import triton.language as tl
import argparse

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig
from transformers.models.llama.modeling_llama import rotate_half


def apply_rotary_pos_emb_custom(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if x.shape[-2] != cos.shape[-2]:
        cos = cos[:, :, -1, :].unsqueeze(2)
        sin = sin[:, :, -1, :].unsqueeze(2)
    embed = (x * cos) + (rotate_half(x) * sin)
    return embed


def set_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@triton.jit
def get_freq_multi_tokens(inv_freq_ptr, starting_idx, NB_TOKENS: tl.constexpr):
    """cos/sin for NB_TOKENS positions from a precomputed inverse-frequency table.

    inv_freq is supplied by the caller rather than derived from a closed-form
    1/theta**(2i/d) here. That closed form only covers default RoPE; llama3,
    yarn, linear and dynamic scaling all reshape inv_freq per dimension (llama3
    on Llama-3.2-3B moves low-frequency dims by up to 32x), and reproducing
    that in-kernel would silently drift from the RoPE applied to the queries.
    """
    DIM_2: tl.constexpr = 64
    inv_freq = tl.load(inv_freq_ptr + tl.arange(0, DIM_2))
    pos = (tl.arange(0, NB_TOKENS) + starting_idx).to(tl.float32)
    freqs = pos[:, None] * inv_freq[None, :]
    return tl.extra.cuda.libdevice.fast_cosf(freqs), tl.extra.cuda.libdevice.fast_sinf(freqs)


def get_configs():
    return [triton.Config({'BLOCK_SIZE_L': 64, 'BLOCK_SIZE_R': 16}, num_warps=4, num_stages=1)]


@triton.autotune(
    configs=get_configs(),
    key=["seq_len"],
)
@triton.jit
def _abx_fwd(
    a_ptr, b_ptr, x_ptr, out_ptr, r_ptr, inv_freq_ptr, off_ptr,
    stride_ab, stride_az, stride_aa, stride_ad,
    stride_br, stride_bd,
    stride_xb, stride_xl, stride_xr,
    stride_ob, stride_oz, stride_oa, stride_ol,
    D, seq_len,
    dtype_tl: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ATTN_SCALING: tl.constexpr,
):
    # int64: pid_b * stride_xb overflows int32 once the K cache passes 2^31
    # elements (e.g. bs=16, seq=64k, 32 heads, rank 80 = 2.6e9), and the wrapped
    # pointer reads whatever else is resident instead of faulting.
    pid_b = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1).to(tl.int64)
    pid_l = tl.program_id(axis=2)

    # GROUP_SIZE = num_query_heads // num_kv_groups (query heads sharing one KV head).
    # Query head pid_h reads the compressed K cache of KV group pid_h // GROUP_SIZE.
    HEAD_GROUPS_ID = pid_h // GROUP_SIZE

    # Per-head rank. Ranks are stored per KV head (B and X are both indexed by
    # HEAD_GROUPS_ID), so every query head in a group shares the trip count.
    R = tl.load(r_ptr + HEAD_GROUPS_ID)

    offs_ds = tl.arange(0, BLOCK_SIZE_D)
    offs_rs = tl.arange(0, BLOCK_SIZE_R)
    offs_ls = (pid_l * BLOCK_SIZE_L) + tl.arange(0, BLOCK_SIZE_L)

    A_ptrs = a_ptr + pid_b * stride_ab + pid_h * stride_az + (0 * stride_aa + offs_ds[None, :] * stride_ad)
    # Heads sit side by side in one flat rank axis, each at its own width, so the
    # head is addressed by a column OFFSET rather than by a uniform per-head
    # stride. X is [B, W, L] and B (k_u U^T) is [W, D], with head h occupying
    # rows [off_h, off_h + width_h).
    col_off = tl.load(off_ptr + HEAD_GROUPS_ID).to(tl.int64)
    B_ptrs = b_ptr + col_off * stride_br + (offs_rs[:, None] * stride_br + offs_ds[None, :] * stride_bd)
    X_ptrs = x_ptr + pid_b * stride_xb + col_off * stride_xr + (offs_ls[:, None] * stride_xl + offs_rs[None, :] * stride_xr)
    O_ptrs = out_ptr + pid_b * stride_ob + pid_h * stride_oz + (0 * stride_oa + offs_ls[None, :] * stride_ol)

    xb_0 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)
    xb_1 = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_D), dtype=tl.float32)

    # Sequence-dim mask: seq_len (KV cache length, e.g. ctx+1 at decode) is rarely
    # a multiple of BLOCK_SIZE_L, so the final L-block runs past the end of X/out.
    # Mask both the X load and the O store on offs_ls to avoid OOB accesses.
    ls_mask = offs_ls < seq_len

    for r in range(0, tl.cdiv(R, BLOCK_SIZE_R)):
        x = tl.load(
            X_ptrs,
            mask=(offs_rs[None, :] < R - r * BLOCK_SIZE_R) & ls_mask[:, None],
            other=0.0,
        )
        b_0 = tl.load(B_ptrs, mask=offs_rs[:, None] < R - r * BLOCK_SIZE_R, other=0.0)
        b_1 = tl.load(B_ptrs + BLOCK_SIZE_D * stride_bd, mask=offs_rs[:, None] < R - r * BLOCK_SIZE_R, other=0.0)
        xb_0 = tl.dot(x, b_0, xb_0)
        xb_1 = tl.dot(x, b_1, xb_1)
        B_ptrs += BLOCK_SIZE_R * stride_br
        X_ptrs += BLOCK_SIZE_R * stride_xr

    xb_0 = xb_0.to(dtype_tl)
    xb_1 = xb_1.to(dtype_tl)

    start_block = pid_l * BLOCK_SIZE_L
    cos, sin = get_freq_multi_tokens(inv_freq_ptr, starting_idx=start_block,
                                     NB_TOKENS=BLOCK_SIZE_L)
    # Rope types such as yarn scale cos/sin by an attention factor; llama3 and
    # default RoPE use 1.0.
    cos = (cos * ATTN_SCALING).to(dtype_tl)
    sin = (sin * ATTN_SCALING).to(dtype_tl)

    xb_rope_0 = xb_0 * cos - xb_1 * sin
    xb_rope_1 = xb_1 * cos + xb_0 * sin
    xb_0 = xb_rope_0.to(dtype_tl)
    xb_1 = xb_rope_1.to(dtype_tl)

    a_0 = tl.load(A_ptrs)
    a_1 = tl.load(A_ptrs + BLOCK_SIZE_D * stride_ad)
    abx_0 = tl.sum(a_0 * xb_0, 1)
    abx_1 = tl.sum(a_1 * xb_1, 1)
    abx = abx_0 + abx_1
    tl.store(O_ptrs, abx[None, :], mask=ls_mask[None, :])


def abx(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor,
        ranks: torch.Tensor, offsets: torch.Tensor,
        inv_freq: torch.Tensor = None, attn_scaling: float = 1.0,
        dtype=torch.float16) -> torch.Tensor:
    """Fused A @ (B @ X^T + RoPE) for decode-step attention.

    x is (batch, total_width, seq_len) -- rank-major, tokens LAST -- and b is
    (total_width, head_dim). KV head h occupies the half-open range
    [offsets[h], offsets[h] + width_h) of the rank axis, where width_h is its
    rank rounded up to BLOCK_SIZE_R, so the cache holds only what each head
    kept, and tokens being last keeps a head's lanes contiguous across them.

    Args:
        a: query states, shape (batch, num_heads, 1, head_dim)
        b: K projection U^T, shape (total_width, head_dim)
        x: compressed K cache, shape (batch, total_width, seq_len)
        ranks: int32 tensor (num_groups,), real rank kept per KV head. The
               kernel stops at ranks[h], so the rest of a head's last tile is
               never read.
        offsets: int32 tensor (num_groups,), per-head column offset into the
               flat rank axis.
        inv_freq: RoPE inverse frequencies, shape (head_dim // 2,). Pass the
               model's own `model.model.rotary_emb.inv_freq` -- it already
               encodes the rope type (llama3/yarn/linear/...). None falls back
               to default RoPE with theta=10000, which is WRONG for any model
               using a different theta or a scaled rope type.
        attn_scaling: cos/sin multiplier (rotary_emb.attention_scaling); 1.0
               for default and llama3 RoPE.
        dtype: compute dtype (float16 or bfloat16)

    Returns:
        attention logits, shape (batch, num_heads, 1, seq_len)
    """
    assert a.dim() == 4
    assert b.dim() == 2, f"expected b [total_width, head_dim], got {tuple(b.shape)}"
    assert x.dim() == 3, f"expected x [batch, total_width, seq_len], got {tuple(x.shape)}"

    # a is per query head (num_heads); ranks/offsets are per KV head (num_groups).
    # Keep them distinct (GQA grids depend on it).
    batch_size, num_heads, _, head_dim = a.shape
    total_width, head_dim = b.shape
    batch_size, x_width, seq_len = x.shape
    assert x_width == total_width, (
        f"x has {x_width} rank columns but b has {total_width}")
    num_groups = offsets.numel()
    assert ranks.numel() == num_groups, (
        f"ranks has {ranks.numel()} entries but offsets has {num_groups}")
    ranks = ranks.to(device=x.device, dtype=torch.int32).contiguous()
    offsets = offsets.to(device=x.device, dtype=torch.int32).contiguous()

    if inv_freq is None:
        # Default RoPE, theta=10000. Only correct for models that actually use it.
        i = torch.arange(0, head_dim // 2, dtype=torch.float32, device=x.device)
        inv_freq = 1.0 / (10000.0 ** (i * 2 / head_dim))
    inv_freq = inv_freq.to(device=x.device, dtype=torch.float32).contiguous()
    assert inv_freq.numel() == head_dim // 2, (
        f"inv_freq has {inv_freq.numel()} entries, expected head_dim//2 = {head_dim // 2}"
    )
    # The kernel splits head_dim into two BLOCK_SIZE_D=64 halves for rotate_half
    # and get_freq_multi_tokens loads a fixed 64 inverse frequencies, so head_dim
    # is not a free parameter. Without this check a 64-wide head reads past the
    # end of inv_freq and returns garbage instead of failing.
    assert head_dim == 128, (
        f"the fused kernel is specialised for head_dim=128, got {head_dim}"
    )

    out = torch.empty((batch_size, num_heads, 1, seq_len), dtype=x.dtype, device=x.device)
    BLOCK_SIZE_D = 64
    NUM_GROUPS = num_groups
    # query heads per KV group (GQA). For MHA this is 1.
    GROUP_SIZE = num_heads // num_groups

    if dtype == torch.float16:
        dtype_tl = tl.float16
    elif dtype == torch.bfloat16:
        dtype_tl = tl.bfloat16
    elif dtype == torch.float32:
        dtype_tl = tl.float32

    grid = lambda META: (batch_size, num_heads, triton.cdiv(seq_len, META["BLOCK_SIZE_L"]))
    _abx_fwd[grid](
        a, b, x, out, ranks, inv_freq, offsets,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1),
        # x is [B, W, L]: the token stride is the LAST axis, the rank stride the
        # middle one.
        x.stride(0), x.stride(2), x.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        D=head_dim,
        seq_len=seq_len,
        dtype_tl=dtype_tl,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        NUM_GROUPS=NUM_GROUPS,
        GROUP_SIZE=GROUP_SIZE,
        ATTN_SCALING=float(attn_scaling),
    )
    return out


def torch_abx(a, b, x, ranks, offsets, dtype=torch.float16):
    """Reference PyTorch implementation of the fused ABX+RoPE operation.

    Shapes match the kernel. Each KV head's keys are rebuilt from its own rank
    only, and query head h reads KV head h // (num_q_heads // num_kv_heads).
    """
    num_q_heads = a.shape[1]
    num_kv_heads = offsets.numel()
    group_size = num_q_heads // num_kv_heads

    # Reconstruct full K per KV head: (batch, num_kv_heads, seq_len, head_dim)
    xb = torch.stack([
        x[:, o:o + r, :].float().transpose(1, 2) @ b[o:o + r].float()
        for o, r in zip(offsets.tolist(), ranks.tolist())
    ], dim=1)

    config = LlamaConfig()
    rotary_emb = LlamaRotaryEmbedding(config=config).to(xb.device)
    position_ids = torch.arange(0, x.shape[-1], device=xb.device).unsqueeze(0)
    cos, sin = rotary_emb(xb, position_ids)
    xb_rope = apply_rotary_pos_emb_custom(x=xb, cos=cos, sin=sin)

    # Broadcast each KV head across the query heads that share it (GQA).
    xb_rope = xb_rope.repeat_interleave(group_size, dim=1)
    return (a.float() @ xb_rope.transpose(-1, -2)).to(dtype)


def random_ragged_cache(batch_size, num_groups, max_rank, seq_len, head_dim,
                        dtype=torch.float16, device="cuda"):
    """Random per-KV-head ranks in [8, max_rank] and a ragged B/X built for them."""
    ranks = torch.randint(8, max_rank + 1, (num_groups,), dtype=torch.int32)
    widths = (ranks + 15) // 16 * 16
    offsets = (torch.cumsum(widths, 0) - widths).to(torch.int32)
    total_width = int(widths.sum())
    B = torch.randn(total_width, head_dim, dtype=dtype, device=device)
    X = torch.randn(batch_size, total_width, seq_len, dtype=dtype, device=device)
    return B, X, ranks.to(device), offsets.to(device), widths.tolist()


def run_benchmark(args):
    print(f"{'seq_len':>8} {'dense QK^T (us)':>16} {'abx (us)':>10} {'speedup':>8}")
    for seq_len in args.target_seq_lens:
        A = torch.randn(args.batch_size, args.num_heads, 1, args.head_dim,
                        dtype=torch.float16, device="cuda")
        B, X, ranks, offsets, _ = random_ragged_cache(
            args.batch_size, args.num_groups, args.max_rank, seq_len, args.head_dim)
        K = torch.randn(args.batch_size, args.num_heads, seq_len, args.head_dim,
                        dtype=torch.float16, device="cuda")
        t_ours = triton.testing.do_bench(lambda: abx(A, B, X, ranks, offsets))
        t_dense = triton.testing.do_bench(lambda: A @ K.transpose(-1, -2))
        print(f"{seq_len:>8} {t_dense * 1000:>16.1f} {t_ours * 1000:>10.1f} "
              f"{t_dense / t_ours:>7.2f}x")


def run_test(args):
    batch_size, seq_len, dtype = 4, 1024, torch.float16
    A = torch.randn(batch_size, args.num_heads, 1, args.head_dim, dtype=dtype, device="cuda")
    B, X, ranks, offsets, widths = random_ragged_cache(
        batch_size, args.num_groups, args.max_rank, seq_len, args.head_dim, dtype)
    print(f"  ranks    : {ranks.tolist()}")

    ref = torch_abx(A, B, X, ranks, offsets, dtype)
    ours = abx(A, B, X, ranks, offsets, dtype=dtype)
    denom = ref.float().abs().mean().clamp_min(1e-6)
    print(f"  mean abs diff vs torch reference: {(ref - ours).float().abs().mean().item():.4f} "
          f"(mean |ref| = {denom.item():.4f})")

    # The kernel must stop at each head's rank: zeroing the rest of every
    # head's last tile cannot change a single bit of the result.
    Bz, Xz = B.clone(), X.clone()
    for o, r, w in zip(offsets.tolist(), ranks.tolist(), widths):
        Bz[o + r:o + w] = 0
        Xz[:, o + r:o + w] = 0
    max_diff = (abx(A, Bz, Xz, ranks, offsets, dtype=dtype).float() - ours.float()).abs().max().item()
    print(f"  tile tail ignored: max diff {max_diff:.3e}  "
          + ("PASS" if max_diff == 0.0 else "FAIL (expected bit-exact)"))


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark or test the fused ABX+RoPE Triton kernel.")
    parser.add_argument("--max_rank", type=int, default=80,
                        help="Per-KV-head ranks are drawn uniformly from [8, max_rank]")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads (32 for LLaMA-7B)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension (128 for LLaMA-7B)")
    parser.add_argument("--group_size", type=int, default=4, help="Number of heads per KV group (for GQA models). For MHA choose 1")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the benchmark")
    parser.add_argument("--target_seq_lens", nargs="+", type=int, default=[4096, 16384, 65536, 262144])
    parser.add_argument("--check", action="store_true", help="Run correctness check instead of benchmark")
    return parser.parse_args()


def main(args):
    args.num_groups = args.num_heads // args.group_size
    print("Fused low-rank KV Cache kernel (ABX+RoPE)")
    print(f"  Heads:         {args.num_heads}")
    print(f"  Head dim:      {args.head_dim}")
    print(f"  Group size:    {args.group_size}")
    print(f"  Groups:        {args.num_groups}")
    print(f"  Max rank:      {args.max_rank}")
    if args.check:
        run_test(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    set_random_seed()
    args = parse_args()
    main(args)
