"""Fused low-rank KV + RoPE Triton kernel.

Implements the fused ABX operation: out = A @ (B @ X^T + RoPE(B @ X^T))
where:
    A: query states       (batch, num_heads, 1, head_dim)
    B: U^T of K SVD      (num_heads, rank_per_group, head_dim)
    X: compressed K cache (batch, num_groups, seq_len, rank_per_group)

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
def get_freq_multi_tokens(starting_idx, theta: tl.constexpr, NB_TOKENS: tl.constexpr):
    DIM: tl.constexpr = 128
    DIM_2: tl.constexpr = 64
    freqs = tl.arange(0, DIM_2) * 2
    freqs = freqs.to(tl.float32) / DIM
    freqs = tl.extra.cuda.libdevice.fast_powf(theta, freqs)
    freqs = (tl.arange(0, NB_TOKENS) + starting_idx)[:, None] / freqs[None, :]
    return tl.extra.cuda.libdevice.fast_cosf(freqs), tl.extra.cuda.libdevice.fast_sinf(freqs)


def get_configs():
    return [triton.Config({'BLOCK_SIZE_L': 64, 'BLOCK_SIZE_R': 16}, num_warps=4, num_stages=1)]


@triton.autotune(
    configs=get_configs(),
    key=["seq_len"],
)
@triton.jit
def _abx_fwd(
    a_ptr, b_ptr, x_ptr, out_ptr,
    stride_ab, stride_az, stride_aa, stride_ad,
    stride_bz, stride_br, stride_bd,
    stride_xb, stride_xhg, stride_xl, stride_xr,
    stride_ob, stride_oz, stride_oa, stride_ol,
    R, D, seq_len,
    dtype_tl: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    THETA: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_l = tl.program_id(axis=2)

    # GROUP_SIZE = num_query_heads // num_kv_groups (query heads sharing one KV head).
    # Query head pid_h reads the compressed K cache of KV group pid_h // GROUP_SIZE.
    HEAD_GROUPS_ID = pid_h // GROUP_SIZE
    offs_ds = tl.arange(0, BLOCK_SIZE_D)
    offs_rs = tl.arange(0, BLOCK_SIZE_R)
    offs_ls = (pid_l * BLOCK_SIZE_L) + tl.arange(0, BLOCK_SIZE_L)

    A_ptrs = a_ptr + pid_b * stride_ab + pid_h * stride_az + (0 * stride_aa + offs_ds[None, :] * stride_ad)
    # B (k_u U^T) is stored per KV head, so it is indexed by the KV group, not the
    # query head. For MHA (GROUP_SIZE=1) HEAD_GROUPS_ID == pid_h, so this is a no-op.
    B_ptrs = b_ptr + HEAD_GROUPS_ID * stride_bz + (offs_rs[:, None] * stride_br + offs_ds[None, :] * stride_bd)
    X_ptrs = x_ptr + pid_b * stride_xb + HEAD_GROUPS_ID * stride_xhg + (offs_ls[:, None] * stride_xl + offs_rs[None, :] * stride_xr)
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
    cos, sin = get_freq_multi_tokens(starting_idx=start_block, theta=THETA, NB_TOKENS=BLOCK_SIZE_L)
    cos = cos.to(dtype_tl)
    sin = sin.to(dtype_tl)

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


def abx(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor, dtype=torch.float16) -> torch.Tensor:
    """Fused A @ (B @ X^T + RoPE) for decode-step attention.

    Args:
        a: query states, shape (batch, num_heads, 1, head_dim)
        b: K projection U^T, shape (num_heads, rank_per_group, head_dim)
        x: compressed K cache, shape (batch, num_groups, seq_len, rank_per_group)
        dtype: compute dtype (float16 or bfloat16)

    Returns:
        attention logits, shape (batch, num_heads, 1, seq_len)
    """
    assert a.dim() == 4
    assert b.dim() == 3
    assert x.dim() == 4

    # a is per query head (num_heads); b is per KV head (num_groups). Keep them
    # distinct — do NOT let b's head count clobber num_heads (breaks GQA grids).
    batch_size, num_heads, _, head_dim = a.shape
    _num_kv_heads, rank_per_head_groups, head_dim = b.shape
    batch_size, num_groups, seq_len, rank_per_head_groups = x.shape

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
        a, b, x, out,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1), b.stride(2),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        R=rank_per_head_groups,
        D=head_dim,
        seq_len=seq_len,
        dtype_tl=dtype_tl,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        NUM_GROUPS=NUM_GROUPS,
        GROUP_SIZE=GROUP_SIZE,
        THETA=10000.,
    )
    return out


def torch_abx(a, b, x, dtype=torch.float16):
    """Reference PyTorch implementation of the fused ABX+RoPE operation."""
    x_expand = x.unsqueeze(2)
    b_reshape = b.reshape(-1, b.shape[0] // x.shape[1], b.shape[-2], b.shape[-1]).unsqueeze(0)
    xb = x_expand @ b_reshape
    xb = xb.reshape(x.shape[0], b.shape[0], -1, b.shape[-1])

    config = LlamaConfig()
    rotary_emb = LlamaRotaryEmbedding(config=config)
    position_ids = torch.arange(0, x.shape[-2]).unsqueeze(0)
    cos, sin = rotary_emb(xb, position_ids)
    xb_rope = apply_rotary_pos_emb_custom(x=xb, cos=cos, sin=sin)
    axb = a @ xb_rope.transpose(-1, -2).to(dtype)
    return axb


def run_benchmark(args):
    configs = [
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=args.target_seq_lens,
            line_arg="provider",
            line_vals=["WX", "torch", "ours"],
            line_names=["WX", "Torch", "Ours"],
            styles=[("gray", "--"), ("green", "--"), ("blue", "-")],
            ylabel="us",
            plot_name=f"low-rank-rank-{args.total_rank}-group-{args.num_groups}",
            args={
                "dtype": torch.float16,
                "num_heads": args.num_heads,
                "head_dim": args.head_dim,
                "total_rank": args.total_rank,
                "num_groups": args.num_groups,
            },
        )
    ]

    @triton.testing.perf_report(configs)
    def bench_low_rank(num_heads, head_dim, total_rank, seq_len, num_groups, provider, dtype=torch.float16, device="cuda"):
        rank_per_groups = total_rank // num_groups
        warmup = 25
        rep = 100
        A = torch.randn(num_heads, 1, head_dim, dtype=dtype, device=device)
        B = torch.randn(num_heads, rank_per_groups, head_dim, dtype=dtype, device=device)
        X = torch.randn(num_groups, seq_len, rank_per_groups, dtype=dtype, device=device)
        org_A = torch.randn(num_heads, 1, head_dim, dtype=dtype, device=device)
        org_X = torch.randn(num_heads, seq_len, head_dim, dtype=dtype, device=device)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "torch":
            fn = lambda: torch_abx(A, B, X)
        elif provider == "ours":
            fn = lambda: abx(A, B, X)
        elif provider == "WX":
            fn = lambda: torch.matmul(org_A, org_X.transpose(-1, -2))

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)
        return ms * 1000, min_ms * 1000, max_ms * 1000

    import os
    os.makedirs('results', exist_ok=True)
    bench_low_rank.run(print_data=True, show_plots=True, save_path='results/')


def run_test(args):
    num_heads = args.num_heads
    head_dim = args.head_dim
    total_rank = args.total_rank
    seq_len = 1024
    batch_size = 4
    num_groups = args.num_groups
    rank_per_groups = total_rank // num_groups
    dtype = torch.float16
    device = "cuda"

    A = torch.randn(batch_size, num_heads, 1, head_dim, dtype=dtype, device=device)
    B = torch.randn(num_heads, rank_per_groups, head_dim, dtype=dtype, device=device)
    X = torch.randn(batch_size, num_groups, seq_len, rank_per_groups, dtype=dtype, device=device)

    axb = torch_abx(A, B, X, dtype)
    ours = abx(A, B, X, dtype=dtype)

    print("Mean diff: ", torch.mean(torch.abs(axb - ours)))


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark or test the fused ABX+RoPE Triton kernel.")
    parser.add_argument("--total_rank", type=int, default=int(4096 * 0.2), help="Total compressed ranks")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads (32 for LLaMA-7B)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension (128 for LLaMA-7B)")
    parser.add_argument("--group_size", type=int, default=4, help="Number of heads per KV group (for GQA models). For MHA choose 1")
    parser.add_argument("--target_seq_lens", nargs="+", type=int, default=[4096, 16384, 65536, 262144])
    parser.add_argument("--check", action="store_true", help="Run correctness check instead of benchmark")
    return parser.parse_args()


def main(args):
    args.num_groups = args.num_heads // args.group_size
    args.group_rank = args.total_rank // args.num_groups
    print("Benchmarking fused low-rank KV Cache kernel (ABX+RoPE)...")
    print(f"  Total rank:    {args.total_rank}")
    print(f"  Heads:         {args.num_heads}")
    print(f"  Head dim:      {args.head_dim}")
    print(f"  Group size:    {args.group_size}")
    print(f"  Groups:        {args.num_groups}")
    print(f"  Rank/group:    {args.group_rank}")
    if args.check:
        run_test(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    set_random_seed()
    args = parse_args()
    main(args)
