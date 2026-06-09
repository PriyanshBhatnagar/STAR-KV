"""Quantized BX Triton kernel for low-rank V cache decode.

Implements: out = B @ dequant(X), where X is split into an int8 upper block
and a packed-int4 lower block with per-token scales.

Run `python bx_quant.py --check` to verify correctness.
Run `python bx_quant.py` to benchmark.
"""

import torch
import triton
import triton.language as tl
import argparse

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig
from transformers.models.llama.modeling_llama import rotate_half

from quant_utils import quantize_r_split_int8_int4_packed, unpack_uint8_to_int4_signed


def set_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_configs():
    return [triton.Config({'BLOCK_SIZE_L': 64, 'BLOCK_SIZE_R': 64, 'SPLIT_L': 6}, num_warps=8, num_stages=2)]


@triton.autotune(
    configs=get_configs(),
    key=["seq_len"],
)
@triton.jit
def _bx_fwd(
    b_ptr, x1_ptr, x2_ptr, out_ptr,
    scale_x1, scale_x2,
    stride_bb, stride_bh, stride_bl,
    stride_x1b, stride_x1l, stride_x1r,
    stride_x2b, stride_x2l, stride_x2r,
    stride_ob, stride_oh, stride_or,
    stride_s1b, stride_s1l,
    stride_s2b, stride_s2l,
    R, H, L,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    SPLIT_L: tl.constexpr,
    UPPER_RANK,
):
    pid_b = tl.program_id(axis=0)
    pid_l = tl.program_id(axis=1)
    pid_r = tl.program_id(axis=2)

    offs_hs = tl.arange(0, BLOCK_SIZE_H)
    offs_rs = (pid_r * BLOCK_SIZE_R) + tl.arange(0, BLOCK_SIZE_R)
    offs_ls = (pid_l * BLOCK_SIZE_L) + tl.arange(0, BLOCK_SIZE_L)
    offs_ls = tl.max_contiguous(tl.multiple_of(offs_ls, BLOCK_SIZE_L), BLOCK_SIZE_L)

    B_ptrs = b_ptr + pid_b * stride_bb + (offs_hs[:, None] * stride_bh + offs_ls[None, :] * stride_bl)
    X1_ptrs = x1_ptr + pid_b * stride_x1b + (offs_ls[:, None] * stride_x1l)
    SCALE_X1_ptrs = scale_x1 + pid_b * stride_s1b + (offs_ls[:, None] * stride_s1l)
    X2_ptrs = x2_ptr + pid_b * stride_x2b + (offs_ls[:, None] * stride_x2l)
    SCALE_X2_ptrs = scale_x2 + pid_b * stride_s2b + (offs_ls[:, None] * stride_s2l)
    O_ptrs = out_ptr + pid_b * stride_ob + (offs_hs[:, None] * stride_oh + offs_rs[None, :] * stride_or)

    bx = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_R), dtype=tl.float32)

    if (pid_r < (UPPER_RANK + BLOCK_SIZE_R - 1) // BLOCK_SIZE_R):
        for l in range(0, tl.cdiv(L, BLOCK_SIZE_L * SPLIT_L)):
            r_idx = offs_rs
            byte_idx = r_idx >> 1
            is_hi = (r_idx & 1).to(tl.int1)

            x = tl.load(X1_ptrs + byte_idx[None, :] * stride_x1r,
                        mask=(offs_ls[:, None] < L - l * BLOCK_SIZE_L * SPLIT_L) & (offs_rs[None, :] < R), other=0.0)
            b = tl.load(B_ptrs, mask=offs_ls[None, :] < L - l * BLOCK_SIZE_L * SPLIT_L, other=0.0)
            scale_x = tl.load(SCALE_X1_ptrs, mask=offs_ls[:, None] < L - l * BLOCK_SIZE_L * SPLIT_L, other=0.0)

            lo = x & 0x0F
            hi = (x >> 4) & 0x0F
            nib = tl.where(is_hi[None, :], hi, lo).to(tl.int8)
            x = (nib - 8).to(tl.float16) * scale_x.to(tl.float16)

            bx = tl.dot(b, x, bx)
            B_ptrs += BLOCK_SIZE_L * SPLIT_L * stride_bl
            X1_ptrs += BLOCK_SIZE_L * SPLIT_L * stride_x1l
            SCALE_X1_ptrs += BLOCK_SIZE_L * SPLIT_L * stride_s1l
    else:
        for l in range(0, tl.cdiv(L, BLOCK_SIZE_L * SPLIT_L)):
            r_idx = offs_rs - UPPER_RANK
            byte_idx = r_idx >> 1
            is_hi = (r_idx & 1).to(tl.int1)

            x = tl.load(X2_ptrs + byte_idx[None, :] * stride_x2r,
                        mask=(offs_ls[:, None] < L - l * BLOCK_SIZE_L * SPLIT_L) & (offs_rs[None, :] < R), other=0.0)
            b = tl.load(B_ptrs, mask=offs_ls[None, :] < L - l * BLOCK_SIZE_L * SPLIT_L, other=0.0)
            scale_x = tl.load(SCALE_X2_ptrs, mask=offs_ls[:, None] < L - l * BLOCK_SIZE_L * SPLIT_L, other=0.0)

            lo = x & 0x0F
            hi = (x >> 4) & 0x0F
            nib = tl.where(is_hi[None, :], hi, lo).to(tl.int8)
            x = (nib - 8).to(tl.float16) * scale_x.to(tl.float16)

            bx = tl.dot(b, x, bx)
            B_ptrs += BLOCK_SIZE_L * SPLIT_L * stride_bl
            X2_ptrs += BLOCK_SIZE_L * SPLIT_L * stride_x2l
            SCALE_X2_ptrs += BLOCK_SIZE_L * SPLIT_L * stride_s2l

    bx = bx.to(tl.float16)
    tl.atomic_add(O_ptrs, bx, mask=(offs_hs[:, None] < H) & (offs_rs[None, :] < R), sem="relaxed")


def bx(
    b: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale_v_upper,
    scale_v_lower,
    original_rank,
    upper_rank=256,
) -> torch.Tensor:
    """Fused dequantize + B @ X for decode-step V computation.

    Args:
        b:             attention probabilities, shape (batch, num_heads, seq_len)
        x1:            int4-packed upper V cache, shape (batch, seq_len, upper_rank/2)
        x2:            int4-packed lower V cache, shape (batch, seq_len, lower_rank/2)
        scale_v_upper: per-token scales for upper block, shape (batch, seq_len, 1)
        scale_v_lower: per-token scales for lower block, shape (batch, seq_len, 1)
        original_rank: total rank (upper + lower)
        upper_rank:    number of int8 ranks (default 256)

    Returns:
        weighted V sum, shape (batch, num_heads, original_rank)
    """
    assert b.dim() == 3
    assert x1.dim() == 3
    assert x2.dim() == 3

    batch_size, num_heads, seq_len = b.shape
    batch_size, seq_len, rank = x1.shape

    out = torch.empty((batch_size, num_heads, original_rank), dtype=b.dtype, device=b.device)
    BLOCK_SIZE_H = 32

    grid = lambda META: (batch_size, META["SPLIT_L"], triton.cdiv(original_rank, META["BLOCK_SIZE_R"]))
    _bx_fwd[grid](
        b, x1, x2, out,
        scale_v_upper, scale_v_lower,
        b.stride(0), b.stride(1), b.stride(2),
        x1.stride(0), x1.stride(1), x1.stride(2),
        x2.stride(0), x2.stride(1), x2.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        scale_v_upper.stride(0), scale_v_upper.stride(1),
        scale_v_lower.stride(0), scale_v_lower.stride(1),
        R=int(original_rank),
        H=num_heads,
        L=seq_len,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        UPPER_RANK=int(upper_rank),
    )
    return out


def torch_bx(b, x_upper_q, x_lower_q, scale_x_upper, scale_x_lower, r4_k):
    """Reference PyTorch implementation."""
    x_upper = (unpack_uint8_to_int4_signed(x_upper_q, 256) * scale_x_upper).to(b.dtype)
    x_lower = (unpack_uint8_to_int4_signed(x_lower_q, r4_k) * scale_x_lower).to(b.dtype)
    x = torch.cat([x_upper, x_lower], dim=-1)
    return b @ x


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
        batch_size = 2
        warmup = 25
        rep = 100
        B = torch.randn(batch_size, num_heads, seq_len, dtype=dtype, device=device)
        X = torch.randn(batch_size, seq_len, total_rank, dtype=dtype, device=device)
        org_B = torch.randn(num_heads, 1, head_dim, dtype=dtype, device=device)
        org_X = torch.randn(num_heads, seq_len, head_dim, dtype=dtype, device=device)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "torch":
            fn = lambda: torch_bx(B, X)
        elif provider == "ours":
            fn = lambda: bx(B, X)
        elif provider == "WX":
            fn = lambda: torch.matmul(B, X)

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles, warmup=warmup, rep=rep)
        return ms * 1000, min_ms * 1000, max_ms * 1000

    import os
    os.makedirs('results', exist_ok=True)
    bench_low_rank.run(print_data=True, show_plots=True, save_path='results/')


def run_test(args):
    num_heads = args.num_heads
    total_rank = args.total_rank
    seq_len = 4 * 1024
    batch_size = 16
    dtype = torch.float16
    device = "cuda"
    upper_rank = 256

    B = torch.randn(batch_size, num_heads, seq_len, dtype=dtype, device=device)
    X = torch.randn(batch_size, seq_len, total_rank, dtype=dtype, device=device)

    x_upper_q, scale_x_upper, x_lower_q, scale_x_lower, r4_v = quantize_r_split_int8_int4_packed(X, split_r=upper_rank)
    rank_total = r4_v + upper_rank

    xb = torch_bx(B, x_upper_q, x_lower_q, scale_x_upper, scale_x_lower, r4_v)
    ours = bx(B, x_upper_q, x_lower_q, scale_x_upper, scale_x_lower, rank_total, upper_rank=upper_rank)

    print("Mean values: ", torch.mean(torch.abs(xb)))
    print("Mean diff:   ", torch.mean(torch.abs(xb - ours)))


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark or test the quantized BX Triton kernel.")
    parser.add_argument("--total_rank", type=int, default=int(0.3 * 4096), help="Total compressed ranks")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads (32 for LLaMA-7B)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension (128 for LLaMA-7B)")
    parser.add_argument("--group_size", type=int, default=4, help="Number of heads per KV group (for GQA models). For MHA choose 1")
    parser.add_argument("--target_seq_lens", nargs="+", type=int, default=[4096, 16384, 65536])
    parser.add_argument("--check", action="store_true", help="Run correctness check instead of benchmark")
    return parser.parse_args()


def main(args):
    args.num_groups = args.num_heads // args.group_size
    args.group_rank = args.total_rank // args.num_groups
    print("Benchmarking quantized BX kernel...")
    print(f"  Total rank:  {args.total_rank}")
    print(f"  Heads:       {args.num_heads}")
    print(f"  Head dim:    {args.head_dim}")
    print(f"  Group size:  {args.group_size}")
    print(f"  Groups:      {args.num_groups}")
    print(f"  Rank/group:  {args.group_rank}")
    if args.check:
        run_test(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    set_random_seed()
    args = parse_args()
    main(args)
