"""Measure decode latency for low-rank KV cache compression vs BF16 baseline.

Two benchmarks are available:
  e2e         End-to-end model decode across context lengths (1 token generated per step).
              Compares BF16 SDPA baseline vs low-rank + Triton kernel.
  layerwise   Per-layer synthetic attention benchmark using the actual exported ranks.
              Reports speedup for each transformer layer.

Example
-------
  # Run both benchmarks (baseline + low-rank)
  python latency.py \\
    --model lmsys/longchat-7b-v1.5-32k \\
    --weights trained_weights.pt \\
    --mode both \\
    --ctx-lens 256 512 1024 2048 4096 8192 16384 32000 64000 128000 \\
    --output-dir results/

  # Baseline only (no weights needed)
  python latency.py \\
    --model lmsys/longchat-7b-v1.5-32k \\
    --mode e2e \\
    --baseline \\
    --output-dir results/

  # Layer-wise only
  python latency.py \\
    --model lmsys/longchat-7b-v1.5-32k \\
    --weights trained_weights.pt \\
    --mode layerwise \\
    --lw-seq 32768 --lw-batch 16
"""

import argparse
import copy
import gc
import json
import math
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import login
from transformers import AutoTokenizer, LlamaForCausalLM

from model import (
    replace_linear_layer,
    replace_attn_with_triton,
    set_model_mode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _median_ms(ts):
    ts.sort()
    return ts[len(ts) // 2]


def _bench_decode(model, tokenizer, ctx_lens, n_runs, device):
    """Time one decode step (single new token) for each KV cache length."""
    results = {}
    for ctx_len in ctx_lens:
        torch.cuda.empty_cache()
        gc.collect()
        prompt = "Hello world. " * (ctx_len // 3)
        inp = tokenizer(
            prompt, return_tensors="pt", max_length=ctx_len, truncation=True
        ).to(device)
        tok = inp.input_ids[:, -1:]
        try:
            with torch.no_grad():
                pkv = model(**inp, use_cache=True).past_key_values
            ts = []
            with torch.no_grad():
                for _ in range(n_runs):
                    pkv_copy = copy.deepcopy(pkv)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model(tok, past_key_values=pkv_copy, use_cache=True)
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
            results[ctx_len] = _median_ms(ts)
            print(f"  ctx={ctx_len:7d}  {results[ctx_len]:.1f} ms")
        except torch.cuda.OutOfMemoryError:
            results[ctx_len] = None
            print(f"  ctx={ctx_len:7d}  OOM")
            torch.cuda.empty_cache()
            gc.collect()
    return results


# ---------------------------------------------------------------------------
# End-to-end benchmark
# ---------------------------------------------------------------------------

def run_e2e(args, model, tokenizer, device, output_dir):
    ctx_lens = args.ctx_lens
    n_runs = args.runs

    if args.baseline:
        print("\n=== Baseline BF16 SDPA ===")
        baseline_results = _bench_decode(model, tokenizer, ctx_lens, n_runs, device)
        out_path = os.path.join(output_dir, "baseline_latency.json")
        with open(out_path, "w") as f:
            json.dump(baseline_results, f, indent=2)
        print(f"Saved → {out_path}")
        return

    # Low-rank modes
    print("\n=== Low-rank + Triton ===")
    set_model_mode(model, "triton", skip_layers=tuple(args.skip_layers))
    triton_results = _bench_decode(model, tokenizer, ctx_lens, n_runs, device)

    print("\n=== Low-rank no Triton ===")
    set_model_mode(model, "no_triton", skip_layers=tuple(args.skip_layers))
    no_triton_results = _bench_decode(model, tokenizer, ctx_lens, n_runs, device)

    out_triton = os.path.join(output_dir, "triton_latency.json")
    out_no_triton = os.path.join(output_dir, "no_triton_latency.json")
    with open(out_triton, "w") as f:
        json.dump(triton_results, f, indent=2)
    with open(out_no_triton, "w") as f:
        json.dump(no_triton_results, f, indent=2)
    print(f"Saved → {out_triton}, {out_no_triton}")

    # Load baseline if it exists and print comparison
    baseline_path = os.path.join(output_dir, "baseline_latency.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = {int(k): v for k, v in json.load(f).items()}
        print(f"\n{'ctx':>8}  {'sdpa (ms)':>10}  {'no-tri (ms)':>12}  {'triton (ms)':>12}  {'speedup':>8}")
        print("-" * 60)
        for c in ctx_lens:
            s = baseline.get(c)
            nt = no_triton_results.get(c)
            t = triton_results.get(c)
            s_str = f"{s:10.1f}" if s else f"{'OOM':>10}"
            nt_str = f"{nt:12.1f}" if nt else f"{'OOM':>12}"
            t_str = f"{t:12.1f}" if t else f"{'OOM':>12}"
            sp = f"{s/t:8.2f}x" if (s and t) else f"{'—':>8}"
            print(f"{c:8d}  {s_str}  {nt_str}  {t_str}  {sp}")

        if not args.no_plot:
            _plot_e2e(ctx_lens, baseline, no_triton_results, triton_results, output_dir)


def _plot_e2e(ctx_lens, baseline, no_triton, triton, output_dir):
    valid = [(c, baseline.get(c), no_triton.get(c), triton.get(c))
             for c in ctx_lens if baseline.get(c) and triton.get(c)]
    if not valid:
        return
    cs, bs, nts, ts = zip(*valid)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(cs, bs, "b-o", label="BF16 SDPA")
    ax.plot(cs, nts, "g--s", label="Low-rank no Triton")
    ax.plot(cs, ts, "r-^", label="Low-rank + Triton")
    ax.set_xlabel("KV cache length (tokens)")
    ax.set_ylabel("Decode latency (ms)")
    ax.set_title("End-to-end decode latency (bs=1)")
    ax.legend()

    ax = axes[1]
    sp_nt = [b / n for b, n in zip(bs, nts)]
    sp_t = [b / t for b, t in zip(bs, ts)]
    ax.plot(cs, sp_nt, "g--s", label="Low-rank no-Triton / SDPA")
    ax.plot(cs, sp_t, "r-^", label="Low-rank + Triton / SDPA")
    ax.axhline(1.0, color="gray", linestyle="--", label="1× (no speedup)")
    ax.set_xlabel("KV cache length (tokens)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("End-to-end speedup vs BF16 SDPA")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "e2e_speedup.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved → {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Layer-wise benchmark
# ---------------------------------------------------------------------------

def run_layerwise(args, model, config, output_dir):
    import triton.testing as _tt
    from abx_rope_batched import abx as _abx

    seq = args.lw_seq
    bs = args.lw_batch
    dtype = torch.bfloat16
    device = get_device()
    scale = 1.0 / math.sqrt(config.head_dim)
    H = config.num_attention_heads
    Hd = config.head_dim
    skip = set(args.skip_layers)

    layer_ranks = {}
    for i, block in enumerate(model.model.layers):
        if i in skip:
            continue
        rk = block.self_attn.k_proj.VS.weight.shape[0] // H
        rv = block.self_attn.v_proj.VS.weight.shape[0]
        layer_ranks[i] = (rk, rv)

    print(f"\nPer-layer ranks (seq_len={seq}, batch={bs}):")
    for i, (rk, rv) in sorted(layer_ranks.items()):
        print(f"  Layer {i:2d}: rank_k/head={rk:3d}, rank_v={rv:4d}")

    def _ms(fn):
        ms, _, _ = _tt.do_bench(fn, quantiles=[0.5, 0.2, 0.8], warmup=10, rep=30)
        return ms * 1000  # ms → μs

    rows = []
    for i, (rk, rv) in sorted(layer_ranks.items()):
        torch.cuda.empty_cache()
        gc.collect()
        try:
            Q = torch.randn(bs, H, 1, Hd, dtype=dtype, device=device)
            K_c = torch.randn(bs, H, seq, rk, dtype=dtype, device=device)
            V_c = torch.randn(bs, seq, rv, dtype=dtype, device=device)
            k_u = torch.randn(H, Hd, rk, dtype=dtype, device=device)
            v_u = torch.randn(H, Hd, rv, dtype=dtype, device=device)
            k_u_T = k_u.transpose(-2, -1).contiguous()
            v_u_T = v_u.transpose(-1, -2).contiguous()

            # BF16 SDPA baseline
            try:
                K_full = torch.randn(bs, H, seq, Hd, dtype=dtype, device=device)
                V_full = torch.randn(bs, H, seq, Hd, dtype=dtype, device=device)
                t_sdpa = _ms(lambda: F.scaled_dot_product_attention(Q, K_full, V_full))
                del K_full, V_full
            except torch.cuda.OutOfMemoryError:
                t_sdpa = float("nan")
                torch.cuda.empty_cache()
                gc.collect()

            # Low-rank, no Triton
            def _no_tri():
                K_exp = torch.matmul(K_c, k_u_T)
                attn = F.softmax(
                    torch.matmul(Q, K_exp.transpose(-2, -1)) * scale,
                    dim=-1, dtype=torch.float32
                ).to(dtype)
                pv = attn.squeeze(2) @ V_c
                return pv.unsqueeze(-2) @ v_u_T

            t_no_tri = _ms(_no_tri)

            # Low-rank + Triton
            def _tri():
                attn = F.softmax(
                    _abx(Q, k_u_T, K_c, dtype=dtype) * scale,
                    dim=-1, dtype=torch.float32
                ).to(dtype)
                pv = attn.squeeze(2) @ V_c
                return pv.unsqueeze(-2) @ v_u_T

            t_tri = _ms(_tri)

            del Q, K_c, V_c, k_u, v_u, k_u_T, v_u_T
            torch.cuda.empty_cache()

            sp_nt = t_sdpa / t_no_tri if not math.isnan(t_sdpa) else float("nan")
            sp_t = t_sdpa / t_tri if not math.isnan(t_sdpa) else float("nan")
            print(
                f"  Layer {i:2d}  rk={rk:3d} rv={rv:4d}  "
                f"sdpa={t_sdpa:7.1f}μs  no-tri={t_no_tri:7.1f}μs  tri={t_tri:7.1f}μs  "
                f"sp(no-tri/sdpa)={sp_nt:.2f}×  sp(tri/sdpa)={sp_t:.2f}×"
            )
            rows.append(dict(
                layer=i, rank_k=rk, rank_v=rv,
                sdpa=t_sdpa, no_triton=t_no_tri, triton=t_tri,
                sp_no_triton=sp_nt, sp_triton=sp_t,
            ))
        except torch.cuda.OutOfMemoryError:
            print(f"  Layer {i:2d}: OOM — skipping")
            torch.cuda.empty_cache()
            gc.collect()

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "layerwise_latency.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nLayer-wise results saved → {csv_path}")

    if not args.no_plot:
        _plot_layerwise(df, bs, seq, output_dir)


def _plot_layerwise(df, bs, seq, output_dir):
    valid = df.dropna(subset=["sp_no_triton", "sp_triton"])
    if valid.empty:
        return
    layers = valid["layer"].tolist()
    x = list(range(len(layers)))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    ax = axes[0]
    ax.bar([xi - w / 2 for xi in x], valid["sp_no_triton"], w,
           label="Low-rank no-Triton / SDPA", color="steelblue", alpha=0.85)
    ax.bar([xi + w / 2 for xi in x], valid["sp_triton"], w,
           label="Low-rank + Triton / SDPA", color="tomato", alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers], rotation=45, fontsize=8)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Speedup (×)")
    ax.set_title(f"Layer-wise attention speedup vs BF16 SDPA (bs={bs}, seq_len={seq})")
    ax.legend()

    ax = axes[1]
    ax.plot(layers, valid["sdpa"].tolist(), "b-o", label="BF16 SDPA (full K,V)")
    ax.plot(layers, valid["no_triton"].tolist(), "g--s", label="Low-rank no Triton")
    ax.plot(layers, valid["triton"].tolist(), "r-^", label="Low-rank + Triton")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Attention latency (μs)")
    ax.set_title("Layer-wise raw attention latency")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "layerwise_speedup.png")
    plt.savefig(path, dpi=150)
    print(f"Plot saved → {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark low-rank KV decode latency.")
    p.add_argument("--model", required=True,
                   help="HuggingFace model name or local path")
    p.add_argument("--weights", default=None,
                   help="Path to trained weights (not needed for --baseline)")
    p.add_argument("--mode", choices=["e2e", "layerwise", "both"], default="both",
                   help="Which benchmark to run")
    p.add_argument("--baseline", action="store_true",
                   help="Run BF16 SDPA baseline only (no weights needed)")
    p.add_argument("--ctx-lens", type=int, nargs="+",
                   default=[256, 512, 1024, 2048, 4096, 8192, 16384, 32000, 64000, 128000],
                   help="Context lengths for end-to-end benchmark")
    p.add_argument("--runs", type=int, default=30,
                   help="Number of timed decode steps per context length")
    p.add_argument("--lw-seq", type=int, default=32768,
                   help="Sequence length for layer-wise benchmark")
    p.add_argument("--lw-batch", type=int, default=16,
                   help="Batch size for layer-wise benchmark")
    p.add_argument("--skip-layers", type=int, nargs="+", default=[0, 1, 31])
    p.add_argument("--output-dir", default="results/",
                   help="Directory for JSON results and plots")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip matplotlib plot generation")
    p.add_argument("--cuda-devices", default="0",
                   help="CUDA_VISIBLE_DEVICES string")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Baseline: original uncompressed model ───────────────────────────────
    if args.baseline:
        print(f"Loading baseline model: {args.model}")
        model = LlamaForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            use_cache=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        if args.mode in ("e2e", "both"):
            run_e2e(args, model, tokenizer, device, args.output_dir)
        return

    # ── Low-rank model ───────────────────────────────────────────────────────
    if args.weights is None:
        raise ValueError("--weights is required unless --baseline is set")

    print(f"Loading model: {args.model}")
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        use_cache=False,
        use_safetensors=True,
    )
    config = model.config

    model = model.float()
    replace_linear_layer(model, config, skip_layers=tuple(args.skip_layers))
    print(f"Loading weights: {args.weights}")
    model.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=False),
        strict=False,
    )
    model = model.bfloat16()

    print("Exporting and replacing attention modules...")
    model = replace_attn_with_triton(
        model, config, skip_layers=tuple(args.skip_layers), dtype=torch.bfloat16
    )
    model.eval()
    model.config.use_cache = True

    if args.mode in ("e2e", "both"):
        run_e2e(args, model, tokenizer, device, args.output_dir)

    if args.mode in ("layerwise", "both"):
        run_layerwise(args, model, config, args.output_dir)


if __name__ == "__main__":
    main()
