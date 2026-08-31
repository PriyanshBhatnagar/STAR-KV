"""Report the KV-cache compression actually encoded in a checkpoint.

Reads the checkpoint directly (no model build, no GPU) and reports the per-token
KV cache size that replace_attn_with_triton would export, using the same
`keep = diag > alpha` rule the exporters use.

Worth running before any long eval: a checkpoint can be numerically perfect --
identical perplexity -- while carrying no compression at all, if the rank
bookkeeping in Sigma has been flattened. Perplexity will not reveal that; this
will.

  python check_compression.py --weights fused_weights.pt
"""

import argparse

import torch


def main():
    p = argparse.ArgumentParser(description="Report KV compression encoded in a checkpoint.")
    p.add_argument("--weights", required=True)
    p.add_argument("--num-layers", type=int, default=28)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--skip-layers", type=int, nargs="+", default=[0, 1, 31])
    p.add_argument("--per-layer", action="store_true", help="Print each layer's ranks")
    args = p.parse_args()

    sd = torch.load(args.weights, map_location="cpu", mmap=True, weights_only=False)
    skip = set(args.skip_layers)
    full_per_layer = args.num_kv_heads * args.head_dim

    K = V = 0
    rows = []
    for li in range(args.num_layers):
        if li in skip:
            K += full_per_layer
            V += full_per_layer
            rows.append((li, None, None, True))
            continue

        pre = f"model.layers.{li}.self_attn."

        if f"{pre}k_proj.head_ranks" in sd:
            # Fused format: ranks are the tensor shapes, nothing to threshold.
            keeps = [int(r) for r in sd[f"{pre}k_proj.head_ranks"].tolist()]
            v_rank = int(sd[f"{pre}v_proj.VS.weight"].shape[0])
        else:
            keeps, h = [], 0
            while f"{pre}k_proj.Sigma_blocks.{h}.diag" in sd:
                diag = sd[f"{pre}k_proj.Sigma_blocks.{h}.diag"].float()
                alpha = float(sd[f"{pre}k_proj.Sigma_blocks.{h}.soft_thres_layer.alpha"])
                keeps.append(int((diag > alpha).sum()))
                h += 1
            if not keeps:
                raise RuntimeError(f"layer {li}: no k_proj ranks found in checkpoint")

            d = sd[f"{pre}v_proj.Sigma.diag"].float()
            a = float(sd[f"{pre}v_proj.Sigma.soft_thres_layer.alpha"])
            v_rank = int((d > a).sum())

        # Export pads every head up to the layer's max kept rank.
        K += args.num_kv_heads * max(keeps)
        V += v_rank
        rows.append((li, keeps, v_rank, False))

    if args.per_layer:
        print(f"{'layer':>6} {'k rank/head (per KV head)':<46} {'k max':>6} {'v rank':>7}")
        for li, keeps, v_rank, skipped in rows:
            if skipped:
                print(f"{li:>6} {'(uncompressed)':<46} {args.head_dim:>6} {full_per_layer:>7}")
            else:
                print(f"{li:>6} {str(keeps):<46} {max(keeps):>6} {v_rank:>7}")
        print()

    full = args.num_layers * full_per_layer
    total, full_total = K + V, 2 * full
    print(f"checkpoint : {args.weights}")
    print(f"  K per token : {K:7d}  (uncompressed {full})")
    print(f"  V per token : {V:7d}  (uncompressed {full})")
    print(f"  K+V         : {total:7d}  (uncompressed {full_total})")
    print(f"  COMPRESSION : {100 * (1 - total / full_total):.1f}%   "
          f"cache is {total / full_total:.3f}x of full size")
    if total >= full_total:
        print("  WARNING: no compression encoded -- Sigma's rank bookkeeping looks flattened.")


if __name__ == "__main__":
    main()
