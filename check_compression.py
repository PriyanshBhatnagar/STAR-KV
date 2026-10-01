"""Report the KV-cache compression actually encoded in a checkpoint.

Reads the checkpoint directly (no model build, no GPU) and reports the per-token
KV cache size that replace_attn_with_triton would export, using the same
same keep rule the exporters use (`diag > alpha` in phase 1, the pinned
keep mask from phase 2 on).

Worth running before any long eval: a checkpoint can be numerically perfect --
identical perplexity -- while carrying no compression at all, if the rank
bookkeeping in Sigma has been flattened. Perplexity will not reveal that; this
will.

  python check_compression.py --weights fused_weights.pt
"""

import argparse

import torch

# Mirrors model.RANK_TILE. Duplicated rather than imported so this script stays
# dependency-free (no transformers, no triton, no GPU); keep the two in step.
RANK_TILE = 16


def _align_up(r: int, multiple: int = RANK_TILE) -> int:
    return int(-(-int(r) // multiple) * multiple)


def _is_compressed(sd, li: int) -> bool:
    """Does layer `li` carry a decomposed k_proj, in either checkpoint format?"""
    pre = f"model.layers.{li}.self_attn.k_proj."
    return (f"{pre}head_ranks" in sd            # fused
            or f"{pre}Sigma_blocks.0.diag" in sd)  # pre-fusion


def _infer_geometry(sd):
    """Read num_layers, skip_layers, kv_heads and head_dim off the checkpoint.

    Everything these flags used to encode is already in the tensors, and a
    mismatch is not a small error -- pointing the wrong --skip-layers at a
    checkpoint either crashes ("no k_proj ranks found") or silently counts an
    uncompressed layer as compressed and misreports the compression.
    """
    n_layers = 1 + max(
        (int(k.split("model.layers.")[1].split(".")[0])
         for k in sd if k.startswith("model.layers.")),
        default=-1,
    )
    if n_layers == 0:
        raise RuntimeError("no 'model.layers.*' keys in checkpoint")

    compressed = [i for i in range(n_layers) if _is_compressed(sd, i)]
    if not compressed:
        raise RuntimeError("no decomposed k_proj in any layer; is this a base model?")
    skip = tuple(i for i in range(n_layers) if i not in set(compressed))

    pre = f"model.layers.{compressed[0]}.self_attn.k_proj."
    # U is [out_features, rank]; out_features == num_kv_heads * head_dim, which
    # is exactly the uncompressed per-layer K cache.
    out_features = int(sd[f"{pre}U.weight"].shape[0])
    if f"{pre}head_ranks" in sd:
        kv_heads = int(sd[f"{pre}head_ranks"].numel())
    else:
        kv_heads = sum(1 for k in sd if k.startswith(f"{pre}Sigma_blocks.")
                       and k.endswith(".diag"))
    return n_layers, skip, kv_heads, out_features // kv_heads, out_features


def _rank_of(sd, prefix: str) -> int:
    """Surviving rank of one Sigma, in whichever mode the checkpoint is in.

    Phase 1 checkpoints gate on `diag > alpha`. From phase 2 the threshold is
    retired and the rank is pinned by a binary keep mask, so the mask is
    authoritative -- reading `diag > alpha` on a frozen checkpoint reports the
    PRE-freeze rank and understates the cache.
    """
    frozen = sd.get(f"{prefix}.rank_frozen")
    if frozen is not None and bool(frozen):
        return int(sd[f"{prefix}.keep_mask"].sum())
    diag = sd[f"{prefix}.diag"].float()
    alpha = float(sd[f"{prefix}.soft_thres_layer.alpha"])
    return int((diag > alpha).sum())


def main():
    p = argparse.ArgumentParser(description="Report KV compression encoded in a checkpoint.")
    p.add_argument("--weights", required=True)
    p.add_argument("--num-layers", type=int, default=None,
                   help="Override; inferred from the checkpoint by default.")
    p.add_argument("--num-kv-heads", type=int, default=None,
                   help="Override; inferred from the checkpoint by default.")
    p.add_argument("--head-dim", type=int, default=None,
                   help="Override; inferred from the checkpoint by default.")
    p.add_argument("--skip-layers", type=int, nargs="+", default=None,
                   help="Override; inferred as the layers with no decomposed "
                        "k_proj. Defaults used to be [0 1 2 31], which silently "
                        "mismatched any model that is not 32 layers.")
    p.add_argument("--per-layer", action="store_true", help="Print each layer's ranks")
    p.add_argument("--compressed-only", action="store_true",
                   help="Report over the compressed layers only, the basis "
                        "train.py's --comp-ratio uses. Default counts all "
                        "layers, with the skipped ones at full rank -- the two "
                        "differ by a lot and must not be quoted interchangeably.")
    args = p.parse_args()

    sd = torch.load(args.weights, map_location="cpu", mmap=True, weights_only=False)
    n_inf, skip_inf, kvh_inf, hd_inf, out_inf = _infer_geometry(sd)
    num_layers = args.num_layers if args.num_layers is not None else n_inf
    skip = set(args.skip_layers) if args.skip_layers is not None else set(skip_inf)
    kv_heads = args.num_kv_heads if args.num_kv_heads is not None else kvh_inf
    head_dim = args.head_dim if args.head_dim is not None else hd_inf
    full_per_layer = kv_heads * head_dim
    print(f"detected: {n_inf} layers, {kvh_inf} kv heads x {hd_inf} head_dim, "
          f"skip={sorted(skip_inf)}")
    for name, got, inf in (("--num-layers", args.num_layers, n_inf),
                           ("--num-kv-heads", args.num_kv_heads, kvh_inf),
                           ("--head-dim", args.head_dim, hd_inf)):
        if got is not None and got != inf:
            print(f"  WARNING: {name}={got} overrides the detected {inf}")
    if args.skip_layers is not None and set(args.skip_layers) != set(skip_inf):
        print(f"  WARNING: --skip-layers {sorted(set(args.skip_layers))} overrides "
              f"the detected {sorted(skip_inf)}")

    K = V = 0
    rows = []
    for li in range(num_layers):
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
                keeps.append(_rank_of(sd, f"{pre}k_proj.Sigma_blocks.{h}"))
                h += 1
            if not keeps:
                raise RuntimeError(f"layer {li}: no k_proj ranks found in checkpoint")

            v_rank = _rank_of(sd, f"{pre}v_proj.Sigma")

        # The export stores each head at its own rank rounded up to RANK_TILE and
        # packs them side by side.
        K += sum(_align_up(r) for r in keeps)
        V += v_rank
        rows.append((li, keeps, v_rank, False))

    if args.per_layer:
        print(f"{'layer':>6} {'k rank/head (per KV head)':<46} {'k/token':>8} "
              f"{'v rank':>7}")
        for li, keeps, v_rank, skipped in rows:
            if skipped:
                print(f"{li:>6} {'(uncompressed)':<46} {full_per_layer:>8} "
                      f"{full_per_layer:>7}")
            else:
                print(f"{li:>6} {str(keeps):<46} "
                      f"{sum(_align_up(r) for r in keeps):>8} {v_rank:>7}")
        print()

    n_counted = (num_layers - len(skip)) if args.compressed_only else num_layers
    if args.compressed_only:
        K -= len(skip) * full_per_layer
        V -= len(skip) * full_per_layer
    full = n_counted * full_per_layer
    total, full_total = K + V, 2 * full
    _over = (f"{n_counted} compressed layers only"
             if args.compressed_only
             else f"all {num_layers} layers, {len(skip)} of them uncompressed")
    print(f"checkpoint : {args.weights}")
    print(f"  measured over: {_over}")
    print(f"  K per token : {K:7d}  (uncompressed {full})")
    print(f"  V per token : {V:7d}  (uncompressed {full})")
    print(f"  K+V         : {total:7d}  (uncompressed {full_total})")
    print(f"  COMPRESSION : {100 * (1 - total / full_total):.1f}%   "
          f"cache is {total / full_total:.3f}x of full size")
    if total >= full_total:
        print("  WARNING: no compression encoded -- Sigma's rank bookkeeping looks flattened.")


if __name__ == "__main__":
    main()
