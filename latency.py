"""Measure STAR-KV's decode speedup on the real model.

Two modes, both reporting the same thing -- how much faster a decode step is:

  layerwise (default)  per-layer attention, at any (seq, batch)
  e2e                  the whole model, end to end

Layer-wise holds ONE attention layer on the GPU at a time. That matters: the
full model cannot reach the configs worth reporting, because 32 layers of dense
cache at 32k x batch 16 is 275 GB, while one layer is 8 GB and one layer's
weights are 134 MB. Both models stay on the CPU and each layer is moved across
in turn, so a single command measures both arms at the paper's config.

Everything timed is the real thing: the real attention module, the real exported
factors, the real cache object, and the real torch.cat append that
transformers' DynamicCache does on every step -- which is two thirds of the
traffic a decode step moves, and which shrinks with the codes.

  python latency.py --weights fused_weights.pt --seq 32768 --batch 16
  python latency.py --weights fused_weights.pt --seq 4096 --batch 1 --mode e2e
  python latency.py --weights fused_weights.pt --seq 32768 --batch 16 --kv-quant

--kv-quant stores the compressed layers' cache packed to 4 bits and decodes
through kv_quant.py's kernels (the accuracy of that format is eval.py --kv-quant).
"""
import argparse
import gc
import os
import sys

import torch
from transformers import AutoConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DTYPE = torch.bfloat16


def load_baseline(model_id):
    """The uncompressed model, on the CPU. SDPA attention, i.e. FlashAttention."""
    m = LlamaForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPE, use_safetensors=True, attn_implementation="sdpa")
    m.eval()
    m.config.use_cache = True
    return m


def load_starkv(model_id, weights, kv_quant=False):
    """The compressed model, on the CPU, built the way deployment builds it.

    Loaded in fp32 because load_compressed_checkpoint installs the pruned factors
    at that precision (and the Hadamard for --kv-quant is folded there), then
    cast to bf16.
    """
    from model import fold_kv_hadamard, load_compressed_checkpoint, replace_attn_with_triton
    m = LlamaForCausalLM.from_pretrained(
        model_id, device_map=None, use_cache=False, use_safetensors=True)
    cfg = m.config
    m, fused, skip = load_compressed_checkpoint(m.float(), cfg, weights, skip_layers=())
    if kv_quant:
        fold_kv_hadamard(m)
    m = replace_attn_with_triton(m.bfloat16(), cfg, skip_layers=skip, dtype=DTYPE,
                                 kv_quant=kv_quant)
    m.eval()
    m.config.use_cache = True
    print(f"  {'fused U/VS (pruned)' if fused else 'legacy U/Sigma/V'} checkpoint, "
          f"uncompressed layers: {list(skip)}" + ("; 4-bit packed KV cache" if kv_quant else ""))
    return m, tuple(skip)


def _time(fn, reps, warm, before=None):
    """Best of `reps`. The minimum is the closest reading to an idle card, and
    these machines are usually shared."""
    for _ in range(warm):
        if before:
            before()
        fn()
    best = float("inf")
    for _ in range(reps):
        if before:
            before()
        torch.cuda.synchronize()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1000.0)
    return best


def _cache_tensors(lay):
    """(attribute, token axis) for everything a cache layer holds."""
    from model import PackedKVLayer
    if isinstance(lay, PackedKVLayer):
        return [(n, 1) for n in ("keys", "key_scales", "values", "value_scales")]
    # Rank-major K is [B, W, seq] with tokens LAST; everything else is at -2.
    return [("keys", -1 if lay.keys.dim() == 3 else 2), ("values", 1 if lay.values.dim() == 3 else 2)]


def time_layer(model, li, compressed, bs, seq, reps, warm, hidden_size):
    """One attention layer's decode step in microseconds, and its cache bytes per token.

    The cache is seeded by running one step against an empty cache and repeating
    the codes the module itself produced, so every shape, dtype and stride is
    what the model would really build. Only the values repeat, and a
    bandwidth-bound measurement does not care what they are.

    Between reps the cache is rewound by rebinding to the seed -- torch.cat never
    mutates its input -- so every rep is timed at exactly `seq` tokens.
    """
    from model import StarKVCache
    attn = model.model.layers[li].self_attn
    attn.layer_idx = 0
    attn = attn.to("cuda")
    rot = model.model.rotary_emb.to("cuda")
    # An uncompressed layer runs stock attention and needs a stock cache layer,
    # whichever model it belongs to; a --kv-quant layer needs the packed one.
    skip = () if compressed else (0,)
    packed = {0: attn.kvq} if compressed and hasattr(attn, "kvq") else None

    hidden = torch.randn(bs, 1, hidden_size, dtype=DTYPE, device="cuda")
    pos = torch.full((bs, 1), seq, dtype=torch.long, device="cuda")
    cos, sin = rot(hidden, pos)
    cpos = torch.tensor([seq], dtype=torch.long, device="cuda")

    def call(cache):
        with torch.no_grad():
            attn(hidden, position_embeddings=(cos, sin), attention_mask=None,
                 past_key_values=cache, cache_position=cpos)

    probe = StarKVCache(1, skip, packed)
    call(probe)
    seed = {}
    for name, ax in _cache_tensors(probe.layers[0]):
        t = getattr(probe.layers[0], name)
        reps_ = [1] * t.dim()
        reps_[ax] = seq
        seed[name] = t.repeat(*reps_).contiguous()
    del probe
    torch.cuda.empty_cache()

    cache = StarKVCache(1, skip, packed)
    lay = cache.layers[0]
    first = next(iter(seed.values()))
    lay.dtype, lay.device, lay.is_initialized = first.dtype, first.device, True
    per_token = sum(t.numel() * t.element_size() for t in seed.values()) // (bs * seq)

    def rewind():
        for name, t in seed.items():
            setattr(lay, name, t)

    us = _time(lambda: call(cache), reps, warm, before=rewind)
    seed.clear()                   # release the cache before empty_cache below
    cache = lay = None
    attn.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    return us, per_token


def run_layerwise(base, star, skip, cfg, args):
    """Compressed layers only.

    The layers the checkpoint left uncompressed run stock attention on a full
    cache, so they are 1.00x by construction. Averaging them in would report a
    number that says as much about how many layers were skipped as about the
    method, so they are left out and named instead.
    """
    skip = sorted(skip)
    layers = [i for i in range(cfg.num_hidden_layers) if i not in set(skip)]
    print(f"\nPer-layer decode attention -- seq={args.seq}, batch={args.batch}, bf16, "
          f"best of {args.reps}")
    print("Each figure is one whole self_attn module: q_proj, the K/V projection, "
          "the cache\nappend, attention, and o_proj.")
    if skip:
        print(f"Uncompressed layers {skip} are excluded; they run stock attention "
              f"on a full cache.")
    print()
    print(f"{'layer':>5} {'baseline us':>12} {'STAR-KV us':>11} {'speedup':>8} "
          f"{'KV B/token':>10}")
    print("-" * 49)
    tb = ts = 0.0
    for li in layers:
        b, kb = time_layer(base, li, False, args.batch, args.seq,
                           args.reps, args.warm, cfg.hidden_size)
        s_, ks = time_layer(star, li, True, args.batch, args.seq,
                            args.reps, args.warm, cfg.hidden_size)
        tb += b
        ts += s_
        print(f"{li:>5} {b:>12.1f} {s_:>11.1f} {b / s_:>7.2f}x {ks:>10}", flush=True)
    n = len(layers)
    print("-" * 49)
    print(f"average over {n} compressed layers: baseline {tb / n:.1f} us, "
          f"STAR-KV {ts / n:.1f} us, speedup {tb / ts:.2f}x")


def _prefill(model, bs, seq, vocab):
    """One forward. The compressed prefill path assumes an empty cache, so it
    cannot be chunked."""
    ids = torch.randint(0, vocab, (bs, seq), device="cuda")
    with torch.no_grad():
        try:
            out = model(ids, use_cache=True, logits_to_keep=1)
        except TypeError:
            out = model(ids, use_cache=True)
    past = out.past_key_values
    del out, ids
    torch.cuda.empty_cache()
    return past


def _e2e_decode_us(model, bs, seq, steps, vocab):
    """Mean microseconds per decode step through the whole model."""
    model.cuda()
    past = _prefill(model, bs, seq, vocab)
    tok = torch.randint(0, vocab, (bs, 1), device="cuda")

    def step():
        nonlocal past
        with torch.no_grad():
            out = model(tok, past_key_values=past, use_cache=True)
        past = out.past_key_values

    for _ in range(4):
        step()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(steps):
        step()
    b.record()
    torch.cuda.synchronize()
    us = a.elapsed_time(b) * 1000.0 / steps
    peak = torch.cuda.max_memory_allocated() / 1024**3
    past = tok = None
    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    return us, peak


def run_e2e(base, star, cfg, args):
    print(f"\nEnd-to-end decode -- seq={args.seq}, batch={args.batch}, "
          f"{args.steps} steps, bf16")
    print("The whole forward pass: attention, MLP and the LM head.\n")
    out = {}
    for name, m in (("baseline", base), ("STAR-KV", star)):
        try:
            us, peak = _e2e_decode_us(m, args.batch, args.seq, args.steps,
                                      cfg.vocab_size)
        except torch.OutOfMemoryError:
            # The dense KV cache is 0.5 MB per token per model; at long context it
            # does not fit beside the weights, which is the point of compressing it.
            print(f"  {name:<9} OOM -- the cache does not fit at this config")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        out[name] = us
        print(f"  {name:<9} {us:>9.0f} us/token   {1e6 / us * args.batch:>7.1f} tok/s"
              f"   peak {peak:.2f} GiB")
    if len(out) == 2:
        print(f"\n  speedup   {out['baseline'] / out['STAR-KV']:.2f}x")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="lmsys/longchat-7b-v1.5-32k")
    p.add_argument("--weights", required=True, help="fused checkpoint from train.py")
    p.add_argument("--mode", choices=("layerwise", "e2e"), default="layerwise")
    p.add_argument("--kv-quant", action="store_true",
                   help="4-bit packed KV cache for the compressed layers (kv_quant.py)")
    p.add_argument("--seq", type=int, default=32768, help="cached context length")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--reps", type=int, default=5, help="layerwise: best of N")
    p.add_argument("--warm", type=int, default=4)
    p.add_argument("--steps", type=int, default=16, help="e2e: decode steps to average")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    cfg = AutoConfig.from_pretrained(args.model)
    if getattr(cfg, "head_dim", None) is None:
        cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads

    print(f"Loading {args.model} (uncompressed) ...")
    base = load_baseline(args.model)
    print(f"Loading {args.weights} (STAR-KV) ...")
    star, skip = load_starkv(args.model, args.weights, kv_quant=args.kv_quant)
    # Both models stay on the CPU. Layer-wise moves one attention module across at
    # a time; e2e moves one whole model at a time. Neither holds both on the card.

    if args.mode == "e2e":
        run_e2e(base, star, cfg, args)
    else:
        run_layerwise(base, star, skip, cfg, args)


if __name__ == "__main__":
    main()
