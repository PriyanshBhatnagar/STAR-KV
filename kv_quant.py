"""4-bit quantization of the low-rank KV cache: the format, and Triton decode kernels.

An add-on to the low-rank cache, not part of it. model.fold_kv_hadamard rotates
every latent by a block Hadamard (folded offline into VS and U, so it is free at
run time); then either

  accuracy  model.set_kv_fake_quant   fp32 fake quantization on the PyTorch path
  speed     replace_attn_with_triton(kv_quant=True)   the packed cache + kernels here

Both apply the SAME quantizer (_q below), so what is measured for accuracy is
exactly what the kernels decode.

The format: per token, each head's K latent (and the single V latent) is split
into a leading OUTLIER block (int(0.2 * rank) channels -- the latent's channels
follow VS's descending singular values, so the leading ones carry the range) at
4-bit, and an INLIER block at 3-bit, each with one symmetric scale per token.
Codes are stored as 4-bit nibbles either way, so 3-bit inliers spend one unused
bit; that keeps every field power-of-two aligned, which the kernels need.

  K  kq  [B, L, W/2]    uint8; byte k of a token holds lanes 2k, 2k+1
     ks  [B, L, Hkv, 2] bf16  (outlier scale, inlier scale) per head per token
  V  vq  [B, L, NBO+NBI] uint8; outlier channels packed first, byte-aligned,
                          then inlier channels, so no byte straddles the blocks
     vs  [B, L, 2]      bf16

Everything is token-major, so appending a token is a contiguous copy of whole
fixed-size rows. The bf16 cache keeps K rank-major ([B, W, L]) for the old
one-program-per-head abx; with that layout every row is L bytes, unaligned
whenever L is odd, and torch.cat of the packed cache ran at 621 GB/s. abx_z owns
all heads of a token tile in one program, so token-major also turns its reads
into one contiguous block per tile.

Four kernels, one launch each per decode step and layer:

  pack_k_step / pack_v_step
      Quantize and pack the new token's codes. One launch replaces ~8 PyTorch ops.

  fold_q
      A|B = the query folded into U, per step; see abx_z. Also absorbs the 1/sqrt(d)
      logit scale and the RoPE attention scaling, so the K kernel has neither.

  abx_z   (K: logits)
      K reconstruction is 2 * W * 128 FLOPs per token whatever K is stored in, so a
      4-bit K kernel is compute-bound, not bandwidth-bound. Rewriting
          logit_l = q . RoPE_l(U^T x_l)
                  = sum_r x[l,r] * z[l,r],   z[l,:] = [cos_l | sin_l] @ [A | B]^T
      moves the MMA onto the cos/sin table (computed once per token tile and reused
      by every head) times the per-step A|B, with K = 128. The 4-bit cache is only
      ever used elementwise, so its dequantized values never have to be shuffled
      into tensor-core operand layout. Algebraically identical to lift-then-rotate.
      The whole rank axis is one flat loop of 16-lane tiles (see k_tile_meta).

  pv_fold (V: probs @ V)
      Split over the token axis (the whole-sequence-per-program version was
      latency-bound at ~2.5 programs per SM), and since a tile never straddles the
      two blocks, the per-token scale folds into the probabilities instead of
      multiplying every V element.

Dequantization uses fp32 bits 0x4B000000 | n == 2^23 + n, which turns a nibble into
a float with an OR and a bitcast. The int16 equivalent (bf16 bits 0x4300 | n) is
cheaper but Triton 3.8 miscompiles it on the K tile's layout whenever L is not a
multiple of 16 -- every token comes out wrong -- so int32 is used throughout.
"""
import math

import torch
import triton
import triton.language as tl

OUTLIER_RATIO = 0.2
BITS_OUT, BITS_IN = 4, 3
EPS = 1e-5                       # amax is clamped here so an all-zero block has a scale
EPS_ = tl.constexpr(EPS)          # the same, visible inside kernels


def n_outlier(rank: int, ratio: float = OUTLIER_RATIO) -> int:
    return int(ratio * rank)


# ---------------------------------------------------------------------------
# The rotation
# ---------------------------------------------------------------------------

def block_hadamard(rank: int, n_out: int, generator: torch.Generator) -> torch.Tensor:
    """rank x rank orthonormal T = diag(H_outlier, H_inlier), float64.

    Each block is rotated on its own: one rotation over the whole latent would mix
    the leading high-energy channels into the tail and inflate the inlier block's
    scale, undoing the reason for splitting. A block is a Hadamard on its largest
    power-of-two leading part (identity on the remainder, so T stays exactly
    rank x rank) with random sign flips on the rows.
    """
    T = torch.eye(rank, dtype=torch.float64)
    for a, n in ((0, n_out), (n_out, rank - n_out)):
        if n <= 1:
            continue
        m = 1 << (n.bit_length() - 1)
        H = torch.ones(1, 1, dtype=torch.float64)
        while H.shape[0] < m:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        signs = torch.where(torch.rand(m, generator=generator) < 0.5, -1.0, 1.0).to(torch.float64)
        T[a:a + m, a:a + m] = signs[:, None] * H / math.sqrt(m)
    return T


# ---------------------------------------------------------------------------
# The quantizer. _q is the definition: fake_quant applies it for accuracy runs,
# the reference packers use it, and the Triton packers match it bit for bit.
# ---------------------------------------------------------------------------

def _q(x, bits):
    qmax = 2 ** (bits - 1) - 1
    amax = x.abs().amax(-1, keepdim=True).clamp_min(EPS)
    # Tensor / tensor, not `amax / qmax`: PyTorch divides by a Python scalar by
    # multiplying with its reciprocal, which is not correctly rounded (measured: 54%
    # of fp32 results off by an ulp). Those ulps then move a code by one at exact
    # rounding ties. The kernels use IEEE division (div_rn), so this is the
    # definition they match bit for bit.
    s = amax / torch.full_like(amax, qmax)
    return torch.round(x / s).clamp_(-(2 ** (bits - 1)), qmax) + 8, s.squeeze(-1)


def fake_quant(x: torch.Tensor, blocks) -> torch.Tensor:
    """Quantize and dequantize the latent x [..., rank] in fp32, per token.

    blocks: (start, n_out, width) per head -- the outlier block is
    [start, start + n_out) at 4-bit, the inlier block the rest of the head at
    3-bit. Returns x's dtype.
    """
    f = x.float()
    out = torch.empty_like(f)
    for a, no, w in blocks:
        for lo, hi, bits in ((a, a + no, BITS_OUT), (a + no, a + w, BITS_IN)):
            if hi > lo:
                n, s = _q(f[..., lo:hi], bits)
                out[..., lo:hi] = (n - 8) * s.unsqueeze(-1)
    return out.to(x.dtype)


def pack_k_ref(x, offsets, widths, nouts):
    """x [B, L, W] -> (kq [B, L, W/2] uint8, ks [B, L, H, 2] bf16)."""
    B, L, W = x.shape
    f = x.float()
    n = torch.full_like(f, 8.0)
    ks = torch.ones(B, L, len(offsets), 2, device=x.device)
    for h, (o, w, no) in enumerate(zip(offsets, widths, nouts)):
        if no:
            n[..., o:o + no], ks[:, :, h, 0] = _q(f[..., o:o + no], BITS_OUT)
        n[..., o + no:o + w], ks[:, :, h, 1] = _q(f[..., o + no:o + w], BITS_IN)
    n = n.to(torch.uint8)
    return (n[..., 0::2] | (n[..., 1::2] << 4)).contiguous(), ks.to(torch.bfloat16).contiguous()


def unpack_k_ref(kq, ks, offsets, widths, nouts):
    """-> [B, L, W] float."""
    n = torch.stack([(kq & 15).float(), ((kq >> 4) & 15).float()], -1).flatten(-2) - 8
    out = torch.zeros_like(n)
    for h, (o, w, no) in enumerate(zip(offsets, widths, nouts)):
        out[..., o:o + no] = n[..., o:o + no] * ks[:, :, h, 0].float().unsqueeze(-1)
        out[..., o + no:o + w] = n[..., o + no:o + w] * ks[:, :, h, 1].float().unsqueeze(-1)
    return out


def _nib_pack(n):
    if n.shape[-1] % 2:
        n = torch.nn.functional.pad(n, (0, 1), value=8)
    return n[..., 0::2] | (n[..., 1::2] << 4)


def pack_v_ref(v, nout):
    """v [B, L, R] -> (vq [B, L, NBO+NBI] uint8, vs [B, L, 2] bf16)."""
    f = v.float()
    no, so = _q(f[..., :nout], BITS_OUT) if nout else (f[..., :0], torch.ones_like(f[..., 0]))
    ni, si = _q(f[..., nout:], BITS_IN)
    vq = torch.cat([_nib_pack(no.to(torch.uint8)), _nib_pack(ni.to(torch.uint8))], -1)
    pad = v_row(v.shape[-1], nout) - vq.shape[-1]
    if pad:
        vq = torch.nn.functional.pad(vq, (0, pad), value=0x88)
    return vq.contiguous(), torch.stack([so, si], -1).to(torch.bfloat16).contiguous()


def unpack_v_ref(vq, vs, nout, R):
    nbo = (nout + 1) // 2

    def un(b, cnt):
        return (torch.stack([(b & 15).float(), ((b >> 4) & 15).float()], -1).flatten(-2) - 8)[..., :cnt]
    return torch.cat([un(vq[..., :nbo], nout) * vs[..., 0].float().unsqueeze(-1),
                      un(vq[..., nbo:], R - nout) * vs[..., 1].float().unsqueeze(-1)], -1)


def v_bytes(R: int, nout: int) -> int:
    """Bytes of codes per token (outlier block, then inlier block)."""
    return (nout + 1) // 2 + (R - nout + 1) // 2


def v_row(R: int, nout: int) -> int:
    """Stored row: v_bytes padded to 8, so the cache can be appended as int64."""
    return -(-v_bytes(R, nout) // 8) * 8


def append(cache: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    """torch.cat(dim=1) of a token-major packed cache, through a wide view.

    torch.cat copies one element per thread-step, so on uint8 it moves a byte at a
    time: 581 GB/s on a 713 MB cache, against 864 GB/s for the same bytes viewed
    as int64. Every packed row is a multiple of 8 bytes (K: W/2 with W a multiple
    of 16; V: padded by v_row; scales: 128 and 4 bytes), so the view is free.
    """
    B, L = cache.shape[:2]
    row = cache[0, 0].numel() * cache.element_size()
    wide = torch.int64 if row % 8 == 0 else torch.int32
    c = cache.reshape(B, L, -1).view(wide)
    n = new.reshape(B, new.shape[1], -1).view(wide)
    return torch.cat([c, n], 1).view(cache.dtype).view(B, L + new.shape[1], *cache.shape[2:])


# ---------------------------------------------------------------------------
# Write side: quantize + pack one token
# ---------------------------------------------------------------------------

@triton.jit
def _pack_k_step(x_ptr, kq_ptr, ks_ptr, r_ptr, off_ptr, nout_ptr,
                 x_s0, x_s1, k_s0, k_s1, s_s0, s_s1, s_s2,
                 BLOCK: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    R = tl.load(r_ptr + h)
    NOUT = tl.load(nout_ptr + h)
    off = tl.load(off_ptr + h).to(tl.int64)
    W_H = ((R + 15) // 16) * 16               # stored width; lanes past R are zero
    lane = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + b * x_s0 + (off + lane) * x_s1, mask=lane < W_H, other=0.0).to(tl.float32)
    is_out = lane < NOUT
    a_out = tl.maximum(tl.max(tl.where(is_out, tl.abs(x), 0.0), 0), EPS_)
    a_in = tl.maximum(tl.max(tl.where((lane >= NOUT) & (lane < W_H), tl.abs(x), 0.0), 0), EPS_)
    s_out = tl.extra.cuda.libdevice.div_rn(a_out, 7.0)
    s_in = tl.extra.cuda.libdevice.div_rn(a_in, 3.0)
    # div_rn, not `/`: Triton's default fp32 divide is approximate (~2 ulp), which
    # moves a code by one at exact rounding ties relative to the torch quantizer
    # the accuracy was validated with.
    q = tl.where(is_out,
                 tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(x, s_out)), -8.0), 7.0),
                 tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(x, s_in)), -4.0), 3.0))
    n = (q + 8.0).to(tl.int32)
    even, odd = tl.split(tl.reshape(n, (BLOCK // 2, 2)))
    byte = (even | (odd << 4)).to(tl.uint8)
    k = tl.arange(0, BLOCK // 2)
    tl.store(kq_ptr + b * k_s0 + (off // 2 + k) * k_s1, byte, mask=2 * k < W_H)
    tl.store(ks_ptr + b * s_s0 + h * s_s1, s_out.to(tl.bfloat16), mask=NOUT > 0)
    tl.store(ks_ptr + b * s_s0 + h * s_s1 + s_s2, s_in.to(tl.bfloat16))



def pack_k_step(kc, ranks, offsets, nouts, W):
    """kc [B, 1, W] codes of the new token -> (kq [B, 1, W/2], ks [B, 1, H, 2]), one launch."""
    B, H = kc.shape[0], ranks.numel()
    kq = torch.empty(B, 1, W // 2, device=kc.device, dtype=torch.uint8)
    ks = torch.ones(B, 1, H, 2, device=kc.device, dtype=torch.bfloat16)
    _pack_k_step[(B, H)](kc, kq, ks, ranks, offsets, nouts,
                         kc.stride(0), kc.stride(2), kq.stride(0), kq.stride(2),
                         ks.stride(0), ks.stride(2), ks.stride(3), BLOCK=128)
    return kq, ks


@triton.jit
def _pack_v_step(x_ptr, vq_ptr, vs_ptr, x_s0, x_s2, v_s0, v_s2, s_s0, s_s1,
                 R, NOUT, NBO, NB, BLOCK: tl.constexpr, BLOCK_B: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + b * x_s0 + c * x_s2, mask=c < R, other=0.0).to(tl.float32)
    s_out = tl.extra.cuda.libdevice.div_rn(tl.maximum(tl.max(tl.where(c < NOUT, tl.abs(x), 0.0), 0), EPS_), 7.0)
    s_in = tl.extra.cuda.libdevice.div_rn(tl.maximum(tl.max(tl.where((c >= NOUT) & (c < R), tl.abs(x), 0.0), 0), EPS_), 3.0)
    # Each byte takes its two channels from its own block, so a byte is never split
    # across the outlier/inlier boundary; an odd block ends in a padding nibble.
    j = tl.arange(0, BLOCK_B)
    in_out = j < NBO
    c_lo = tl.where(in_out, 2 * j, NOUT + 2 * (j - NBO))
    c_end = tl.where(in_out, NOUT, R)
    ok_lo = (j < NB) & (c_lo < c_end)
    ok_hi = (j < NB) & (c_lo + 1 < c_end)
    xl = tl.load(x_ptr + b * x_s0 + c_lo * x_s2, mask=ok_lo, other=0.0).to(tl.float32)
    xh = tl.load(x_ptr + b * x_s0 + (c_lo + 1) * x_s2, mask=ok_hi, other=0.0).to(tl.float32)
    s = tl.where(in_out, s_out, s_in)
    lo_q = tl.where(in_out, 7.0, 3.0)
    lo_m = tl.where(in_out, -8.0, -4.0)
    nl = tl.where(ok_lo, tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(xl, s)), lo_m), lo_q) + 8.0, 8.0)
    nh = tl.where(ok_hi, tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(xh, s)), lo_m), lo_q) + 8.0, 8.0)
    byte = (nl.to(tl.int32) | (nh.to(tl.int32) << 4)).to(tl.uint8)
    tl.store(vq_ptr + b * v_s0 + j * v_s2, byte, mask=j < NB)
    tl.store(vs_ptr + b * s_s0, tl.where(NOUT > 0, s_out, 1.0).to(tl.bfloat16))
    tl.store(vs_ptr + b * s_s0 + s_s1, s_in.to(tl.bfloat16))


def pack_v_step(vc, nout):
    """vc [B, 1, R] -> (vq [B, 1, NB], vs [B, 1, 2]), one launch."""
    B, _, R = vc.shape
    nbo = (nout + 1) // 2
    nb = v_bytes(R, nout)
    vq = torch.full((B, 1, v_row(R, nout)), 0x88, device=vc.device, dtype=torch.uint8)
    vs = torch.empty(B, 1, 2, device=vc.device, dtype=torch.bfloat16)
    _pack_v_step[(B,)](vc, vq, vs, vc.stride(0), vc.stride(2), vq.stride(0), vq.stride(2),
                       vs.stride(0), vs.stride(2), R, nout, nbo, nb,
                       BLOCK=triton.next_power_of_2(R), BLOCK_B=triton.next_power_of_2(nb))
    return vq, vs


# ---------------------------------------------------------------------------
# K: fold the query into U, then the z-formulation kernel
# ---------------------------------------------------------------------------

def k_tile_meta(ranks, nouts, device):
    """Per 16-lane tile of the flat rank axis: its head, how many of its lanes are
    outliers, and whether it is its head's last tile.

    Every head's stored width is a multiple of 16, so no 16-lane tile straddles two
    heads. That lets abx_z walk the whole rank axis as ONE loop -- 85 iterations for
    a 1360-lane layer, long enough for Triton to pipeline the loads -- where a
    head-by-rank-tile double loop gave it trip counts of 1 to 5.
    """
    head, n_out, last = [], [], []
    for h, (r, no) in enumerate(zip(ranks, nouts)):
        tiles = -(-int(r) // 16)
        for j in range(tiles):
            head.append(h)
            n_out.append(max(0, min(16, int(no) - 16 * j)))
            last.append(int(j == tiles - 1))
    t = lambda v: torch.tensor(v, device=device, dtype=torch.int32)
    return t(head), t(n_out), t(last)


@triton.jit
def _fold_q(q_ptr, u_ptr, hol_ptr, ab_ptr, q_s0, q_s1, q_s3, u_s0, u_s1,
            ab_s0, ab_s1, ab_s2, W, SCALE, BLOCK_W: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    lane = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    ok = lane < W
    i = tl.arange(0, 64)
    head = tl.load(hol_ptr + lane, mask=ok, other=0).to(tl.int64)
    qb = q_ptr + b * q_s0 + head[:, None] * q_s1
    a0 = tl.load(qb + i[None, :] * q_s3, mask=ok[:, None], other=0.0).to(tl.float32)
    a1 = tl.load(qb + (i + 64)[None, :] * q_s3, mask=ok[:, None], other=0.0).to(tl.float32)
    u0 = tl.load(u_ptr + lane[:, None] * u_s0 + i[None, :] * u_s1, mask=ok[:, None], other=0.0).to(tl.float32)
    u1 = tl.load(u_ptr + lane[:, None] * u_s0 + (i + 64)[None, :] * u_s1, mask=ok[:, None], other=0.0).to(tl.float32)
    base = ab_ptr + b * ab_s0 + lane[:, None] * ab_s1
    tl.store(base + i[None, :] * ab_s2, ((a0 * u0 + a1 * u1) * SCALE).to(tl.float16), mask=ok[:, None])
    tl.store(base + (i + 64)[None, :] * ab_s2, ((a1 * u0 - a0 * u1) * SCALE).to(tl.float16), mask=ok[:, None])


def fold_q(q, U, head_of_lane, sm_scale, attn_scaling=1.0, out=None):
    """q [B,H,1,128] (RoPE applied), U [W,128] -> AB [B,W,128] fp16, lanes-major.

      A[r,i] = q_i U[r,i] + q_{i+64} U[r,i+64]      B[r,i] = q_{i+64} U[r,i] - q_i U[r,i+64]

    so that q . RoPE_l(U^T x) == sum_r x[r] * ([cos_l | sin_l] . AB[r]). The logit
    scale and the RoPE attention scaling multiply every term of the logit, so they
    fold in here and cost nothing downstream. Lanes-major, so the kernel reads a
    16-lane tile as 16 contiguous 256-byte rows.
    """
    B, W = q.shape[0], U.shape[0]
    if out is None:
        out = torch.empty(B, W, 128, device=q.device, dtype=torch.float16)
    _fold_q[(B, triton.cdiv(W, 64))](q, U, head_of_lane, out, q.stride(0), q.stride(1), q.stride(3),
                                     U.stride(0), U.stride(1), *out.stride(), W,
                                     float(sm_scale * attn_scaling), BLOCK_W=64)
    return out


@triton.jit
def _abx_z(ab_ptr, kq_ptr, ks_ptr, out_ptr, th_ptr, tn_ptr, tl_ptr, inv_ptr,
           ab_s0, ab_s1, ab_s2, k_s0, k_s1, k_s2, s_s0, s_s1, s_s2, s_s3, o_s0, o_s1, o_s3,
           L, NT, BLOCK_L: tl.constexpr, ACC16: tl.constexpr):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_l = tl.program_id(1)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = offs_l < L
    offs_c = tl.arange(0, 128)
    offs_k = tl.arange(0, 8)
    offs_r = tl.arange(0, 16)
    # The RoPE table for this token tile, once, shared by every head. Computed per
    # (batch, head) it was 2e9 sin/cos at 32k x batch 16.
    inv = tl.load(inv_ptr + tl.arange(0, 64))
    fr = offs_l.to(tl.float32)[:, None] * inv[None, :]
    cs = tl.reshape(tl.permute(tl.reshape(tl.join(tl.extra.cuda.libdevice.fast_cosf(fr),
                                                  tl.extra.cuda.libdevice.fast_sinf(fr)),
                                          (BLOCK_L, 64, 2)), (0, 2, 1)), (BLOCK_L, 128)).to(tl.float16)
    kq_row = kq_ptr + pid_b * k_s0 + offs_l[:, None] * k_s1
    ab_b = ab_ptr + pid_b * ab_s0
    sb_row = ks_ptr + pid_b * s_s0 + offs_l * s_s1
    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)
    for t in range(0, NT):
        h = tl.load(th_ptr + t)
        n_out = tl.load(tn_ptr + t)
        is_last = tl.load(tl_ptr + t)
        ab = tl.load(ab_b + (t * 16 + offs_r)[:, None] * ab_s1 + offs_c[None, :] * ab_s2)      # [16, 128]
        if ACC16:
            # fp16 accumulation over K=128 of |cos|,|sin| <= 1 times A|B: ~0.1% error,
            # well under the 4-bit step, at twice the fp32-accumulate MMA rate.
            z = tl.dot(cs, tl.trans(ab), out_dtype=tl.float16).to(tl.float32)
        else:
            z = tl.dot(cs, tl.trans(ab))
        raw = tl.load(kq_row + (t * 8 + offs_k)[None, :] * k_s2, mask=l_mask[:, None], other=0x88)
        lo = ((raw & 0x0F).to(tl.int32) | 0x4B000000).to(tl.float32, bitcast=True)
        hi = (((raw >> 4) & 0x0F).to(tl.int32) | 0x4B000000).to(tl.float32, bitcast=True)
        v = tl.reshape(tl.join(lo, hi), (BLOCK_L, 16)) - 8388616.0                              # n - 8
        so = tl.load(sb_row + h * s_s2, mask=l_mask, other=0.0).to(tl.float32)
        si = tl.load(sb_row + h * s_s2 + s_s3, mask=l_mask, other=0.0).to(tl.float32)
        sc = tl.where((offs_r < n_out)[None, :], so[:, None], si[:, None])
        acc += tl.sum(v * sc * z, 1)
        # A head's logit is complete at its last tile: store it, start the next.
        # Branch-free, so the loop stays pipelineable.
        tl.store(out_ptr + pid_b * o_s0 + h * o_s1 + offs_l * o_s3, acc, mask=l_mask & (is_last != 0))
        acc = tl.where(is_last != 0, 0.0, acc)


def abx_z(ab, kq, ks, meta, inv_freq, out=None, block_l=None, acc16=True, warps=4, stages=None):
    """Scaled attention logits [B, H, 1, L] bf16 from the packed K cache. MHA only.

    ab from fold_q; meta from k_tile_meta. Tiling defaults were swept on an RTX
    4090 at batch 16: 128-token tiles with 4 pipeline stages at long context, 64 and
    3 below 16k where there are too few programs to fill the card otherwise.
    """
    th, tn, tlast = meta
    B, L, H = kq.shape[0], kq.shape[1], ks.shape[2]
    if block_l is None:
        block_l = 128 if L >= 16384 else 64
    if stages is None:
        stages = 4 if L >= 16384 else 3
    if out is None:
        out = torch.empty(B, H, 1, L, device=kq.device, dtype=torch.bfloat16)
    _abx_z[(B, triton.cdiv(L, block_l))](ab, kq, ks, out, th, tn, tlast, inv_freq, *ab.stride(),
                                         kq.stride(0), kq.stride(1), kq.stride(2), *ks.stride(),
                                         out.stride(0), out.stride(1), out.stride(3), L, th.numel(),
                                         BLOCK_L=block_l, ACC16=acc16, num_warps=warps, num_stages=stages)
    return out


# ---------------------------------------------------------------------------
# V: split over tokens, scale folded into the probabilities
# ---------------------------------------------------------------------------

@triton.jit
def _pv_fold(p_ptr, vq_ptr, vs_ptr, out_ptr,
             p_s0, p_s1, p_s2, v_s0, v_s1, v_s2, s_s0, s_s1, s_s2, o_s0, o_s1, o_s2,
             H, L, R, NOUT, NBO, T_OUT, SEG_LEN,
             BLOCK_H: tl.constexpr, BLOCK_VB: tl.constexpr, BLOCK_L: tl.constexpr):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_t = tl.program_id(1)
    pid_s = tl.program_id(2)
    # A tile sits wholly inside one block, so one scale per token covers all of it
    # and folds into p instead of multiplying every V element.
    is_out = pid_t < T_OUT
    t = tl.where(is_out, pid_t, pid_t - T_OUT)
    byte0 = tl.where(is_out, 0, NBO)
    ch0 = tl.where(is_out, 0, NOUT)
    ch_end = tl.where(is_out, NOUT, R)
    blk = tl.where(is_out, 0, 1)
    offs_h = tl.arange(0, BLOCK_H)
    offs_bt = t * BLOCK_VB + tl.arange(0, BLOCK_VB)
    offs_c = ch0 + t * 2 * BLOCK_VB + tl.arange(0, 2 * BLOCK_VB)
    h_mask = offs_h < H
    c_mask = offs_c < ch_end
    b_mask = offs_bt * 2 < ch_end - ch0
    acc = tl.zeros((BLOCK_H, 2 * BLOCK_VB), dtype=tl.float32)
    l_beg = pid_s * SEG_LEN
    for l0 in range(0, tl.cdiv(SEG_LEN, BLOCK_L)):
        offs_l = l_beg + l0 * BLOCK_L + tl.arange(0, BLOCK_L)
        l_mask = (offs_l < L) & (offs_l < l_beg + SEG_LEN)
        p = tl.load(p_ptr + pid_b * p_s0 + offs_h[:, None] * p_s1 + offs_l[None, :] * p_s2,
                    mask=h_mask[:, None] & l_mask[None, :], other=0.0)
        s = tl.load(vs_ptr + pid_b * s_s0 + offs_l * s_s1 + blk * s_s2, mask=l_mask, other=0.0)
        ps = (p.to(tl.float32) * s.to(tl.float32)[None, :]).to(tl.bfloat16)
        raw = tl.load(vq_ptr + pid_b * v_s0 + offs_l[:, None] * v_s1 + (byte0 + offs_bt)[None, :] * v_s2,
                      mask=l_mask[:, None] & b_mask[None, :], other=0x88)
        lo = ((raw & 0x0F).to(tl.int32) | 0x4B000000).to(tl.float32, bitcast=True)
        hi = (((raw >> 4) & 0x0F).to(tl.int32) | 0x4B000000).to(tl.float32, bitcast=True)
        x = (tl.reshape(tl.join(lo, hi), (BLOCK_L, 2 * BLOCK_VB)) - 8388616.0).to(tl.bfloat16)
        acc = tl.dot(ps, x, acc)
    tl.atomic_add(out_ptr + pid_b * o_s0 + offs_h[:, None] * o_s1 + offs_c[None, :] * o_s2,
                  acc, mask=h_mask[:, None] & c_mask[None, :], sem="relaxed")


def pv_fold(p, vq, vs, nout, R, out=None, block_vb=64, block_l=32, target=2048, warps=4):
    """p [B, H, L] bf16 @ packed V -> [B, H, R] fp32."""
    B, H, L = p.shape
    nbo = (nout + 1) // 2
    t_out = triton.cdiv(nout, 2 * block_vb) if nout else 0
    nt = t_out + triton.cdiv(R - nout, 2 * block_vb)
    if out is None:
        out = torch.zeros(B, H, R, device=p.device, dtype=torch.float32)
    else:
        out.zero_()
    nseg = max(1, min(triton.cdiv(target, B * nt), triton.cdiv(L, block_l)))
    seg = triton.cdiv(triton.cdiv(L, nseg), block_l) * block_l
    nseg = triton.cdiv(L, seg)
    _pv_fold[(B, nt, nseg)](p, vq, vs, out, *p.stride(), *vq.stride(), *vs.stride(), *out.stride(),
                            H, L, R, nout, nbo, t_out, seg, BLOCK_H=max(16, triton.next_power_of_2(H)),
                            BLOCK_VB=block_vb, BLOCK_L=block_l, num_warps=warps)
    return out
