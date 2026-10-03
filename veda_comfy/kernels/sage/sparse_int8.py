"""Block-sparse INT8 attention in Triton, forked from SageAttention v1.

Upstream is `sageattention` 1.0.6 (BSD-3-Clause, Thu-ML), specifically
`quant_per_block.py` and `attn_qk_int8_per_block.py` - the same code
ComfyUI runs behind `--use-sage-attention`. Keeping its arithmetic is the
point: INT8 Q and K with one scale per block, the softmax scale folded into
Q's quantisation so the kernel can use `exp2`, and P and V in fp16. Our
output then matches what ComfyUI's low-precision path would have produced,
which is the accuracy bar a sparse kernel has to clear.

What we changed:

  * The key loop walks a per (head, query tile) list of kept tiles instead
    of the whole sequence. That is the whole point of Veda.
  * Padding slots are masked by `valid_count`, exploiting the tiling
    invariant that a tile's real rows are a prefix. Upstream instead loads
    out-of-range keys as zeros, which quietly lets them into the softmax;
    harmless for its own last block, wrong for tiles that are half padding.
  * Veda's tile is 128 rows and upstream's key block is 64, so the kept
    tiles are expanded to key blocks on the host and the kernel keeps a
    flat loop. That preserves upstream's scale granularity (and so its
    accuracy), fits SM120's shared memory, and gives Triton's pipeliner
    the same loop shape it already handles well.
  * Tile-ordered [N, H, D] tensors throughout, so no layout branching.

Why INT8 and not FP8: e4m3 keeps three mantissa bits, so its relative
error is ~3.6% per value, where INT8 with a per-block scale resolves 127
uniform steps (~0.9%). Measured on attention output, FP8 Q/K alone cost
3.9% against an fp32 reference where this path costs 1.3%. That is why
SageAttention chose INT8. It is also why the kernel is written in Triton:
the CuTe DSL exposes no integer warp MMA below SM100, so a CuTe kernel
cannot reach this arithmetic at all on consumer cards.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

TILE = 128
# Upstream's key block. One Veda tile is two of these.
KEY_BLOCK = 64
SUB = TILE // KEY_BLOCK
LOG2E = 1.4426950408889634


@triton.jit
def _quantize_kernel(X, Xq, Scale, stride_xn, stride_xh, stride_sh,
                     pre_scale, D: tl.constexpr, BLK: tl.constexpr):
    """One INT8 scale per (block of BLK slots, head), upstream's rounding."""
    blk = tl.program_id(0)
    head = tl.program_id(1)
    offs_n = blk * BLK + tl.arange(0, BLK)
    offs_d = tl.arange(0, D)
    offset = head * stride_xh + offs_n[:, None] * stride_xn + offs_d[None, :]
    x = tl.load(X + offset).to(tl.float32) * pre_scale
    scale = tl.max(tl.abs(x)) / 127.0
    # A block that is all padding has no scale; 1.0 keeps the zeros zero.
    scale = tl.where(scale > 0.0, scale, 1.0)
    quantized = x / scale
    quantized += 0.5 * tl.where(quantized >= 0, 1, -1)
    tl.store(Xq + offset, quantized.to(tl.int8))
    tl.store(Scale + head * stride_sh + blk, scale)


def quantize(x: torch.Tensor, block: int,
             pre_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, H, D] -> (int8 of the same shape, [H, N // block] fp32 scales).

    `pre_scale` is multiplied in before quantising; the caller folds the
    softmax scale into Q that way, exactly as upstream does, so the kernel
    never applies it.
    """
    slots, heads, dim = x.shape
    out = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((heads, slots // block), dtype=torch.float32,
                        device=x.device)
    _quantize_kernel[(slots // block, heads)](
        x, out, scale, x.stride(0), x.stride(1), scale.stride(0),
        pre_scale=pre_scale, D=dim, BLK=block, num_warps=4)
    return out, scale


@triton.jit
def _attention_kernel(Q, K, V, Q_scale, K_scale, Index, Count, Valid, Out,
                      stride_n, stride_h, stride_vn, stride_vh,
                      stride_qs, stride_ks, stride_ih, stride_iq,
                      n_q_tiles, D: tl.constexpr, BLK: tl.constexpr,
                      BLOCK_M: tl.constexpr):
    query_tile = tl.program_id(0)
    head = tl.program_id(1)
    offs_m = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLK)
    offs_d = tl.arange(0, D)

    q = tl.load(Q + head * stride_h + offs_m[:, None] * stride_n
                + offs_d[None, :])
    q_scale = tl.load(Q_scale + head * stride_qs + query_tile)

    # Finite, not -inf: a key block that is entirely padding leaves m_i
    # untouched, and -inf would make (-inf) - (-inf) a NaN on the way in.
    m_i = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    count = tl.load(Count + head * n_q_tiles + query_tile)
    index_row = Index + head * stride_ih + query_tile * stride_iq
    for i in range(0, count):
        block = tl.load(index_row + i)
        start = block * BLK
        k = tl.load(K + head * stride_h + (start + offs_n)[None, :] * stride_n
                    + offs_d[:, None])
        k_scale = tl.load(K_scale + head * stride_ks + block)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
        # Real rows are a prefix of every block, so one comparison masks
        # all the padding. Upstream has no equivalent: it loads keys past
        # the end as zeros, which quietly lets them into the softmax.
        qk = tl.where(offs_n[None, :] < tl.load(Valid + block), qk,
                      -float('inf'))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + head * stride_vh + (start + offs_n)[:, None] * stride_vn
                    + offs_d[None, :])
        acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float16)
        m_i = m_ij

    # A query tile with nothing kept would divide by zero; selection always
    # keeps the diagonal, so this only guards against a malformed mask.
    l_i = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_i[:, None]
    tl.store(Out + head * stride_h + offs_m[:, None] * stride_n
             + offs_d[None, :], acc.to(Out.type.element_ty))


def key_blocks(index: torch.Tensor, count: torch.Tensor,
               valid_count: torch.Tensor):
    """Kept tiles -> kept key blocks, which is what the kernel walks.

    One 128-row tile is SUB key blocks. Expanding on the host keeps the
    kernel's loop flat and lets the per-block valid counts (a tile's real
    rows are a prefix) be sliced the same way.
    """
    blocks = (index.to(torch.int32) * SUB)[..., None] + torch.arange(
        SUB, device=index.device, dtype=torch.int32)
    offsets = torch.arange(SUB, device=valid_count.device) * KEY_BLOCK
    valid = (valid_count[:, None] - offsets).clamp_(0, KEY_BLOCK)
    return (blocks.flatten(-2).contiguous(), (count * SUB).contiguous(),
            valid.flatten().to(torch.int32).contiguous())


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           index: torch.Tensor, count: torch.Tensor,
           valid_count: torch.Tensor,
           softmax_scale: float | None = None) -> torch.Tensor:
    """Block-sparse INT8 attention on tile-ordered tensors.

    Args:
        q, k, v: [N, H, D] contiguous, 16-bit, N a multiple of TILE.
        index: [H, N // TILE, max_kept] int32 kept key tiles per query
            tile, ascending; entries past `count` are ignored.
        count: [H, N // TILE] int32 how many of `index` are real.
        valid_count: [n_tiles] int32 real rows per tile (a prefix).
        softmax_scale: defaults to D ** -0.5.

    Returns:
        [N, H, D] in q's dtype.
    """
    slots, heads, dim = q.shape
    scale = (dim ** -0.5) if softmax_scale is None else softmax_scale
    q_int8, q_scale = quantize(q, TILE, pre_scale=scale * LOG2E)
    k_int8, k_scale = quantize(k, KEY_BLOCK)
    blocks, block_count, valid = key_blocks(index, count, valid_count)
    v16 = v.to(torch.float16)
    out = torch.empty_like(q)
    _attention_kernel[(slots // TILE, heads)](
        q_int8, k_int8, v16, q_scale, k_scale, blocks,
        block_count, valid, out,
        q.stride(0), q.stride(1), v16.stride(0), v16.stride(1),
        q_scale.stride(0), k_scale.stride(0), blocks.stride(0),
        blocks.stride(1),
        n_q_tiles=slots // TILE, D=dim, BLK=KEY_BLOCK, BLOCK_M=TILE,
        num_warps=8, num_stages=4)
    return out


def available() -> bool:
    """Whether Triton can compile for the current device."""
    return triton is not None and torch.cuda.is_available()
