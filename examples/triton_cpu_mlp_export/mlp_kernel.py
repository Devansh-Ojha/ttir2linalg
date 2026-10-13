"""The playground's fused FC1-ReLU-FC2-ReLU kernel, standalone for export."""
import triton
import triton.language as tl


@triton.jit
def fused_mlp2_kernel(
    x_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    out_ptr,
    M,
    K: tl.constexpr,
    H1: tl.constexpr,
    H2: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Each program handles BLOCK_M rows and retains the intermediate on-chip.
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rk, r1, r2 = tl.arange(0, K), tl.arange(0, H1), tl.arange(0, H2)
    x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=rm[:, None] < M, other=0.0)
    # nn.Linear stores weights as [out, in]; load transposed for tl.dot.
    w1 = tl.load(w1_ptr + r1[None, :] * K + rk[:, None])
    h = tl.dot(x, w1, input_precision="ieee") + tl.load(b1_ptr + r1)[None, :]
    h = tl.maximum(h, 0.0)
    w2 = tl.load(w2_ptr + r2[None, :] * H1 + r1[:, None])
    o = tl.dot(h, w2, input_precision="ieee") + tl.load(b2_ptr + r2)[None, :]
    o = tl.maximum(o, 0.0)
    tl.store(out_ptr + rm[:, None] * H2 + r2[None, :], o, mask=rm[:, None] < M)
