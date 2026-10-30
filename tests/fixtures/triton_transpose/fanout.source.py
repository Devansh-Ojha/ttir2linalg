"""Single-program 32x32 kernels for genuine offline Triton compilation.

Atlas execution requires explicitly selected target numerical policies. These
kernels are source programs, not assertions of GPU/Atlas numerical equivalence.
"""
import triton
import triton.language as tl


@triton.jit
def bf16_add_relu(X, Y, Out):
    rows = tl.arange(0, 32)[:, None]
    cols = tl.arange(0, 32)[None, :]
    offsets = rows * 32 + cols
    x = tl.load(X + offsets)
    y = tl.load(Y + offsets)
    added = (x + y).to(tl.bfloat16)
    result = tl.maximum(added, 0.0).to(tl.bfloat16)
    tl.store(Out + offsets, result)


@triton.jit
def fp8_matmul(X, W, Out):
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    x = tl.load(X + r * 32 + c)
    w = tl.load(W + r * 32 + c)
    result = tl.dot(x, w, input_precision='ieee').to(tl.bfloat16)
    tl.store(Out + r * 32 + c, result)


@triton.jit
def fp8_two_layer(X, W0, W1, Out):
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    x = tl.load(X + r * 32 + c)
    w0 = tl.load(W0 + r * 32 + c)
    w1 = tl.load(W1 + r * 32 + c)
    hidden = tl.dot(x, w0, input_precision='ieee').to(tl.bfloat16)
    hidden = tl.maximum(hidden, 0.0).to(tl.bfloat16)
    packed = hidden.to(tl.float8e4nv, fp_downcast_rounding='rtne')
    output = tl.dot(packed, w1, input_precision='ieee').to(tl.bfloat16)
    tl.store(Out + r * 32 + c, output)


@triton.jit
def bf16_row_softmax(X, Out):
    """Stable row softmax with explicit BF16 boundaries and FP32 row sum."""
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    x = tl.load(X + r * 32 + c)
    maximum = tl.max(x.to(tl.float32), 1).to(tl.bfloat16)
    shifted = (x - maximum[:, None]).to(tl.bfloat16)
    exponential = tl.exp(shifted.to(tl.float32)).to(tl.bfloat16)
    denominator = tl.sum(exponential.to(tl.float32), 1).to(tl.bfloat16)
    result = (exponential / denominator[:, None]).to(tl.bfloat16)
    tl.store(Out + r * 32 + c, result)


@triton.jit
def bf16_row_reduce(X, Out, REDUCE_MAX: tl.constexpr):
    """Independent operation-composition probe, without softmax arithmetic."""
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    x = tl.load(X + r * 32 + c).to(tl.float32)
    if REDUCE_MAX:
        reduced = tl.max(x, 1).to(tl.bfloat16)
    else:
        reduced = tl.sum(x, 1).to(tl.bfloat16)
    result = tl.broadcast_to(reduced[:, None], (32, 32))
    tl.store(Out + r * 32 + c, result)


@triton.jit
def bf16_simple_attention(Q, K, V, Out):
    """Conventional static attention feasibility probe; no causal/online logic.

    Score and probability BF16 boundaries are explicit. Neither dot is
    rewritten into FP8 merely to fit the currently qualified MXU interface.
    """
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    q = tl.load(Q + r * 32 + c)
    k = tl.load(K + r * 32 + c)
    v = tl.load(V + r * 32 + c)
    scores = (tl.dot(q, tl.trans(k), input_precision='ieee') * 0.1767766952966369).to(tl.bfloat16)
    maximum = tl.max(scores.to(tl.float32), 1).to(tl.bfloat16)
    shifted = (scores - maximum[:, None]).to(tl.bfloat16)
    exponential = tl.exp(shifted.to(tl.float32)).to(tl.bfloat16)
    denominator = tl.sum(exponential.to(tl.float32), 1).to(tl.bfloat16)
    probability = (exponential / denominator[:, None]).to(tl.bfloat16)
    result = tl.dot(probability, v, input_precision='ieee').to(tl.bfloat16)
    tl.store(Out + r * 32 + c, result)


@triton.jit
def tile_transpose(X, Out):
    """Standalone typed transpose; captures separately with FP8/BF16 pointers."""
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    x = tl.load(X + r * 32 + c)
    result = tl.trans(x)
    tl.store(Out + r * 32 + c, result)


@triton.jit
def fp8_transposed_matmul(X, W, Out):
    """Exercise source transpose as an actual contraction SSA dependency."""
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    x = tl.load(X + r * 32 + c)
    w = tl.load(W + r * 32 + c)
    result = tl.dot(tl.trans(x), w, input_precision='ieee').to(tl.bfloat16)
    tl.store(Out + r * 32 + c, result)


@triton.jit
def fp8_transpose_fanout(X, Direct, Transposed):
    """Keep the original producer live across a transpose and two stores."""
    r = tl.arange(0, 32)[:, None]
    c = tl.arange(0, 32)[None, :]
    x = tl.load(X + r * 32 + c)
    transposed = tl.trans(x)
    tl.store(Direct + r * 32 + c, x)
    tl.store(Transposed + r * 32 + c, transposed)
