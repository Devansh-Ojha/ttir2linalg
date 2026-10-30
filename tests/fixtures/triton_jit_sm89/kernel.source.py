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
