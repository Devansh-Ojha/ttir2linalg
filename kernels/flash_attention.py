"""Small, inference-only Triton FlashAttention kernel (online softmax)."""
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    SEQ: tl.constexpr, HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr,
    stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr,
    stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute non-causal attention using tiled Q/K/V and online softmax."""
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    # Batch and head are flattened into one grid axis.
    batch = batch_head // HEADS
    head = batch_head % HEADS
    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_D)
    q = tl.load(
        q_ptr + batch * stride_qb + head * stride_qh
        + rows[:, None] * stride_qm + cols[None, :],
        (rows[:, None] < SEQ) & (cols[None, :] < HEAD_DIM), other=0.0,
    )
    running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    running_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    for start_n in range(tl.cdiv(SEQ, BLOCK_N)):
        keys = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k = tl.load(
            k_ptr + batch * stride_kb + head * stride_kh
            + keys[None, :] * stride_kn + cols[:, None],
            (keys[None, :] < SEQ) & (cols[:, None] < HEAD_DIM), other=0.0,
        )
        v = tl.load(
            v_ptr + batch * stride_vb + head * stride_vh
            + keys[:, None] * stride_vn + cols[None, :],
            (keys[:, None] < SEQ) & (cols[None, :] < HEAD_DIM), other=0.0,
        )
        scores = tl.dot(q, k) * scale
        scores = tl.where(keys[None, :] < SEQ, scores, float("-inf"))
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        alpha = tl.exp(running_max - new_max)
        probs = tl.exp(scores - new_max[:, None])
        running_sum = running_sum * alpha + tl.sum(probs, axis=1)
        running_acc = running_acc * alpha[:, None] + tl.dot(probs.to(v.dtype), v)
        running_max = new_max
    result = running_acc / running_sum[:, None]
    tl.store(
        out_ptr + batch * stride_ob + head * stride_oh
        + rows[:, None] * stride_om + cols[None, :],
        result,
        (rows[:, None] < SEQ) & (cols[None, :] < HEAD_DIM),
    )
