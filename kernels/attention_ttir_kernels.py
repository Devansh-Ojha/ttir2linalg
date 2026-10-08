"""Explicit Triton lowering kernels for compiler-only attention artifacts."""
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    SEQUENCE_LENGTH: tl.constexpr,
    EMBED_SIZE: tl.constexpr,
):
    query_id = tl.program_id(0)
    dims = tl.arange(0, EMBED_SIZE)
    keys = tl.arange(0, SEQUENCE_LENGTH)
    query = tl.load(q_ptr + query_id * EMBED_SIZE + dims)
    key = tl.load(
        k_ptr + keys[:, None] * EMBED_SIZE + dims[None, :]
    )
    scores = tl.sum(query[None, :] * key, axis=1)
    scores = scores * (1.0 / tl.sqrt(EMBED_SIZE))
    weights = tl.softmax(scores)
    values = tl.load(
        v_ptr + keys[:, None] * EMBED_SIZE + dims[None, :]
    )
    result = tl.sum(weights[:, None] * values, axis=0)
    tl.store(output_ptr + query_id * EMBED_SIZE + dims, result)
