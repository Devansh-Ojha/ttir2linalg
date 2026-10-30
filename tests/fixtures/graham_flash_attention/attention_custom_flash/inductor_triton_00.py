import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[{'num_warps': 4, 'num_stages': 3}],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'declared_constexpr_names': ['SEQ', 'HEADS', 'HEAD_DIM', 'stride_qb', 'stride_qh', 'stride_qm', 'stride_kb', 'stride_kh', 'stride_kn', 'stride_vb', 'stride_vh', 'stride_vn', 'stride_ob', 'stride_oh', 'stride_om', 'BLOCK_M', 'BLOCK_N', 'BLOCK_D'], 'kernel_name': 'flash_attention_kernel_0', 'backend_hash': '99112F46E34FFE0C7B9EA02CA7CCE3EA132FFF264769496D789923144AFED6EF', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    triton_meta={'signature': {'q_ptr': '*fp16', 'k_ptr': '*fp16', 'v_ptr': '*fp16', 'out_ptr': '*fp16', 'SEQ': 'constexpr', 'HEADS': 'constexpr', 'HEAD_DIM': 'constexpr', 'stride_qb': 'constexpr', 'stride_qh': 'constexpr', 'stride_qm': 'constexpr', 'stride_kb': 'constexpr', 'stride_kh': 'constexpr', 'stride_kn': 'constexpr', 'stride_vb': 'constexpr', 'stride_vh': 'constexpr', 'stride_vn': 'constexpr', 'stride_ob': 'constexpr', 'stride_oh': 'constexpr', 'stride_om': 'constexpr', 'BLOCK_M': 'constexpr', 'BLOCK_N': 'constexpr', 'BLOCK_D': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'SEQ': 32, 'HEADS': 1, 'HEAD_DIM': 32, 'stride_qb': 1024, 'stride_qh': 32, 'stride_qm': 32, 'stride_kb': 1024, 'stride_kh': 32, 'stride_kn': 32, 'stride_vb': 1024, 'stride_vh': 32, 'stride_vn': 32, 'stride_ob': 1024, 'stride_oh': 32, 'stride_om': 32, 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_D': 32}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
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
