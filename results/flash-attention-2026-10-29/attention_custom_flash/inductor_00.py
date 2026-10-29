# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# Original path: /home/grammy/blast/triton-shit/ttir2linalg/kernels/flash_attention.py:6
flash_attention_kernel_0 = async_compile.triton('flash_attention_kernel', '''

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
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
        args.clear()
        assert_size_stride_grouped((arg2_1, arg0_1, arg1_1), ((32, ), (1, 32, 32), (32, 32)), ((1, ), (1024, 32, 1), (32, 1)), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_cuda((32, 32), (32, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg2_1, reinterpret_tensor(arg0_1, (32, 32), (32, 1), 0), reinterpret_tensor(arg1_1, (32, 32), (1, 32), 0), alpha=1, beta=1, out=buf0)
            del arg1_1
            del arg2_1
            assert_size_stride_grouped((arg4_1, arg3_1), ((32, ), (32, 32)), ((1, ), (32, 1)), 'input')
            buf1 = empty_strided_cuda((32, 32), (32, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg4_1, reinterpret_tensor(arg0_1, (32, 32), (32, 1), 0), reinterpret_tensor(arg3_1, (32, 32), (1, 32), 0), alpha=1, beta=1, out=buf1)
            del arg3_1
            del arg4_1
            assert_size_stride_grouped((arg6_1, arg5_1), ((32, ), (32, 32)), ((1, ), (32, 1)), 'input')
            buf2 = empty_strided_cuda((32, 32), (32, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg6_1, reinterpret_tensor(arg0_1, (32, 32), (32, 1), 0), reinterpret_tensor(arg5_1, (32, 32), (1, 32), 0), alpha=1, beta=1, out=buf2)
            del arg0_1
            del arg5_1
            del arg6_1
            buf3 = empty_strided_cuda((1, 32, 1, 32), (1024, 32, 32, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear, view, transpose, linear_1, view_1, transpose_1, linear_2, view_2, transpose_2, result], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.empty_like]
            raw_stream0 = get_raw_stream(0)
            flash_attention_kernel_0.run(reinterpret_tensor(buf0, (1, 1, 32, 32), (1024, 32, 32, 1), 0), reinterpret_tensor(buf1, (1, 1, 32, 32), (1024, 32, 32, 1), 0), reinterpret_tensor(buf2, (1, 1, 32, 32), (1024, 32, 32, 1), 0), reinterpret_tensor(buf3, (1, 1, 32, 32), (1024, 32, 32, 1), 0), 2, 1, 1, stream=raw_stream0)
            del buf0
            del buf1
            assert_size_stride_grouped((arg8_1, arg7_1), ((32, ), (32, 32)), ((1, ), (32, 1)), 'input')
            buf5 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [result, linear_3], Original ATen: [aten.empty_like, aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg8_1, reinterpret_tensor(buf3, (32, 32), (32, 1), 0), reinterpret_tensor(arg7_1, (32, 32), (1, 32), 0), alpha=1, beta=1, out=buf5)
            del arg7_1
            del arg8_1
            del buf3
        return (reinterpret_tensor(buf5, (1, 32, 32), (1024, 32, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((32, 32), (32, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((32, 32), (32, 1), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((32, 32), (32, 1), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((32, 32), (32, 1), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
