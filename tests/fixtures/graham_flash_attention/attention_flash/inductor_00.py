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
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.runtime_utils import assert_tensor_metadata

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


# kernel path: /tmp/torchinductor_grammy/5z/c5zxbr65tmj2bf4qs4qn5ogavfwyuihixhrzr2rue34q4ig57whh.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention]
# Source node to ATen node mapping:
#   multi_head_attention_forward => _scaled_dot_product_flash_attention, clone, permute_2, permute_3, permute_4, permute_5, select, select_1, select_2, squeeze, unsqueeze, view_1, view_2, view_6, view_7, view_8
# Graph fragment:
#   %addmm : Tensor "f16[32, 96][96, 1]cuda:0" = PlaceHolder[target=addmm]
#   %view_1 : Tensor "f16[32, 1, 96][96, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [32, 1, 96]), kwargs = {})
#   %view_2 : Tensor "f16[32, 1, 3, 32][96, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [32, 1, 3, 32]), kwargs = {})
#   %unsqueeze : Tensor "f16[1, 32, 1, 3, 32][3072, 96, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 0), kwargs = {})
#   %permute_2 : Tensor "f16[3, 32, 1, 1, 32][32, 96, 96, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f16[3, 32, 1, 32][32, 96, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_2, -2), kwargs = {})
#   %clone : Tensor "f16[3, 32, 1, 32][1024, 32, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f16[32, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone, 0, 0), kwargs = {})
#   %permute_3 : Tensor "f16[1, 32, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select, [1, 0, 2]), kwargs = {})
#   %view_6 : Tensor "f16[1, 1, 32, 32][32, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_3, [1, 1, 32, 32]), kwargs = {})
#   %select_1 : Tensor "f16[32, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone, 0, 1), kwargs = {})
#   %permute_4 : Tensor "f16[1, 32, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select_1, [1, 0, 2]), kwargs = {})
#   %view_7 : Tensor "f16[1, 1, 32, 32][32, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_4, [1, 1, 32, 32]), kwargs = {})
#   %select_2 : Tensor "f16[32, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone, 0, 2), kwargs = {})
#   %permute_5 : Tensor "f16[1, 32, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select_2, [1, 0, 2]), kwargs = {})
#   %view_8 : Tensor "f16[1, 1, 32, 32][32, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_5, [1, 1, 32, 32]), kwargs = {})
#   %_scaled_dot_product_flash_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention.default](args = (%view_6, %view_7, %view_8), kwargs = {scale: 0.17677669529663687})
#   return %buf1
triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_select_squeeze_transpose_unsqueeze_view_0 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_select_squeeze_transpose_unsqueeze_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_select_squeeze_transpose_unsqueeze_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 6144}, 'backend_hash': '99112F46E34FFE0C7B9EA02CA7CCE3EA132FFF264769496D789923144AFED6EF', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_select_squeeze_transpose_unsqueeze_view_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 96*x1), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_grammy/f6/cf6p5ytndjd3wfes7ezx6actfukc6gqfrcoi7kc756jylhgtxwih.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention]
# Source node to ATen node mapping:
#   multi_head_attention_forward => _scaled_dot_product_flash_attention, clone, permute_2, permute_3, permute_4, permute_5, select, select_1, select_2, squeeze, unsqueeze, view_1, view_2, view_6, view_7, view_8
# Graph fragment:
#   %addmm : Tensor "f16[32, 96][96, 1]cuda:0" = PlaceHolder[target=addmm]
#   %view_1 : Tensor "f16[32, 1, 96][96, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [32, 1, 96]), kwargs = {})
#   %view_2 : Tensor "f16[32, 1, 3, 32][96, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [32, 1, 3, 32]), kwargs = {})
#   %unsqueeze : Tensor "f16[1, 32, 1, 3, 32][3072, 96, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 0), kwargs = {})
#   %permute_2 : Tensor "f16[3, 32, 1, 1, 32][32, 96, 96, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f16[3, 32, 1, 32][32, 96, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_2, -2), kwargs = {})
#   %clone : Tensor "f16[3, 32, 1, 32][1024, 32, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f16[32, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone, 0, 0), kwargs = {})
#   %permute_3 : Tensor "f16[1, 32, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select, [1, 0, 2]), kwargs = {})
#   %view_6 : Tensor "f16[1, 1, 32, 32][32, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_3, [1, 1, 32, 32]), kwargs = {})
#   %select_1 : Tensor "f16[32, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone, 0, 1), kwargs = {})
#   %permute_4 : Tensor "f16[1, 32, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select_1, [1, 0, 2]), kwargs = {})
#   %view_7 : Tensor "f16[1, 1, 32, 32][32, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_4, [1, 1, 32, 32]), kwargs = {})
#   %select_2 : Tensor "f16[32, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone, 0, 2), kwargs = {})
#   %permute_5 : Tensor "f16[1, 32, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select_2, [1, 0, 2]), kwargs = {})
#   %view_8 : Tensor "f16[1, 1, 32, 32][32, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_5, [1, 1, 32, 32]), kwargs = {})
#   %_scaled_dot_product_flash_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention.default](args = (%view_6, %view_7, %view_8), kwargs = {scale: 0.17677669529663687})
#   return %buf2
triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_select_squeeze_transpose_unsqueeze_view_1 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_select_squeeze_transpose_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_select_squeeze_transpose_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 6144}, 'backend_hash': '99112F46E34FFE0C7B9EA02CA7CCE3EA132FFF264769496D789923144AFED6EF', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_select_squeeze_transpose_unsqueeze_view_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (32 + x0 + 96*x1), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_grammy/3o/c3oamjbwf4kqj3vkx5fefdoteaadl2vlvj2poa3iroqixzkmmqcf.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention]
# Source node to ATen node mapping:
#   multi_head_attention_forward => _scaled_dot_product_flash_attention, clone, permute_2, permute_3, permute_4, permute_5, select, select_1, select_2, squeeze, unsqueeze, view_1, view_2, view_6, view_7, view_8
# Graph fragment:
#   %addmm : Tensor "f16[32, 96][96, 1]cuda:0" = PlaceHolder[target=addmm]
#   %view_1 : Tensor "f16[32, 1, 96][96, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [32, 1, 96]), kwargs = {})
#   %view_2 : Tensor "f16[32, 1, 3, 32][96, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [32, 1, 3, 32]), kwargs = {})
#   %unsqueeze : Tensor "f16[1, 32, 1, 3, 32][3072, 96, 96, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 0), kwargs = {})
#   %permute_2 : Tensor "f16[3, 32, 1, 1, 32][32, 96, 96, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f16[3, 32, 1, 32][32, 96, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_2, -2), kwargs = {})
#   %clone : Tensor "f16[3, 32, 1, 32][1024, 32, 32, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f16[32, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone, 0, 0), kwargs = {})
#   %permute_3 : Tensor "f16[1, 32, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select, [1, 0, 2]), kwargs = {})
#   %view_6 : Tensor "f16[1, 1, 32, 32][32, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_3, [1, 1, 32, 32]), kwargs = {})
#   %select_1 : Tensor "f16[32, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone, 0, 1), kwargs = {})
#   %permute_4 : Tensor "f16[1, 32, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select_1, [1, 0, 2]), kwargs = {})
#   %view_7 : Tensor "f16[1, 1, 32, 32][32, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_4, [1, 1, 32, 32]), kwargs = {})
#   %select_2 : Tensor "f16[32, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone, 0, 2), kwargs = {})
#   %permute_5 : Tensor "f16[1, 32, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%select_2, [1, 0, 2]), kwargs = {})
#   %view_8 : Tensor "f16[1, 1, 32, 32][32, 1024, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_5, [1, 1, 32, 32]), kwargs = {})
#   %_scaled_dot_product_flash_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention.default](args = (%view_6, %view_7, %view_8), kwargs = {scale: 0.17677669529663687})
#   return %buf3
triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_select_squeeze_transpose_unsqueeze_view_2 = async_compile.triton('triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_select_squeeze_transpose_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=26, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_select_squeeze_transpose_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 6144}, 'backend_hash': '99112F46E34FFE0C7B9EA02CA7CCE3EA132FFF264769496D789923144AFED6EF', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_select_squeeze_transpose_unsqueeze_view_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (64 + x0 + 96*x1), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
        args.clear()
        assert_size_stride_grouped((arg1_1, arg0_1, arg2_1), ((96, ), (1, 32, 32), (96, 32)), ((1, ), (1024, 32, 1), (32, 1)), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_cuda((32, 96), (96, 1), torch.float16)
            # Topologically Sorted Source Nodes: [query, multi_head_attention_forward], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg1_1, reinterpret_tensor(arg0_1, (32, 32), (32, 1), 0), reinterpret_tensor(arg2_1, (32, 96), (1, 32), 0), alpha=1, beta=1, out=buf0)
            del arg0_1
            del arg1_1
            del arg2_1
            buf1 = empty_strided_cuda((1, 1, 32, 32), (1024, 1024, 32, 1), torch.float16)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_select_squeeze_transpose_unsqueeze_view_0.run(buf0, buf1, 1024, stream=raw_stream0)
            buf2 = empty_strided_cuda((1, 1, 32, 32), (1024, 1024, 32, 1), torch.float16)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_select_squeeze_transpose_unsqueeze_view_1.run(buf0, buf2, 1024, stream=raw_stream0)
            buf3 = empty_strided_cuda((1, 1, 32, 32), (1024, 1024, 32, 1), torch.float16)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_flash_attention__unsafe_view_clone_select_squeeze_transpose_unsqueeze_view_2.run(buf0, buf3, 1024, stream=raw_stream0)
            del buf0
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten._scaled_dot_product_flash_attention]
            buf4 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1, buf2, buf3, scale=0.17677669529663687)
            del buf1
            del buf2
            buf5 = buf4[0]
            assert_tensor_metadata(buf5, (1, 1, 32, 32), (1024, 1024, 32, 1), torch.float16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            assert_alignment(buf5, 16, 'torch.ops.aten._scaled_dot_product_flash_attention.default')
            del buf4
            assert_size_stride_grouped((arg4_1, arg3_1), ((32, ), (32, 32)), ((1, ), (32, 1)), 'input')
            buf10 = reinterpret_tensor(buf3, (32, 32), (32, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.permute, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg4_1, reinterpret_tensor(buf5, (32, 32), (32, 1), 0), reinterpret_tensor(arg3_1, (32, 32), (1, 32), 0), alpha=1, beta=1, out=buf10)
            del arg3_1
            del arg4_1
            del buf5
        return (reinterpret_tensor(buf10, (1, 32, 32), (32, 32, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((96, 32), (32, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((32, 32), (32, 1), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
