"""Explicit Triton kernel definitions for combined Attention + MLP pipeline."""
import triton
import triton.language as tl
from kernels.attention_ttir_kernels import attention_kernel
from kernels.mlp_ttir_kernels import linear_kernel, linear_relu_kernel
