"""Multi-head attention using this repository's Triton FlashAttention kernel."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.library import triton_op, wrap_triton

import triton

from kernels.flash_attention import flash_attention_kernel


@triton_op("ttir2linalg::custom_flash_attention", mutates_args={})
def _custom_flash_attention(q, k, v, heads: int):
    batch, _, seq, head_dim = q.shape
    result = torch.empty_like(q)
    block_m, block_n = 16, 32
    wrap_triton(flash_attention_kernel)[
        (triton.cdiv(seq, block_m), batch * heads)
    ](
        q, k, v, result,
        seq, heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        result.stride(0), result.stride(1), result.stride(2),
        block_m, block_n, triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return result


class CustomFlashAttention(nn.Module):
    """Projection layers plus a custom non-causal online-softmax Triton kernel."""

    def __init__(self, embed_size: int = 64, num_heads: int = 2):
        super().__init__()
        if embed_size % num_heads:
            raise ValueError("embed_size must be divisible by num_heads")
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)

    def _project(self, x):
        batch, seq, _ = x.shape
        shape = (batch, seq, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(shape).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(shape).transpose(1, 2).contiguous()
        return q, k, v

    def forward(self, x):
        q, k, v = self._project(x)
        batch, heads, seq, head_dim = q.shape
        result = _custom_flash_attention(q, k, v, heads)
        result = result.transpose(1, 2).contiguous().view(batch, seq, self.embed_size)
        return self.out_proj(result)

    def reference_forward(self, x):
        """Independent PyTorch math-SDPA reference for correctness comparisons."""
        q, k, v = self._project(x)
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.MATH):
                result = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        except ImportError:
            result = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        batch, _, seq, _ = result.shape
        result = result.transpose(1, 2).contiguous().view(batch, seq, self.embed_size)
        return self.out_proj(result)
