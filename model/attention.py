import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):
        # need_weights=False lets MultiheadAttention route through PyTorch's
        # scaled_dot_product_attention implementation. Inductor can then select
        # an available fused/FlashAttention backend for supported devices and
        # shapes, while PyTorch retains its normal fallback behavior.
        out, _ = self.attention(x, x, x, need_weights=False)
        return out

    def reference_forward(self, x):
        """Evaluate the same learned attention with PyTorch's math SDPA path."""
        layer = self.attention
        q, k, v = F.linear(x, layer.in_proj_weight, layer.in_proj_bias).chunk(3, dim=-1)
        batch, seq, _ = q.shape
        head_dim = self.embed_size // self.num_heads
        def heads(t):
            return t.view(batch, seq, self.num_heads, head_dim).transpose(1, 2)
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.MATH):
                attended = F.scaled_dot_product_attention(
                    heads(q), heads(k), heads(v), dropout_p=0.0
                )
        except ImportError:
            attended = F.scaled_dot_product_attention(
                heads(q), heads(k), heads(v), dropout_p=0.0
            )
        attended = attended.transpose(1, 2).reshape(batch, seq, self.embed_size)
        return F.linear(attended, layer.out_proj.weight, layer.out_proj.bias)


if __name__ == "__main__":
    torch.manual_seed(0)

    embed_size = 8
    num_heads = 2
    sequence_length = 4
    batch_size = 1

    model = SimpleAttention(embed_size, num_heads)

    x = torch.randn(
        batch_size,
        sequence_length,
        embed_size,
    )

    y = model(x)

    print("Input:")
    print(x)
    print("Input shape:", x.shape)

    print("\nOutput:")
    print(y)
    print("Output shape:", y.shape)
