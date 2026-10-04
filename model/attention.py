import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.attention(x, x, x)
        return out


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