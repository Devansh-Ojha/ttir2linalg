"""A small pre-norm-style Transformer encoder block for compiler capture."""
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Self-attention and a two-layer feed-forward network with residuals."""

    def __init__(self, embed_size: int = 8, num_heads: int = 2,
                 hidden_size: int = 16):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=num_heads, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(embed_size)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, embed_size)
        self.mlp_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.attention_norm(x + attended)
        feed_forward = self.fc2(self.activation(self.fc1(x)))
        return self.mlp_norm(x + feed_forward)
