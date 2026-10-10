import torch
import torch.nn as nn
from model.attention import SimpleAttention
from model.mlp import SimpleMLP


class AttentionMLP(nn.Module):
    """Combined model: Attention layer followed by an MLP layer."""

    def __init__(self, embed_size=8, num_heads=2, hidden_size=8):
        super().__init__()
        self.attention = SimpleAttention(embed_size, num_heads)
        self.mlp = SimpleMLP(embed_size, hidden_size, embed_size)

    def forward(self, x):
        attn_out = self.attention(x)
        # Average over sequence dimension for feeding into MLP
        mlp_in = attn_out.mean(dim=1)
        out = self.mlp(mlp_in)
        return out
