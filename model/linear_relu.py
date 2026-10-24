"""Single-layer projection followed by ReLU, for the compiler baseline."""
import torch.nn as nn


class LinearRelu(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.proj = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.proj(x))
