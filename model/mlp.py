import torch 
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

if __name__ == "__main__":
    model = SimpleMLP(
        input_size=4,
        hidden_size=8,
        output_size=8,
    )

    x = torch.randn(1, 4)

    y = model(x)

    print("Input:")
    print(x)
    print("Input shape:", x.shape)

    print("\nOutput:")
    print(y)
    print("Output shape:", y.shape)