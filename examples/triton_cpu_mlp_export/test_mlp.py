
import torch
import triton

from mlp_kernel import fused_mlp2_kernel


def main():
    torch.manual_seed(0)

    # Match the dimensions used by the LLVM IR export
    M, K, H1, H2 = 16, 16, 32, 32
    BLOCK_M = 16

    # Create random inputs and weights
    x = torch.randn(M, K)
    w1 = torch.randn(H1, K)
    b1 = torch.randn(H1)
    w2 = torch.randn(H2, H1)
    b2 = torch.randn(H2)

    output = torch.empty(M, H2)

    # Run the fused MLP using Triton CPU
    fused_mlp2_kernel[(triton.cdiv(M, BLOCK_M),)](
        x, w1, b1, w2, b2, output,
        M, K, H1, H2, BLOCK_M,
    )

    # Compute the same result using PyTorch
    expected = torch.relu(
        torch.relu(x @ w1.T + b1) @ w2.T + b2
    )

    # Compare results
    print("Triton output:", output[0, :8])
    print("PyTorch output:", expected[0, :8])

    print(
        "Correct:",
        torch.allclose(output, expected, atol=1e-4, rtol=1e-4)
    )
    print("Max error:", (output - expected).abs().max().item())


if __name__ == "__main__":
    main()
