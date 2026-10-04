import torch
import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    INPUT_SIZE: tl.constexpr,
    OUTPUT_SIZE: tl.constexpr,
):
    output_idx = tl.program_id(0)

    input_idx = tl.arange(0, INPUT_SIZE)

    x = tl.load(x_ptr + input_idx)

    weight = tl.load(
        weight_ptr + output_idx * INPUT_SIZE + input_idx
    )

    bias = tl.load(bias_ptr + output_idx)

    result = tl.sum(x * weight) + bias

    tl.store(output_ptr + output_idx, result)


def linear(x, weight, bias):
    output_size = weight.shape[0]

    output = torch.empty(
        (output_size,),
        device=x.device,
        dtype=x.dtype,
    )

    linear_kernel[(output_size,)](
        x,
        weight,
        bias,
        output,
        INPUT_SIZE=x.shape[0],
        OUTPUT_SIZE=output_size,
    )

    return output
if __name__ == "__main__":
    from model.mlp import SimpleMLP

    torch.manual_seed(0)

    model = SimpleMLP(4, 8, 8)

    x = torch.randn(4)

    pytorch_output = model.fc1(x)

    triton_output = linear(
        x,
        model.fc1.weight,
        model.fc1.bias,
    )

    print("PyTorch:")
    print(pytorch_output)

    print("\nTriton:")
    print(triton_output)

    print("\nCorrect:")
    print(
        torch.allclose(
            pytorch_output,
            triton_output,
            atol=1e-5,
            rtol=1e-5,
        )
    )