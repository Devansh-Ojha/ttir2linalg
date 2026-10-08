@triton.jit
def linear_relu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    INPUT_SIZE: tl.constexpr,
    OUTPUT_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, INPUT_SIZE)
    x = tl.load(x_ptr + offsets)
    weight = tl.load(weight_ptr + pid * INPUT_SIZE + offsets)
    bias = tl.load(bias_ptr + pid)
    value = tl.sum(x * weight) + bias
    value = tl.maximum(value, 0.0)
    tl.store(output_ptr + pid, value)
