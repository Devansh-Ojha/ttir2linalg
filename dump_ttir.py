import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

from kernels.mlp_kernel import linear_kernel

signature = {
    "x_ptr": "*fp32",
    "weight_ptr": "*fp32",
    "bias_ptr": "*fp32",
    "output_ptr": "*fp32",
    "INPUT_SIZE": "i32",
    "OUTPUT_SIZE": "i32",
}

constexprs = {
    "INPUT_SIZE": 4,
    "OUTPUT_SIZE": 8,
}

src = ASTSource(
    linear_kernel,
    signature=signature,
    constexprs=constexprs,
)

target = GPUTarget("cuda", 80, 32)

compiled = triton.compile(
    src,
    target=target,
)

print(compiled.asm["ttir"])