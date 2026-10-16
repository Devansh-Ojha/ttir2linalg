import torch
import triton
import triton.language as tl
from pathlib import Path

triton.runtime.driver.set_active_to_cpu()


@triton.jit
def matmul_relu_kernel(A, B, C):
    rows = tl.arange(0, 32)
    cols = tl.arange(0, 32)

    a = tl.load(A + rows[:, None] * 32 + cols[None, :])
    b = tl.load(B + rows[:, None] * 32 + cols[None, :])

    acc = tl.full((32, 32), 0.0, tl.float32)

    result = tl.dot(a, b, acc)
    result = tl.maximum(result, 0.0)

    tl.store(C + rows[:, None] * 32 + cols[None, :], result)


a = torch.randn(32, 32).to(torch.float8_e4m3fn)
b = torch.randn(32, 32).to(torch.float8_e4m3fn)
c = torch.empty((32, 32), dtype=torch.float32)

kernel = matmul_relu_kernel[(1,)](a, b, c, num_warps=4)

ttir = kernel.asm["ttir"]
Path("matmul_relu_generated.ttir").write_text(ttir)

print("Generated TTIR from Triton")
print(ttir)
