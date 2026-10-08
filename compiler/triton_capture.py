"""Optional Triton TTIR capture without launching a kernel.

This module keeps Triton imports lazy so the CPU-only FX/HGIR path works in
environments that do not install Triton.  Compilation produces TTIR artifacts;
it does not allocate device tensors or execute a kernel.
"""
from __future__ import annotations


def compile_linear_ttir(input_size: int = 4, output_size: int = 8) -> str:
    try:
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource, compile as triton_compile
    except ImportError as exc:
        raise RuntimeError(
            "Triton is not installed; use CPU FX capture or run this mode in "
            "the Triton environment"
        ) from exc

    from kernels.mlp_kernel import linear_kernel

    source = ASTSource(
        linear_kernel,
        signature={
            "x_ptr": "*fp32",
            "weight_ptr": "*fp32",
            "bias_ptr": "*fp32",
            "output_ptr": "*fp32",
        },
        constexprs={
            "INPUT_SIZE": input_size,
            "OUTPUT_SIZE": output_size,
        },
    )
    compiled = triton_compile(
        source,
        target=GPUTarget("cuda", 80, 32),
        options={"num_warps": 1, "num_ctas": 1},
    )
    try:
        return compiled.asm["ttir"]
    except KeyError as exc:
        available = ", ".join(sorted(compiled.asm))
        raise RuntimeError(
            "Triton compilation succeeded but did not produce TTIR; "
            "available artifacts: %s" % available
        ) from exc
