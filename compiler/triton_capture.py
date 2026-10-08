"""Optional Triton TTIR capture without launching a kernel.

This module keeps Triton imports lazy so the CPU-only FX/HGIR path works in
environments that do not install Triton.  Compilation produces TTIR artifacts;
it does not allocate device tensors or execute a kernel.
"""
from __future__ import annotations


def _compile_kernel(kernel, input_size: int, output_size: int) -> str:
    try:
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource, compile as triton_compile
    except ImportError as exc:
        raise RuntimeError(
            "Triton is not installed; use CPU FX capture or run this mode in "
            "the Triton environment"
        ) from exc

    source = ASTSource(
        kernel,
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


def compile_linear_ttir(input_size: int = 4, output_size: int = 8) -> str:
    """Compile the existing linear kernel and return genuine TTIR text."""
    from kernels.mlp_kernel import linear_kernel

    return _compile_kernel(linear_kernel, input_size, output_size)


def compile_linear_relu_ttir(input_size: int = 4, output_size: int = 8) -> str:
    """Compile a fused linear+ReLU kernel without launching it."""
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError(
            "Triton 3.4.0 is required for direct TTIR capture"
        ) from exc

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

    return _compile_kernel(linear_relu_kernel, input_size, output_size)


def write_verified_ttir(
    path,
    input_size: int = 4,
    output_size: int = 8,
    fused_relu: bool = True,
) -> str:
    """Write compiler-produced TTIR and verify it with libtriton.ir."""
    from pathlib import Path

    output = Path(path)
    text = (
        compile_linear_relu_ttir(input_size, output_size)
        if fused_relu
        else compile_linear_ttir(input_size, output_size)
    )
    output.write_text(text)
    from compiler.ttir_reader import parse_ttir

    parsed = parse_ttir(text)
    if not parsed.mod.verify():
        raise RuntimeError("libtriton.ir rejected the generated TTIR")
    return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="mlp_linear_relu.ttir")
    parser.add_argument("--input-size", type=int, default=4)
    parser.add_argument("--output-size", type=int, default=8)
    args = parser.parse_args()
    text = write_verified_ttir(args.out, args.input_size, args.output_size)
    print("wrote verified TTIR: %s (%d bytes)" % (args.out, len(text)))
