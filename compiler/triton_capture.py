"""Optional Triton TTIR capture without launching a kernel.

This module keeps Triton imports lazy so the CPU-only FX/HGIR path works in
environments that do not install Triton.  Compilation produces TTIR artifacts;
it does not allocate device tensors or execute a kernel.
"""
from __future__ import annotations

import re
import ast
from pathlib import Path


def _triton_sources_from_inductor_code(source_codes):
    sources = []
    for code in source_codes:
        for match in re.finditer(r"'''(.*?)'''", code, re.DOTALL):
            source = match.group(1)
            if "@triton.jit" in source or "triton_" in source:
                sources.append(source.strip() + "\n")
    if not sources:
        raise RuntimeError(
            "Inductor generated code, but no Triton kernel source was found; "
            "the selected backend may not be Triton"
        )
    return sources


def capture_model_triton_source(
    model_name="mlp", input_size=4, hidden_size=8, output_size=8,
    sequence_length=4, embed_size=8, num_heads=2,
):
    """Return Triton source emitted by Inductor without compiling or launching."""
    try:
        from torch._inductor.utils import get_code
    except ImportError as exc:
        raise RuntimeError("this PyTorch build has no Inductor source hook") from exc

    import torch
    if not torch.cuda.is_available():
        return _explicit_model_triton_sources(model_name)
    if model_name == "mlp":
        from model.mlp import SimpleMLP

        model = SimpleMLP(input_size, hidden_size, output_size).eval().cuda()
        sample = torch.randn(1, input_size, device="cuda")
    elif model_name == "attention":
        from model.attention import SimpleAttention

        model = SimpleAttention(embed_size, num_heads).eval().cuda()
        sample = torch.randn(
            1, sequence_length, embed_size, device="cuda"
        )
    else:
        raise ValueError("unsupported model %r" % model_name)

    compiled = torch.compile(model, backend="inductor")
    source_codes = get_code(compiled, sample)
    return _triton_sources_from_inductor_code(source_codes)


def _explicit_model_triton_sources(model_name):
    """Return source from the project's explicit Triton lowering path.

    This is intentionally not labeled as Inductor output. The kernel source is
    authored as a target-independent model lowering and then can be compiled
    by Triton to obtain genuine TTIR.
    """
    kernel_files = {
        "mlp": (
            Path(__file__).parent.parent / "kernels" / "mlp_ttir_kernels.py",
            ("linear_relu_kernel", "linear_kernel"),
        ),
        "attention": (
            Path(__file__).parent.parent / "kernels" / "attention_ttir_kernels.py",
            ("attention_kernel",),
        ),
    }
    try:
        source_path, names = kernel_files[model_name]
    except KeyError as exc:
        raise ValueError("unsupported model %r" % model_name) from exc
    return _read_kernel_sources(source_path, names)


def _read_kernel_sources(path: Path, names: tuple[str, ...]) -> list[str]:
    """Extract decorated Triton functions directly from their source file."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    lines = source.splitlines(keepends=True)
    result = []
    for name in names:
        try:
            node = functions[name]
        except KeyError as exc:
            raise RuntimeError(
                "kernel %s is missing from %s" % (name, path)
            ) from exc
        start = min(
            [decorator.lineno for decorator in node.decorator_list]
            + [node.lineno]
        ) - 1
        result.append("".join(lines[start:node.end_lineno]).strip() + "\n")
    return result


def capture_mlp_triton_source(input_size=4, hidden_size=8, output_size=8):
    return capture_model_triton_source(
        "mlp", input_size, hidden_size, output_size
    )


def write_model_triton_source(
    path, model_name="mlp", input_size=4, hidden_size=8, output_size=8,
    sequence_length=4, embed_size=8, num_heads=2,
):
    """Write one file per Triton kernel emitted by Inductor."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = capture_model_triton_source(
        model_name, input_size, hidden_size, output_size,
        sequence_length, embed_size, num_heads,
    )
    files = []
    for index, source in enumerate(sources):
        match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)", source)
        name = match.group(1) if match else "kernel_%d" % index
        target = output_dir / ("kernel_%d_%s.py" % (index, name))
        target.write_text(source)
        files.append(target)
    return files


def model_triton_source_origin():
    import torch

    return "inductor" if torch.cuda.is_available() else "explicit-triton-lowering"


def write_mlp_triton_source(path, input_size=4, hidden_size=8, output_size=8):
    return write_model_triton_source(
        path, "mlp", input_size, hidden_size, output_size
    )


def _compile_kernel(
    kernel, input_size: int | None = None, output_size: int | None = None,
    signature=None, constexprs=None,
) -> str:
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
        signature=signature or {
            "x_ptr": "*fp32", "weight_ptr": "*fp32",
            "bias_ptr": "*fp32", "output_ptr": "*fp32",
        },
        constexprs=constexprs or {
            "INPUT_SIZE": input_size, "OUTPUT_SIZE": output_size,
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


def _compile_kernel_family(kernels):
    artifacts = []
    for name, kernel, input_size, output_size in kernels:
        artifacts.append((
            name,
            _compile_kernel(kernel, input_size, output_size),
        ))
    return artifacts


def compile_linear_ttir(input_size: int = 4, output_size: int = 8) -> str:
    """Compile the existing linear kernel and return genuine TTIR text."""
    from kernels.mlp_kernel import linear_kernel

    return _compile_kernel(linear_kernel, input_size, output_size)


def compile_mlp_ttir(input_size: int = 4, hidden_size: int = 8,
                     output_size: int = 8):
    """Compile the two MLP stages into genuine TTIR artifacts.

    The first kernel computes FC1 followed by ReLU; the second computes FC2.
    Compilation only produces compiler artifacts and never launches either
    kernel.
    """
    try:
        from kernels.mlp_ttir_kernels import linear_kernel, linear_relu_kernel
    except ImportError as exc:
        raise RuntimeError(
            "Triton 3.4.0 is required for direct MLP TTIR capture"
        ) from exc

    return _compile_kernel_family(
        [
            ("fc1_relu", linear_relu_kernel, input_size, hidden_size),
            ("fc2", linear_kernel, hidden_size, output_size),
        ]
    )


def compile_attention_ttir(sequence_length=4, embed_size=8):
    try:
        from kernels.attention_ttir_kernels import attention_kernel
    except ImportError as exc:
        raise RuntimeError(
            "Triton 3.4.0 is required for direct attention TTIR capture"
        ) from exc
    return [(
        "attention",
        _compile_kernel(
            attention_kernel,
            signature={
                "q_ptr": "*fp32", "k_ptr": "*fp32",
                "v_ptr": "*fp32", "output_ptr": "*fp32",
            },
            constexprs={
                "SEQUENCE_LENGTH": sequence_length,
                "EMBED_SIZE": embed_size,
            },
        ),
    )]


def write_mlp_ttir(path, input_size=4, hidden_size=8, output_size=8):
    """Write and libtriton-verify one TTIR file for each MLP stage."""
    from pathlib import Path

    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    artifacts = compile_mlp_ttir(input_size, hidden_size, output_size)
    from compiler.ttir_reader import parse_ttir

    for index, (name, text) in enumerate(artifacts):
        parsed = parse_ttir(text)
        if not parsed.mod.verify():
            raise RuntimeError("libtriton.ir rejected TTIR for %s" % name)
        artifact = output_dir / ("kernel_%d_%s.ttir" % (index, name))
        artifact.write_text(text)
        files.append(artifact)
    return files


def write_model_ttir(
    path, model_name="mlp", input_size=4, hidden_size=8, output_size=8,
    sequence_length=4, embed_size=8,
):
    if model_name == "mlp":
        artifacts = compile_mlp_ttir(input_size, hidden_size, output_size)
    elif model_name == "attention":
        artifacts = compile_attention_ttir(sequence_length, embed_size)
    else:
        raise ValueError("unsupported model %r" % model_name)
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    from compiler.ttir_reader import parse_ttir

    files = []
    for index, (name, text) in enumerate(artifacts):
        parsed = parse_ttir(text)
        if not parsed.mod.verify():
            raise RuntimeError("libtriton.ir rejected TTIR for %s" % name)
        artifact = output_dir / ("kernel_%d_%s.ttir" % (index, name))
        artifact.write_text(text)
        files.append(artifact)
    return files


def compile_linear_relu_ttir(input_size: int = 4, output_size: int = 8) -> str:
    """Compile the fused FC1+ReLU kernel without launching it."""
    try:
        from kernels.mlp_ttir_kernels import linear_relu_kernel
    except ImportError as exc:
        raise RuntimeError(
            "Triton 3.4.0 is required for direct TTIR capture"
        ) from exc

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
    parser.add_argument("--mlp-dir", default=None)
    parser.add_argument("--input-size", type=int, default=4)
    parser.add_argument("--output-size", type=int, default=8)
    args = parser.parse_args()
    if args.mlp_dir:
        files = write_mlp_ttir(
            args.mlp_dir,
            input_size=args.input_size,
            hidden_size=args.output_size,
            output_size=args.output_size,
        )
        print("wrote verified TTIR files: %s" % ", ".join(map(str, files)))
    else:
        text = write_verified_ttir(args.out, args.input_size, args.output_size)
        print("wrote verified TTIR: %s (%d bytes)" % (args.out, len(text)))
