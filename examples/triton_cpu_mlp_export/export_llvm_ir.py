"""Compile the standalone fused MLP Triton kernel for Triton CPU and save LLVM IR."""
from __future__ import annotations

import argparse
from pathlib import Path

from mlp_kernel import fused_mlp2_kernel


def compile_kernel(input_size: int, hidden_size: int, block_m: int) -> str:
    try:
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource, compile as triton_compile
        from triton._C.libtriton import cpu
    except ImportError as exc:
        raise RuntimeError(
            "Install/build Triton from the triton-cpu checkout before exporting LLVM IR."
        ) from exc

    arch = cpu.llvm.get_cpu_triple().split("-")[0]
    source = ASTSource(
        fused_mlp2_kernel,
        signature={
            "x_ptr": "*fp32",
            "w1_ptr": "*fp32",
            "b1_ptr": "*fp32",
            "w2_ptr": "*fp32",
            "b2_ptr": "*fp32",
            "out_ptr": "*fp32",
            "M": "i32",
        },
        constexprs={
            "K": input_size,
            "H1": hidden_size,
            "H2": hidden_size,
            "BLOCK_M": block_m,
        },
    )
    compiled = triton_compile(
        source,
        target=GPUTarget("cpu", arch, 0),
        options={"num_warps": 0, "num_stages": 0, "num_ctas": 0},
    )
    try:
        return compiled.asm["llir"]
    except KeyError as exc:
        raise RuntimeError(
            "Triton CPU compilation produced no LLVM IR; available artifacts: "
            + ", ".join(sorted(compiled.asm))
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("out/mlp2.llir"))
    args = parser.parse_args()

    llir = compile_kernel(args.input_size, args.hidden_size, args.block_m)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(llir)
    print(f"Wrote Triton CPU LLVM IR to {args.out}")


if __name__ == "__main__":
    main()
