"""Convert a recognized TTIR matmul+ReLU tile to Atlas virtual SSA.

The input slot bindings are part of the runtime ABI: the caller stages the
exact TTIR matmul operand tiles into the selected Atlas FP8 slots. Address
generation and data movement from arbitrary Triton pointer arithmetic are not
lowered by this first bounded conversion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from compiler.atlas_mapping import AtlasTileABI, map_matmul_relu
from compiler.triton_analysis import analyze_matmul_relu
from compiler.ttir_reader import parse_ttir


def convert_file(source: Path, abi: AtlasTileABI = AtlasTileABI()) -> str:
    parsed = parse_ttir(source.read_text())
    if len(parsed.funcs) != 1:
        raise NotImplementedError("Atlas tile conversion expects one TTIR function")
    plan = analyze_matmul_relu(parsed.funcs[0])
    return map_matmul_relu(plan, abi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="TTIR file with one 32x32 FP8 matmul+ReLU")
    parser.add_argument("--out", type=Path, required=True, help="Atlas virtual SSA output MLIR")
    parser.add_argument("--lhs-slot", type=int, default=0)
    parser.add_argument("--rhs-slot", type=int, default=1)
    parser.add_argument("--mxu-unit", type=int, default=0, choices=(0, 1))
    args = parser.parse_args()
    abi = AtlasTileABI(lhs_input=args.lhs_slot, rhs_input=args.rhs_slot,
                       mxu_unit=args.mxu_unit)
    result = convert_file(args.input, abi)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result)
    print(f"Wrote Atlas virtual SSA to {args.out}")


if __name__ == "__main__":
    main()
