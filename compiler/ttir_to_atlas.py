"""Convert bounded TTIR matmul/ReLU or explicitly selected two-layer MLP tiles.

The input slot bindings are part of the runtime ABI: the caller stages the
exact TTIR matmul operand tiles into the selected Atlas FP8 slots. Address
generation and data movement from arbitrary Triton pointer arithmetic are not
lowered by this first bounded conversion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from compiler.atlas_mapping import AtlasTileABI, map_matmul_relu, AtlasMlpABI, map_two_layer_mlp
from compiler.triton_analysis import analyze_matmul_relu, analyze_two_layer_mlp
from compiler.ttir_reader import parse_ttir


def convert_file(source: Path, abi: AtlasTileABI = AtlasTileABI(), *,
                 pattern: str = "matmul-relu", numerical_policy: str | None = None,
                 mlp_abi: AtlasMlpABI = AtlasMlpABI()) -> str:
    parsed = parse_ttir(source.read_text())
    if len(parsed.funcs) != 1:
        raise NotImplementedError("Atlas tile conversion expects one TTIR function")
    if pattern == "two-layer-mlp":
        functions = [op for op in parsed.all_ops if op.name == "tt.func"]
        if len(functions[0].regions) != 1 or len(functions[0].regions[0]) != 1:
            raise NotImplementedError("two-layer MLP requires one straight-line block")
        return map_two_layer_mlp(analyze_two_layer_mlp(parsed.funcs[0]), mlp_abi,
                                 numerical_policy=numerical_policy)
    if pattern != "matmul-relu" or numerical_policy is not None:
        raise NotImplementedError("unsupported pattern/policy combination")
    plan = analyze_matmul_relu(parsed.funcs[0])
    return map_matmul_relu(plan, abi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="TTIR file with one 32x32 FP8 matmul+ReLU")
    parser.add_argument("--out", type=Path, required=True, help="Atlas virtual SSA output MLIR")
    parser.add_argument("--lhs-slot", type=int, default=0)
    parser.add_argument("--rhs-slot", type=int, default=1)
    parser.add_argument("--mxu-unit", type=int, default=0, choices=(0, 1))
    parser.add_argument("--pattern", choices=("matmul-relu", "two-layer-mlp"), default="matmul-relu")
    parser.add_argument("--numerical-policy", help="two-layer MLP requires atlas-fp8-mlp-v1")
    parser.add_argument("--second-rhs-slot", type=int, default=2)
    args = parser.parse_args()
    if args.pattern == "two-layer-mlp" and args.mxu_unit != 0:
        parser.error("two-layer MLP fixes MXU units to 0 then 1")
    abi = AtlasTileABI(lhs_input=args.lhs_slot, rhs_input=args.rhs_slot,
                       mxu_unit=args.mxu_unit)
    result = convert_file(args.input, abi, pattern=args.pattern, numerical_policy=args.numerical_policy,
                          mlp_abi=AtlasMlpABI(input_slots=(args.lhs_slot, args.rhs_slot, args.second_rhs_slot)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result)
    print(f"Wrote Atlas virtual SSA to {args.out}")


if __name__ == "__main__":
    main()
