"""Convert supported TTIR workload plans to Atlas virtual SSA.

The caller stages inputs into the selected Atlas tile slots, including padding
and explicit target precision conversion. This bounded frontend recognizes
the checked-in static workload patterns; it does not lower arbitrary Triton
pointer arithmetic or perform runtime data movement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from compiler.atlas_mapping import (AtlasAttentionABI, AtlasLinearReluABI,
                                    AtlasMlpABI, AtlasTargetMapper, AtlasTileABI)
from compiler.target_mapping import MatmulReluMapper, TritonTargetMapper
from compiler.triton_analysis import (analyze_attention, analyze_linear_relu,
                                      analyze_matmul_relu,
                                      analyze_two_layer_mlp)
from compiler.ttir_model import TFunc


def _plan_for_pattern(function: TFunc, pattern: str):
    if pattern == "matmul-relu":
        return analyze_matmul_relu(function)
    if pattern == "two-layer-mlp":
        return analyze_two_layer_mlp(function)
    if pattern == "linear-relu":
        return analyze_linear_relu(function)
    if pattern == "attention":
        return analyze_attention(function)
    raise NotImplementedError(f"unsupported TTIR pattern {pattern!r}")


def convert_function(function: TFunc, mapper: MatmulReluMapper | None = None) -> str:
    """Compatibility entry point for the original matmul + ReLU pattern."""
    plan = analyze_matmul_relu(function)
    selected = mapper or AtlasTargetMapper()
    if hasattr(selected, "map_matmul_relu"):
        return selected.map_matmul_relu(plan)
    return selected.map(plan)


def convert_file(source: Path, abi: AtlasTileABI = AtlasTileABI(), *,
                 pattern: str = "matmul-relu", numerical_policy: str | None = None,
                 mlp_abi: AtlasMlpABI = AtlasMlpABI(),
                 linear_abi: AtlasLinearReluABI = AtlasLinearReluABI(),
                 attention_abi: AtlasAttentionABI = AtlasAttentionABI(),
                 target_mapper: TritonTargetMapper | None = None) -> str:
    from compiler.ttir_reader import parse_ttir

    parsed = parse_ttir(source.read_text())
    if len(parsed.funcs) != 1:
        raise NotImplementedError("workload conversion expects one TTIR function")
    if pattern == "two-layer-mlp":
        functions = [op for op in parsed.all_ops if op.name == "tt.func"]
        if len(functions[0].regions) != 1 or len(functions[0].regions[0]) != 1:
            raise NotImplementedError("two-layer MLP requires one straight-line block")
    plan = _plan_for_pattern(parsed.funcs[0], pattern)
    mapper = target_mapper or AtlasTargetMapper(
        matmul_abi=abi, mlp_abi=mlp_abi, linear_abi=linear_abi,
        attention_abi=attention_abi, numerical_policy=numerical_policy)
    return mapper.map(plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="TTIR file matching the selected workload pattern")
    parser.add_argument("--out", type=Path, required=True, help="Atlas virtual SSA output MLIR")
    parser.add_argument("--lhs-slot", type=int, default=0)
    parser.add_argument("--rhs-slot", type=int, default=1)
    parser.add_argument("--mxu-unit", type=int, default=0, choices=(0, 1))
    parser.add_argument("--pattern", choices=("linear-relu", "two-layer-mlp", "attention", "matmul-relu"), default="matmul-relu")
    parser.add_argument("--numerical-policy", help="must explicitly select the pattern's Atlas precision mapping")
    parser.add_argument("--second-rhs-slot", type=int, default=2)
    parser.add_argument("--mask-slot", type=int, default=3)
    parser.add_argument("--output-features", type=int, default=8,
                        help="active output columns in the padded single-layer MLP tile")
    args = parser.parse_args()
    if args.pattern == "two-layer-mlp" and args.mxu_unit != 0:
        parser.error("two-layer MLP fixes MXU units to 0 then 1")
    abi = AtlasTileABI(lhs_input=args.lhs_slot, rhs_input=args.rhs_slot,
                       mxu_unit=args.mxu_unit)
    result = convert_file(
        args.input, abi, pattern=args.pattern, numerical_policy=args.numerical_policy,
        mlp_abi=AtlasMlpABI(input_slots=(args.lhs_slot, args.rhs_slot, args.second_rhs_slot)),
        linear_abi=AtlasLinearReluABI(x_slot=args.lhs_slot, weight_slot=args.rhs_slot,
                                      bias_slot=args.second_rhs_slot,
                                      output_features=args.output_features),
        attention_abi=AtlasAttentionABI(input_slots=(args.lhs_slot, args.rhs_slot,
                                                     args.second_rhs_slot, args.mask_slot)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(result)
    print(f"Wrote Atlas virtual SSA to {args.out}")


if __name__ == "__main__":
    main()
