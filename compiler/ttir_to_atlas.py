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
                                      analyze_matmul_relu, analyze_mlp_chain,
                                      analyze_two_layer_mlp)
from compiler.ttir_model import TFunc


def _plan_for_pattern(function: TFunc, pattern: str):
    if pattern == "matmul-relu":
        return analyze_matmul_relu(function)
    if pattern == "two-layer-mlp":
        return analyze_two_layer_mlp(function)
    if pattern == "mlp-chain":
        return analyze_mlp_chain(function)
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
                 mlp_abi: AtlasMlpABI | None = None,
                 linear_abi: AtlasLinearReluABI = AtlasLinearReluABI(),
                 attention_abi: AtlasAttentionABI = AtlasAttentionABI(),
                 operation_abi=None, operation_memory=None,
                 target_mapper: TritonTargetMapper | None = None) -> str:
    from compiler.ttir_reader import parse_ttir

    parsed = parse_ttir(source.read_text())
    if len(parsed.funcs) != 1:
        raise NotImplementedError("workload conversion expects one TTIR function")
    if pattern == "atlas-ops":
        from compiler.atlas_operation_mapping import emit_atlas_operations, AtlasOperationMemory
        from compiler.ttir_operation_analysis import analyze_operations
        if target_mapper is not None:
            raise ValueError("atlas-ops uses its separate operation emitter")
        return emit_atlas_operations(analyze_operations(parsed.funcs[0]),
                                     numerical_policy=numerical_policy, abi=operation_abi,
                                     memory=operation_memory if operation_memory is not None else AtlasOperationMemory())
    if pattern in {"two-layer-mlp", "mlp-chain"}:
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
    parser.add_argument("--pattern", choices=("atlas-ops", "linear-relu", "two-layer-mlp", "mlp-chain", "attention", "matmul-relu"), default="matmul-relu")
    parser.add_argument("--tile-binding", action="append", metavar="ARG:ROLE:SLOT",
                        help="atlas-ops explicit argument index, input/output role, slot; repeat for every argument; authorizes the BF16TileABI staging contract")
    parser.add_argument("--input-slots", type=int, nargs="+",
                        help="mlp-chain slots in X, W0, W1, ... order")
    parser.add_argument("--input-base", type=lambda s: int(s, 0), default=0x90000000)
    parser.add_argument("--output-base", type=lambda s: int(s, 0),
                        help="mlp-chain DRAM base; default follows all input slots")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--numerical-policy", help="must explicitly select the pattern's Atlas precision mapping")
    parser.add_argument("--second-rhs-slot", type=int, default=2)
    parser.add_argument("--mask-slot", type=int, default=3)
    parser.add_argument("--output-features", type=int, default=8,
                        help="active output columns in the padded single-layer MLP tile")
    args = parser.parse_args()
    if args.pattern == "atlas-ops":
        from compiler.atlas_eligibility import BF16TileABI, TileBinding
        from compiler.atlas_operation_mapping import AtlasOperationMemory
        if args.out.exists():
            parser.error("atlas-ops output already exists; choose a new artifact path")
        if (args.input_slots is not None or args.output_index != 0 or
                (args.lhs_slot, args.rhs_slot, args.second_rhs_slot, args.mask_slot, args.mxu_unit, args.output_features) != (0, 1, 2, 3, 0, 8)):
            parser.error("atlas-ops uses --tile-binding instead of legacy workload options")
        try:
            bindings = []
            for specification in args.tile_binding or []:
                argument, role, slot = specification.split(":")
                bindings.append(TileBinding(int(argument), role, int(slot)))
            result = convert_file(args.input, pattern="atlas-ops", numerical_policy=args.numerical_policy,
                                  operation_abi=BF16TileABI(tuple(bindings)),
                                  operation_memory=AtlasOperationMemory(args.input_base,
                                      args.output_base if args.output_base is not None else 0x90020000))
        except (ValueError, RuntimeError, NotImplementedError) as error:
            parser.error(str(error))
        # No path is created until full source/eligibility/ABI validation succeeds.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as output:
            output.write(result)
        print(f"Wrote Atlas virtual SSA to {args.out}")
        return
    if args.tile_binding is not None:
        parser.error("--tile-binding requires --pattern atlas-ops")
    if args.pattern in {"two-layer-mlp", "mlp-chain"} and args.mxu_unit != 0:
        parser.error("MLP layers fix MXU units to alternating 0 then 1")
    mlp_abi = AtlasMlpABI(input_slots=(args.lhs_slot, args.rhs_slot, args.second_rhs_slot))
    if args.pattern == "mlp-chain":
        if (args.lhs_slot, args.rhs_slot, args.second_rhs_slot) != (0, 1, 2):
            parser.error("mlp-chain uses --input-slots instead of individual slot options")
        if args.out.exists():
            parser.error("mlp-chain output already exists; choose a new artifact path")
        from compiler.ttir_reader import parse_ttir
        module = parse_ttir(args.input.read_text())
        if len(module.funcs) != 1:
            parser.error("mlp-chain requires exactly one function")
        depth = len(analyze_mlp_chain(module.funcs[0]).layers)
        slots = tuple(args.input_slots) if args.input_slots is not None else tuple(range(depth + 1))
        output_base = (args.output_base if args.output_base is not None else
                       args.input_base + max(4, max(slots, default=0) + 1) * 2048)
        mlp_abi = AtlasMlpABI(slots, args.input_base, output_base, args.output_index)
    elif (args.input_slots is not None or args.output_base is not None or
          args.input_base != 0x90000000 or args.output_index != 0):
        parser.error("chain ABI options require --pattern mlp-chain")
    abi = AtlasTileABI(lhs_input=args.lhs_slot, rhs_input=args.rhs_slot,
                       mxu_unit=args.mxu_unit)
    result = convert_file(
        args.input, abi, pattern=args.pattern, numerical_policy=args.numerical_policy,
        mlp_abi=mlp_abi,
        linear_abi=AtlasLinearReluABI(x_slot=args.lhs_slot, weight_slot=args.rhs_slot,
                                      bias_slot=args.second_rhs_slot,
                                      output_features=args.output_features),
        attention_abi=AtlasAttentionABI(input_slots=(args.lhs_slot, args.rhs_slot,
                                                     args.second_rhs_slot, args.mask_slot)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.pattern == "mlp-chain":
        with args.out.open("x") as output:
            output.write(result)
    else:
        args.out.write_text(result)
    print(f"Wrote Atlas virtual SSA to {args.out}")


if __name__ == "__main__":
    main()
