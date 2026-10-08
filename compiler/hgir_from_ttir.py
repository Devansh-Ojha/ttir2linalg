"""Convert the existing parsed TTIR representation into the project HIR."""
from __future__ import annotations

from compiler.hgir import HGraph, HOp, HValue
from compiler.ttir_reader import TTIRModule


_OP_NAMES = {
    "arith.constant": "constant",
    "arith.muli": "integer_multiply",
    "arith.mulf": "multiply",
    "arith.addf": "add",
    "tt.get_program_id": "partition_id",
    "tt.make_range": "range",
    "tt.splat": "broadcast",
    "tt.addptr": "address_offset",
    "tt.load": "load",
    "tt.store": "store",
    "tt.reshape": "reshape",
    "tt.reduce": "reduce",
}


def _type_name(value) -> str:
    if value.type.kind == "ptr":
        return "buffer<%s>" % value.type.elem
    if value.type.kind == "tensor" and value.type.elem_is_ptr:
        return "tensor<%s x buffer<%s>>" % (
            "x".join(str(dim) for dim in value.type.shape),
            value.type.elem,
        )
    if value.type.kind == "tensor":
        return "tensor<%s x %s>" % (
            "x".join(str(dim) for dim in value.type.shape),
            value.type.elem,
        )
    return value.type.elem


def from_ttir_module(tmod: TTIRModule) -> HGraph:
    if len(tmod.funcs) != 1:
        raise NotImplementedError("HIR demo expects one TTIR function")
    function = tmod.funcs[0]
    inputs = {
        arg.name: HValue(arg.name, _type_name(arg), arg.type.shape)
        for arg in function.args
    }
    values = dict(inputs)
    ops: list[HOp] = []
    outputs: list[str] = []

    def convert_op(op) -> HOp | None:
        if op.name in {"tt.return", "tt.reduce.return"}:
            return None
        result_values = [result.name for result in op.results]
        for result in op.results:
            values[result.name] = HValue(
                result.name, _type_name(result), result.type.shape
            )
        regions = [
            [
                nested_op
                for nested in block.ops
                if (nested_op := convert_op(nested)) is not None
            ]
            for region in op.regions
            for block in region
        ]
        attrs = dict(op.attrs)
        if op.regions:
            attrs["region_count"] = len(op.regions)
        return HOp(
            _OP_NAMES.get(op.name, "target_operation"),
            [operand.name for operand in op.operands],
            result_values,
            attrs,
            regions,
        )

    for op in function.body.ops:
        if op.name == "tt.return":
            outputs = [operand.name for operand in op.operands]
            continue
        converted = convert_op(op)
        if converted is not None:
            ops.append(converted)
    graph = HGraph(function.name, inputs, values, ops, outputs)
    errors = graph.verify()
    if errors:
        raise RuntimeError("invalid TTIR-derived HIR: " + "; ".join(errors))
    return graph
