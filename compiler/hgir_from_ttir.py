"""Convert the existing parsed TTIR representation into the project HIR."""
from __future__ import annotations

from compiler.hgir import HGraph, HOp, HValue
from compiler.ttir_reader import TTIRModule


def from_ttir_module(tmod: TTIRModule) -> HGraph:
    if len(tmod.funcs) != 1:
        raise NotImplementedError("HIR demo expects one TTIR function")
    function = tmod.funcs[0]
    inputs = {
        arg.name: HValue(arg.name, arg.type.raw, arg.type.shape)
        for arg in function.args
    }
    values = dict(inputs)
    ops: list[HOp] = []
    for op in function.body.ops:
        if op.name in {"tt.return", "tt.reduce.return"}:
            continue
        outputs = [result.name for result in op.results]
        for result in op.results:
            values[result.name] = HValue(
                result.name, result.type.raw, result.type.shape
            )
        attrs = dict(op.attrs)
        if op.regions:
            attrs["region_count"] = len(op.regions)
        ops.append(HOp(
            op.name,
            [operand.name for operand in op.operands],
            outputs,
            attrs,
        ))
    graph = HGraph(function.name, inputs, values, ops, [])
    errors = graph.verify()
    if errors:
        raise RuntimeError("invalid TTIR-derived HIR: " + "; ".join(errors))
    return graph
