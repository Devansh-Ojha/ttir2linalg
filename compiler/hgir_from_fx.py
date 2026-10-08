"""Convert a small torch.export/FX graph into the hardware-agnostic IR."""
from __future__ import annotations

from compiler.hgir import HGraph, HOp, HValue


def _meta(node):
    value = node.meta.get("val")
    shape = tuple(getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", "unknown")).replace("torch.", "")
    return dtype, shape


def from_fx_graph(graph) -> HGraph:
    inputs: dict[str, HValue] = {}
    values: dict[str, HValue] = {}
    ops: list[HOp] = []
    outputs: list[str] = []

    for node in graph.nodes:
        if node.op == "placeholder":
            dtype, shape = _meta(node)
            inputs[node.name] = HValue(node.name, dtype, shape)
            continue
        if node.op == "output":
            result = node.args[0]
            result = result if isinstance(result, tuple) else (result,)
            outputs = [
                item.name if hasattr(item, "name") else str(item)
                for item in result
            ]
            continue
        if node.op not in {"call_function", "call_module"}:
            continue

        dtype, shape = _meta(node)
        values[node.name] = HValue(node.name, dtype, shape)
        inputs_for_op = [
            item.name for item in node.all_input_nodes
        ]
        target = str(node.target)
        if "linear" in target or "addmm" in target:
            name = "linear"
        elif "relu" in target:
            name = "relu"
        else:
            name = target.split(".")[-1]
        attrs = {}
        if node.op == "call_module":
            attrs["module"] = target
        ops.append(HOp(name, inputs_for_op, [node.name], attrs))

    graph_ir = HGraph("exported_mlp", inputs, values, ops, outputs)
    errors = graph_ir.verify()
    if errors:
        raise RuntimeError("invalid hardware-agnostic graph: " + "; ".join(errors))
    return graph_ir


def from_exported_graph(exported) -> HGraph:
    return from_fx_graph(exported.graph)
