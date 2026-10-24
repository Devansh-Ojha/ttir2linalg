"""Convert a small torch.export/FX graph into the hardware-agnostic IR."""
from __future__ import annotations

from compiler.hgir import HGraph, HOp, HValue


def _meta(node):
    value = node.meta.get("val")
    shape = tuple(getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", "unknown")).replace("torch.", "")
    return dtype, shape


def _target_name(node) -> str:
    """Return the operator name, not an overload suffix such as ``.Tensor``."""
    target = node.target
    schema = getattr(target, "_schema", None)
    if schema is not None:
        raw = str(schema.name)
        return raw.split("::")[-1]
    if node.op == "call_method":
        return str(target)
    if node.op == "call_module":
        return str(target).split(".")[-1]
    name = getattr(target, "__name__", None)
    if name:
        return name
    raw = str(target)
    if raw.startswith("aten."):
        return raw.split(".")[1]
    if raw.startswith("operator."):
        return raw.split(".")[-1]
    return raw.split(".")[-1]


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
        if node.op not in {"call_function", "call_method", "call_module"}:
            continue

        dtype, shape = _meta(node)
        values[node.name] = HValue(node.name, dtype, shape)
        inputs_for_op = [
            item.name for item in node.all_input_nodes
        ]
        target = str(node.target)
        target_name = _target_name(node)
        module_type = None
        if node.op == "call_module":
            try:
                module = node.graph.owning_module.get_submodule(target)
                module_type = type(module).__name__
            except (AttributeError, KeyError):
                pass
            target_name = {
                "Linear": "linear",
                "ReLU": "relu",
                "GELU": "gelu",
                "LayerNorm": "layer_norm",
                "MultiheadAttention": "multihead_attention",
            }.get(module_type, target_name)
        if target_name in {"linear", "addmm", "mm", "matmul"}:
            name = "linear"
        elif target_name in {"relu", "relu_"}:
            name = "relu"
        else:
            name = target_name
        attrs = {"fx_target": target}
        schema = getattr(node.target, "_schema", None)
        if schema is not None:
            attrs["schema"] = str(schema)
        if node.op == "call_module":
            attrs["module"] = target
            if module_type:
                attrs["module_type"] = module_type
        ops.append(HOp(name, inputs_for_op, [node.name], attrs))

    graph_ir = HGraph("exported_mlp", inputs, values, ops, outputs)
    errors = graph_ir.verify()
    if errors:
        raise RuntimeError("invalid hardware-agnostic graph: " + "; ".join(errors))
    return graph_ir


def from_exported_graph(exported) -> HGraph:
    return from_fx_graph(exported.graph)
