"""Target-neutral workload classification for the exported HGraph IR."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from compiler.hgir import HGraph


@dataclass(frozen=True)
class HGraphWorkload:
    """Structural facts needed before selecting a backend mapping."""

    pattern: str
    operation_counts: dict[str, int]
    has_scaled_dot_product_attention: bool
    has_residual_add: bool
    has_normalization: bool
    has_gelu: bool
    input_shapes: dict[str, tuple[int | str, ...]]
    output_shapes: dict[str, tuple[int | str, ...]]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HGraphDenseLayer:
    input: str
    weight: str
    bias: str
    output: str
    batch: int
    input_features: int
    output_features: int


@dataclass(frozen=True)
class HGraphDensePlan:
    graph_name: str
    pattern: str
    layers: tuple[HGraphDenseLayer, ...]
    activation: str | None


def _static_dense_shape(graph: HGraph, name: str, rank: int) -> tuple[int, ...]:
    value = graph.inputs.get(name) or graph.values.get(name)
    if value is None:
        raise NotImplementedError("missing HGraph value %s" % name)
    if len(value.shape) != rank or any(type(dim) is not int for dim in value.shape):
        raise NotImplementedError("%s must have a static rank-%d shape" % (name, rank))
    return tuple(value.shape)


def analyze_dense_graph(graph: HGraph) -> HGraphDensePlan:
    """Recognize biased one-layer or two-layer PyTorch dense MLP graphs."""
    errors = graph.verify()
    if errors:
        raise ValueError("cannot analyze invalid HGraph: " + "; ".join(errors))
    ops = graph.ops
    is_linear_relu = (
        len(ops) == 2 and ops[0].name == "linear" and ops[1].name == "relu"
        and len(ops[0].outputs) == 1 and len(ops[1].outputs) == 1
        and ops[1].inputs == [ops[0].outputs[0]]
        and graph.outputs == ops[1].outputs
    )
    is_two_layer = (
        len(ops) == 3 and ops[0].name == "linear" and ops[1].name == "relu"
        and ops[2].name == "linear" and len(ops[0].outputs) == 1
        and len(ops[2].inputs) == 3 and len(ops[2].outputs) == 1
        and ops[1].inputs == [ops[0].outputs[0]]
        and ops[2].inputs[0] == ops[1].outputs[0]
        and graph.outputs == ops[2].outputs
    )
    if not (is_linear_relu or is_two_layer):
        raise NotImplementedError(
            "HGraph dense mapping requires exactly linear+ReLU or "
            "linear+ReLU+linear with a single graph output"
        )

    linear_ops = [ops[0]] + ([ops[2]] if is_two_layer else [])
    layers = []
    for op in linear_ops:
        if len(op.inputs) != 3 or len(op.outputs) != 1:
            raise NotImplementedError("dense layer must have input, weight, bias, and one result")
        input_name, weight_name, bias_name = op.inputs
        output_name = op.outputs[0]
        for operand_name in (*op.inputs, output_name):
            value = graph.inputs.get(operand_name) or graph.values.get(operand_name)
            if value is None or value.type != "float32":
                raise NotImplementedError(
                    "Atlas dense mapping currently requires float32 source values"
                )
        input_shape = _static_dense_shape(graph, input_name, 2)
        weight_shape = _static_dense_shape(graph, weight_name, 2)
        bias_shape = _static_dense_shape(graph, bias_name, 1)
        output_shape = _static_dense_shape(graph, output_name, 2)
        batch, input_features = input_shape
        output_features, weight_input_features = weight_shape
        if (weight_input_features != input_features or
                bias_shape != (output_features,) or
                output_shape != (batch, output_features)):
            raise NotImplementedError("dense layer shapes do not agree for %s" % op.name)
        layers.append(HGraphDenseLayer(
            input_name, weight_name, bias_name, output_name,
            batch, input_features, output_features,
        ))
    if is_two_layer and layers[0].output_features != layers[1].input_features:
        raise NotImplementedError("first layer output does not match second layer input")
    return HGraphDensePlan(
        graph_name=graph.name,
        pattern="two_layer_mlp" if is_two_layer else "linear_relu",
        layers=tuple(layers),
        activation="relu",
    )


def analyze_hgraph(graph: HGraph) -> HGraphWorkload:
    """Classify supported model families without assigning target semantics.

    This is a structural recognizer, not a proof of numerical equivalence or
    an Atlas mapping. It deliberately retains the complete op inventory for
    downstream passes to decide what they can lower.
    """
    errors = graph.verify()
    if errors:
        raise ValueError("cannot analyze invalid HGraph: " + "; ".join(errors))

    counts = Counter(op.name for op in graph.ops)
    linear_count = counts["linear"]
    has_sdpa = counts["scaled_dot_product_attention"] > 0
    has_gelu = counts["gelu"] > 0
    has_norm = counts["layer_norm"] > 0
    add_count = counts["add"]
    has_relu = counts["relu"] > 0

    if has_sdpa and has_gelu and add_count >= 2 and has_norm:
        pattern = "transformer_block"
    elif has_sdpa and linear_count >= 4:
        pattern = "attention_mlp"
    elif has_sdpa:
        pattern = "attention"
    elif has_gelu and linear_count >= 2:
        pattern = "two_layer_mlp"
    elif has_relu and linear_count >= 2:
        pattern = "two_layer_mlp"
    elif has_relu and linear_count == 1:
        pattern = "linear_relu"
    else:
        pattern = "unclassified"

    outputs = {
        name: graph.values[name].shape
        for name in graph.outputs
        if name in graph.values
    }
    return HGraphWorkload(
        pattern=pattern,
        operation_counts=dict(sorted(counts.items())),
        has_scaled_dot_product_attention=has_sdpa,
        has_residual_add=add_count > 0,
        has_normalization=has_norm,
        has_gelu=has_gelu,
        input_shapes={name: value.shape for name, value in graph.inputs.items()},
        output_shapes=outputs,
    )
