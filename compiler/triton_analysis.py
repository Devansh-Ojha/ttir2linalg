"""Target-independent recognition of a small Triton matmul epilogue.

This module deliberately knows nothing about Atlas or any other target. It
turns parsed TTIR into a semantic description that target-specific mappers can
consume.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Optional

from compiler.ttir_model import TFunc, TOp, TValue


_MATMUL_OPS = {"tt.dot", "ttg.dot"}
_MAX_OPS = {"arith.maxnumf", "arith.maximumf", "arith.maxf"}


@dataclass(frozen=True)
class MatmulRelu:
    """A recognized `matmul -> ReLU` pair, independent of target lowering."""

    function: str
    matmul: TOp
    relu: TOp
    lhs: TValue
    rhs: TValue
    result: TValue
    shape: tuple[int, ...]
    lhs_load: Optional[TOp] = None
    rhs_load: Optional[TOp] = None
    store: Optional[TOp] = None

def _is_zero(value: TValue, definitions: dict[int, TOp]) -> bool:
    defining_op = definitions.get(value.id)
    if defining_op is None or defining_op.name != "arith.constant":
        return False

    text = defining_op.attrs.get("value", "").strip()

    # Handle splat tensor constants like dense<0.000000e+00>
    if text.startswith("dense<") and text.endswith(">"):
        text = text[6:-1].strip()

    try:
        return float(text.split(":", 1)[0].strip()) == 0.0
    except ValueError:
        return False

def analyze_matmul_relu(function: TFunc) -> MatmulRelu:
    """Recognize exactly one direct matmul followed by max(matmul, 0).

    More complex epilogues are rejected instead of silently guessed. The
    returned values retain their TTIR identities and types for target mappers.
    """
    ops = function.body.ops
    definitions = {
        result.id: op
        for op in ops
        for result in op.results
    }
    matmuls = [op for op in ops if op.name in _MATMUL_OPS]
    candidates: list[MatmulRelu] = []

    for matmul in matmuls:
        if len(matmul.operands) < 2 or len(matmul.results) != 1:
            continue
        dot_result = matmul.results[0]
        for relu in ops:
            if relu.name not in _MAX_OPS or len(relu.operands) != 2 or len(relu.results) != 1:
                continue
            lhs, rhs = relu.operands
            if lhs.id == dot_result.id and _is_zero(rhs, definitions):
                result = relu.results[0]
            elif rhs.id == dot_result.id and _is_zero(lhs, definitions):
                result = relu.results[0]
            else:
                continue
            candidates.append(MatmulRelu(
                function=function.name,
                matmul=matmul,
                relu=relu,
                lhs=matmul.operands[0],
                rhs=matmul.operands[1],
                result=result,
                shape=result.type.shape,
            ))

    if len(matmuls) != 1 or len(candidates) != 1:
        raise NotImplementedError(
            "expected exactly one TTIR matmul + ReLU; found %d matmul(s), %d candidate(s)"
            % (len(matmuls), len(candidates))
        )
    plan = candidates[0]
    lhs_load = definitions.get(plan.lhs.id)
    rhs_load = definitions.get(plan.rhs.id)
    if lhs_load is None or lhs_load.name != "tt.load" or rhs_load is None or rhs_load.name != "tt.load":
        raise NotImplementedError("the supported matmul operands must be direct TTIR loads")
    if lhs_load is rhs_load:
        raise NotImplementedError("the supported matmul requires two distinct loaded tiles")
    stores = [op for op in ops if op.name == "tt.store"]
    if len(stores) != 1 or not any(value.id == plan.result.id for value in stores[0].operands):
        raise NotImplementedError("the ReLU result must feed the kernel's only TTIR store")
    return MatmulRelu(
        function=plan.function,
        matmul=plan.matmul,
        relu=plan.relu,
        lhs_load=lhs_load,
        rhs_load=rhs_load,
        store=stores[0],
        lhs=plan.lhs,
        rhs=plan.rhs,
        result=plan.result,
        shape=plan.shape,
    )


@dataclass(frozen=True)
class TwoLayerMlp:
    """Exact source graph; target arithmetic and MXU assignment are mapper policy."""

    function: str
    inputs: tuple[TValue, TValue, TValue]
    dots: tuple[TOp, TOp]
    bf16_casts: tuple[TOp, TOp]
    relu: TOp
    pack: TOp
    store: TOp


@dataclass(frozen=True)
class MlpLayer:
    """One source dot/cast, with ReLU and FP8 conversion on hidden layers."""

    dot: TOp
    bf16_cast: TOp
    weight: TValue
    relu: Optional[TOp] = None
    pack: Optional[TOp] = None


@dataclass(frozen=True)
class MlpChain:
    """Ordered source dataflow; contains no target arithmetic or placement policy."""

    function: str
    input_activation: TValue
    weights: tuple[TValue, ...]
    layers: tuple[MlpLayer, ...]
    store: TOp


@dataclass(frozen=True)
class LinearRelu:
    """One vector dot product, bias add, ReLU, and scalar store per program."""

    function: str
    x_load: TOp
    weight_load: TOp
    bias_load: TOp
    multiply: TOp
    reduction: TOp
    bias_add: TOp
    relu: TOp
    store: TOp
    input_features: int


@dataclass(frozen=True)
class ScaledDotProductAttention:
    """One Triton attention query with a static K/V sequence and softmax."""

    function: str
    query_load: TOp
    key_load: TOp
    value_load: TOp
    query_key_mul: TOp
    score_reduce: TOp
    score_scale: TOp
    max_reduce: TOp
    subtract: TOp
    exp: TOp
    softmax_sum: TOp
    normalize: TOp
    probability_value_mul: TOp
    output_reduce: TOp
    store: TOp
    sequence_length: int
    embedding_size: int


def analyze_mlp_chain(function: TFunc) -> MlpChain:
    """Recognize a bias-free, straight-line FP8 chain of at least two layers.

    Walk backward from the only store, retaining the actual SSA operations.
    Dots require explicit IEEE input precision; optional maxNumImpreciseAcc
    must be zero. No scheduling, quantization policy or allocation is inferred.
    """
    def check(condition, message):
        if not condition:
            raise NotImplementedError("MLP chain: " + message)

    ops = function.body.ops
    definitions = {v.id: op for op in ops for v in op.results}
    positions = {id(op): i for i, op in enumerate(ops)}
    check(len(positions) == len(ops), "duplicate operation")
    check(len(definitions) == sum(len(op.results) for op in ops), "duplicate SSA definition")
    check(all(not op.regions for op in ops), "regions/control flow are unsupported")
    arguments = {v.id: v for v in function.args}
    check(len(arguments) == len(function.args), "duplicate argument")
    check(not arguments.keys() & definitions.keys(), "argument redefined by an operation")
    # Validate all edges before traversing, including undefined values and cycles.
    for op in ops:
        for value in op.operands:
            defining = definitions.get(value.id)
            canonical = (arguments.get(value.id) if defining is None else
                         next(v for v in defining.results if v.id == value.id))
            check(canonical is not None, "undefined SSA dependency")
            check(value.type == canonical.type, "inconsistent SSA value type")
            check(defining is None or positions[id(defining)] < positions[id(op)],
                  "non-dominating SSA dependency")
    used = set()

    def tile(value, element, pointer=False):
        check(value.type.kind == "tensor" and value.type.shape == (32, 32) and
              value.type.elem.lower() == element.lower() and value.type.elem_is_ptr == pointer,
              f"expected 32x32 {'pointer ' if pointer else ''}{element}")

    def producer(value, name, element):
        tile(value, element)
        op = definitions.get(value.id)
        check(op is not None and op.name == name and len(op.results) == 1,
              f"expected {name} producer")
        used.add(id(op))
        return op

    def zero(value, element):
        op = producer(value, "arith.constant", element)
        check(_is_zero(value, definitions), "accumulator/ReLU constant must be proven zero")
        check(set(op.attrs) <= {"value", "type"} and not op.operands, "unsupported constant")

    stores = [op for op in ops if op.name == "tt.store"]
    check(len(stores) == 1, "expected exactly one store")
    store = stores[0]
    check(len(store.operands) == 2 and not store.results and not store.attrs, "unsupported store")
    tile(store.operands[0], "bf16", True)
    used.add(id(store))
    current = store.operands[1]
    reverse_layers = []
    relu = pack = None
    while True:
        cast = producer(current, "arith.truncf", "bf16")
        check(len(cast.operands) == 1 and not cast.attrs, "unsupported BF16 conversion")
        dot = producer(cast.operands[0], "tt.dot", "f32")
        check(len(dot.operands) == 3, "dot requires lhs, rhs and accumulator")
        check(set(dot.attrs) <= {"inputPrecision", "maxNumImpreciseAcc"} and
              dot.attrs.get("inputPrecision") == "ieee" and
              re.fullmatch(r"0(?:\s*:\s*i32)?", dot.attrs.get("maxNumImpreciseAcc", "0"))
              is not None, "unsupported dot precision attributes (requires explicit ieee)")
        zero(dot.operands[2], "f32")
        tile(dot.operands[0], "f8E4M3FN")
        tile(dot.operands[1], "f8E4M3FN")
        reverse_layers.append(MlpLayer(dot, cast, dot.operands[1], relu, pack))
        lhs = dot.operands[0]
        lhs_producer = definitions.get(lhs.id)
        if lhs_producer is not None and lhs_producer.name == "tt.load":
            break
        pack = producer(lhs, "tt.fp_to_fp", "f8E4M3FN")
        check(len(pack.operands) == 1 and pack.attrs == {"rounding": "rtne"},
              "FP8 conversion requires explicit rounding=rtne and no other attributes")
        relu = producer(pack.operands[0], "arith.maxnumf", "bf16")
        check(len(relu.operands) == 2 and not relu.attrs, "unsupported ReLU")
        a, b = relu.operands
        if _is_zero(a, definitions):
            a, b = b, a
        zero(b, "bf16")
        current = a
    layers = tuple(reversed(reverse_layers))
    check(len(layers) >= 2, "expected at least two layers")
    weights = tuple(layer.weight for layer in layers)
    inputs = (lhs,) + weights
    pointers = []
    for value in inputs:
        load = producer(value, "tt.load", "f8E4M3FN")
        check(len(load.operands) == 1 and not load.attrs, "only unmasked direct loads are supported")
        tile(load.operands[0], "f8E4M3FN", True)
        pointers.append(load.operands[0])
    check(len({v.id for v in inputs}) == len(inputs) and
          len({v.id for v in pointers}) == len(inputs),
          "expected distinct direct FP8 loads")
    check([v.id for v in function.args] == [v.id for v in pointers] + [store.operands[0].id],
          "arguments must be X, weights in layer order, output direct pointer tiles")
    returns = [op for op in ops if op.name == "tt.return"]
    check(len(returns) == 1 and returns[0] is ops[-1] and
          not returns[0].operands and not returns[0].results and not returns[0].attrs,
          "expected final void return")
    used.add(id(returns[0]))
    check(used == set(positions), "extra or unsupported operations")
    return MlpChain(function.name, lhs, weights, layers, store)


def analyze_two_layer_mlp(function: TFunc) -> TwoLayerMlp:
    """Compatibility entry point retaining the original two-layer plan shape."""
    plan = analyze_mlp_chain(function)
    if len(plan.layers) != 2:
        raise NotImplementedError("two-layer MLP: expected exactly two layers")
    first, final = plan.layers
    return TwoLayerMlp(plan.function, (plan.input_activation,) + plan.weights,
                       (first.dot, final.dot), (first.bf16_cast, final.bf16_cast),
                       first.relu, first.pack, plan.store)


def _definitions(ops: list[TOp]) -> dict[int, TOp]:
    return {value.id: op for op in ops for value in op.results}


def _depends(value: TValue, target: TValue, definitions: dict[int, TOp], seen=None) -> bool:
    if value.id == target.id:
        return True
    seen = set() if seen is None else seen
    if value.id in seen:
        return False
    seen.add(value.id)
    producer = definitions.get(value.id)
    return bool(producer and any(_depends(v, target, definitions, seen)
                                 for v in producer.operands))


def _reduce_body_is(op: TOp, operation: str) -> bool:
    if not op.regions or not op.regions[0] or len(op.regions[0]) != 1:
        return False
    body = op.regions[0][0]
    returns = [nested for nested in body.ops if nested.name == "tt.reduce.return"]
    arithmetic = [nested for nested in body.ops if nested.name.startswith("arith.")]
    return len(returns) == 1 and len(arithmetic) == 1 and arithmetic[0].name == operation


def _has_only_names(ops: list[TOp], allowed: set[str]) -> bool:
    return all(op.name in allowed for op in ops)


def analyze_linear_relu(function: TFunc) -> LinearRelu:
    """Recognize the checked-in per-output FC + bias + ReLU Triton kernel.

    The plan intentionally describes semantic operands only. Launch dimensions,
    padding and the FP8/BF16 target policy are supplied by the Atlas mapping.
    """
    ops = function.body.ops
    definitions = _definitions(ops)
    loads = [op for op in ops if op.name == "tt.load" and len(op.results) == 1]
    stores = [op for op in ops if op.name == "tt.store"]
    multiplies = [op for op in ops if op.name == "arith.mulf" and len(op.results) == 1]
    reductions = [op for op in ops if op.name == "tt.reduce" and len(op.results) == 1]
    adds = [op for op in ops if op.name == "arith.addf" and len(op.results) == 1]
    relus = [op for op in ops if op.name in _MAX_OPS and len(op.results) == 1]

    def check(condition, message):
        if not condition:
            raise NotImplementedError("linear ReLU: " + message)

    check(len(loads) == 3 and len(stores) == 1, "expected three direct loads and one store")
    check(len(multiplies) == 1 and len(reductions) == 1 and len(relus) == 1,
          "expected one multiply, sum reduction and ReLU")
    check(len(adds) == 1, "expected one bias add outside the reduction region")
    check(_has_only_names(ops, {
        "tt.get_program_id", "tt.make_range", "tt.splat", "tt.addptr", "tt.load",
        "tt.expand_dims", "tt.broadcast", "tt.reshape", "tt.reduce", "arith.constant",
        "arith.muli", "arith.mulf", "arith.addf", "arith.maxnumf", "tt.store", "tt.return",
    }), "unsupported operation in the FC + bias + ReLU kernel")
    returns = [op for op in ops if op.name == "tt.return"]
    check(len(returns) == 1 and returns[0] is ops[-1], "expected one final void return")
    vector_loads = [op for op in loads if op.results[0].type.kind == "tensor" and
                    len(op.results[0].type.shape) == 1]
    scalar_loads = [op for op in loads if op.results[0].type.kind == "scalar"]
    check(len(vector_loads) == 2 and len(scalar_loads) == 1,
          "expected two rank-one input/weight loads and one scalar bias load")
    x_load, weight_load = vector_loads
    bias_load = scalar_loads[0]
    x, weight = x_load.results[0], weight_load.results[0]
    multiply = multiplies[0]
    check(len(multiply.operands) == 2 and all(
        any(v.id == loaded.id for v in multiply.operands) for loaded in (x, weight)),
        "multiply must consume the loaded input and weight vectors")
    check(x.type.shape == weight.type.shape and x.type.shape[0] > 0,
          "input and weight reduction extents must agree")
    reduction = reductions[0]
    check(reduction.attrs.get("axis", "0 : i32").split(":")[0].strip() == "0" and
          _reduce_body_is(reduction, "arith.addf") and len(reduction.results) == 1 and
          _depends(reduction.operands[0], multiply.results[0], definitions),
          "vector multiply must feed an axis-zero sum")
    bias_add_candidates = [op for op in adds if any(
        _depends(value, reduction.results[0], definitions) for value in op.operands) and
        any(value.id == bias_load.results[0].id for value in op.operands)]
    check(len(bias_add_candidates) == 1, "sum must be added to the loaded bias")
    bias_add = bias_add_candidates[0]
    zero_values = [value for op in ops if op.name == "arith.constant" for value in op.results
                   if _is_zero(value, definitions)]
    relu = relus[0]
    check(len(relu.operands) == 2 and any(
        _depends(value, bias_add.results[0], definitions) for value in relu.operands) and
        any(value.id in {v.id for v in zero_values} for value in relu.operands),
        "bias sum must feed max with zero")
    check(len(stores[0].operands) == 2 and any(
        _depends(value, relu.results[0], definitions) for value in stores[0].operands),
        "ReLU result must feed the only store")
    check(len(function.args) == 4, "expected input, weight, bias and output pointer arguments")
    return LinearRelu(function.name, x_load, weight_load, bias_load, multiply,
                      reduction, bias_add, relu, stores[0], x.type.shape[0])


def analyze_attention(function: TFunc) -> ScaledDotProductAttention:
    """Recognize the checked-in 4-key, 8-feature single-query attention TTIR."""
    ops = function.body.ops
    definitions = _definitions(ops)
    loads = [op for op in ops if op.name == "tt.load" and len(op.results) == 1]
    stores = [op for op in ops if op.name == "tt.store"]
    reductions = [op for op in ops if op.name == "tt.reduce" and len(op.results) == 1]
    multiplies = [op for op in ops if op.name == "arith.mulf" and len(op.results) == 1]

    def check(condition, message):
        if not condition:
            raise NotImplementedError("attention: " + message)

    check(len(loads) == 3 and len(stores) == 1 and len(function.args) == 4,
          "expected query/key/value loads, one output store, and four pointer arguments")
    vectors = [op for op in loads if op.results[0].type.shape == (8,)]
    matrices = [op for op in loads if op.results[0].type.shape == (4, 8)]
    check(len(vectors) == 1 and len(matrices) == 2,
          "supported shape is one 8-feature query and 4x8 key/value tiles")
    query_load = vectors[0]
    # Key and value have identical types; identify them by their use in the
    # score and weighted-value branches below instead of textual SSA names.
    check(len(reductions) == 4 and len(multiplies) == 3,
          "expected QK, softmax, and PV reduction sequence")
    check(_has_only_names(ops, {
        "tt.get_program_id", "tt.make_range", "tt.splat", "tt.addptr", "tt.load",
        "tt.expand_dims", "tt.broadcast", "tt.reduce", "arith.constant", "arith.muli",
        "arith.mulf", "arith.subf", "math.exp", "arith.divf", "tt.store", "tt.return",
    }), "unsupported operation in the bounded attention kernel")
    check(sum(op.name == "arith.subf" for op in ops) == 1 and
          sum(op.name == "arith.divf" for op in ops) == 1 and
          sum(op.name == "math.exp" for op in ops) == 1,
          "expected one stable softmax subtract, exponential, and normalization")
    check(all(len(op.operands) == 1 and not op.attrs for op in loads),
          "masked or attributed loads are unsupported")
    score_mul = next((op for op in multiplies if tuple(op.results[0].type.shape) == (4, 8) and
                      any(_depends(v, query_load.results[0], definitions) for v in op.operands)), None)
    check(score_mul is not None, "query must participate in the QK product")
    score_loads = [op for op in matrices if any(_depends(v, op.results[0], definitions)
                                                 for v in score_mul.operands)]
    check(len(score_loads) == 1, "QK product must consume exactly one loaded key tile")
    key_load = score_loads[0]
    value_load = next(op for op in matrices if op is not key_load)

    def reduction_of(axis, value, body_op, result_shape=None):
        matches = [op for op in reductions if op.attrs.get("axis", "").split(":")[0].strip() == str(axis)
                   and _reduce_body_is(op, body_op) and
                   (result_shape is None or op.results[0].type.shape == result_shape) and
                   _depends(op.operands[0], value, definitions)]
        check(len(matches) == 1, f"expected one axis-{axis} {body_op} reduction")
        return matches[0]

    score_reduce = reduction_of(1, score_mul.results[0], "arith.addf", (4,))
    scaled_candidates = [op for op in multiplies if op.results[0].type.shape == (4,) and any(
        _depends(v, score_reduce.results[0], definitions) for v in op.operands)]
    check(len(scaled_candidates) == 1, "QK sum must be scaled once")
    score_scale = scaled_candidates[0]
    scale_inputs = [v for v in score_scale.operands
                    if not _depends(v, score_reduce.results[0], definitions)]
    check(len(scale_inputs) == 1, "score scaling must use one constant")
    scale_def = definitions.get(scale_inputs[0].id)
    scale_text = scale_def.attrs.get("value", "") if scale_def is not None else ""
    match = re.search(r"dense<\s*([-+0-9.eE]+)\s*>", scale_text)
    check(scale_def is not None and scale_def.name == "arith.constant" and match is not None,
          "attention score scale must be a static dense constant")
    source_scale = float(match.group(1))
    expected_scale = 1.0 / math.sqrt(8.0)
    check(abs(source_scale - expected_scale) <= 1e-6,
          "score scale must be 1/sqrt(8) for this static kernel")
    max_reduce = reduction_of(0, score_scale.results[0], "arith.maxnumf")
    subtracts = [op for op in ops if op.name == "arith.subf" and len(op.results) == 1 and
                 len(op.operands) == 2 and
                 _depends(op.operands[0], score_scale.results[0], definitions) and
                 _depends(op.operands[1], max_reduce.results[0], definitions)]
    check(len(subtracts) == 1, "softmax must subtract the row maximum")
    exps = [op for op in ops if op.name == "math.exp" and len(op.results) == 1 and
            _depends(op.operands[0], subtracts[0].results[0], definitions)]
    check(len(exps) == 1, "shifted scores must feed math.exp")
    softmax_sum = reduction_of(0, exps[0].results[0], "arith.addf", ())
    normalizers = [op for op in ops if op.name == "arith.divf" and len(op.results) == 1 and
                   len(op.operands) == 2 and
                   _depends(op.operands[0], exps[0].results[0], definitions) and
                   _depends(op.operands[1], softmax_sum.results[0], definitions)]
    check(len(normalizers) == 1, "softmax exponential must be normalized by its row sum")
    pv_candidates = [op for op in multiplies if any(
        _depends(v, normalizers[0].results[0], definitions) for v in op.operands) and any(
        _depends(v, value_load.results[0], definitions) for v in op.operands)]
    check(len(pv_candidates) == 1, "softmax probabilities must multiply the loaded values")
    pv_mul = pv_candidates[0]
    output_reduce = reduction_of(0, pv_mul.results[0], "arith.addf", (8,))
    check(len(output_reduce.results[0].type.shape) == 1 and
          output_reduce.results[0].type.shape == (8,), "attention output must have 8 features")
    check(any(_depends(v, output_reduce.results[0], definitions) for v in stores[0].operands),
          "attention output reduction must feed the only store")
    check(len(ops) <= 60 and
          len([op for op in ops if op.name in {"tt.load", "tt.store", "tt.reduce",
                                              "arith.mulf", "arith.subf", "arith.divf", "math.exp"}]) == 14,
          "unexpected extra or unsupported operations in the bounded attention kernel")
    returns = [op for op in ops if op.name == "tt.return"]
    check(len(returns) == 1 and returns[0] is ops[-1], "expected one final void return")
    return ScaledDotProductAttention(
        function.name, query_load, key_load, value_load, score_mul, score_reduce,
        score_scale, max_reduce, subtracts[0], exps[0], softmax_sum,
        normalizers[0], pv_mul, output_reduce, stores[0], 4, 8)
