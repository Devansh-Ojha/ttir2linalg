"""Target-independent recognition of a small Triton matmul epilogue.

This module deliberately knows nothing about Atlas or any other target. It
turns parsed TTIR into a semantic description that target-specific mappers can
consume.
"""
from __future__ import annotations

from dataclasses import dataclass
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


def analyze_two_layer_mlp(function: TFunc) -> TwoLayerMlp:
    """Admit only a straight-line, bias-free 32x32 E4M3FN two-dot graph."""
    def check(condition, message):
        if not condition:
            raise NotImplementedError("two-layer MLP: " + message)

    ops = function.body.ops
    definitions = {v.id: op for op in ops for v in op.results}
    positions = {id(op): i for i, op in enumerate(ops)}
    check(len(positions) == len(ops), "duplicate operation")
    check(len(definitions) == sum(len(op.results) for op in ops), "duplicate SSA definition")
    check(all(not op.regions for op in ops), "regions/control flow are unsupported")
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
    final = producer(store.operands[1], "arith.truncf", "bf16")
    check(len(final.operands) == 1 and not final.attrs, "unsupported final conversion")
    dot1 = producer(final.operands[0], "tt.dot", "f32")
    check(len(dot1.operands) == 3 and not dot1.attrs, "unsupported second dot")
    zero(dot1.operands[2], "f32")
    pack = producer(dot1.operands[0], "tt.fp_to_fp", "f8E4M3FN")
    check(len(pack.operands) == 1 and pack.attrs == {"rounding": "rtne"},
          "FP8 conversion requires explicit rounding=rtne and no other attributes")
    relu = producer(pack.operands[0], "arith.maxnumf", "bf16")
    check(len(relu.operands) == 2 and not relu.attrs, "unsupported ReLU")
    a, b = relu.operands
    if _is_zero(a, definitions):
        a, b = b, a
    zero(b, "bf16")
    first = producer(a, "arith.truncf", "bf16")
    check(len(first.operands) == 1 and not first.attrs, "unsupported first BF16 conversion")
    dot0 = producer(first.operands[0], "tt.dot", "f32")
    check(len(dot0.operands) == 3 and not dot0.attrs, "unsupported first dot")
    check(dot0 is not dot1, "expected two distinct dots")
    zero(dot0.operands[2], "f32")
    inputs = (dot0.operands[0], dot0.operands[1], dot1.operands[1])
    pointers = []
    for value in inputs:
        load = producer(value, "tt.load", "f8E4M3FN")
        check(len(load.operands) == 1 and not load.attrs, "only unmasked direct loads are supported")
        tile(load.operands[0], "f8E4M3FN", True)
        pointers.append(load.operands[0])
    check(len({v.id for v in inputs}) == 3 and len({v.id for v in pointers}) == 3,
          "expected three distinct direct FP8 loads")
    check([v.id for v in function.args] == [v.id for v in pointers] + [store.operands[0].id],
          "arguments must be X, B0, B1, output direct pointer tiles")
    returns = [op for op in ops if op.name == "tt.return"]
    check(len(returns) == 1 and returns[0] is ops[-1] and
          not returns[0].operands and not returns[0].results and not returns[0].attrs,
          "expected final void return")
    used.add(id(returns[0]))
    check(used == set(positions), "extra or unsupported operations")
    for op in ops:
        for value in op.operands:
            defining = definitions.get(value.id)
            check(defining is None or positions[id(defining)] < positions[id(op)], "non-dominating SSA dependency")
    return TwoLayerMlp(function.name, inputs, (dot0, dot1), (first, final), relu, pack, store)
