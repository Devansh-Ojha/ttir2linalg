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
