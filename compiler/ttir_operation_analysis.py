"""Conservative operation analysis, independent of target scheduling or numerics.

Admission covers typed arithmetic, contractions, bounded reductions, shape
transforms, conversions, integer indexing and masked memory movement. Admission means source semantics can be inspected, NOT that
Atlas can execute the kernel. No subgraph is returned on failure. The original
TFunc/TOp/TValue objects (including source, regions and attributes) are retained.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import math
from typing import Optional, Tuple

from compiler.ttir_model import TFunc, TOp, TValue
from compiler.ttir_reader import _strip_locs


class UnsupportedKernel(NotImplementedError):
    pass


@dataclass(frozen=True)
class MemoryAccess:
    operation: TOp
    pointer: TValue
    value: Optional[TValue]
    mask: Optional[TValue]
    other: Optional[TValue]
    # An absent load 'other' means unspecified inactive lanes, never zero fill.


@dataclass(frozen=True)
class OperationAnalysis:
    function: TFunc
    operations: Tuple[TOp, ...]
    dependencies: Tuple[Tuple[TValue, ...], ...]
    memory: Tuple[MemoryAccess, ...]
    reductions: Tuple["Reduction", ...] = ()
    relu_candidates: Tuple[TOp, ...] = ()
    resolved_dot_defaults: bool = False


@dataclass(frozen=True)
class Reduction:
    operation: TOp
    axis: int
    combiner: TOp


# Match custom assembly up to its type separator, rather than silently ignoring
# optional overflow/exact/cache/volatile/semantic flags not yet understood.
_V = r"%[\w#]+"
_PATTERNS = {
    "arith.constant": r"(?:dense<[^<>]+>|[-+\w.]+)",
    "tt.get_program_id": r"[xyz]",
    "tt.make_range": r"\{\s*(?:end|start)\s*=\s*\d+\s*:\s*i32\s*,\s*(?:end|start)\s*=\s*\d+\s*:\s*i32\s*\}",
    "tt.splat": _V,
    "tt.addptr": _V + r"\s*,\s*" + _V,
    "arith.cmpi": r"(?:eq|ne|slt|sle|sgt|sge|ult|ule|ugt|uge)\s*,\s*" + _V + r"\s*,\s*" + _V,
    "tt.load": _V + r"(?:\s*,\s*" + _V + r"){0,2}",
    "tt.store": _V + r"\s*,\s*" + _V + r"(?:\s*,\s*" + _V + r")?",
    "tt.return": "",
}
for _name in ("arith.addi", "arith.muli", "arith.divsi", "arith.remsi"):
    _PATTERNS[_name] = _V + r"\s*,\s*" + _V


_FLOAT = {"f16", "bf16", "f32", "f8E4M3FN"}
_ELEM = r"(?:i1|i32|f16|bf16|f32|f8E4M3FN)"
_TYPE = r"(?:" + _ELEM + r"|!tt\.ptr<" + _ELEM + r">|tensor<(?:\d+x)*(?:" + _ELEM + r"|!tt\.ptr<" + _ELEM + r">)>)"
for _name in ("arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.maxnumf"):
    _PATTERNS[_name] = _V + r"\s*,\s*" + _V
for _name in ("math.exp", "arith.truncf", "arith.extf", "tt.broadcast", "tt.reshape"):
    _PATTERNS[_name] = _V
_PATTERNS["tt.trans"] = _V + r"\s*\{order\s*=\s*array<i32:\s*\d+(?:\s*,\s*\d+)*>\s*\}"
_PATTERNS["tt.expand_dims"] = _V + r"\s*\{axis\s*=\s*\d+\s*:\s*i32\}"
_PATTERNS["tt.fp_to_fp"] = _V + r"(?:\s*,\s*rounding\s*=\s*(?:rtne|rtz))?"
_PATTERNS["tt.dot"] = (_V + r"\s*,\s*" + _V + r"\s*,\s*" + _V
                       + r"(?:\s*,\s*inputPrecision\s*=\s*(?:ieee|tf32|tf32x3))?"
                       + r"(?:\s*,\s*maxNumImpreciseAcc\s*=\s*\d+)?"
                       + r"(?:\s*\{maxNumImpreciseAcc\s*=\s*\d+\s*:\s*i32\})?")


def transpose_order(op):
    """Recover the validated logical permutation; never infer from tile shape."""
    match = re.fullmatch(r"array<i32:\s*(\d+(?:\s*,\s*\d+)*)>", op.attrs.get('order',''))
    if match is None:
        raise UnsupportedKernel(f'tt.trans at operation {op.index}: invalid dimension permutation')
    order = tuple(int(x.strip()) for x in match.group(1).split(','))
    printed = re.search(r'order\s*=\s*array<i32:\s*(\d+(?:\s*,\s*\d+)*)>', op.assembly)
    if printed is None or tuple(int(x.strip()) for x in printed.group(1).split(',')) != order:
        raise UnsupportedKernel(f'tt.trans at operation {op.index}: permutation attribute disagrees with source assembly')
    return order


def _reject(op: TOp, reason: str):
    raise UnsupportedKernel(f"{op.name} at operation {op.index}: {reason}")


def _require(op, condition, message):
    if not condition:
        _reject(op, message)


def _same(a, b):
    return (a.kind, a.elem, a.shape, a.elem_is_ptr) == (b.kind, b.elem, b.shape, b.elem_is_ptr)


def _floating(t):
    return t.kind in {"scalar", "tensor"} and not t.elem_is_ptr and t.elem in _FLOAT - {"f8E4M3FN"}


def _validate_spelling(op: TOp):
    if not op.assembly:
        _reject(op, "source assembly required to validate semantic flags")
    assembly = op.assembly.split("//", 1)[0].strip()
    assembly = re.sub(r"^%[\w#]+\s*=\s*", "", assembly)
    if op.name == "tt.reduce":
        pattern = r'"tt.reduce"\(' + _V + r'\)\s*<\{axis\s*=\s*\d+\s*:\s*i32\}>\s*\(\{'
    elif op.name == "tt.reduce.return":
        pattern = r"tt.reduce.return\s+" + _V + r"\s*:\s*" + _TYPE
    else:
        pattern = re.escape(op.name) + r"\s*" + _PATTERNS[op.name]
        if op.name != "tt.return":
            pattern += r"\s*:\s*" + _TYPE + r"(?:\s*(?:,|->|to|\*)\s*" + _TYPE + r")*"
    _require(op, re.fullmatch(pattern, assembly), "unsupported assembly, type, or semantic attribute")
    allowed = {"arith.constant": {"value", "type"}, "arith.cmpi": {"predicate"},
               "tt.make_range": {"start", "end"}, "tt.get_program_id": {"axis"},
               "tt.expand_dims": {"axis"}, "tt.trans": {"order"}, "tt.reduce": {"axis"},
               "tt.fp_to_fp": {"rounding"}, "tt.dot": {"inputPrecision", "maxNumImpreciseAcc"}}
    _require(op, not (set(op.attrs) - allowed.get(op.name, set())), "unsupported attributes")


def _validate_types(op: TOp, resolve_dot_defaults=False):
    operands, results = op.operands, op.results
    def require(condition, message):
        _require(op, condition, message)
    def integer(t):
        return t.kind in {"scalar", "tensor"} and t.elem == "i32" and not t.elem_is_ptr
    def pointer(t):
        return (t.kind == "ptr" or t.elem_is_ptr) and t.elem in _FLOAT | {"i1", "i32"}
    name = op.name
    if name in {"tt.return", "tt.reduce.return"}:
        require(not results and (not operands if name == "tt.return" else len(operands) == 1), "invalid return arity")
        return
    require(len(results) == (0 if name == "tt.store" else 1), "invalid result arity")
    if name == "arith.constant":
        require(not operands and (integer(results[0].type) or _floating(results[0].type)), "expected numeric constant")
        # Initial constants are scalar literals or tensor splats, not arbitrary
        # dense arrays/resources whose syntax the attribute scanner cannot retain.
        value = op.attrs.get("value", "")
        require(re.fullmatch(r"(?:dense<)?(?:[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|0x[0-9A-Fa-f]+)>?", value) is not None, "unsupported constant literal")
    elif name == "tt.get_program_id":
        require(not operands and results[0].type.kind == "scalar" and integer(results[0].type), "expected scalar program id")
        require(op.attr_int("axis") in (0, 1, 2), "invalid program axis")
    elif name == "tt.make_range":
        t = results[0].type
        require(not operands and integer(t) and len(t.shape) == 1, "expected rank-one i32 range")
        require(set(op.attrs) == {"start", "end"} and op.attr_int("end") - op.attr_int("start") == t.shape[0], "invalid range bounds")
    elif name == "tt.splat":
        require(len(operands) == 1, "invalid splat arity")
        a, b = operands[0].type, results[0].type
        require(a.kind in {"scalar", "ptr"} and b.kind == "tensor" and a.elem == b.elem and (a.kind == "ptr") == b.elem_is_ptr, "invalid splat types")
    elif name == "tt.addptr":
        require(len(operands) == 2, "invalid addptr arity")
        a, b, c = operands[0].type, operands[1].type, results[0].type
        require(pointer(a) and integer(b) and a.shape == b.shape and _same(a, c), "invalid pointer offset types")
    elif name in {"arith.addi", "arith.muli", "arith.divsi", "arith.remsi", "arith.cmpi"}:
        require(len(operands) == 2, "invalid integer operation arity")
        a, b, c = operands[0].type, operands[1].type, results[0].type
        require(integer(a) and _same(a, b), "expected matching i32 operands")
        require((c.shape == a.shape and c.kind == a.kind and c.elem == "i1" and not c.elem_is_ptr) if name == "arith.cmpi" else _same(a, c), "invalid integer result type")
        if name == "arith.cmpi":
            require(op.attrs.get("predicate") in {"eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"}, "missing comparison predicate")
    elif name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.maxnumf", "math.exp"}:
        require(len(operands) == (1 if name == "math.exp" else 2), "invalid floating arity")
        require(_floating(results[0].type) and all(_same(v.type, results[0].type) for v in operands), "expected matching floating types")
    elif name in {"arith.truncf", "arith.extf", "tt.fp_to_fp"}:
        require(len(operands) == 1, "invalid conversion arity")
        a, b = operands[0].type, results[0].type
        require(a.kind == b.kind and a.kind in {"scalar", "tensor"} and a.shape == b.shape and not a.elem_is_ptr and not b.elem_is_ptr, "invalid conversion shape")
        pairs = {"arith.truncf": {("f32", "bf16"), ("f32", "f16")},
                 "arith.extf": {("bf16", "f32"), ("f16", "f32")},
                 "tt.fp_to_fp": {(x, "f8E4M3FN") for x in ("bf16", "f16", "f32")} | {("f8E4M3FN", x) for x in ("bf16", "f16", "f32")}}
        require((a.elem, b.elem) in pairs[name], "unsupported conversion pair")
        if name == "tt.fp_to_fp":
            require(op.attrs.get("rounding") in {"rtne", "rtz"} if b.elem == "f8E4M3FN" else not op.attrs, "explicit supported narrowing rounding required; widening has no rounding")
    elif name == "tt.dot":
        require(len(operands) == 3, "dot requires two inputs and accumulator")
        a, b, c = [v.type for v in operands]
        require(all(t.kind == "tensor" and len(t.shape) == 2 and not t.elem_is_ptr for t in (a, b, c)), "only rank-two dot is supported")
        require(a.elem == b.elem and a.elem in _FLOAT and c.elem == "f32" and _same(c, results[0].type), "unsupported dot types/accumulator")
        require(a.shape[1] == b.shape[0] and c.shape == (a.shape[0], b.shape[1]), "dot contraction shape mismatch")
        require(op.attrs.get("inputPrecision", "ieee" if resolve_dot_defaults else None) in {"ieee", "tf32", "tf32x3"}, "unsupported dot precision")
        if "maxNumImpreciseAcc" in op.attrs:
            require(op.attr_int("maxNumImpreciseAcc") >= 0, "invalid dot precision bound")
    elif name == "tt.trans":
        require(len(operands) == 1, "invalid transpose arity")
        a, b = operands[0].type, results[0].type
        require(a.kind == b.kind == "tensor" and not a.elem_is_ptr and not b.elem_is_ptr,
                "transpose requires value tensors, not pointers")
        require(a.elem == b.elem, "transpose changes element type")
        order = transpose_order(op)
        require(sorted(order) == list(range(len(a.shape))), "transpose order must be a complete dimension permutation")
        require(b.shape == tuple(a.shape[i] for i in order), "transpose result shape does not match permutation")
    elif name in {"tt.broadcast", "tt.expand_dims", "tt.reshape"}:
        require(len(operands) == 1, "invalid shape operation arity")
        a, b = operands[0].type, results[0].type
        require(a.kind == b.kind == "tensor" and a.elem == b.elem and a.elem_is_ptr == b.elem_is_ptr, "shape transform changes element type")
        if name == "tt.broadcast":
            require(len(a.shape) == len(b.shape) and all(x == y or x == 1 for x, y in zip(a.shape, b.shape)), "invalid broadcast dimensions")
        elif name == "tt.expand_dims":
            axis = op.attr_int("axis")
            require(0 <= axis <= len(a.shape) and b.shape == a.shape[:axis] + (1,) + a.shape[axis:], "invalid expand_dims axis/shape")
        else:
            from math import prod
            require(prod(a.shape) == prod(b.shape), "reshape element count mismatch")
    elif name == "tt.reduce":
        require(len(operands) == 1, "only single-input reductions are supported")
        a, b = operands[0].type, results[0].type
        axis = op.attr_int("axis")
        require(a.kind == "tensor" and _floating(a) and 0 <= axis < len(a.shape), "invalid reduction input/axis")
        shape = a.shape[:axis] + a.shape[axis+1:]
        require(b.shape == shape and b.elem == a.elem and not b.elem_is_ptr and b.kind == ("tensor" if shape else "scalar"), "invalid reduction result")
        require(len(op.regions) == 1 and len(op.regions[0]) == 1, "expected one reduction block")
        block = op.regions[0][0]
        require(len(block.args) == 2 and all(v.type.kind == "scalar" and v.type.elem == a.elem for v in block.args), "invalid reduction arguments")
        require(len(block.ops) == 2, "only a single sum/max combiner is supported")
        combine, ret = block.ops
        require(combine.name in {"arith.addf", "arith.maxnumf"} and len(combine.operands) == 2 and all(x is y for x,y in zip(combine.operands, block.args)), "unsupported reduction combiner or capture")
        require(ret.name == "tt.reduce.return" and len(combine.results) == 1 and len(ret.operands) == 1 and ret.operands[0] is combine.results[0], "invalid reduction return")
    else:
        load = name == "tt.load"
        require(len(operands) in ((1, 2, 3) if load else (2, 3)), "invalid memory arity")
        p = operands[0].type
        value = results[0].type if load else operands[1].type
        require(pointer(p) and value.elem == p.elem and value.shape == p.shape and not value.elem_is_ptr and value.kind == ("tensor" if p.kind == "tensor" else "scalar"), "invalid memory value type")
        mask = operands[1 if load else 2] if len(operands) >= (2 if load else 3) else None
        if mask:
            require(mask.type.elem == "i1" and mask.type.shape == p.shape and not mask.type.elem_is_ptr and mask.type.kind == value.kind, "invalid mask")
        if load and len(operands) == 3:
            require(_same(operands[2].type, value), "invalid masked-load other value")


def _positive_zero(literal):
    if literal.startswith("dense<") and literal.endswith(">"):
        literal = literal[6:-1]
    try:
        number = float(literal)
        return number == 0 and math.copysign(1, number) > 0
    except ValueError:
        return False


def analyze_operations(function: TFunc, *, resolve_dot_defaults=False) -> OperationAnalysis:
    """Validate a complete operation graph without selecting target numerics.

    Original attributes, reduction bodies and memory accesses remain authoritative.
    No runtime bounds, aliasing, hardware eligibility or reassociation is implied.
    """
    source = _strip_locs(function.source)
    source = "\n".join(line.split("//", 1)[0] for line in source.splitlines())
    arg = r"%\w+\s*:\s*" + _TYPE + r"(?:\s*\{tt\.divisibility\s*=\s*\d+\s*:\s*i32\})?\s*"
    header = (r"tt\.func(?: public)? @" + re.escape(function.name) + r"\(\s*(?:" + arg
              + r"(?:,\s*" + arg + r")*)?\)\s*(?:attributes\s*\{noinline\s*=\s*false\}\s*)?\{")
    if len(re.findall(r"\btt\.func\b", source)) != 1 or not re.search(header, source):
        raise UnsupportedKernel("unsupported function signature, attributes, or multiple functions")
    if not any(line.strip() == "module {" for line in source.splitlines()):
        raise UnsupportedKernel("unsupported module attributes")
    if [id(v) for v in function.body.args] != [id(v) for v in function.args] or not function.body.ops or function.body.ops[-1].name != "tt.return":
        raise UnsupportedKernel("invalid function arguments or missing terminal return")
    available = {v.id: v for v in function.args}
    if len(available) != len(function.args):
        raise UnsupportedKernel("duplicate function argument identity")
    definitions = set(available)
    operations, memory, reductions, relus = [], [], [], []

    def walk(block, scope, reduction=False):
        for op in block.ops:
            if op.name not in _PATTERNS and op.name not in {"tt.reduce", "tt.reduce.return"}:
                _reject(op, "unsupported operation")
            if op.regions and op.name != "tt.reduce":
                _reject(op, "unsupported regions/control flow")
            if op.name == "tt.return" and (reduction or op is not function.body.ops[-1]):
                _reject(op, "nonterminal return")
            if op.name == "tt.reduce.return" and not reduction:
                _reject(op, "reduction return outside combiner")
            _validate_spelling(op)
            try:
                _validate_types(op, resolve_dot_defaults)
            except (KeyError, ValueError, IndexError) as exc:
                _reject(op, f"invalid or missing semantic attribute: {exc}")
            for value in op.operands:
                if scope.get(value.id) is not value:
                    _reject(op, "undefined, non-dominating, or inconsistent SSA dependency")
            operations.append(op)
            if op.name == "tt.reduce":
                body = op.regions[0][0]
                inner = dict(scope)
                for value in body.args:
                    if value.id in definitions or not value.is_block_arg or value.defining_op is not None:
                        _reject(op, "invalid reduction block argument")
                    inner[value.id] = value
                    definitions.add(value.id)
                reductions.append(Reduction(op, op.attr_int("axis"), body.ops[0]))
                walk(body, inner, True)
            for value in op.results:
                if value.id in definitions or value.defining_op is not op or value.is_block_arg:
                    _reject(op, "invalid SSA result definition")
                definitions.add(value.id)
                scope[value.id] = value
            if op.name == "tt.load":
                memory.append(MemoryAccess(op, op.operands[0], op.results[0],
                                          op.operands[1] if len(op.operands) > 1 else None,
                                          op.operands[2] if len(op.operands) > 2 else None))
            elif op.name == "tt.store":
                memory.append(MemoryAccess(op, op.operands[0], op.operands[1],
                                          op.operands[2] if len(op.operands) > 2 else None, None))
            elif op.name == "arith.maxnumf":
                for v in op.operands:
                    producer = v.defining_op
                    if producer and producer.name == "arith.constant" and _positive_zero(producer.attrs.get("value", "")):
                        relus.append(op)
                        break
    walk(function.body, available)
    # Generic region attributes can also be printed after the closing region;
    # the existing header scanner cannot structurally expose those. Fail closed.
    tails = re.findall(r"^\s*\}\).*?$", source, re.M)
    tail_pattern = r"\s*\}\)\s*:\s*\(" + _TYPE + r"\)\s*->\s*" + _TYPE + r"\s*"
    if len(tails) != len(reductions) or any(not re.fullmatch(tail_pattern, tail) for tail in tails):
        raise UnsupportedKernel("unsupported region suffix or attributes")
    return OperationAnalysis(function, tuple(operations), tuple(tuple(op.operands) for op in operations), tuple(memory), tuple(reductions), tuple(relus), resolve_dot_defaults)
