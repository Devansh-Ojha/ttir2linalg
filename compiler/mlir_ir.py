"""A small structured MLIR IR: Values, Ops, Blocks, Funcs, Module.

Ops hold real operand *Value objects* (so SSA dependencies are structural, not
strings). Printing resolves names from those objects; verify_ssa() checks that
every operand is defined before use in a visible scope.

Op format templates use placeholders:
  $oK   operand K name        $rK    result K name
  $bargsK  "%a: T, %b: U" of region K's block arguments
  $regK    region K body ops  $blkK  region K with a "^bb0(args):" header
"""
from __future__ import annotations
import re
from contextlib import contextmanager
from typing import List, Optional


class Value:
    def __init__(self, type_: str, owner=None):
        self.type = type_
        self.owner = owner
        self.name: Optional[str] = None


class Block:
    def __init__(self, arg_types=()):
        self.args = [Value(t, self) for t in arg_types]
        self.ops: List["Op"] = []


class Op:
    def __init__(self, fmt: str, operands, result_types, regions=()):
        self.fmt = fmt
        self.operands = list(operands)
        self.results = [Value(t, self) for t in result_types]
        self.regions = list(regions)


class Func:
    def __init__(self, name: str, arg_types):
        self.name = name
        self.body = Block(arg_types)


class Module:
    def __init__(self):
        self.funcs: List[Func] = []


class Builder:
    def __init__(self):
        self.block: Optional[Block] = None

    @contextmanager
    def at(self, block: Block):
        prev, self.block = self.block, block
        try:
            yield
        finally:
            self.block = prev

    def emit(self, fmt, operands, result_types, regions=()) -> Op:
        op = Op(fmt, operands, result_types, regions)
        self.block.ops.append(op)
        return op


# ------------------------------------------------------------------ printing
_TOK = re.compile(r"\$(bargs|reg|blk|o|r)(\d+)")


class _State:
    def __init__(self):
        self.n = 0      # numbered results  %0 %1 ...
        self.a = 0      # block arguments   %arg0 ...


def _name_args(block: Block, st: _State):
    for a in block.args:
        a.name = "%%arg%d" % st.a
        st.a += 1


def _nm(v: Value) -> str:
    if v.name is None:
        raise ValueError("operand used before it is defined")
    return v.name


def _bargs(block: Block) -> str:
    return ", ".join("%s: %s" % (a.name, a.type) for a in block.args)


def _block_lines(block: Block, pad: str, st: _State) -> List[str]:
    lines: List[str] = []
    for op in block.ops:
        for r in op.results:
            r.name = "%%%d" % st.n
            st.n += 1
        for reg in op.regions:
            _name_args(reg, st)

        def rep(m):
            kind, k = m.group(1), int(m.group(2))
            if kind == "o":
                return _nm(op.operands[k])
            if kind == "r":
                return op.results[k].name
            if kind == "bargs":
                return _bargs(op.regions[k])
            body = _block_lines(op.regions[k], pad + "  ", st)
            head = ["%s  ^bb0(%s):" % (pad, _bargs(op.regions[k]))] if kind == "blk" else []
            return "\n" + "\n".join(head + body) + "\n" + pad

        lines.append(pad + _TOK.sub(rep, op.fmt))
    return lines


def print_module(m: Module) -> str:
    out = ["module {"]
    for f in m.funcs:
        st = _State()
        _name_args(f.body, st)
        out.append("  func.func @%s(%s) {" % (f.name, _bargs(f.body)))
        out.extend(_block_lines(f.body, "    ", st))
        out.append("  }")
    out.append("}")
    return "\n".join(out) + "\n"


# -------------------------------------------------------------- verification
def verify_ssa(m: Module) -> List[str]:
    """Internal check: every operand must be defined before use in a visible scope."""
    errs: List[str] = []

    def walk(block: Block, scopes):
        scope = set(id(a) for a in block.args)
        scopes = scopes + [scope]
        for op in block.ops:
            for v in op.operands:
                if not any(id(v) in s for s in scopes):
                    errs.append("use before def in op: %s" % op.fmt)
            for reg in op.regions:
                walk(reg, scopes)
            scope.update(id(r) for r in op.results)

    for f in m.funcs:
        walk(f.body, [])
    return errs
