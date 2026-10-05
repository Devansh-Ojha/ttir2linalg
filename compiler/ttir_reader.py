"""Read TTIR through Triton 3.4.0's libtriton bindings into a structured Python IR.

Facts about the 3.4.0 bindings this relies on (all verified):
  * ir.parse_mlir_module(path, ctx) takes a FILE PATH, not text.
  * module.walk() visits ops POST-order (region bodies before the owning op).
  * value.id() is a stable identity for op results and block arguments.
  * Types are opaque; only str(type) works.
  * Integer attributes (start/end/axis/constant value) are NOT readable from
    Python, so they are recovered from the TTIR text, matched to walked ops by
    (op name, ordinal) and cross-checked by count.
"""
from __future__ import annotations
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from triton._C.libtriton import ir

_PTR_RE = re.compile(r"^!tt\.ptr<\s*([^,>]+?)\s*(?:,\s*\d+\s*)?>$")


@dataclass(frozen=True)
class TType:
    kind: str                      # "scalar" | "ptr" | "tensor"
    elem: str                      # element type (pointee type for ptr)
    shape: Tuple[int, ...] = ()
    elem_is_ptr: bool = False      # tensor of pointers
    raw: str = ""

    def __str__(self):
        return self.raw


def parse_type(s: str) -> TType:
    s = s.strip()
    m = _PTR_RE.match(s)
    if m:
        return TType("ptr", m.group(1), (), False, s)
    if s.startswith("tensor<") and s.endswith(">"):
        inner = s[len("tensor<"):-1]
        m = re.match(r"^((?:\d+x)*)(.+)$", inner)
        dims = tuple(int(d) for d in m.group(1).split("x") if d)
        et = m.group(2)
        pm = _PTR_RE.match(et)
        if pm:
            return TType("tensor", pm.group(1), dims, True, s)
        return TType("tensor", et, dims, False, s)
    return TType("scalar", s, (), False, s)


@dataclass
class TValue:
    id: int
    type: TType
    name: str                       # best-effort original SSA name, e.g. "%12"
    is_block_arg: bool = False
    defining_op: Optional["TOp"] = None


@dataclass
class TBlock:
    id: int
    args: List[TValue] = field(default_factory=list)
    ops: List["TOp"] = field(default_factory=list)


@dataclass
class TOp:
    name: str
    operands: List[TValue] = field(default_factory=list)
    results: List[TValue] = field(default_factory=list)
    attrs: Dict[str, str] = field(default_factory=dict)   # raw text, e.g. "4 : i32"
    regions: List[List[TBlock]] = field(default_factory=list)
    index: int = 0

    def attr_int(self, key: str) -> int:
        return int(self.attrs[key].split(":")[0].strip())


@dataclass
class TFunc:
    name: str
    args: List[TValue]
    body: TBlock


# ----------------------------------------------------------------- text scan
def _strip_locs(text: str) -> str:
    text = "\n".join(l for l in text.splitlines() if not l.startswith("#loc"))
    out, i = [], 0
    while i < len(text):
        j = text.find("loc(", i)
        if j == -1:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k, d = j + 4, 1
        while d and k < len(text):
            d += (text[k] == "(") - (text[k] == ")")
            k += 1
        i = k
    return "".join(out)


_LINE_RE = re.compile(
    r'^\s*(?:(%[\w#]+)(?::\d+)?\s*=\s*)?(?:"([\w.]+)"|([a-z_]+\.[a-z_0-9.]+))')
_ATTR_DICT_RE = re.compile(r"<?\{([^{}]*)\}>?")


def _split_top(s: str) -> List[str]:
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def _scan_text(ttir: str):
    """Return [(opname, result_name_or_None, attrs_dict)] in textual order."""
    found = []
    for line in _strip_locs(ttir).splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        res, qname, bname = m.group(1), m.group(2), m.group(3)
        name = qname or bname
        attrs: Dict[str, str] = {}
        if name == "tt.get_program_id":
            pm = re.search(r"tt\.get_program_id\s+([xyz])", line)
            if pm:
                attrs["axis"] = str("xyz".index(pm.group(1))) + " : i32"
        elif name == "arith.constant":
            cm = re.search(r"arith\.constant\s+(.+?)\s*:\s*(\S.*)$", line)
            if cm:
                attrs["value"] = cm.group(1)
                attrs["type"] = cm.group(2).strip()
        elif name != "tt.func":
            am = _ATTR_DICT_RE.search(line)
            if am:
                for kv in _split_top(am.group(1)):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        attrs[k.strip()] = v.strip()
        found.append((name, res, attrs))
    return found


# --------------------------------------------------------------------- module
class TTIRModule:
    """Holds the libtriton context + module (kept alive) and the structured IR."""

    def __init__(self, ctx, mod, funcs: List[TFunc], all_ops: List[TOp]):
        self.ctx = ctx
        self.mod = mod
        self.funcs = funcs
        self.all_ops = all_ops


def parse_ttir(ttir: str) -> TTIRModule:
    ctx = ir.context()
    ir.load_dialects(ctx)
    with tempfile.NamedTemporaryFile("w", suffix=".ttir", delete=False) as f:
        f.write(ttir)
        path = f.name
    try:
        mod = ir.parse_mlir_module(path, ctx)      # NOTE: path, not text
    finally:
        os.unlink(path)
    if not mod.verify():
        raise RuntimeError("Triton failed to verify the parsed TTIR module")

    # pass 1: collect all ops (post-order) and all blocks
    raw_ops = []
    mod.walk(lambda op: raw_ops.append(op))

    blocks: Dict[int, TBlock] = {}
    values: Dict[int, TValue] = {}
    region_blocks: Dict[int, List[int]] = {}
    for op in raw_ops:
        if op.get_name() == "builtin.module":
            continue
        blk = op.get_block()
        if blk.id() in blocks:
            continue
        tb = TBlock(blk.id())
        for i in range(blk.get_num_arguments()):
            a = blk.get_argument(i)
            tv = TValue(a.id(), parse_type(str(a.get_type())),
                        "%%blkarg%d_%d" % (len(blocks), i), is_block_arg=True)
            values[tv.id] = tv
            tb.args.append(tv)
        blocks[blk.id()] = tb
        region_blocks.setdefault(blk.get_parent().id(), []).append(blk.id())

    # pass 2: build ops (post-order == program order within each block)
    tops: List[TOp] = []
    for idx, op in enumerate(raw_ops):
        name = op.get_name()
        if name == "builtin.module":
            continue
        t = TOp(name=name, index=idx)
        for i in range(op.get_num_operands()):
            oid = op.get_operand(i).id()
            if oid not in values:
                raise RuntimeError("operand of %s has unknown value id" % name)
            t.operands.append(values[oid])
        for i in range(op.get_num_results()):
            r = op.get_result(i)
            tv = TValue(r.id(), parse_type(str(r.get_type())), "%?", False, t)
            values[tv.id] = tv
            t.results.append(tv)
        for i in range(op.get_num_regions()):
            rid = op.get_region(i).id()
            t.regions.append([blocks[b] for b in region_blocks.get(rid, [])])
        blocks[op.get_block().id()].ops.append(t)
        tops.append(t)
        if name == "tt.func":
            t.attrs["sym_name"] = op.get_str_attr("sym_name")

    # attributes + original SSA names from the text, matched by (name, ordinal)
    by_name: Dict[str, list] = {}
    for name, res, attrs in _scan_text(ttir):
        by_name.setdefault(name, []).append((res, attrs))
    seen: Dict[str, int] = {}
    for t in tops:
        if t.name == "tt.func":
            continue
        k = seen.get(t.name, 0)
        seen[t.name] = k + 1
        lst = by_name.get(t.name, [])
        if k >= len(lst):
            raise RuntimeError("text scan found fewer '%s' ops than the walk" % t.name)
        res, attrs = lst[k]
        t.attrs.update(attrs)
        if res and t.results:
            t.results[0].name = res
    for name, lst in by_name.items():
        if name != "tt.func" and len(lst) != seen.get(name, 0):
            raise RuntimeError("text scan / walk disagree on count of '%s'" % name)

    funcs = []
    for t in tops:
        if t.name == "tt.func":
            body = t.regions[0][0]
            for i, a in enumerate(body.args):
                a.name = "%%arg%d" % i
            funcs.append(TFunc(t.attrs["sym_name"], list(body.args), body))
    return TTIRModule(ctx, mod, funcs, tops)


# ----------------------------------------------------------------- inspection
def dump(tmod: TTIRModule) -> str:
    """Human-readable dump: name, operands, results+types, attrs, regions."""
    lines = []

    def block(b: TBlock, ind: int):
        pad = "  " * ind
        if b.args:
            lines.append("%sblock args: %s" % (
                pad, ", ".join("%s:%s" % (a.name, a.type) for a in b.args)))
        for o in b.ops:
            res = ", ".join("%s:%s" % (r.name, r.type) for r in o.results) or "-"
            opr = ", ".join(v.name for v in o.operands) or "-"
            lines.append("%s%s  operands=[%s]  results=[%s]  attrs=%s" % (
                pad, o.name, opr, res, o.attrs or "{}"))
            for ri, reg in enumerate(o.regions):
                lines.append("%s  region %d:" % (pad, ri))
                for bb in reg:
                    block(bb, ind + 2)

    for f in tmod.funcs:
        lines.append("func @%s(%s)" % (f.name, ", ".join(
            "%s:%s" % (a.name, a.type) for a in f.args)))
        block(f.body, 1)
    return "\n".join(lines)
