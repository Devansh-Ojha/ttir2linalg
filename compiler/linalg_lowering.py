"""Lower parsed TTIR (compiler.ttir_reader) to a real MLIR module (compiler.mlir_ir).

Design (SPMD -> tensor/memref):
  * tt.get_program_id  -> extra leading function argument (%pid_x: i32), one per axis used
  * !tt.ptr<T> kernel arg -> memref<?xT> argument; a pointer value is tracked as
    (base memref, i32 element offset) -- tt.addptr just adds to the offset
  * tensor-of-pointers load -> linalg.generic gather (memref.load in the body)
  * elementwise arith on tensors -> linalg.add/sub/mul/div, scalars stay arith.*
  * tt.reduce -> linalg.reduce with the region translated (reduce.return -> linalg.yield)
Unsupported ops raise NotImplementedError naming the op.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Union

from compiler.mlir_ir import Block, Builder, Func, Module, Value
from compiler.ttir_reader import TBlock, TFunc, TOp, TTIRModule, TType, TValue

_NAMED_BINARY = {"addf": "add", "addi": "add", "subf": "sub", "subi": "sub",
                 "mulf": "mul", "muli": "mul", "divf": "div"}
_ARITH_BINARY = ["addf", "subf", "mulf", "divf", "addi", "subi", "muli",
                 "divsi", "remsi", "andi", "ori", "xori", "maximumf", "minimumf",
                 "maxnumf", "minnumf", "maxsi", "minsi"]
_REDUCE_IDENTITY = {"arith.addf": "0.000000e+00", "arith.mulf": "1.000000e+00",
                    "arith.addi": "0", "arith.muli": "1"}


class PtrVal:
    """A Triton pointer: base memref plus an element offset (None means 0)."""
    def __init__(self, base: Value, offset: Optional[Value]):
        self.base = base
        self.offset = offset


Lowered = Union[Value, PtrVal]


def _tensor(shape, elem) -> str:
    return "tensor<%s%s>" % ("".join("%dx" % d for d in shape), elem)


class Lowerer:
    def __init__(self, limit: Optional[int] = None):
        self.limit = limit
        self.b = Builder()
        self.vmap: Dict[int, Lowered] = {}
        self.pid: Dict[int, Value] = {}

    # ------------------------------------------------------------ plumbing
    def get(self, tv: TValue) -> Lowered:
        return self.vmap[tv.id]

    def val(self, tv: TValue) -> Value:
        v = self.get(tv)
        if not isinstance(v, Value):
            raise NotImplementedError("expected a value, got a pointer for %s" % tv.name)
        return v

    def ptr(self, tv: TValue) -> PtrVal:
        v = self.get(tv)
        if not isinstance(v, PtrVal):
            raise NotImplementedError("expected a pointer for %s" % tv.name)
        return v

    def mtype(self, t: TType) -> str:
        if t.kind == "scalar":
            return t.elem
        if t.kind == "tensor":
            return _tensor(t.shape, "i32" if t.elem_is_ptr else t.elem)
        raise NotImplementedError("no MLIR value type for %s" % t.raw)

    # --------------------------------------------------------- tiny emitters
    def empty(self, ty: str) -> Value:
        return self.b.emit("$r0 = tensor.empty() : %s" % ty, [], [ty]).results[0]

    def const(self, text: str, ty: str) -> Value:
        return self.b.emit("$r0 = arith.constant %s : %s" % (text, ty), [], [ty]).results[0]

    def fill(self, scalar: Value, ty: str) -> Value:
        e = self.empty(ty)
        return self.b.emit("$r0 = linalg.fill ins($o0 : %s) outs($o1 : %s) -> %s"
                           % (scalar.type, ty, ty), [scalar, e], [ty]).results[0]

    def index_cast(self, v: Value, to: str) -> Value:
        return self.b.emit("$r0 = arith.index_cast $o0 : %s to %s" % (v.type, to),
                           [v], [to]).results[0]

    def named_binary(self, name: str, a: Value, c: Value) -> Value:
        ty = a.type
        e = self.empty(ty)
        return self.b.emit("$r0 = linalg.%s ins($o0, $o1 : %s, %s) outs($o2 : %s) -> %s"
                           % (name, ty, ty, ty, ty), [a, c, e], [ty]).results[0]

    @staticmethod
    def _generic_attrs(rank: int, n_maps: int) -> str:
        dims = ", ".join("d%d" % i for i in range(rank))
        m = "affine_map<(%s) -> (%s)>" % (dims, dims)
        return "{indexing_maps = [%s], iterator_types = [%s]}" % (
            ", ".join([m] * n_maps), ", ".join(['"parallel"'] * rank))

    # --------------------------------------------------------------- driver
    def lower_module(self, tmod: TTIRModule) -> Module:
        m = Module()
        for tf in tmod.funcs:
            m.funcs.append(self.lower_func(tf))
        return m

    def lower_func(self, tf: TFunc) -> Func:
        ops = tf.body.ops
        keep = ops if self.limit is None else ops[:self.limit]
        axes = sorted(set(o.attr_int("axis") for o in keep if o.name == "tt.get_program_id"))
        arg_types: List[str] = ["i32"] * len(axes)
        for a in tf.args:
            if a.type.kind == "ptr":
                arg_types.append("memref<?x%s>" % a.type.elem)
            elif a.type.kind == "scalar":
                arg_types.append(a.type.elem)
            else:
                raise NotImplementedError("kernel argument type %s" % a.type.raw)
        f = Func(tf.name, arg_types)
        fargs = list(f.body.args)
        for ax in axes:
            self.pid[ax] = fargs.pop(0)
        for a in tf.args:
            v = fargs.pop(0)
            self.vmap[a.id] = PtrVal(v, None) if a.type.kind == "ptr" else v
        with self.b.at(f.body):
            self.lower_ops(keep)
            if not keep or keep[-1].name != "tt.return":
                self.b.emit("return", [], [])
        return f

    def lower_ops(self, ops: List[TOp]):
        for op in ops:
            fn = getattr(self, "op_" + op.name.replace(".", "_"), None)
            if fn is None:
                if op.name.startswith("arith.") and op.name[6:] in _ARITH_BINARY:
                    fn = self.op_arith_binary
                else:
                    raise NotImplementedError("unsupported op '%s' (op #%d)" % (op.name, op.index))
            fn(op)

    # ------------------------------------------------------------------ ops
    def op_arith_constant(self, op: TOp):
        if op.results[0].type.kind != "scalar":
            raise NotImplementedError("tensor-valued arith.constant")
        self.vmap[op.results[0].id] = self.const(op.attrs["value"], op.attrs["type"])

    def op_tt_get_program_id(self, op: TOp):
        self.vmap[op.results[0].id] = self.pid[op.attr_int("axis")]

    def op_tt_make_range(self, op: TOp):
        rt = op.results[0].type
        ty = self.mtype(rt)
        start = op.attr_int("start")
        init = self.empty(ty)
        blk = Block([rt.elem])
        with self.b.at(blk):
            idx = self.b.emit("$r0 = linalg.index 0 : index", [], ["index"]).results[0]
            v = self.index_cast(idx, rt.elem)
            if start != 0:
                v = self.b.emit("$r0 = arith.addi $o0, $o1 : %s" % rt.elem,
                                [v, self.const(str(start), rt.elem)], [rt.elem]).results[0]
            self.b.emit("linalg.yield $o0 : %s" % rt.elem, [v], [])
        r = self.b.emit("$r0 = linalg.generic %s outs($o0 : %s) {$blk0} -> %s"
                        % (self._generic_attrs(1, 1), ty, ty), [init], [ty], [blk])
        self.vmap[op.results[0].id] = r.results[0]

    def op_tt_splat(self, op: TOp):
        src = self.get(op.operands[0])
        rt = op.results[0].type
        if isinstance(src, PtrVal):
            off = src.offset if src.offset is not None else self.const("0", "i32")
            self.vmap[op.results[0].id] = PtrVal(src.base, self.fill(off, self.mtype(rt)))
        else:
            self.vmap[op.results[0].id] = self.fill(src, self.mtype(rt))

    def op_tt_addptr(self, op: TOp):
        p = self.ptr(op.operands[0])
        off = self.val(op.operands[1])
        if p.offset is None:
            new = off
        elif off.type.startswith("tensor"):
            new = self.named_binary("add", p.offset, off)
        else:
            new = self.b.emit("$r0 = arith.addi $o0, $o1 : %s" % off.type,
                              [p.offset, off], [off.type]).results[0]
        self.vmap[op.results[0].id] = PtrVal(p.base, new)

    def op_tt_load(self, op: TOp):
        if len(op.operands) != 1:
            raise NotImplementedError("masked/other tt.load")
        p = self.ptr(op.operands[0])
        rt = op.results[0].type
        mty = p.base.type
        if rt.kind == "scalar":
            off = p.offset if p.offset is not None else self.const("0", "i32")
            idx = self.index_cast(off, "index")
            r = self.b.emit("$r0 = memref.load $o0[$o1] : %s" % mty, [p.base, idx], [rt.elem])
            self.vmap[op.results[0].id] = r.results[0]
            return
        ty = self.mtype(rt)
        offty = p.offset.type
        init = self.empty(ty)
        blk = Block([offty.split("x")[-1][:-1], rt.elem])
        with self.b.at(blk):
            idx = self.index_cast(blk.args[0], "index")
            v = self.b.emit("$r0 = memref.load $o0[$o1] : %s" % mty, [p.base, idx], [rt.elem]).results[0]
            self.b.emit("linalg.yield $o0 : %s" % rt.elem, [v], [])
        r = self.b.emit("$r0 = linalg.generic %s ins($o0 : %s) outs($o1 : %s) {$blk0} -> %s"
                        % (self._generic_attrs(len(rt.shape), 2), offty, ty, ty),
                        [p.offset, init], [ty], [blk])
        self.vmap[op.results[0].id] = r.results[0]

    def op_arith_binary(self, op: TOp):
        a, c = self.val(op.operands[0]), self.val(op.operands[1])
        short = op.name[6:]
        if a.type.startswith("tensor") and short in _NAMED_BINARY:
            r = self.named_binary(_NAMED_BINARY[short], a, c)
        else:
            r = self.b.emit("$r0 = arith.%s $o0, $o1 : %s" % (short, a.type),
                            [a, c], [a.type]).results[0]
        self.vmap[op.results[0].id] = r

    def op_tt_reshape(self, op: TOp):
        src, dst = op.operands[0].type, op.results[0].type
        if src.shape != dst.shape:
            raise NotImplementedError("tt.reshape that changes shape %s -> %s" % (src.raw, dst.raw))
        self.vmap[op.results[0].id] = self.val(op.operands[0])   # identity reshape

    def op_tt_reduce(self, op: TOp):
        if len(op.operands) != 1 or len(op.results) != 1:
            raise NotImplementedError("multi-operand tt.reduce")
        src_t = op.operands[0].type
        axis = op.attr_int("axis")
        body = op.regions[0][0]
        comb = [o.name for o in body.ops if o.name != "tt.reduce.return"]
        if len(comb) != 1 or comb[0] not in _REDUCE_IDENTITY:
            raise NotImplementedError("tt.reduce combiner %s (supported: %s)"
                                      % (comb, sorted(_REDUCE_IDENTITY)))
        elem = src_t.elem
        out_shape = tuple(d for i, d in enumerate(src_t.shape) if i != axis)
        out_ty = _tensor(out_shape, elem)
        init = self.fill(self.const(_REDUCE_IDENTITY[comb[0]], elem), out_ty)
        blk = Block([elem, elem])                       # (%in, %acc)
        for ba, nb in zip(body.args, blk.args):
            self.vmap[ba.id] = nb
        with self.b.at(blk):
            self.lower_ops(body.ops)
        src = self.val(op.operands[0])
        r = self.b.emit("$r0 = linalg.reduce ins($o0 : %s) outs($o1 : %s) dimensions = [%d] ($bargs0) {$reg0}"
                        % (src.type, out_ty, axis), [src, init], [out_ty], [blk]).results[0]
        if not out_shape:                                # scalar result
            r = self.b.emit("$r0 = tensor.extract $o0[] : %s" % out_ty, [r], [elem]).results[0]
        self.vmap[op.results[0].id] = r

    def op_tt_reduce_return(self, op: TOp):
        v = self.val(op.operands[0])
        self.b.emit("linalg.yield $o0 : %s" % v.type, [v], [])

    def op_tt_store(self, op: TOp):
        if len(op.operands) != 2:
            raise NotImplementedError("masked tt.store")
        p = self.ptr(op.operands[0])
        v = self.val(op.operands[1])
        if v.type.startswith("tensor"):
            raise NotImplementedError("tensor-valued tt.store")
        off = p.offset if p.offset is not None else self.const("0", "i32")
        idx = self.index_cast(off, "index")
        self.b.emit("memref.store $o0, $o1[$o2] : %s" % p.base.type, [v, p.base, idx], [])

    def op_tt_return(self, op: TOp):
        self.b.emit("return", [], [])
