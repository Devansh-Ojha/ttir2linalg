"""TTIR to standard MLIR using the MLIR Python IR bindings.

The Triton ``libtriton.ir`` bindings intentionally expose Triton builders, not
the standard Linalg/MemRef/Tensor builders.  This backend consumes the existing
TTIR parser output and constructs standard operations through
``mlir.ir.Operation.create``.  No MLIR is assembled as text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from compiler.ttir_reader import TFunc, TOp, TTIRModule, TValue


@dataclass
class Ptr:
    base: object
    offset: object | None = None


class StandardLowerer:
    def __init__(self, tmod: TTIRModule):
        try:
            from mlir import ir
        except ImportError as exc:
            raise RuntimeError(
                "standard lowering requires the EDA MLIR Python bindings "
                "(import mlir.ir)"
            ) from exc
        try:
            from mlir.dialects import arith, func, linalg, memref, tensor  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "standard lowering requires arith, func, linalg, memref, "
                "and tensor MLIR dialect bindings"
            ) from exc
        self.ir = ir
        self.tmod = tmod
        self.ssa: Dict[int, object] = {}
        self.ptrs: Dict[int, Ptr] = {}
        self.block = None

    def type(self, t):
        ir = self.ir
        if t.kind == "scalar":
            return {
                "i1": ir.IntegerType.get_signless(1),
                "i32": ir.IntegerType.get_signless(32),
                "f32": ir.F32Type.get(),
                "f64": ir.F64Type.get(),
            }[t.elem]
        if t.kind == "tensor":
            elem = self.type(type("_T", (), {"kind": "scalar", "elem": t.elem})())
            return ir.RankedTensorType.get(list(t.shape), elem)
        if t.kind == "ptr":
            elem = self.type(type("_T", (), {"kind": "scalar", "elem": t.elem})())
            return ir.MemRefType.get([-1], elem)
        raise NotImplementedError("standard type conversion: %s" % t.raw)

    def emit(self, name, operands=(), results=(), attrs=None, regions=()):
        op = self.ir.Operation.create(
            name,
            results=list(results),
            operands=list(operands),
            attributes=attrs or {},
            regions=list(regions),
        )
        self.block.append(op)
        return op

    def constant(self, op: TOp):
        typ = self.type(op.results[0].type)
        value = op.attrs["value"].strip()
        attr = self.ir.IntegerAttr.get(typ, int(value)) if value.lstrip("-").isdigit() \
            else self.ir.FloatAttr.get(typ, float(value))
        return self.emit("arith.constant", results=[typ],
                         attrs={"value": attr}).results[0]

    def get(self, value: TValue):
        if value.id in self.ssa:
            return self.ssa[value.id]
        raise RuntimeError("unlowered TTIR SSA value %s" % value.name)

    def _tensor_binary(self, name, a, b, result_type):
        rank = len(result_type.shape)
        identity = self.ir.Attribute.parse(
            "affine_map<(%s) -> (%s)>" %
            (", ".join("d%d" % i for i in range(rank)),
             ", ".join("d%d" % i for i in range(rank)))
        )
        region = self.ir.Region()
        elem = result_type.element_type
        region.blocks.append(self.ir.Block([elem, elem, elem]))
        body = region.blocks[0]
        binary = self.ir.Operation.create(
            name, operands=list(body.arguments[:2]), results=[elem]
        )
        body.append(binary)
        body.append(self.ir.Operation.create(
            "linalg.yield", operands=[binary.results[0]]
        ))
        return self.emit(
            "linalg.generic", [a, b], [result_type],
            {
                "indexing_maps": self.ir.ArrayAttr.get([identity, identity, identity]),
                "iterator_types": self.ir.ArrayAttr.get(
                    [self.ir.StringAttr.get("parallel") for _ in range(rank)]
                ),
            },
            [region],
        ).results[0]

    def _load_tensor(self, pointer: Ptr, result_type):
        ir = self.ir
        shape = list(result_type.shape)
        empty = self.emit("tensor.empty", results=[result_type]).results[0]
        rank = len(shape)
        maps = ir.ArrayAttr.get([
            ir.Attribute.parse("affine_map<(%s) -> (%s)>" %
                              (", ".join("d%d" % i for i in range(rank)),
                               ", ".join("d%d" % i for i in range(rank))))
            for _ in range(2)
        ])
        iterators = ir.ArrayAttr.get(
            [ir.StringAttr.get("parallel") for _ in range(rank)]
        )
        region = ir.Region()
        region.blocks.append(ir.Block([
            pointer.offset.type.element_type, result_type.element_type
        ]))
        body = region.blocks[0]
        cast = ir.Operation.create(
            "arith.index_cast", operands=[body.arguments[0]],
            results=[ir.IndexType.get()]
        )
        body.append(cast)
        index = cast.results[0]
        load = ir.Operation.create(
            "memref.load", operands=[pointer.base, index],
            results=[result_type.element_type]
        )
        body.append(load)
        body.append(ir.Operation.create(
            "linalg.yield", operands=[load.results[0]]
        ))
        return self.emit(
            "linalg.generic", [pointer.offset, empty], [result_type],
            {"indexing_maps": maps, "iterator_types": iterators},
            [region],
        ).results[0]

    def lower(self, op: TOp):
        ir = self.ir
        if op.name == "arith.constant":
            self.ssa[op.results[0].id] = self.constant(op)
        elif op.name in {"arith.muli", "arith.mulf", "arith.addf"}:
            names = {"arith.muli": "arith.muli", "arith.mulf": "arith.mulf",
                     "arith.addf": "arith.addf"}
            a, b = (self.get(v) for v in op.operands)
            result_type = self.type(op.results[0].type)
            if op.results[0].type.kind == "tensor":
                value = self._tensor_binary(names[op.name], a, b, result_type)
            else:
                value = self.emit(names[op.name], [a, b], [result_type]).results[0]
            self.ssa[op.results[0].id] = value
        elif op.name == "tt.get_program_id":
            typ = self.type(op.results[0].type)
            zero = ir.IntegerAttr.get(typ, 0)
            self.ssa[op.results[0].id] = self.emit(
                "arith.constant", results=[typ], attrs={"value": zero}
            ).results[0]
        elif op.name == "tt.make_range":
            typ = self.type(op.results[0].type)
            elems = []
            for value in range(op.attr_int("start"), op.attr_int("end")):
                c = ir.IntegerAttr.get(typ.element_type, value)
                elems.append(self.emit(
                    "arith.constant", results=[typ.element_type],
                    attrs={"value": c}
                ).results[0])
            self.ssa[op.results[0].id] = self.emit(
                "tensor.from_elements", elems, [typ]
            ).results[0]
        elif op.name == "tt.splat":
            source = op.operands[0]
            if source.id in self.ptrs:
                self.ptrs[op.results[0].id] = self.ptrs[source.id]
            else:
                src = self.get(source)
                typ = self.type(op.results[0].type)
                self.ssa[op.results[0].id] = self.emit(
                    "tensor.splat", [src], [typ]
                ).results[0]
        elif op.name == "tt.reshape":
            src = self.get(op.operands[0])
            typ = self.type(op.results[0].type)
            shape = ir.DenseI64ArrayAttr.get(list(op.results[0].type.shape))
            self.ssa[op.results[0].id] = self.emit(
                "tensor.reshape", [src], [typ],
                {"shape": shape},
            ).results[0]
        elif op.name == "tt.addptr":
            base = self.ptrs[op.operands[0].id]
            offset = self.get(op.operands[1])
            self.ptrs[op.results[0].id] = Ptr(base.base, offset)
        elif op.name == "tt.load":
            pointer = self.ptrs[op.operands[0].id]
            result_type = self.type(op.results[0].type)
            if op.results[0].type.kind == "tensor":
                self.ssa[op.results[0].id] = self._load_tensor(pointer, result_type)
                return
            index = pointer.offset
            if index is None:
                index = self.emit(
                    "arith.constant", results=[ir.IndexType.get()],
                    attrs={"value": ir.IntegerAttr.get(ir.IndexType.get(), 0)},
                ).results[0]
            elif str(index.type) != "index":
                index = self.emit(
                    "arith.index_cast", [index], [ir.IndexType.get()]
                ).results[0]
            self.ssa[op.results[0].id] = self.emit(
                "memref.load", [pointer.base, index], [result_type]
            ).results[0]
        elif op.name == "tt.store":
            pointer = self.ptrs[op.operands[0].id]
            index = pointer.offset
            if index is None:
                index = self.emit(
                    "arith.constant", results=[ir.IndexType.get()],
                    attrs={"value": ir.IntegerAttr.get(ir.IndexType.get(), 0)},
                ).results[0]
            elif str(index.type) != "index":
                index = self.emit("arith.index_cast", [index], [ir.IndexType.get()]).results[0]
            self.emit("memref.store", [self.get(op.operands[1]), pointer.base, index])
        elif op.name == "tt.reduce":
            source = self.get(op.operands[0])
            elem = self.type(type("_T", (), {
                "kind": "scalar", "elem": op.results[0].type.elem
            })())
            init_value = self.emit(
                "arith.constant", results=[elem],
                attrs={"value": self.ir.FloatAttr.get(elem, 0.0)},
            ).results[0]
            init_type = self.ir.RankedTensorType.get([], elem)
            init = self.emit("tensor.from_elements", [init_value], [init_type]).results[0]
            result_type = self.ir.RankedTensorType.get([], elem)
            region = self.ir.Region()
            region.blocks.append(self.ir.Block([elem, elem]))
            body = region.blocks[0]
            add = self.ir.Operation.create(
                "arith.addf",
                operands=list(body.arguments),
                results=[elem],
            )
            body.append(add)
            body.append(self.ir.Operation.create(
                "linalg.yield", operands=[add.results[0]]
            ))
            reduce = self.emit(
                "linalg.reduce", [source, init], [result_type],
                {"dimensions": self.ir.DenseI64ArrayAttr.get(
                    [op.attr_int("axis")]
                )},
                [region],
            ).results[0]
            self.ssa[op.results[0].id] = self.emit(
                "tensor.extract", [reduce], [elem]
            ).results[0]
        else:
            raise NotImplementedError("standard lowering not implemented: %s" % op.name)

    def module(self):
        ir = self.ir
        ctx = ir.Context()
        with ctx, ir.Location.unknown():
            module = ir.Module.create()
            for tf in self.tmod.funcs:
                arg_types = [self.type(a.type) for a in tf.args]
                fn_type = ir.FunctionType.get(arg_types, [])
                region = ir.Region()
                block = ir.Block(arg_types)
                region.blocks.append(block)
                fn = ir.Operation.create(
                    "func.func",
                    attributes={
                        "sym_name": ir.StringAttr.get(tf.name),
                        "function_type": ir.TypeAttr.get(fn_type),
                    },
                    regions=[region],
                )
                module.body.append(fn)
                self.block = block
                for arg, value in zip(tf.args, block.arguments):
                    if arg.type.kind == "ptr":
                        self.ptrs[arg.id] = Ptr(value)
                    else:
                        self.ssa[arg.id] = value
                for op in tf.body.ops:
                    if op.name == "tt.return":
                        self.emit("func.return")
                    else:
                        self.lower(op)
                if not block.operations or block.operations[-1].name != "func.return":
                    self.emit("func.return")
            module.operation.verify()
            return module


def lower_standard(tmod: TTIRModule):
    lowerer = StandardLowerer(tmod)
    return lowerer.module(), lowerer.ssa
