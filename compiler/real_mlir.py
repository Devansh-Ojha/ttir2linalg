"""Minimal real MLIR construction through Triton 3.4.0's bindings.

This is intentionally a first milestone, not the complete TTIR lowering.  The
existing ``compiler.linalg_lowering`` path remains the semantic/textual
reference.  This module proves the complete plumbing:

    TTIR text -> parsed Triton IR -> real MLIR operation -> verified module

Only ``arith.constant`` is materialized for now.  Values are kept in an SSA
map keyed by Triton's stable ``Value.id()``.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

from triton._C.libtriton import ir

from compiler.ttir_reader import TFunc, TTIRModule, TOp, TValue


def _scalar_type(builder, name: str):
    """Return the Triton binding type corresponding to a TTIR scalar."""
    types = {
        "i1": builder.get_int1_ty,
        "i8": builder.get_int8_ty,
        "i16": builder.get_int16_ty,
        "i32": builder.get_int32_ty,
        "i64": builder.get_int64_ty,
        "f16": builder.get_half_ty,
        "bf16": builder.get_bf16_ty,
        "f32": builder.get_float_ty,
        "f64": builder.get_double_ty,
    }
    try:
        return types[name]()
    except KeyError as exc:
        raise NotImplementedError("unsupported TTIR type %s" % name) from exc


def _function_types(builder, tf: TFunc):
    """Build a function signature while preserving TTIR argument types."""
    result = []
    for arg in tf.args:
        if arg.type.kind == "scalar":
            result.append(_scalar_type(builder, arg.type.elem))
        elif arg.type.kind == "ptr":
            result.append(_scalar_type(builder, arg.type.elem))
        else:
            raise NotImplementedError(
                "tensor arguments are not part of the real-IR milestone: %s"
                % arg.type.raw
            )
    return result


def _constant(builder, op: TOp):
    """Materialize one parsed arith.constant using a typed real operation."""
    text = op.attrs.get("value", "0").strip()
    ty = op.attrs.get("type", op.results[0].type.elem if op.results else "i32")
    integer_builders = {
        "i1": builder.get_int1,
        "i8": builder.get_int8,
        "i16": builder.get_int16,
        "i32": builder.get_int32,
        "i64": builder.get_int64,
    }
    if ty in integer_builders and text.lstrip("-").isdigit():
        return integer_builders[ty](int(text))
    if ty == "f32":
        return builder.get_fp32(float(text))
    if ty == "f64":
        return builder.get_fp64(float(text))
    raise NotImplementedError("unsupported constant %s : %s" % (text, ty))


def _ops_in_order(tf: TFunc) -> Iterable[TOp]:
    # The parser stores each block in textual/program order.  This helper
    # makes the traversal boundary explicit before nested regions are added.
    return tf.body.ops


def build_real_module(tmod: TTIRModule, limit: int | None = None):
    """Build and verify a real Triton/MLIR module plus its SSA mapping."""
    ctx = tmod.ctx
    builder = ir.builder(ctx)
    module = builder.create_module()
    ssa: Dict[int, object] = {}

    for tf in tmod.funcs:
        arg_types = _function_types(builder, tf)
        fn_type = builder.get_function_ty(arg_types, [])
        fn = builder.get_or_insert_function(
            module, tf.name, fn_type, "public", False
        )
        module.push_back(fn)
        entry = fn.add_entry_block()
        builder.set_insertion_point_to_start(entry)

        for index, arg in enumerate(tf.args):
            ssa[arg.id] = fn.args(index)

        ops = _ops_in_order(tf)
        if limit is not None:
            ops = ops[:limit]
        for op in ops:
            if op.name != "arith.constant":
                continue
            if len(op.results) != 1:
                raise RuntimeError("arith.constant without one result")
            ssa[op.results[0].id] = _constant(builder, op)
            # One real operation is sufficient for this milestone; later
            # milestones will consume the mapping for all TTIR operations.
            break

        builder.ret([])
        fn.finalize()

    if not module.verify():
        raise RuntimeError("generated real MLIR module failed verification")
    return module, ssa


def lower_ttir(ttir: str, limit: int | None = None):
    """Parse TTIR and construct the verified real module."""
    from compiler.ttir_reader import parse_ttir

    tmod = parse_ttir(ttir)
    return tmod, *build_real_module(tmod, limit=limit)
