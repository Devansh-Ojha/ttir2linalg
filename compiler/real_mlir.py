"""Minimal real MLIR construction through Triton 3.4.0's bindings.

This is intentionally a first milestone, not the complete TTIR lowering.  The
existing ``compiler.linalg_lowering`` path remains the semantic/textual
reference.  This module proves the complete plumbing:

    TTIR text -> parsed Triton IR -> real MLIR operation -> verified module

Values are kept in an SSA map keyed by Triton's stable ``Value.id()``.
"""
from __future__ import annotations

from typing import Dict, Iterable

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
            result.append(builder.get_ptr_ty(_scalar_type(builder, arg.type.elem), 1))
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


def _lower_arithmetic(builder, op: TOp, ssa: Dict[int, object]):
    if len(op.operands) != 2 or len(op.results) != 1:
        raise RuntimeError(
            "%s expects two operands and one result" % op.name
        )
    try:
        lhs = ssa[op.operands[0].id]
        rhs = ssa[op.operands[1].id]
    except KeyError as exc:
        raise RuntimeError(
            "%s uses an SSA value that has not been lowered" % op.name
        ) from exc

    creators = {
        "arith.muli": builder.create_mul,
        "arith.mulf": builder.create_fmul,
        "arith.addf": builder.create_fadd,
    }
    value = creators[op.name](lhs, rhs)
    ssa[op.results[0].id] = value


def _lower_make_range(builder, op: TOp, ssa: Dict[int, object]):
    if len(op.results) != 1:
        raise RuntimeError("tt.make_range without one result")
    result_type = _type(builder, op.results[0].type)
    ssa[op.results[0].id] = builder.create_make_range(
        result_type,
        op.attr_int("start"),
        op.attr_int("end"),
    )


def _lower_reshape(builder, op: TOp, ssa: Dict[int, object]):
    if len(op.operands) != 1 or len(op.results) != 1:
        raise RuntimeError("tt.reshape expects one operand and one result")
    ssa[op.results[0].id] = builder.create_reshape(
        _get(ssa, op.operands[0]),
        list(op.results[0].type.shape),
        True,
    )


def _type(builder, t):
    if t.kind == "scalar":
        return _scalar_type(builder, t.elem)
    if t.kind == "ptr":
        return builder.get_ptr_ty(_scalar_type(builder, t.elem), 1)
    if t.kind == "tensor":
        elem = (_scalar_type(builder, t.elem) if not t.elem_is_ptr
                else builder.get_ptr_ty(_scalar_type(builder, t.elem), 1))
        return builder.get_block_ty(elem, list(t.shape))
    raise NotImplementedError("unsupported TTIR type %s" % t.raw)


def _get(ssa, value: TValue):
    try:
        return ssa[value.id]
    except KeyError as exc:
        raise RuntimeError("SSA value %s has not been lowered" % value.name) from exc


def _operands_ready(op: TOp, ssa: Dict[int, object]) -> bool:
    return all(value.id in ssa for value in op.operands)


def _lower_op(builder, op: TOp, ssa: Dict[int, object]):
    if op.name == "arith.constant":
        if len(op.results) != 1:
            raise RuntimeError("arith.constant without one result")
        ssa[op.results[0].id] = _constant(builder, op)
    elif op.name in {"arith.muli", "arith.mulf", "arith.addf"}:
        _lower_arithmetic(builder, op, ssa)
    elif op.name == "tt.get_program_id":
        ssa[op.results[0].id] = builder.create_get_program_id(
            op.attr_int("axis")
        )
    elif op.name == "tt.make_range":
        _lower_make_range(builder, op, ssa)
    elif op.name == "tt.splat":
        ssa[op.results[0].id] = builder.create_splat(
            _type(builder, op.results[0].type),
            _get(ssa, op.operands[0]),
        )
    elif op.name == "tt.addptr":
        ssa[op.results[0].id] = builder.create_addptr(
            _get(ssa, op.operands[0]), _get(ssa, op.operands[1])
        )
    elif op.name == "tt.load":
        ssa[op.results[0].id] = builder.create_load(
            _get(ssa, op.operands[0]),
            ir.CACHE_MODIFIER.NONE,
            ir.EVICTION_POLICY.NORMAL,
            False,
        )
    elif op.name == "tt.store":
        builder.create_store(
            _get(ssa, op.operands[0]),
            _get(ssa, op.operands[1]),
            ir.CACHE_MODIFIER.NONE,
            ir.EVICTION_POLICY.NORMAL,
        )
    elif op.name == "tt.reshape":
        _lower_reshape(builder, op, ssa)
    elif op.name == "tt.reduce":
        insertion_point = builder.get_insertion_point()
        _lower_reduce(builder, op, ssa)
        builder.restore_insertion_point(insertion_point)
    elif op.name in {"tt.reduce.return", "tt.return"}:
        return
    else:
        raise NotImplementedError("unsupported TTIR operation %s" % op.name)


def _lower_reduce(builder, op: TOp, ssa: Dict[int, object]):
    if len(op.operands) != 1 or len(op.results) != 1 or not op.regions:
        raise RuntimeError("malformed tt.reduce")
    body = op.regions[0][0]
    combiner = [item for item in body.ops if item.name != "tt.reduce.return"]
    if len(combiner) != 1 or len(body.args) != 2:
        raise NotImplementedError("only a binary tt.reduce combiner is supported")
    reduce_op = builder.create_reduce([_get(ssa, op.operands[0])],
                                      op.attr_int("axis"))
    region = reduce_op.get_region(0)
    block = builder.create_block_with_parent(
        region, [_type(builder, body.args[0].type), _type(builder, body.args[1].type)]
    )
    builder.set_insertion_point_to_start(block)
    lhs, rhs = block.arg(0), block.arg(1)
    creators = {
        "arith.addf": builder.create_fadd,
        "arith.mulf": builder.create_fmul,
        "arith.muli": builder.create_mul,
    }
    try:
        combined = creators[combiner[0].name](lhs, rhs)
    except KeyError as exc:
        raise NotImplementedError(
            "unsupported tt.reduce combiner %s" % combiner[0].name
        ) from exc
    builder.create_reduce_ret(combined)
    ssa[op.results[0].id] = reduce_op.get_result(0)


def build_real_module(
    tmod: TTIRModule,
    limit: int | None = None,
    arithmetic_only: bool = False,
):
    """Build and verify a real Triton/MLIR module plus its SSA mapping."""
    ctx = tmod.ctx
    builder = ir.builder(ctx)
    module = builder.create_module()
    ssa: Dict[int, object] = {}
    skipped = []
    enabled_ops = None
    if arithmetic_only:
        enabled_ops = {
            "arith.constant",
            "arith.muli",
            "arith.mulf",
            "arith.addf",
            "tt.get_program_id",
            "tt.make_range",
            "tt.return",
        }

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
        pending = list(ops)
        while pending:
            progress = False
            next_pending = []
            for op in pending:
                if not _operands_ready(op, ssa):
                    next_pending.append(op)
                    continue
                if enabled_ops is not None and op.name not in enabled_ops:
                    skipped.append("%s: lowering not enabled in this stage" % op.name)
                    progress = True
                    continue
                try:
                    _lower_op(builder, op, ssa)
                except NotImplementedError as exc:
                    if not arithmetic_only:
                        raise
                    skipped.append("%s: %s" % (op.name, exc))
                    progress = True
                    continue
                progress = True
            if not progress:
                if arithmetic_only:
                    skipped.extend(
                        "%s: operands unavailable (%s)" % (
                            op.name,
                            ", ".join(value.name for value in op.operands),
                        )
                        for op in next_pending
                    )
                    break
                details = ", ".join(
                    "%s(%s)" % (
                        op.name,
                        ", ".join(value.name for value in op.operands),
                    )
                    for op in next_pending
                )
                raise RuntimeError("cannot resolve TTIR SSA dependencies: " + details)
            pending = next_pending

        builder.ret([])
        fn.finalize()

    if not module.verify():
        raise RuntimeError("generated real MLIR module failed verification")
    return module, ssa, skipped


def lower_ttir(
    ttir: str,
    limit: int | None = None,
    arithmetic_only: bool = False,
):
    """Parse TTIR and construct the verified real module."""
    from compiler.ttir_reader import parse_ttir

    tmod = parse_ttir(ttir)
    return tmod, *build_real_module(
        tmod, limit=limit, arithmetic_only=arithmetic_only
    )
