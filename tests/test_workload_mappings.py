"""Synthetic-plan coverage for the padded MLP and attention mappings."""
from __future__ import annotations

import unittest
from pathlib import Path

from compiler.atlas_mapping import (ATTENTION_NUMERICAL_POLICY,
                                    LINEAR_NUMERICAL_POLICY,
                                    AtlasAttentionABI, AtlasLinearReluABI,
                                    map_attention, map_linear_relu)
from compiler.triton_analysis import (LinearRelu, ScaledDotProductAttention,
                                      analyze_attention, analyze_linear_relu)
from compiler.ttir_model import TBlock, TFunc, TOp, TType, TValue


def value(identity: int, elem: str, shape: tuple[int, ...] = (), *, kind=None) -> TValue:
    return TValue(identity, TType(kind or ("tensor" if shape else "scalar"), elem, shape), f"%v{identity}")


def linear_relu_function() -> TFunc:
    args = [value(i, "f32", kind="ptr") for i in range(4)]
    x, w, bias = value(4, "f32", (4,)), value(5, "f32", (4,)), value(6, "f32")
    loaded = [TOp("tt.load", [args[i]], [v]) for i, v in enumerate((x, w, bias))]
    product = value(7, "f32", (4,))
    mul = TOp("arith.mulf", [x, w], [product])
    reshaped = value(8, "f32", (4,))
    reshape = TOp("tt.reshape", [product], [reshaped])
    lhs, rhs, summed = value(9, "f32"), value(10, "f32"), value(11, "f32")
    reduce_body = TBlock(1, args=[lhs, rhs], ops=[
        TOp("arith.addf", [lhs, rhs], [value(12, "f32")]),
        TOp("tt.reduce.return", [value(12, "f32")]),
    ])
    reduce = TOp("tt.reduce", [reshaped], [summed], {"axis": "0 : i32"}, [[reduce_body]])
    zero = value(13, "f32")
    constant = TOp("arith.constant", [], [zero], {"value": "0.000000e+00"})
    biased = value(14, "f32")
    add = TOp("arith.addf", [summed, bias], [biased])
    result = value(15, "f32")
    relu = TOp("arith.maxnumf", [biased, zero], [result])
    store = TOp("tt.store", [args[3], result])
    ret = TOp("tt.return")
    return TFunc("linear_relu", args, TBlock(0, ops=loaded + [mul, reshape, reduce, constant, add, relu, store, ret]))


def attention_function() -> TFunc:
    args = [value(i, "f32", kind="ptr") for i in range(4)]
    query, key, val = value(4, "f32", (8,)), value(5, "f32", (4, 8)), value(6, "f32", (4, 8))
    qload, kload, vload = [TOp("tt.load", [args[i]], [x]) for i, x in enumerate((query, key, val))]
    qwide, qmatrix = value(7, "f32", (1, 8)), value(8, "f32", (4, 8))
    expand_q = TOp("tt.expand_dims", [query], [qwide])
    broadcast_q = TOp("tt.broadcast", [qwide], [qmatrix])
    qk = value(9, "f32", (4, 8))
    qk_mul = TOp("arith.mulf", [qmatrix, key], [qk])

    def reduction(identity, operand, shape, axis, reducer):
        lhs, rhs, result = value(identity, "f32"), value(identity + 1, "f32"), value(identity + 2, "f32", shape)
        combined = value(identity + 3, "f32")
        body = TBlock(identity, args=[lhs, rhs], ops=[
            TOp(reducer, [lhs, rhs], [combined]), TOp("tt.reduce.return", [combined])])
        return TOp("tt.reduce", [operand], [result], {"axis": f"{axis} : i32"}, [[body]])

    scores = reduction(10, qk, (4,), 1, "arith.addf")
    scale_value = value(14, "f32", (4,))
    scale_const = TOp("arith.constant", [], [scale_value], {"value": "dense<0.353553385>"})
    scaled = value(15, "f32", (4,))
    scale_mul = TOp("arith.mulf", [scores.results[0], scale_value], [scaled])
    maxed = reduction(16, scaled, (), 0, "arith.maxnumf")
    maxwide = value(20, "f32", (4,))
    max_splat = TOp("tt.splat", [maxed.results[0]], [maxwide])
    shifted = value(21, "f32", (4,))
    sub = TOp("arith.subf", [scaled, maxwide], [shifted])
    exp_value = value(22, "f32", (4,))
    exp = TOp("math.exp", [shifted], [exp_value])
    total = reduction(23, exp_value, (), 0, "arith.addf")
    total_wide = value(27, "f32", (4,))
    total_splat = TOp("tt.splat", [total.results[0]], [total_wide])
    probs = value(28, "f32", (4,))
    div = TOp("arith.divf", [exp_value, total_wide], [probs])
    probs_wide, probs_matrix = value(29, "f32", (4, 1)), value(30, "f32", (4, 8))
    expand_p = TOp("tt.expand_dims", [probs], [probs_wide])
    broadcast_p = TOp("tt.broadcast", [probs_wide], [probs_matrix])
    weighted = value(31, "f32", (4, 8))
    pv_mul = TOp("arith.mulf", [probs_matrix, val], [weighted])
    result = reduction(32, weighted, (8,), 0, "arith.addf")
    store = TOp("tt.store", [args[3], result.results[0]])
    ops = [qload, kload, vload, expand_q, broadcast_q, qk_mul, scores,
           scale_const, scale_mul, maxed, max_splat, sub, exp, total,
           total_splat, div, expand_p, broadcast_p, pv_mul, result, store, TOp("tt.return")]
    return TFunc("attention", args, TBlock(0, ops=ops))


class WorkloadMappingTest(unittest.TestCase):
    def assert_fixture(self, name: str, text: str) -> None:
        fixture = Path(__file__).parents[1] / "examples/ttir_to_atlas" / name
        reference = "\n".join(line for line in fixture.read_text().splitlines()
                               if not line.lstrip().startswith("//"))
        self.assertEqual(text.strip(), reference.strip())

    def test_analyzes_linear_relu_and_maps_with_explicit_precision_policy(self):
        plan = analyze_linear_relu(linear_relu_function())
        self.assertEqual(plan.input_features, 4)
        with self.assertRaisesRegex(NotImplementedError, "numerical policy"):
            map_linear_relu(plan)
        text = map_linear_relu(plan, AtlasLinearReluABI(output_features=8),
                               numerical_policy=LINEAR_NUMERICAL_POLICY)
        self.assertIn('"atlas.virtual_input_bf16"(%io2) {index = 2 : i32}', text)
        self.assertIn('"atlas.virtual_mxu_matmul"(%x, %weight_nk) {unit = 0 : i32}', text)
        self.assertIn('"atlas.virtual_vpu_binary"(%dot, %bias) {kind = "add"}', text)
        self.assertIn('"atlas.virtual_vpu_unary"(%biased) {kind = "relu"}', text)
        self.assert_fixture("linear_relu.virtual.mlir", map_linear_relu(
            plan, numerical_policy=LINEAR_NUMERICAL_POLICY))

    def test_maps_attention_with_explicit_padding_mask_and_precision_policy(self):
        plan = analyze_attention(attention_function())
        self.assertEqual((plan.sequence_length, plan.embedding_size), (4, 8))
        with self.assertRaisesRegex(NotImplementedError, "numerical policy"):
            map_attention(plan)
        text = map_attention(plan, AtlasAttentionABI(),
                             numerical_policy=ATTENTION_NUMERICAL_POLICY)
        self.assertIn('"atlas.virtual_input_bf16"(%io3) {index = 3 : i32}', text)
        self.assertIn('"atlas.virtual_vpu_binary"(%scores0, %mask) {kind = "add"}', text)
        self.assertIn('"atlas.virtual_vpu_unary"(%scaled) {kind = "exp2"}', text)
        self.assertIn('"atlas.virtual_pack_fp8"(%p) {scale_code = 127 : i32}', text)
        self.assertIn('"atlas.virtual_mxu_matmul"(%p8, %v_t) {unit = 1 : i32}', text)
        self.assert_fixture("attention.virtual.mlir", map_attention(
            plan, numerical_policy=ATTENTION_NUMERICAL_POLICY))


if __name__ == "__main__":
    unittest.main()
