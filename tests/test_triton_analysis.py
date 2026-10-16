"""Target-independent recognition of the first TTIR matmul epilogue."""
from __future__ import annotations

import unittest

from compiler.ttir_model import TBlock, TFunc, TOp, TType, TValue
from compiler.triton_analysis import analyze_matmul_relu
from compiler.atlas_mapping import AtlasTileABI, map_matmul_relu


def value(identity: int, kind: str, shape: tuple[int, ...] = ()) -> TValue:
    return TValue(identity, TType("tensor" if shape else "scalar", kind, shape), f"%{identity}")


class TritonAnalysisTest(unittest.TestCase):
    def function(self, *, zero: str = "0.000000e+00", reverse_relu: bool = False) -> TFunc:
        lhs_ptr, rhs_ptr, out_ptr = (value(i, "f8E4M3FN") for i in range(3))
        lhs, rhs = value(3, "f8E4M3FN", (32, 32)), value(4, "f8E4M3FN", (32, 32))
        dot_result = value(5, "f32", (32, 32))
        zero_result = value(6, "f32")
        relu_result = value(7, "f32", (32, 32))
        lhs_load, rhs_load = TOp("tt.load", [lhs_ptr], [lhs]), TOp("tt.load", [rhs_ptr], [rhs])
        dot = TOp("tt.dot", [lhs, rhs], [dot_result])
        constant = TOp("arith.constant", [], [zero_result], {"value": zero})
        operands = [zero_result, dot_result] if reverse_relu else [dot_result, zero_result]
        relu = TOp("arith.maxnumf", operands, [relu_result])
        store = TOp("tt.store", [out_ptr, relu_result])
        return TFunc("matmul_relu", [lhs_ptr, rhs_ptr, out_ptr],
                     TBlock(0, ops=[lhs_load, rhs_load, dot, constant, relu, store]))

    def test_finds_matmul_relu_plan_without_target_types(self) -> None:
        function = self.function()
        plan = analyze_matmul_relu(function)
        self.assertEqual(plan.function, "matmul_relu")
        self.assertIs(plan.matmul, function.body.ops[2])
        self.assertIs(plan.relu, function.body.ops[4])
        self.assertIs(plan.store, function.body.ops[5])
        self.assertEqual((plan.lhs.name, plan.rhs.name, plan.shape), ("%3", "%4", (32, 32)))

    def test_accepts_zero_on_either_relu_side(self) -> None:
        self.assertEqual(analyze_matmul_relu(self.function(reverse_relu=True)).shape, (32, 32))

    def test_rejects_nonzero_clamp_and_ambiguous_patterns(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "0 candidate"):
            analyze_matmul_relu(self.function(zero="1.000000e+00"))
        function = self.function()
        function.body.ops.append(function.body.ops[2])
        with self.assertRaisesRegex(NotImplementedError, "2 candidate"):
            analyze_matmul_relu(function)

    def test_atlas_mapper_emits_reference_virtual_operation_chain(self) -> None:
        plan = analyze_matmul_relu(self.function())
        mlir = map_matmul_relu(plan, AtlasTileABI(lhs_input=2, rhs_input=3, mxu_unit=1))
        self.assertIn('"atlas.virtual_input_fp8"(%io0) {index = 2 : i32}', mlir)
        self.assertIn('"atlas.virtual_input_fp8"(%io1) {index = 3 : i32}', mlir)
        self.assertIn('"atlas.virtual_xlu_transpose"(%rhs)', mlir)
        self.assertIn('"atlas.virtual_mxu_matmul"(%lhs, %rhs_nk) {unit = 1 : i32}', mlir)
        self.assertIn('"atlas.virtual_vpu_unary"(%dot) {kind = "relu"}', mlir)
        self.assertIn('"atlas.virtual_output_bf16"(%io2, %relu)', mlir)
        self.assertNotIn("atlas.machine", mlir)


if __name__ == "__main__":
    unittest.main()
