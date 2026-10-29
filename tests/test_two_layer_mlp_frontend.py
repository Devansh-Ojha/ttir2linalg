"""Bounded frontend tests; no backend lowering or persistent output artifacts."""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from compiler.atlas_mapping import AtlasMlpABI, MLP_NUMERICAL_POLICY, map_two_layer_mlp
from compiler.triton_analysis import analyze_two_layer_mlp
from compiler.ttir_model import TOp, TValue
from compiler.ttir_reader import _scan_text, parse_ttir
from compiler.ttir_to_atlas import convert_file, main

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/two_layer_mlp_fp8.ttir"
sys.path.insert(0, str(ROOT / "tools" / "validation"))


class TwoLayerMlpFrontendTest(unittest.TestCase):
    def parse(self, text=None):
        return parse_ttir(SOURCE.read_text() if text is None else text)

    def function(self):
        # Keep the parser/context alive as well as the plain Python graph.
        self.module = self.parse()
        return self.module.funcs[0]

    def test_real_parser_and_dependency_plan(self):
        function = self.function()
        plan = analyze_two_layer_mlp(function)
        self.assertEqual(plan.pack.attrs, {"rounding": "rtne"})
        self.assertEqual([v.name for v in plan.inputs], ["%x", "%b0", "%b1"])
        self.assertEqual(plan.dots[1].operands[0].id, plan.pack.results[0].id)
        self.assertEqual(plan.store.operands[1].id, plan.bf16_casts[1].results[0].id)

    def test_emits_units_transposes_quantization_and_slots(self):
        plan = analyze_two_layer_mlp(self.function())
        text = map_two_layer_mlp(plan, AtlasMlpABI(input_slots=(1, 2, 3)),
                                 numerical_policy=MLP_NUMERICAL_POLICY)
        self.assertEqual(text.count('"atlas.virtual_input_fp8"'), 3)
        self.assertEqual(text.count('"atlas.virtual_xlu_transpose"'), 2)
        for slot in (1, 2, 3):
            self.assertIn(f"index = {slot} : i32", text)
        self.assertIn('matmul"(%x, %w0) {unit = 0 : i32}', text)
        self.assertIn('matmul"(%h2, %w1) {unit = 1 : i32}', text)
        self.assertIn('pack_fp8"(%h1) {scale_code = 127 : i32}', text)
        self.assertIn('output_bf16"(%io3, %y)', text)
        self.assertEqual(text.count('kind = "relu"'), 1)
        # Independent parser verifies emitted SSA and virtual interface.
        from atlas_virtual_evaluator import parse_program
        program = parse_program(text)
        self.assertEqual(dict(program.input_formats), {1: "fp8", 2: "fp8", 3: "fp8"})
        self.assertEqual(tuple(program.output_indices), (0,))

    def test_multiline_attributes_fail_closed_from_source(self):
        source = SOURCE.read_text()
        cases = (
            source.replace("tt.load %xp :", "tt.load %xp\n      {isVolatile = true} :"),
            source.replace("tt.load %xp :", "tt.load %xp {isVolatile = true\n      } :"),
            source.replace("tt.load %xp :", "tt.load %xp\n      {\n        isVolatile = true\n      } :"),
            source.replace(", rounding = rtne", ",\n      rounding = rtne"),
            source.replace(", rounding = rtne", ",\n      rounding = rtz"),
        )
        for text in cases:
            with self.subTest(text=text), self.assertRaisesRegex(RuntimeError, "multiline"):
                self.parse(text)
        # A parsed attribute must also reach the strict recognizer when inline.
        inline = source.replace("tt.load %xp :", "tt.load %xp {isVolatile = true} :")
        parsed = self.parse(inline)
        self.assertEqual(parsed.funcs[0].body.ops[0].attrs["isVolatile"], "true")
        with self.assertRaisesRegex(NotImplementedError, "unmasked direct loads"):
            analyze_two_layer_mlp(parsed.funcs[0])

    def test_independent_role_to_slot_abi(self):
        from validate_two_layer_mlp_frontend import inspect
        plan = analyze_two_layer_mlp(self.function())
        for slots in ((0, 1, 2), (1, 2, 3)):
            text = map_two_layer_mlp(plan, AtlasMlpABI(input_slots=slots),
                                     numerical_policy=MLP_NUMERICAL_POLICY)
            with redirect_stdout(StringIO()):
                inspect(SOURCE, True, expected_slots=slots, source=text)
            swapped = text.replace(f"index = {slots[1]} : i32", "index = SWAPPED : i32")
            swapped = swapped.replace(f"index = {slots[2]} : i32", f"index = {slots[1]} : i32")
            swapped = swapped.replace("index = SWAPPED : i32", f"index = {slots[2]} : i32")
            with self.subTest(slots=slots), self.assertRaisesRegex(ValueError, "role-to-slot ABI mismatch"):
                inspect(SOURCE, True, expected_slots=slots, source=swapped)

    def test_fresh_conversion_has_known_permutation_result(self):
        import torch
        from atlas_virtual_evaluator import Tile
        from validate_two_layer_mlp_frontend import execute, inspect
        text = convert_file(SOURCE, pattern="two-layer-mlp", numerical_policy=MLP_NUMERICAL_POLICY)
        with redirect_stdout(StringIO()):
            spec = inspect(SOURCE, True, source=text)
        values = (0.0, 0.5, 1.0, 2.0, -1.0)
        x = torch.tensor([[values[(r * 7 + c * 3) % 5] for c in range(32)]
                          for r in range(32)], dtype=torch.bfloat16)
        # Independent source [K,N] permutations: output c takes input c+8.
        b0 = torch.zeros((32, 32), dtype=torch.bfloat16)
        b1 = torch.zeros_like(b0)
        for c in range(32):
            b0[(c + 3) % 32, c] = 1
            b1[(c + 5) % 32, c] = 1
        def exact_tile(value):
            raw = value.to(torch.float8_e4m3fn).view(torch.uint8).flatten().tolist()
            return Tile("fp8", tuple(raw))
        actual, _ = execute(spec, tuple(exact_tile(v) for v in (x, b0, b1)))
        expected = torch.tensor([[max(float(x[r, (c + 8) % 32]), 0.0) for c in range(32)]
                                 for r in range(32)], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected.view(torch.uint16)))

    def test_policy_required_and_checked(self):
        plan = analyze_two_layer_mlp(self.function())
        for policy in (None, "", "standard", "atlas-fp8-mlp-v2"):
            with self.subTest(policy=policy), self.assertRaisesRegex(NotImplementedError, "policy"):
                map_two_layer_mlp(plan, numerical_policy=policy)
        with self.assertRaisesRegex(NotImplementedError, "policy"):
            convert_file(SOURCE, pattern="two-layer-mlp")

    def test_nonzero_accumulators_both_layers(self):
        for layer in (0, 1):
            function = self.function()
            plan = analyze_two_layer_mlp(function)
            # Separate constant so each dot's accumulator is tested independently.
            accumulator = plan.dots[layer].operands[2]
            identity = max(v.id for o in function.body.ops for v in o.results) + 1
            value = TValue(identity, accumulator.type, "%nonzero")
            constant = TOp("arith.constant", [], [value], {"value": "dense<1.000000e+00>"})
            function.body.ops.insert(function.body.ops.index(plan.dots[layer]), constant)
            plan.dots[layer].operands[2] = value
            with self.assertRaisesRegex(NotImplementedError, "proven zero"):
                analyze_two_layer_mlp(function)

    def test_rounding_is_never_assumed(self):
        for rounding in ("rtz", None):
            function = self.function()
            pack = next(o for o in function.body.ops if o.name == "tt.fp_to_fp")
            pack.attrs = {} if rounding is None else {"rounding": rounding}
            with self.subTest(rounding=rounding), self.assertRaisesRegex(NotImplementedError, "rounding"):
                analyze_two_layer_mlp(function)
        text = SOURCE.read_text().replace("rounding = rtne", "rounding = rtz")
        self.assertEqual(next(a for n, _, a in _scan_text(text) if n == "tt.fp_to_fp"), {"rounding": "rtz"})
        with self.assertRaisesRegex(NotImplementedError, "rounding"):
            analyze_two_layer_mlp(self.parse(text).funcs[0])
        self.assertEqual(next(a for n, _, a in _scan_text(text.replace(", rounding = rtz", "")) if n == "tt.fp_to_fp"), {})

        comment = "    %h8 = tt.fp_to_fp %relu : tensor<32x32xbf16> -> tensor<32x32xf8E4M3FN> // , rounding = rtne"
        self.assertEqual(_scan_text(comment)[0][2], {})

    def test_broken_dependencies(self):
        for edge in ("second_dot", "relu", "output"):
            function = self.function()
            plan = analyze_two_layer_mlp(function)
            if edge == "second_dot":
                plan.dots[1].operands[0] = plan.inputs[0]
            elif edge == "relu":
                plan.relu.operands[0] = plan.bf16_casts[0].operands[0]
            else:
                plan.store.operands[1] = plan.relu.results[0]
            with self.subTest(edge=edge), self.assertRaises(NotImplementedError):
                analyze_two_layer_mlp(function)

    def test_shapes_encodings_and_conversion_types(self):
        for changed_type in ("shape", "fp8_format", "cast"):
            function = self.function()
            plan = analyze_two_layer_mlp(function)
            value = plan.inputs[0] if changed_type != "cast" else plan.pack.results[0]
            if changed_type == "shape":
                value.type = replace(value.type, shape=(16, 32))
            else:
                value.type = replace(value.type, elem="f8E4M3FNUZ")
            with self.subTest(changed_type=changed_type), self.assertRaises(NotImplementedError):
                analyze_two_layer_mlp(function)

    def test_extra_stores_operations_attributes_and_regions(self):
        for extra in ("store", "arithmetic", "attribute", "region"):
            function = self.function()
            plan = analyze_two_layer_mlp(function)
            if extra == "store":
                function.body.ops.insert(-1, TOp("tt.store", list(plan.store.operands)))
            elif extra == "arithmetic":
                function.body.ops.insert(-1, TOp("arith.addf"))
            elif extra == "attribute":
                plan.pack.attrs["unsupported"] = "true"
            else:
                plan.relu.regions = [[]]
            with self.subTest(extra=extra), self.assertRaises(NotImplementedError):
                analyze_two_layer_mlp(function)

    def test_slot_and_address_validation(self):
        plan = analyze_two_layer_mlp(self.function())
        for abi in (AtlasMlpABI(input_slots=(0, 0, 2)), AtlasMlpABI(input_slots=(0, 1, 64)),
                    AtlasMlpABI(output_base=0x90000000), AtlasMlpABI(input_base=0x90000001),
                    AtlasMlpABI(output_base=0xFFFFF800, output_index=1)):
            with self.subTest(abi=abi), self.assertRaises(NotImplementedError):
                map_two_layer_mlp(plan, abi, numerical_policy=MLP_NUMERICAL_POLICY)

    def test_api_and_cli_to_dev_null(self):
        text = convert_file(SOURCE, pattern="two-layer-mlp", numerical_policy=MLP_NUMERICAL_POLICY)
        self.assertIn("ttir_two_layer_mlp_atlas", text)
        with patch("sys.argv", ["ttir_to_atlas", str(SOURCE), "--out", "/dev/null",
                               "--pattern", "two-layer-mlp", "--numerical-policy", MLP_NUMERICAL_POLICY]), redirect_stdout(StringIO()):
            main()
        # Original API and CLI defaults continue to select the one-dot path.
        self.assertIn('"atlas.virtual_xlu_transpose"', convert_file(ROOT / "matmul_relu_generated.ttir"))
        with patch("sys.argv", ["ttir_to_atlas", str(ROOT / "matmul_relu_generated.ttir"), "--out", "/dev/null"]), redirect_stdout(StringIO()):
            main()


if __name__ == "__main__":
    unittest.main()
