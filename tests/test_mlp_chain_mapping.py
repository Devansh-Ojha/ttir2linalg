"""Generalized emission, compatibility, CLI and real backend resource checks."""
from compiler.toolchain_paths import atlas_root as default_atlas_root
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from compiler.atlas_mapping import (
    AtlasMlpABI, MLP_NUMERICAL_POLICY, map_mlp_chain, map_two_layer_mlp,
)
from compiler.triton_analysis import analyze_mlp_chain, analyze_two_layer_mlp
from compiler.ttir_reader import parse_ttir
from compiler.ttir_to_atlas import convert_file, main
from test_mlp_chain_analysis import chain_function, chain_source

ROOT = Path(__file__).parents[1]
BACKEND = default_atlas_root() / "build/bin/atlas-opt"


class MlpChainMappingTest(unittest.TestCase):
    def plan(self, depth):
        return analyze_mlp_chain(chain_function(depth))

    def test_two_layer_exact_compatibility(self):
        module = parse_ttir((ROOT / "examples/two_layer_mlp_fp8.ttir").read_text())
        chain = analyze_mlp_chain(module.funcs[0])
        old = analyze_two_layer_mlp(module.funcs[0])
        for abi in (None, AtlasMlpABI(input_slots=(1, 2, 3), output_index=2)):
            expected = map_two_layer_mlp(old, abi or AtlasMlpABI(),
                                         numerical_policy=MLP_NUMERICAL_POLICY)
            self.assertEqual(map_mlp_chain(chain, abi, numerical_policy=MLP_NUMERICAL_POLICY),
                             expected)

    def test_two_three_four_layer_inventory_and_dependencies(self):
        from atlas_virtual_evaluator import operation_name, parse_program
        for depth in (2, 3, 4):
            with self.subTest(depth=depth):
                text = map_mlp_chain(self.plan(depth), numerical_policy=MLP_NUMERICAL_POLICY)
                program = parse_program(text)
                ops = program.operations
                dots = [op for op in ops if operation_name(op) == "atlas.virtual_mxu_matmul"]
                packs = [op for op in ops if operation_name(op) == "atlas.virtual_pack_fp8"]
                self.assertEqual(len(dots), depth)
                self.assertEqual(len(packs), depth - 1)
                self.assertEqual(text.count('"atlas.virtual_xlu_transpose"'), depth)
                self.assertEqual(text.count('kind = "relu"'), depth - 1)
                self.assertEqual(dict(program.input_formats),
                                 {i: "fp8" for i in range(depth + 1)})
                for i in range(depth - 1):
                    self.assertIs(dots[i + 1].operands[0], packs[i].results[0])
                if depth > 2:
                    self.assertLess(text.index('%b1 ='), text.index('%d1 ='))
                    self.assertGreater(text.index('%b1 ='), text.index('%h0 ='))

    def test_abi_and_policy_rejections(self):
        plan = self.plan(3)
        for policy in (None, "", "standard"):
            with self.subTest(policy=policy), self.assertRaises(NotImplementedError):
                map_mlp_chain(plan, numerical_policy=policy)
        for abi in (
            AtlasMlpABI(),
            AtlasMlpABI((0, 1, 1, 3)),
            AtlasMlpABI((0, 1, 2, 64)),
            AtlasMlpABI((0, 1, 2, -1)),
            AtlasMlpABI((0, 1, 2, True)),
            AtlasMlpABI((0, 1, 2, 3), input_base=0x90000001),
            AtlasMlpABI((0, 1, 2, 3), input_base=0x1000),
            AtlasMlpABI((0, 1, 2, 3), output_base=0x90001800),
            AtlasMlpABI((0, 1, 2, 3), output_base=0xFFFFF800, output_index=1),
            AtlasMlpABI((0, 1, 2, 3), output_index=64),
            AtlasMlpABI((0, 1, 2, 3), input_base=0xFFFFF800),
            # Output slot 1 overlaps input span even though output base is below it.
            AtlasMlpABI((0, 1, 2, 3), input_base=0x90000800,
                        output_base=0x90000000, output_index=1),
        ):
            with self.subTest(abi=abi), self.assertRaises(NotImplementedError):
                map_mlp_chain(plan, abi, numerical_policy=MLP_NUMERICAL_POLICY)
        with self.assertRaises(NotImplementedError):
            map_mlp_chain(self.plan(64), numerical_policy=MLP_NUMERICAL_POLICY)

    def test_rejects_mutated_layer_dependencies(self):
        plan = self.plan(3)
        plan.layers[1].dot.operands[0] = plan.input_activation
        with self.assertRaisesRegex(NotImplementedError, "dependencies"):
            map_mlp_chain(plan, numerical_policy=MLP_NUMERICAL_POLICY)
        plan = self.plan(3)
        with self.assertRaises(NotImplementedError):
            map_mlp_chain(replace(plan, layers=plan.layers[:-1]),
                          numerical_policy=MLP_NUMERICAL_POLICY)

    def test_cli_and_conversion_api(self):
        fixture = ROOT / "examples/three_layer_mlp_fp8.ttir"
        default = convert_file(fixture, pattern="mlp-chain",
                               numerical_policy=MLP_NUMERICAL_POLICY)
        self.assertIn('%d2 = "atlas.virtual_mxu_matmul"(%h1, %w2) {unit = 0', default)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "chain.mlir"
            argv = ["ttir_to_atlas", str(fixture), "--pattern", "mlp-chain",
                    "--numerical-policy", MLP_NUMERICAL_POLICY,
                    "--input-slots", "4", "5", "6", "7",
                    "--output-base", "0x90010000", "--out", str(output)]
            with patch("sys.argv", argv), redirect_stdout(StringIO()):
                main()
            self.assertIn("index = 7 : i32", output.read_text())
            previous = output.read_bytes()
            with patch("sys.argv", argv), patch("sys.stderr", StringIO()), self.assertRaises(SystemExit):
                main()
            self.assertEqual(output.read_bytes(), previous)

    @unittest.skipUnless(BACKEND.is_file(), "Atlas verifier binary unavailable")
    def test_backend_rejects_resource_overflow(self):
        # Input slot admission is not proof of machine resource fit.
        text = map_mlp_chain(self.plan(63), numerical_policy=MLP_NUMERICAL_POLICY)
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "allocation.json"
            result = subprocess.run([
                str(BACKEND), "--verify-atlas-virtual-stream", "--schedule-atlas-virtual",
                f"--lower-atlas-virtual-to-machine=allocation-report={report}",
                "-o", "/dev/null"], input=text, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            allocation = json.loads(report.read_text())
            self.assertFalse(allocation["fits"])
            self.assertTrue(allocation["failures"])

    @unittest.skipUnless(BACKEND.is_file(), "Atlas verifier binary unavailable")
    def test_real_backend_two_three_four_layers_fit(self):
        for depth in (2, 3, 4):
            with self.subTest(depth=depth), tempfile.TemporaryDirectory() as directory:
                module = parse_ttir(chain_source(depth))
                text = map_mlp_chain(analyze_mlp_chain(module.funcs[0]),
                                     numerical_policy=MLP_NUMERICAL_POLICY)
                report = Path(directory) / "allocation.json"
                result = subprocess.run([
                    str(BACKEND), "--verify-atlas-virtual-stream",
                    "--schedule-atlas-virtual",
                    f"--lower-atlas-virtual-to-machine=allocation-report={report}",
                    "--insert-atlas-dma-waits", "--verify-atlas-generated-schedule",
                    "--insert-atlas-delays", "--verify-atlas-timing",
                    "-o", "/dev/null"], input=text, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                allocation = json.loads(report.read_text())
                self.assertTrue(allocation["fits"])
                self.assertFalse(allocation["failures"])
                # The scheduler can hoist weight loads (four FP8 tiles for
                # depth four), so check actual hardware capacity, not source order.
                for demand in allocation["registers"].values():
                    self.assertTrue(demand["fits"])
                    self.assertLessEqual(demand["peak_live"], demand["capacity"])
                    self.assertLessEqual(demand["used"], demand["capacity"])


if __name__ == "__main__":
    unittest.main()
