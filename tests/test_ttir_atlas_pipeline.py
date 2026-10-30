import json
from pathlib import Path
import tempfile
import unittest
from compiler.ttir_atlas_pipeline import compile_ttir_to_atlas, AtlasTarget
from compiler.atlas_eligibility import TileBinding

FIXTURE = Path(__file__).parent/'fixtures/triton_quantized_attention/kernel.ttir'
POLICY = 'atlas-fp8-attention-v1'
BINDINGS = tuple(TileBinding(i,'input',2*i) for i in range(3))+tuple(TileBinding(i,'output',2*(i-3)+1) for i in range(3,7))

class PipelineTests(unittest.TestCase):
    def test_preserves_genuine_input_and_emits_existing_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'artifacts'
            path = compile_ttir_to_atlas(FIXTURE, out_dir=out, numerical_policy=POLICY, bindings=BINDINGS)
            self.assertEqual((out/'input.ttir').read_bytes(), FIXTURE.read_bytes())
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertTrue(manifest['success'])
            self.assertEqual(manifest['diagnostics'], [])
            self.assertIn('atlas.virtual_mxu_matmul', path.read_text())
            with self.assertRaises(FileExistsError):
                compile_ttir_to_atlas(FIXTURE, out_dir=out, numerical_policy=POLICY, bindings=BINDINGS)

    def test_rejects_unsupported_target_before_creating_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'artifacts'
            with self.assertRaises(ValueError):
                compile_ttir_to_atlas(FIXTURE, out_dir=out, numerical_policy=POLICY, bindings=BINDINGS, target=AtlasTarget(tile=(16,16)))
            self.assertFalse(out.exists())

    def test_diagnostic_manifest_for_invalid_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'artifacts'
            with self.assertRaises(NotImplementedError):
                compile_ttir_to_atlas(FIXTURE, out_dir=out, numerical_policy=POLICY, bindings=())
            report = json.loads((out/'manifest.json').read_text())
            self.assertFalse(report['success'])
            self.assertEqual(report['error']['stage'], 'eligibility')
            self.assertTrue(report['diagnostics'])
            self.assertFalse((out/'program.virtual.mlir').exists())
