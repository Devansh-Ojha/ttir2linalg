"""Operation-level qualification of real Triton 3.6 SM89 compiler output."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/validation'))
from compiler.ttir_reader import parse_ttir
from compiler.triton_tile_normalization import normalize_static_tiles
from compiler.atlas_operation_mapping import emit_atlas_operations
from compiler.atlas_eligibility import FP8TileABI,TileBinding,analyze_atlas_eligibility
from atlas_fp8_reference import dot, pack

FIXTURES=Path(__file__).parent/'fixtures/triton_jit_sm89'
POLICY='atlas-fp8-mlp-v1'

class TritonFP8Tests(unittest.TestCase):
    def graph(self,name):
        text=(FIXTURES/f'{name}.ttir').read_text()
        a=normalize_static_tiles(parse_ttir(text).funcs[0],grid=(1,1,1),numerical_policy=POLICY,resolve_dot_defaults=True).analysis
        n=len(a.function.args)-1
        abi=FP8TileABI(tuple(TileBinding(i,'input',i) for i in range(n))+(TileBinding(n,'output',0),))
        return a,abi

    def test_genuine_capture_hashes(self):
        for name in ('fp8_matmul','fp8_two_layer'):
            meta=json.loads((FIXTURES/f'{name}.capture.json').read_text())
            self.assertEqual(meta['triton_version'],'3.6.0')
            for filename,key in ((f'{name}.ttir','ttir_sha256'),('kernel.source.py','source_sha256')):
                self.assertEqual(hashlib.sha256((FIXTURES/filename).read_bytes()).hexdigest(),meta[key])

    def test_operation_emission_and_dependencies(self):
        for name,depth in (('fp8_matmul',1),('fp8_two_layer',2)):
            a,abi=self.graph(name)
            text=emit_atlas_operations(a,numerical_policy=POLICY,abi=abi)
            self.assertEqual(text.count('"atlas.virtual_mxu_matmul"'),depth)
            self.assertEqual(text.count('"atlas.virtual_xlu_transpose"'),depth)
            self.assertEqual(text.count('"atlas.virtual_pack_fp8"'),depth-1)
            self.assertEqual([e.unit for e in analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=abi).plan.operations if e.candidate=='atlas.virtual_mxu_matmul'],list(range(depth)))

    def test_missing_policy_rejected(self):
        a,abi=self.graph('fp8_matmul')
        self.assertFalse(analyze_atlas_eligibility(a,abi=abi).eligible)

    def test_precision_rejected(self):
        # Use the actual printed SSA names rather than relying on source names.
        import re
        original=(FIXTURES/'fp8_matmul.ttir').read_text()
        text=re.sub(r'(tt.dot %[^,]+, %[^,]+, %\w+)',r'\1, inputPrecision = tf32',original)
        a=normalize_static_tiles(parse_ttir(text).funcs[0],grid=(1,1,1),numerical_policy=POLICY,resolve_dot_defaults=True).analysis
        _,abi=self.graph('fp8_matmul')
        result=analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=abi)
        self.assertIn('dot-policy',[d.code for d in result.diagnostics])

    def test_nonzero_accumulator_rejected(self):
        a,abi=self.graph('fp8_matmul')
        constant=next(o for o in a.operations if o.name=='arith.constant')
        constant.attrs['value']='dense<1.000000e+00>'
        result=analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=abi)
        self.assertFalse(result.eligible)

    def test_imprecise_accumulation_rejected(self):
        import re
        original=(FIXTURES/'fp8_matmul.ttir').read_text()
        text=re.sub(r'(tt.dot %[^,]+, %[^,]+, %\w+)',r'\1 {maxNumImpreciseAcc = 1 : i32}',original)
        a=normalize_static_tiles(parse_ttir(text).funcs[0],grid=(1,1,1),numerical_policy=POLICY,resolve_dot_defaults=True).analysis
        _,abi=self.graph('fp8_matmul')
        self.assertFalse(analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=abi).eligible)

    def test_pack_boundaries(self):
        self.assertEqual([pack(v) for v in (0x3f88,0x3f98,0x3b80,0x4460,0xbf80)], [0x38,0x3a,0,0x7e,0xb8])

    def test_independent_known_permutation(self):
        x=[0x38 if r==c else 0 for r in range(32) for c in range(32)]
        w=[0x40 if r==(c+3)%32 else 0 for r in range(32) for c in range(32)]
        expected=[0x4000 if r==(c+3)%32 else 0 for r in range(32) for c in range(32)]
        self.assertEqual(dot(x,w,0),expected)
        self.assertEqual(dot(x,w,1),expected)

if __name__=='__main__':unittest.main()
