"""Genuine offline captures, static-address proofs, policy rejection and execution."""
from compiler.toolchain_paths import atlas_root as default_atlas_root
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

from compiler.jit_capture import capture_jit_kernel
from compiler.ttir_reader import parse_ttir
from compiler.triton_tile_normalization import normalize_static_tiles
from compiler.ttir_operation_analysis import UnsupportedKernel,analyze_operations
from compiler.atlas_eligibility import BF16TileABI,FP8TileABI,TileBinding,analyze_atlas_eligibility
from compiler.atlas_operation_mapping import emit_atlas_operations

ROOT=Path(__file__).parents[1]
FIX=ROOT/'tests/fixtures/jit_atlas'
VALIDATION=ROOT/'tools/validation'
if str(VALIDATION) not in sys.path:sys.path.insert(0,str(VALIDATION))
from validate_jit_atlas import validate_example,EXAMPLES
from validate_atlas_operations import validation_abi


def normalized(name,text=None,grid=(1,1,1)):
    fp8=name.startswith('fp8')
    f=parse_ttir(text or (FIX/(name+'.ttir')).read_text()).funcs[0]
    return normalize_static_tiles(f,grid=grid,numerical_policy='atlas-fp8-mlp-v1' if fp8 else 'atlas-bf16-elementwise-rtl-v1',resolve_dot_defaults=fp8)


def fp8_abi(a):
    n=len(a.function.args)-1
    return FP8TileABI(tuple([TileBinding(i,'input',i) for i in range(n)]+[TileBinding(n,'output',0)]))


class JitAtlasTests(unittest.TestCase):
    def test_capture_provenance(self):
        source=(FIX/'kernel.source.py').read_bytes()
        for name in ['bf16_add_relu','fp8_matmul','fp8_two_layer']:
            m=json.loads((FIX/(name+'.capture.json')).read_text())
            self.assertEqual(m['origin'],'triton.compile(ASTSource).asm[ttir]')
            self.assertEqual(hashlib.sha256((FIX/(name+'.ttir')).read_bytes()).hexdigest(),m['ttir_sha256'])
            self.assertEqual(hashlib.sha256(source).hexdigest(),m['source_sha256'])
            self.assertFalse(m['gpu_launched'])
            self.assertEqual(m['target']['arch'],89)

    def test_bf16_original_and_derived_graph(self):
        n=normalized('bf16_add_relu')
        self.assertTrue(n.address_operations)
        self.assertTrue(n.precision_operations)
        self.assertTrue(any(o.name=='arith.extf' for o in n.original.operations))
        self.assertFalse(any(o.name in {'tt.addptr','arith.extf','arith.truncf'} for o in n.analysis.operations))
        self.assertTrue(all(v.type.shape==(32,32) for v in n.analysis.function.args))
        self.assertTrue(all(v.type.kind=='ptr' for v in n.original.function.args))
        e=analyze_atlas_eligibility(n.analysis,numerical_policy='atlas-bf16-elementwise-rtl-v1',abi=validation_abi(n.analysis))
        self.assertTrue(e.eligible,e.diagnostics)

    def test_fp8_generic_operations(self):
        for name,count in [('fp8_matmul',1),('fp8_two_layer',2)]:
            n=normalized(name);a=n.analysis
            e=analyze_atlas_eligibility(a,numerical_policy='atlas-fp8-mlp-v1',abi=fp8_abi(a))
            self.assertTrue(e.eligible,e.diagnostics)
            out=emit_atlas_operations(a,numerical_policy='atlas-fp8-mlp-v1',abi=fp8_abi(a))
            self.assertEqual(out.count('"atlas.virtual_mxu_matmul"'),count)
            self.assertEqual(out.count('"atlas.virtual_xlu_transpose"'),count)
            self.assertEqual(out.count('"atlas.virtual_pack_fp8"'),count-1)

    def test_kernel_name_does_not_select_mapping(self):
        text=(FIX/'fp8_two_layer.ttir').read_text().replace('@fp8_two_layer','@unrelated_graph')
        a=normalized('fp8_two_layer',text).analysis
        self.assertIn('@ttir_unrelated_graph_atlas_ops',emit_atlas_operations(a,numerical_policy='atlas-fp8-mlp-v1',abi=fp8_abi(a)))

    def test_dot_defaults_are_opt_in_and_not_fabricated_attributes(self):
        f=parse_ttir((FIX/'fp8_matmul.ttir').read_text()).funcs[0]
        with self.assertRaises(UnsupportedKernel):analyze_operations(f)
        a=analyze_operations(f,resolve_dot_defaults=True)
        dot=next(o for o in a.operations if o.name=='tt.dot')
        self.assertNotIn('inputPrecision',dot.attrs)
        self.assertTrue(a.resolved_dot_defaults)

    def test_launch_geometry_rejection(self):
        for grid in [(2,1,1),(1,2,1),(1,1),(0,1,1)]:
            with self.assertRaises(UnsupportedKernel):normalized('bf16_add_relu',grid=grid)

    def test_strided_or_transposed_addresses_rejected(self):
        text=(FIX/'bf16_add_relu.ttir').read_text()
        for value in ['16','64','-32']:
            with self.assertRaises(UnsupportedKernel):normalized('bf16_add_relu',text.replace('dense<32>','dense<'+value+'>'))

    def test_masks_rejected(self):
        text=(FIX/'bf16_add_relu.ttir').read_text()
        # Valid comparison mask, deliberately added to a genuine capture only
        # for this negative test; the saved fixture itself is unchanged.
        text=text.replace('    %x_6 = tt.load %x_5', '    %mask = arith.cmpi eq, %offsets_4, %offsets_4 : tensor<32x32xi32>\n    %x_6 = tt.load %x_5, %mask')
        with self.assertRaises(UnsupportedKernel):normalized('bf16_add_relu',text)

    def test_unknown_operations_and_flags_rejected(self):
        text=(FIX/'bf16_add_relu.ttir').read_text()
        for altered in [text.replace('arith.addf','arith.divf'),text.replace('arith.addf %x_6, %y_8','arith.addf %x_6, %y_8 fastmath<nnan>')]:
            with self.assertRaises(UnsupportedKernel):normalized('bf16_add_relu',altered)

    def test_nonzero_fp32_maximum_not_demoted(self):
        text=(FIX/'bf16_add_relu.ttir').read_text().replace('dense<0.000000e+00>','dense<1.000000e+00>')
        a=normalized('bf16_add_relu',text).analysis
        e=analyze_atlas_eligibility(a,numerical_policy='atlas-bf16-elementwise-rtl-v1',abi=validation_abi(a))
        self.assertFalse(e.eligible);self.assertIsNone(e.plan)

    def test_wrong_policy_rejected(self):
        a=normalized('fp8_matmul').analysis
        for p in [None,'ieee','atlas-bf16-elementwise-rtl-v1']:
            self.assertFalse(analyze_atlas_eligibility(a,numerical_policy=p,abi=fp8_abi(a)).eligible)

    def test_fp8_nonzero_accumulator_and_rounding_rejected(self):
        for name,text in [
          ('fp8_matmul',(FIX/'fp8_matmul.ttir').read_text().replace('dense<0.000000e+00>','dense<1.000000e+00>')),
          ('fp8_two_layer',(FIX/'fp8_two_layer.ttir').read_text().replace('rounding = rtne','rounding = rtz'))]:
            a=normalized(name,text).analysis
            e=analyze_atlas_eligibility(a,numerical_policy='atlas-fp8-mlp-v1',abi=fp8_abi(a))
            self.assertFalse(e.eligible);self.assertIsNone(e.plan)

    def test_fp32_dot_extra_consumer_rejected(self):
        text=(FIX/'fp8_matmul.ttir').read_text()
        text=text.replace('    %result_12 = arith.truncf','    %extra = arith.addf %result_11, %result_11 : tensor<32x32xf32>\n    %result_12 = arith.truncf')
        a=normalized('fp8_matmul',text).analysis
        self.assertFalse(analyze_atlas_eligibility(a,numerical_policy='atlas-fp8-mlp-v1',abi=fp8_abi(a)).eligible)

    def test_stale_fp8_analysis_and_binding_rejected(self):
        a=normalized('fp8_matmul').analysis
        e=analyze_atlas_eligibility(replace(a,operations=a.operations[:-1]),numerical_policy='atlas-fp8-mlp-v1',abi=fp8_abi(a))
        self.assertFalse(e.eligible)
        e=analyze_atlas_eligibility(a,numerical_policy='atlas-fp8-mlp-v1',abi=replace(fp8_abi(a),disjoint=False))
        self.assertFalse(e.eligible)

    def test_real_offline_capture_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'capture'
            ttir=capture_jit_kernel('kernels.atlas_jit_kernels','bf16_add_relu',EXAMPLES['bf16-add-relu'][1],out_dir=p)
            self.assertIn('tt.func public @bf16_add_relu',ttir.read_text())
            self.assertTrue((p/'capture.json').exists())
            self.assertTrue((p/'triton-cache').exists())
            before=ttir.read_bytes()
            with self.assertRaises(FileExistsError):capture_jit_kernel('kernels.atlas_jit_kernels','bf16_add_relu',{},out_dir=p)
            self.assertEqual(ttir.read_bytes(),before)

    def test_generic_cli_accepts_user_module_signature_and_bindings(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'compiled'
            command=[sys.executable,'-m','compiler.jit_to_atlas','--module','kernels.atlas_jit_kernels',
                     '--kernel','bf16_add_relu','--signature',json.dumps(EXAMPLES['bf16-add-relu'][1]),
                     '--out-dir',str(out),'--numerical-policy','atlas-bf16-elementwise-rtl-v1',
                     '--tile-binding','0:input:0','--tile-binding','1:input:1','--tile-binding','2:output:0']
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((out/'normalization.json').exists())
            before=(out/'program.virtual.mlir').read_bytes()
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(before,(out/'program.virtual.mlir').read_bytes())

    @unittest.skipUnless((default_atlas_root()/'build/bin/atlas-opt').exists(),'Atlas backend unavailable')
    def test_real_offline_compile_to_machine_all_examples(self):
        with tempfile.TemporaryDirectory() as d:
            for example in EXAMPLES:
                with self.subTest(example=example),redirect_stdout(StringIO()):
                    r=validate_example(example,Path(d)/example,numerical_policy=EXAMPLES[example][2])
                    self.assertEqual(len(r['cases']),8 if example=='bf16-add-relu' else 21)
                    self.assertTrue(all(c['mismatches']==0 for c in r['cases']))


if __name__=='__main__':unittest.main()
