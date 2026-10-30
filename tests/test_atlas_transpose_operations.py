"""Genuine typed transpose, hardware format rejection and exact machine execution."""
import ast
from contextlib import redirect_stdout
from dataclasses import replace
from fractions import Fraction
from io import StringIO
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/validation'))
from compiler.ttir_reader import parse_ttir
from compiler.ttir_operation_analysis import analyze_operations,transpose_order,UnsupportedKernel
from compiler.triton_tile_normalization import normalize_static_tiles
from compiler.atlas_eligibility import FP8LayoutABI,FP8TileABI,BF16TileABI,TileBinding,analyze_atlas_eligibility
from compiler.atlas_layout_operations import LAYOUT_POLICY
from compiler.atlas_operation_mapping import emit_atlas_operations,AtlasOperationMemory
from compiler.jit_to_atlas import compile_jit_to_atlas
from validate_transpose_operations import validate,layout_reference,instruction_adapter
from atlas_bf16_reference import round_binary,_value
FIX=ROOT/'tests/fixtures/triton_transpose'
LAYOUT_ABI=FP8LayoutABI((TileBinding(0,'input',0),TileBinding(1,'output',0)))
FANOUT=(TileBinding(0,'input',2),TileBinding(1,'output',1),TileBinding(2,'output',3))


def normalized(name,policy=None):
    policy=policy or ('atlas-fp8-mlp-v1' if name=='fp8_dot' else LAYOUT_POLICY if name in ('fp8','fanout') else 'atlas-bf16-rowwise-rtl-v1')
    return normalize_static_tiles(parse_ttir((FIX/f'{name}.ttir').read_text()).funcs[0],grid=(1,1,1),numerical_policy=policy,resolve_dot_defaults=name=='fp8_dot')


class TransposeTests(unittest.TestCase):
    def test_compiler_capture_hashes(self):
        for name in ('fp8','bf16','fp8_dot','fanout'):
            m=json.loads((FIX/f'{name}.capture.json').read_text())
            self.assertEqual(m['origin'],'triton.compile(ASTSource).asm[ttir]')
            self.assertEqual(m['triton_version'],'3.6.0')
            self.assertEqual(m['target'],{'backend':'cuda','arch':89,'warp_size':32})
            for suffix,key in [('ttir','ttir_sha256'),('source.py','source_sha256')]:
                self.assertEqual(hashlib.sha256((FIX/f'{name}.{suffix}').read_bytes()).hexdigest(),m[key])

    def test_shape_type_ssa_and_order_preserved(self):
        for name in ('fp8','bf16'):
            n=normalized(name)
            before=next(o for o in n.original.operations if o.name=='tt.trans')
            after=next(o for o in n.analysis.operations if o.name=='tt.trans')
            self.assertEqual(transpose_order(after),(1,0))
            self.assertEqual(after.operands[0].type,before.operands[0].type)
            self.assertEqual(after.results[0].type,before.results[0].type)
            self.assertEqual(after.attrs,before.attrs)
            self.assertEqual(after.operands[0].defining_op.name,'tt.load')
            self.assertIs(n.analysis.memory[1].value,after.results[0])

    def test_standalone_emits_physical_movement_and_raw_store(self):
        a=normalized('fp8').analysis
        text=emit_atlas_operations(a,numerical_policy=LAYOUT_POLICY,abi=LAYOUT_ABI)
        self.assertEqual(text.count('"atlas.virtual_xlu_transpose"'),1)
        self.assertEqual(text.count('"atlas.virtual_dma_store_fp8"'),1)
        self.assertNotIn('"atlas.virtual_pack_fp8"',text)
        self.assertNotIn('"atlas.virtual_mxu_matmul"',text)

    def test_transpose_not_ignored_in_contraction(self):
        a=normalized('fp8_dot').analysis
        abi=FP8TileABI((TileBinding(0,'input',0),TileBinding(1,'input',1),TileBinding(2,'output',0)))
        text=emit_atlas_operations(a,numerical_policy='atlas-fp8-mlp-v1',abi=abi)
        self.assertEqual(text.count('"atlas.virtual_xlu_transpose"'),2)
        self.assertEqual(text.count('"atlas.virtual_mxu_matmul"'),1)
        self.assertIn('"atlas.virtual_mxu_matmul"(%v2, %v3_weight)',text)

    def test_fanout_keeps_original_producer_and_bindings(self):
        a=normalized('fanout').analysis
        result=analyze_atlas_eligibility(a,numerical_policy=LAYOUT_POLICY,abi=FP8LayoutABI(FANOUT))
        self.assertTrue(result.eligible,result.diagnostics)
        trans=next(o for o in a.operations if o.name=='tt.trans')
        stores=[o for o in a.operations if o.name=='tt.store']
        self.assertIs(stores[0].operands[1],trans.operands[0])
        self.assertIs(stores[1].operands[1],trans.results[0])
        text=emit_atlas_operations(a,numerical_policy=LAYOUT_POLICY,abi=FP8LayoutABI(FANOUT))
        self.assertIn('index = 2 : i32',text)
        self.assertEqual(text.count('"atlas.virtual_dma_store_fp8"'),2)

    def test_no_function_name_recognition(self):
        text=(FIX/'fp8.ttir').read_text().replace('@tile_transpose','@unrelated_kernel')
        a=normalize_static_tiles(parse_ttir(text).funcs[0],grid=(1,1,1),numerical_policy=LAYOUT_POLICY).analysis
        self.assertIn('"atlas.virtual_xlu_transpose"',emit_atlas_operations(a,numerical_policy=LAYOUT_POLICY,abi=LAYOUT_ABI))

    def test_stale_permutation_rejected(self):
        a=normalized('fp8').analysis
        trans=next(o for o in a.operations if o.name=='tt.trans')
        trans.attrs['order']='array<i32: 0, 1>'
        with self.assertRaisesRegex(UnsupportedKernel,'disagrees with source'):
            analyze_operations(a.function)
        self.assertFalse(analyze_atlas_eligibility(a,numerical_policy=LAYOUT_POLICY,abi=LAYOUT_ABI).eligible)

    def test_partial_graph_rejected(self):
        a=normalized('fp8').analysis
        self.assertFalse(analyze_atlas_eligibility(replace(a,operations=a.operations[:-1]),numerical_policy=LAYOUT_POLICY,abi=LAYOUT_ABI).eligible)

    def test_bf16_transpose_rejected_without_quantization(self):
        a=normalized('bf16').analysis
        abi=BF16TileABI(LAYOUT_ABI.bindings)
        r=analyze_atlas_eligibility(a,numerical_policy='atlas-bf16-rowwise-rtl-v1',abi=abi)
        self.assertFalse(r.eligible)
        self.assertIn('transpose-hardware-format',[d.code for d in r.diagnostics])
        with self.assertRaisesRegex(NotImplementedError,'XLU transposes 32x32 bytes'):
            emit_atlas_operations(a,numerical_policy='atlas-bf16-rowwise-rtl-v1',abi=abi)

    def test_wrong_policy_and_format_rejected(self):
        a=normalized('fp8').analysis
        self.assertFalse(analyze_atlas_eligibility(a,abi=LAYOUT_ABI).eligible)
        self.assertFalse(analyze_atlas_eligibility(a,numerical_policy=LAYOUT_POLICY,abi=BF16TileABI(LAYOUT_ABI.bindings)).eligible)
        self.assertFalse(analyze_atlas_eligibility(normalized('bf16').analysis,numerical_policy=LAYOUT_POLICY,abi=LAYOUT_ABI).eligible)

    def test_aliasing_and_overlapping_memory_rejected(self):
        a=normalized('fp8').analysis
        self.assertFalse(analyze_atlas_eligibility(a,numerical_policy=LAYOUT_POLICY,abi=replace(LAYOUT_ABI,disjoint=False)).eligible)
        with self.assertRaisesRegex(ValueError,'overlap'):
            emit_atlas_operations(a,numerical_policy=LAYOUT_POLICY,abi=LAYOUT_ABI,memory=AtlasOperationMemory(output_base=0x90000000))

    def test_known_asymmetric_raw_permutation(self):
        data=[0]*1024;data[3*32+17]=0xff;data[7*32+2]=0x81
        expected=[0]*1024;expected[17*32+3]=0xff;expected[2*32+7]=0x81
        self.assertEqual(layout_reference(normalized('fp8').analysis,{0:data})[1],expected)

    def test_attention_source_ast_unchanged(self):
        old=ROOT/'tests/fixtures/triton_rowwise/probes.source.py'
        def kernel(path):
            return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='bf16_simple_attention')
        self.assertEqual(ast.dump(kernel(old)),ast.dump(kernel(ROOT/'kernels/atlas_jit_kernels.py')))

    def test_attention_analyzed_but_two_bf16_dots_rejected(self):
        f=parse_ttir((ROOT/'tests/fixtures/triton_rowwise/attention.ttir').read_text()).funcs[0]
        n=normalize_static_tiles(f,grid=(1,1,1),numerical_policy='atlas-bf16-rowwise-rtl-v1',resolve_dot_defaults=True)
        abi=BF16TileABI(tuple(TileBinding(i,'input',i) for i in range(3))+(TileBinding(3,'output',0),))
        r=analyze_atlas_eligibility(n.analysis,numerical_policy='atlas-bf16-rowwise-rtl-v1',abi=abi)
        self.assertFalse(r.eligible)
        self.assertEqual(sum(d.code=='dot-hardware-format' for d in r.diagnostics),2)
        self.assertTrue(any(o.name=='arith.mulf' and o.results[0].type.elem=='f32' for o in n.analysis.operations))
        self.assertFalse(any(o.name=='tt.fp_to_fp' for o in n.original.operations))

    def test_scaling_rounding_point_counterexample(self):
        scale=_value(round_binary(Fraction('0.176776692'),23),23)
        score=Fraction(257,256)
        source=round_binary(_value(round_binary(score*scale,23),23))
        early=round_binary(_value(round_binary(score))*_value(round_binary(scale)))
        self.assertEqual(source,0x3e36)
        self.assertEqual(early,0x3e35)

    def test_decoder_metadata_is_not_an_instruction_field(self):
        row={'operation':'atlas.upper','fields':{'kind':'lui','dst':10,'immediate':589856,'atlas.virtual_scalar_result':2}}
        instruction_adapter(row)
        with self.assertRaises(ValueError):
            instruction_adapter(dict(row,fields=dict(row['fields'],unexpected=1)))
        with self.assertRaises(ValueError):
            instruction_adapter(dict(row,fields=dict(row['fields'],**{'atlas.virtual_scalar_result':-1})))

    def test_generic_jit_api_layout_policy(self):
        with tempfile.TemporaryDirectory(prefix='transpose-jit-') as temporary:
            r=compile_jit_to_atlas('kernels.atlas_jit_kernels','tile_transpose',{'X':'*fp8e4nv','Out':'*fp8e4nv'},
                out_dir=Path(temporary)/'compiled',bindings=LAYOUT_ABI.bindings,numerical_policy=LAYOUT_POLICY,lower=True)
            self.assertTrue((r.lowered/'5-words.hex').exists())
            self.assertIsInstance(r.abi,FP8LayoutABI)

    def test_all_genuine_transpose_machine_cases(self):
        with tempfile.TemporaryDirectory(prefix='transpose-machine-') as temporary,redirect_stdout(StringIO()):
            root=Path(temporary)
            for name,policy,bindings in [('fp8',LAYOUT_POLICY,None),('fanout',LAYOUT_POLICY,FANOUT),('fp8_dot','atlas-fp8-mlp-v1',None)]:
                capture=root/name/'capture';capture.mkdir(parents=True)
                for suffix,target in [('ttir','kernel.ttir'),('capture.json','capture.json'),('source.py','kernel.source.py')]:
                    shutil.copyfile(FIX/f'{name}.{suffix}',capture/target)
                self.assertTrue(validate(capture,root/name/'validation',numerical_policy=policy,bindings=bindings))

if __name__=='__main__':unittest.main()
