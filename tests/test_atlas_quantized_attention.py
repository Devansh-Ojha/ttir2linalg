"""Real Triton mixed-operation attention, independent arithmetic and execution."""
import ast
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/validation'))
from compiler.ttir_reader import parse_ttir
from compiler.triton_tile_normalization import normalize_static_tiles
from compiler.atlas_eligibility import MixedTileABI,BF16TileABI,TileBinding,analyze_atlas_eligibility
from compiler.atlas_rowwise_operations import FP8_ROWWISE_POLICY as POLICY
from compiler.atlas_operation_mapping import emit_atlas_operations,AtlasOperationMemory
from compiler.jit_to_atlas import compile_jit_to_atlas
from atlas_fp8_reference import pack
from atlas_mixed_reference import source_reference
from validate_quantized_attention import validate,BINDINGS,cases,instruction_adapter,accuracy
from validate_transpose_operations import run_case
from validate_two_layer_mlp_generated_machine import load_instructions
FIX=ROOT/'tests/fixtures/triton_quantized_attention'


def normalized(text=None):
    return normalize_static_tiles(parse_ttir(text or (FIX/'kernel.ttir').read_text()).funcs[0],grid=(1,1,1),numerical_policy=POLICY,resolve_dot_defaults=True)


def abi_for(a):return MixedTileABI(BINDINGS,formats=tuple(v.type.elem for v in a.function.args))


class QuantizedAttentionTests(unittest.TestCase):
    def test_real_compiler_capture_hashes(self):
        m=json.loads((FIX/'capture.json').read_text())
        self.assertEqual(m['origin'],'triton.compile(ASTSource).asm[ttir]')
        self.assertEqual(m['triton_version'],'3.6.0')
        self.assertEqual(m['target'],{'backend':'cuda','arch':89,'warp_size':32})
        for filename,key in [('kernel.ttir','ttir_sha256'),('kernel.source.py','source_sha256')]:
            self.assertEqual(hashlib.sha256((FIX/filename).read_bytes()).hexdigest(),m[key])

    def test_original_bf16_attention_unchanged(self):
        def original(path):return next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='bf16_simple_attention')
        self.assertEqual(ast.dump(original(ROOT/'tests/fixtures/triton_rowwise/probes.source.py')),ast.dump(original(ROOT/'kernels/atlas_jit_kernels.py')))

    def test_source_and_plan_cover_all_operations(self):
        n=normalized();a=n.analysis;r=analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=abi_for(a))
        self.assertTrue(r.eligible,r.diagnostics)
        self.assertEqual([id(e.source) for e in r.plan.operations],[id(o) for o in a.function.body.ops])
        self.assertEqual(len(a.reductions),2)
        self.assertEqual(sum(o.name=='tt.fp_to_fp' for o in n.original.operations),4)
        self.assertEqual([e.unit for e in r.plan.operations if e.candidate=='atlas.virtual_mxu_matmul'],[0,1])
        self.assertEqual([e.scale_code for e in r.plan.operations if e.candidate=='atlas.virtual_pack_fp8'],[127]*4)
        self.assertEqual(sum(e.candidate=='atlas.virtual_xlu_transpose' for e in r.plan.operations),1)

    def test_explicit_source_scales_and_mixed_stores(self):
        a=normalized().analysis;r=analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=abi_for(a))
        self.assertEqual({e.constant_bits for e in r.plan.operations if e.candidate=='atlas.virtual_vpu_splat'},{0x3e35,0x4200,0x3d00})
        text=emit_atlas_operations(a,numerical_policy=POLICY,abi=abi_for(a))
        self.assertEqual(text.count('"atlas.virtual_output_bf16"'),3)
        self.assertEqual(text.count('"atlas.virtual_dma_store_fp8"'),1)
        self.assertEqual(text.count('"atlas.virtual_xlu_transpose"'),3)

    def test_wrong_format_rejected(self):
        a=normalized().analysis;abi=abi_for(a)
        self.assertFalse(analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=replace(abi,formats=('bf16',)*7)).eligible)
        self.assertFalse(analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=replace(abi,formats=())).eligible)

    def test_wrong_policy_rejected(self):
        a=normalized().analysis
        for policy in (None,'atlas-bf16-rowwise-rtl-v1','atlas-fp8-mlp-v1'):
            self.assertFalse(analyze_atlas_eligibility(a,numerical_policy=policy,abi=abi_for(a)).eligible)
        self.assertFalse(analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=BF16TileABI(BINDINGS)).eligible)

    def test_original_bf16_dots_not_admitted(self):
        f=parse_ttir((ROOT/'tests/fixtures/triton_rowwise/attention.ttir').read_text()).funcs[0]
        a=normalize_static_tiles(f,grid=(1,1,1),numerical_policy=POLICY,resolve_dot_defaults=True).analysis
        abi=MixedTileABI(tuple(TileBinding(i,'input',i) for i in range(3))+(TileBinding(3,'output',0),),formats=('bf16',)*4)
        r=analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=abi)
        self.assertFalse(r.eligible)
        self.assertEqual(sum(d.code=='dot-policy' for d in r.diagnostics),2)

    def test_nonzero_accumulator_rejected(self):
        text=(FIX/'kernel.ttir').read_text().replace('dense<0.000000e+00> : tensor<32x32xf32>','dense<1.000000e+00> : tensor<32x32xf32>')
        a=normalized(text).analysis
        self.assertFalse(analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=abi_for(a)).eligible)

    def test_unsupported_rounding_rejected(self):
        text=(FIX/'kernel.ttir').read_text().replace('rounding = rtne','rounding = rtz')
        a=normalized(text).analysis
        self.assertFalse(analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=abi_for(a)).eligible)

    def test_dot_precision_rejected(self):
        original=(FIX/'kernel.ttir').read_text()
        text=re.sub(r'(tt.dot %[^,]+, %[^,]+, %\w+)',r'\1, inputPrecision = tf32',original)
        a=normalized(text).analysis
        self.assertIn('dot-policy',[d.code for d in analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=abi_for(a)).diagnostics])

    def test_extra_fp32_consumer_rejected(self):
        original=(FIX/'kernel.ttir').read_text()
        text=original.replace('    %weighted_29 = arith.truncf','    %extra = arith.addf %weighted, %weighted : tensor<32x32xf32>\n    %weighted_29 = arith.truncf')
        self.assertNotEqual(text,original)
        a=normalized(text).analysis
        self.assertIn('precision-policy',[d.code for d in analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=abi_for(a)).diagnostics])

    def test_partial_graph_and_aliasing_rejected(self):
        a=normalized().analysis;abi=abi_for(a)
        self.assertFalse(analyze_atlas_eligibility(replace(a,operations=a.operations[:-1]),numerical_policy=POLICY,abi=abi).eligible)
        self.assertFalse(analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=replace(abi,disjoint=False)).eligible)
        with self.assertRaisesRegex(ValueError,'overlap'):
            emit_atlas_operations(a,numerical_policy=POLICY,abi=abi,memory=AtlasOperationMemory(output_base=0x90000000))

    def test_pack_normal_round_before_subnormal_flush(self):
        self.assertEqual([pack(b) for b in range(0x3c70,0x3c78)],[0]*8)
        self.assertEqual([pack(b) for b in range(0xbc70,0xbc78)],[0]*8)
        self.assertEqual(pack(0x3c78),8)
        self.assertEqual(pack(0xbc78),0x88)

    def test_exhaustive_independent_pack_matches_model(self):
        from npu_model.configs.numerics import RtlNumerics
        inputs=torch.arange(65536,dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
        wanted=RtlNumerics.to_fp8(inputs,127).view(torch.uint8).tolist()
        self.assertEqual([pack(b) for b in range(65536)],wanted)

    def test_known_uniform_and_scale_negative_control(self):
        n=normalized();inputs={0:[0]*1024,1:[0]*1024,2:[0x4000]*1024}
        expected=source_reference(n.analysis,inputs)
        self.assertEqual(expected[3],[0x4000]*1024)
        self.assertEqual(expected[5],[0x3d00]*1024)
        self.assertEqual(expected[6],[0x38]*1024)
        # Synthetic mutation only for the reference negative control. Positive
        # machine fixtures remain untouched genuine compiler output.
        text=(FIX/'kernel.ttir').read_text().replace('dense<3.200000e+01>','dense<1.600000e+01>')
        a=normalized(text).analysis
        self.assertTrue(analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=abi_for(a)).eligible)
        self.assertEqual(source_reference(a,inputs)[3],[0x3f80]*1024)

    def test_bounded_accuracy_is_separate(self):
        a=normalized().analysis
        _,tiles,_=next(c for c in cases() if c[0]=='bounded_random_0')
        expected=source_reference(a,dict(enumerate(tiles)))
        metrics=accuracy(tiles,expected[3])
        self.assertLess(metrics['max_absolute_error'],0.1)
        self.assertGreater(metrics['max_absolute_error'],0)

    def test_policy_reuses_non_attention_graphs(self):
        with tempfile.TemporaryDirectory(prefix='mixed-composition-') as temporary,redirect_stdout(StringIO()):
            for name,signature,constexprs,data in [
                ('tile_transpose',{'X':'*fp8e4nv','Out':'*fp8e4nv'},{},[i%256 for i in range(1024)]),
                ('bf16_row_reduce',{'X':'*bf16','Out':'*bf16'},{'REDUCE_MAX':False},[0x3f80]*1024)]:
                r=compile_jit_to_atlas('kernels.atlas_jit_kernels',name,signature,out_dir=Path(temporary)/name,
                    bindings=(TileBinding(0,'input',0),TileBinding(1,'output',0)),numerical_policy=POLICY,constexprs=constexprs,lower=True)
                instructions,_=load_instructions(r.lowered,expected_word_count=None,adapter=instruction_adapter)
                result=run_case(r.normalization.analysis,r.abi,r.memory,instructions,name,[data],source_reference)
                self.assertTrue(result['passed'])

    def test_real_fused_machine_and_intermediate_outputs(self):
        with tempfile.TemporaryDirectory(prefix='quantized-attention-') as temporary,redirect_stdout(StringIO()):
            self.assertTrue(validate(FIX,Path(temporary)/'validation',numerical_policy=POLICY,
                selected=['zero_all','uniform_constant_value','bounded_random_0','input_pack_ties','pack_underflow_and_subnormals','nonfinite_inputs']))

if __name__=='__main__':unittest.main()
