"""Genuine compiler fixtures, reusable planning, numerics and machine execution."""
from compiler.toolchain_paths import npu_model_root
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/validation'))
from compiler.ttir_reader import parse_ttir
from compiler.ttir_operation_analysis import analyze_operations,UnsupportedKernel
from compiler.triton_tile_normalization import normalize_static_tiles
from compiler.atlas_eligibility import BF16TileABI,TileBinding,analyze_atlas_eligibility
from compiler.atlas_operation_mapping import emit_atlas_operations
from compiler.atlas_rowwise_operations import ROWWISE_POLICY
from atlas_rowwise_reference import exponential,reciprocal,reduce_rows,source_reference
from validate_rowwise_operations import validate

FIX=ROOT/'tests/fixtures/triton_rowwise'
ABI=BF16TileABI((TileBinding(0,'input',0),TileBinding(1,'output',0)))


def normalized(name='softmax',text=None):
    f=parse_ttir(text or (FIX/f'{name}.ttir').read_text()).funcs[0]
    return normalize_static_tiles(f,grid=(1,1,1),numerical_policy=ROWWISE_POLICY)


class RowwiseTests(unittest.TestCase):
    def test_genuine_capture_hashes(self):
        for name in ('softmax','row_max','row_sum','attention'):
            m=json.loads((FIX/f'{name}.capture.json').read_text())
            self.assertEqual(m['triton_version'],'3.6.0')
            self.assertEqual(m['origin'],'triton.compile(ASTSource).asm[ttir]')
            self.assertEqual(m['target']['arch'],89)
            self.assertFalse(m['gpu_launched'])
            self.assertEqual(hashlib.sha256((FIX/f'{name}.ttir').read_bytes()).hexdigest(),m['ttir_sha256'])
            src=FIX/('softmax.source.py' if name=='softmax' else 'probes.source.py')
            self.assertEqual(hashlib.sha256(src.read_bytes()).hexdigest(),m['source_sha256'])

    def test_regions_and_original_source_preserved(self):
        n=normalized()
        self.assertEqual(n.original.function.source,(FIX/'softmax.ttir').read_text())
        self.assertEqual(len(n.original.reductions),2)
        for original,derived in zip(n.original.reductions,n.analysis.reductions):
            self.assertIsNot(original.operation,derived.operation)
            self.assertEqual(original.axis,derived.axis)
            self.assertEqual(original.combiner.name,derived.combiner.name)
            block=derived.operation.regions[0][0]
            self.assertTrue(all(x is y for x,y in zip(block.args,block.ops[0].operands)))
            self.assertIs(block.ops[1].operands[0],block.ops[0].results[0])

    def test_layout_and_precision_plan(self):
        a=normalized().analysis
        plan=analyze_atlas_eligibility(a,numerical_policy=ROWWISE_POLICY,abi=ABI)
        self.assertTrue(plan.eligible,plan.diagnostics)
        reductions=[e for e in plan.plan.operations if e.source.name=='tt.reduce']
        self.assertEqual([e.candidate for e in reductions],['atlas.virtual_vpu_reduce:row_max','atlas.virtual_vpu_reduce:row_sum'])
        self.assertTrue(all(e.result_layout=='row' and e.precision_mode=='readout' for e in reductions))
        text=emit_atlas_operations(a,numerical_policy=ROWWISE_POLICY,abi=ABI)
        for op in ('row_max','row_sum','exp','recip','sub','mul'):
            self.assertIn(f'kind = "{op}"',text)
        self.assertNotIn('softmax"',text)

    def test_reduction_compositions_not_name_matched(self):
        for name,kind in [('row_max','row_max'),('row_sum','row_sum')]:
            original=(FIX/f'{name}.ttir').read_text()
            renamed=original.replace('@bf16_row_reduce','@unrelated_computation')
            a=normalized(name,renamed).analysis
            text=emit_atlas_operations(a,numerical_policy=ROWWISE_POLICY,abi=ABI)
            self.assertEqual(text.count('"atlas.virtual_vpu_reduce"'),1)
            self.assertIn(f'kind = "{kind}"',text)
            self.assertNotIn('"atlas.virtual_vpu_unary"',text)

    def test_wrong_policy_rejected(self):
        a=normalized().analysis
        self.assertFalse(analyze_atlas_eligibility(a,numerical_policy='atlas-bf16-elementwise-rtl-v1',abi=ABI).eligible)

    def test_partial_graph_rejected(self):
        a=normalized().analysis
        self.assertFalse(analyze_atlas_eligibility(replace(a,operations=a.operations[:-1]),numerical_policy=ROWWISE_POLICY,abi=ABI).eligible)
        self.assertFalse(analyze_atlas_eligibility(replace(a,reductions=()),numerical_policy=ROWWISE_POLICY,abi=ABI).eligible)

    def test_extra_fp32_computation_rejected(self):
        original=(FIX/'softmax.ttir').read_text()
        marker='    %exponential_12 = arith.truncf %exponential_11'
        text=original.replace(marker,'    %extra = math.exp %exponential_11 : tensor<32x32xf32>\n    %exponential_12 = arith.truncf %extra')
        self.assertNotEqual(text,original)
        a=normalized(text=text).analysis
        result=analyze_atlas_eligibility(a,numerical_policy=ROWWISE_POLICY,abi=ABI)
        self.assertFalse(result.eligible)
        self.assertIn('precision-policy',[d.code for d in result.diagnostics])

    def test_non_row_reduction_rejected(self):
        a=normalized('row_sum', (FIX/'row_sum.ttir').read_text().replace('axis = 1 : i32}>','axis = 0 : i32}>')).analysis
        result=analyze_atlas_eligibility(a,numerical_policy=ROWWISE_POLICY,abi=ABI)
        self.assertFalse(result.eligible)
        self.assertIn('reduction-policy',[d.code for d in result.diagnostics])

    def test_bad_abi_and_grid_rejected(self):
        a=normalized().analysis
        self.assertFalse(analyze_atlas_eligibility(a,numerical_policy=ROWWISE_POLICY,abi=replace(ABI,disjoint=False)).eligible)
        with self.assertRaises(UnsupportedKernel):
            normalize_static_tiles(parse_ttir((FIX/'softmax.ttir').read_text()).funcs[0],grid=(2,1,1),numerical_policy=ROWWISE_POLICY)

    def test_attention_reports_transpose_blocker(self):
        original=parse_ttir((FIX/'attention.ttir').read_text()).funcs[0]
        analyzed=analyze_operations(original,resolve_dot_defaults=True)
        self.assertTrue(any(o.name=='tt.trans' for o in analyzed.operations))
        a=normalize_static_tiles(original,grid=(1,1,1),numerical_policy=ROWWISE_POLICY,resolve_dot_defaults=True).analysis
        abi=BF16TileABI(tuple(TileBinding(i,'input',i) for i in range(3))+(TileBinding(3,'output',0),))
        result=analyze_atlas_eligibility(a,numerical_policy=ROWWISE_POLICY,abi=abi)
        self.assertFalse(result.eligible)
        self.assertIn('transpose-hardware-format',[d.code for d in result.diagnostics])
        self.assertEqual(sum(d.code=='dot-hardware-format' for d in result.diagnostics),2)

    def test_unary_known_values(self):
        self.assertEqual(exponential(0),0x3f80)
        self.assertEqual(exponential(0xff80),0)
        self.assertEqual(reciprocal(0x4040),0x3eaa) # LUT/truncation 1/3, not RNE 0x3eab.
        self.assertEqual(reciprocal(0x7fc0),0)
        self.assertEqual(reciprocal(1),0x7f80)

    def test_reference_matches_all_rtl_unary_inputs(self):
        # Read-only exhaustive cross-check. The reference derives its own LUTs;
        # these files are comparison data only, never used during evaluation.
        root=npu_model_root()/'npu_model/configs/data'
        for name,fn in [('exp',exponential),('rcp',reciprocal)]:
            wanted=struct.unpack('<65536H',gzip.decompress((root/f'{name}.bin.gz').read_bytes()))
            self.assertEqual([fn(i) for i in range(65536)],list(wanted))

    def test_known_row_results(self):
        bits=[0x3f80]*1024
        self.assertEqual(reduce_rows(bits,'arith.addf'),[0x4200]*32)
        self.assertEqual(reduce_rows(bits,'arith.maxnumf'),[0x3f80]*32)
        self.assertEqual(source_reference(normalized().analysis,{0:[0]*1024})[1],[0x3d00]*1024)

    def test_generic_jit_compiler_accepts_rowwise_policy(self):
        from compiler.jit_to_atlas import compile_jit_to_atlas
        with tempfile.TemporaryDirectory(prefix='rowwise-jit-') as temporary:
            result=compile_jit_to_atlas('kernels.atlas_jit_kernels','bf16_row_softmax',
                {'X':'*bf16','Out':'*bf16'},out_dir=Path(temporary)/'compiled',
                bindings=ABI.bindings,numerical_policy=ROWWISE_POLICY,lower=True)
            self.assertTrue(result.captured_ttir.exists())
            self.assertTrue((result.lowered/'5-words.hex').exists())
            self.assertEqual(len(result.normalization.analysis.reductions),2)

    def test_genuine_verifier_machine_simulation(self):
        with tempfile.TemporaryDirectory(prefix='rowwise-test-') as temporary,redirect_stdout(StringIO()):
            root=Path(temporary)
            for name in ('softmax','row_max','row_sum'):
                capture=root/name/'capture';capture.mkdir(parents=True)
                for original,target in [(f'{name}.ttir','kernel.ttir'),(f'{name}.capture.json','capture.json'),
                                        ('softmax.source.py' if name=='softmax' else 'probes.source.py','kernel.source.py')]:
                    shutil.copyfile(FIX/original,capture/target)
                self.assertTrue(validate(capture,root/name/'validation',numerical_policy=ROWWISE_POLICY,check_softmax=name=='softmax'))

if __name__=='__main__':unittest.main()
