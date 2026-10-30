"""Per-operation emission, non-overwriting CLI, backend and machine checks."""
from compiler.toolchain_paths import atlas_root as default_atlas_root
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from compiler.atlas_eligibility import BF16_RTL_POLICY, BF16TileABI, TileBinding
from compiler.atlas_operation_mapping import emit_atlas_operations, AtlasOperationMemory
from compiler.ttir_operation_analysis import analyze_operations
from compiler.ttir_reader import parse_ttir
from compiler.ttir_to_atlas import convert_file, main
from test_atlas_eligibility import source, abi

ROOT=Path(__file__).parents[1]
FIXTURES=ROOT/'tests/fixtures/atlas_ops'
VALIDATION=ROOT/'tools/validation'
if str(VALIDATION) not in sys.path:
    sys.path.insert(0,str(VALIDATION))
from atlas_bf16_reference import binary,round_binary


def analysis(name='add'):
    return analyze_operations(parse_ttir((FIXTURES/(name+'.ttir')).read_text()).funcs[0])


def contract(a):
    from validate_atlas_operations import validation_abi
    return validation_abi(a)


class AtlasOperationMappingTests(unittest.TestCase):
    def emit(self,name='add',**kw):
        a=analysis(name)
        options=dict(numerical_policy=BF16_RTL_POLICY,abi=contract(a))
        options.update(kw)
        return emit_atlas_operations(a,**options)

    def test_all_compositions(self):
        for path in FIXTURES.glob('*.ttir'):
            text=self.emit(path.stem)
            self.assertIn('"atlas.virtual_start"',text)
            self.assertIn('"atlas.virtual_output_bf16"',text)
            self.assertNotIn('virtual_mxu',text)
            self.assertNotIn('virtual_pack',text)

    def test_ssa_operand_order_and_fanout(self):
        from atlas_virtual_evaluator import parse_program,operation_name
        a=analyze_operations(parse_ttir(source(['%v = arith.subf %a, %b : T','%r = arith.mulf %v, %v : T'])).funcs[0])
        for op in a.operations:
            for value in op.results:
                value.name='%same'
        text=emit_atlas_operations(a,numerical_policy=BF16_RTL_POLICY,abi=abi())
        p=parse_program(text)
        loads=[o for o in p.operations if operation_name(o)=='atlas.virtual_input_bf16']
        bins=[o for o in p.operations if operation_name(o)=='atlas.virtual_vpu_binary']
        self.assertIs(bins[0].operands[0],loads[0].results[1])
        self.assertIs(bins[0].operands[1],loads[1].results[1])
        self.assertIs(bins[1].operands[0],bins[0].results[0])
        self.assertIs(bins[1].operands[1],bins[0].results[0])

    def test_state_threading_and_bindings(self):
        from atlas_virtual_evaluator import parse_program,operation_name
        a=analysis()
        b=BF16TileABI((TileBinding(0,'input',5),TileBinding(1,'input',1),TileBinding(2,'output',3)))
        p=parse_program(emit_atlas_operations(a,numerical_policy=BF16_RTL_POLICY,abi=b))
        ops=[o for o in p.operations if operation_name(o) in ('atlas.virtual_start','atlas.virtual_input_bf16','atlas.virtual_output_bf16')]
        for prev,nxt in zip(ops,ops[1:]):
            self.assertIs(nxt.operands[0],prev.results[0])
        self.assertIn('index = 5 : i32',self.emit(abi=b))
        self.assertIn('index = 3 : i32',self.emit(abi=b))

    def test_constant_bits_and_relu(self):
        self.assertIn('bits = 48896 : i32',self.emit('constants'))
        self.assertIn('kind = "relu"',self.emit('mul_add_relu'))
        self.assertNotIn('constant-scalar',self.emit('mul_add_relu'))

    def test_policy_and_layout_rejection(self):
        for policy in (None,'ieee','atlas-fp8-mlp-v1'):
            with self.assertRaises(NotImplementedError):self.emit(numerical_policy=policy)
        with self.assertRaises(NotImplementedError):self.emit(abi=replace(contract(analysis()),disjoint=False))

    def test_physical_abi_rejection(self):
        for m in [AtlasOperationMemory(1,0x90020000),AtlasOperationMemory(0x90000001,0x90020000),
                  AtlasOperationMemory(0x90000000,0x90000800),AtlasOperationMemory(0xfffff800,0x90020000),
                  AtlasOperationMemory(0x90000000,0xfffff800)]:
            with self.subTest(memory=m),self.assertRaises(ValueError):self.emit(memory=m)

    def test_revalidate_complete_graph(self):
        a=analysis()
        with self.assertRaises(NotImplementedError):
            emit_atlas_operations(replace(a,operations=a.operations[:-1]),numerical_policy=BF16_RTL_POLICY,abi=contract(a))
        a.function.body.ops[2].name='unknown.side_effect'
        with self.assertRaises(NotImplementedError):
            emit_atlas_operations(a,numerical_policy=BF16_RTL_POLICY,abi=abi())

    def test_multiple_outputs_preserve_bindings(self):
        pointer='tensor<32x32x!tt.ptr<bf16>>'
        text=source().replace('%yp: '+pointer+')','%yp: '+pointer+', %zp: '+pointer+')')
        text=text.replace('    tt.return','    tt.store %zp, %a : '+pointer+'\n    tt.return')
        a=analyze_operations(parse_ttir(text).funcs[0])
        b=replace(abi(),bindings=abi().bindings+(TileBinding(3,'output',5),))
        out=emit_atlas_operations(a,numerical_policy=BF16_RTL_POLICY,abi=b)
        self.assertIn('%io3 = "atlas.virtual_output_bf16"(%io2,',out)
        self.assertIn('%io4 = "atlas.virtual_output_bf16"(%io3, %v0) {index = 5',out)
        self.assertIn('return %io4',out)

    def test_validator_rejection_creates_no_directory(self):
        from validate_atlas_operations import validate
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.ttir'
            p.write_text(source(['%r = math.exp %a : T']))
            out=Path(d)/'artifacts'
            with self.assertRaises(NotImplementedError):validate(p,out)
            self.assertFalse(out.exists())

    def cli(self,path,out,*extra):
        argv=['ttir_to_atlas',str(path),'--pattern','atlas-ops','--out',str(out),
              '--tile-binding','0:input:0','--tile-binding','1:input:1','--tile-binding','2:output:0']
        with patch('sys.argv',argv+list(extra)),redirect_stdout(StringIO()),redirect_stderr(StringIO()):main()

    def test_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out/program.mlir'
            self.cli(FIXTURES/'add.ttir',p,'--numerical-policy',BF16_RTL_POLICY)
            original=p.read_bytes()
            with self.assertRaises(SystemExit):self.cli(FIXTURES/'add.ttir',p,'--numerical-policy',BF16_RTL_POLICY)
            self.assertEqual(p.read_bytes(),original)
            api=convert_file(FIXTURES/'add.ttir',pattern='atlas-ops',numerical_policy=BF16_RTL_POLICY,operation_abi=abi())
            self.assertEqual(p.read_text(),api)

    def test_unsupported_cli_produces_no_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for i,options in enumerate([[],['--numerical-policy',BF16_RTL_POLICY,'--output-base','0x90000000']]):
                out=root/f'failed-{i}'/'program.mlir'
                with self.assertRaises(SystemExit):self.cli(FIXTURES/'add.ttir',out,*options)
                self.assertFalse(out.parent.exists())
            bad=root/'bad.ttir';bad.write_text((FIXTURES/'add.ttir').read_text().replace('arith.addf','arith.divf'))
            out=root/'unsupported'/'program.mlir'
            with self.assertRaises(SystemExit):self.cli(bad,out,'--numerical-policy',BF16_RTL_POLICY)
            self.assertFalse(out.parent.exists())

    def test_existing_two_layer_bytes(self):
        from compiler.atlas_mapping import MLP_NUMERICAL_POLICY
        original=ROOT/'out/atlas/two_layer_mlp_generated.virtual.mlir'
        if not original.exists():self.skipTest('saved two-layer artifact absent')
        actual=convert_file(ROOT/'examples/two_layer_mlp_fp8.ttir',pattern='two-layer-mlp',numerical_policy=MLP_NUMERICAL_POLICY)
        self.assertEqual(actual.encode(),original.read_bytes())

    def test_independent_reference_known_answers(self):
        self.assertEqual(binary('add',0x3f80,0x3bc0),0x3f80)
        self.assertEqual(binary('sub',0x3f80,0xbbc0),0x3f80)
        self.assertEqual(binary('mul',0x3fc0,0x3fc0),0x4010)
        self.assertEqual(binary('mul',0,0x7f80),0x7fc0)
        self.assertEqual(binary('add',0x7f80,0xff80),0x7fc0)
        self.assertEqual(binary('mul',0x8000,0x4000),0x8000)
        self.assertEqual(binary('mul',1,0x3f00),0)
        self.assertEqual(binary('mul',3,0x3f00),2)
        self.assertEqual(binary('mul',0x7f7f,0x4000),0x7f80)

    @unittest.skipUnless((default_atlas_root()/'build/bin/atlas-opt').exists(),'Atlas backend unavailable')
    def test_real_verifier_lowering_and_machine_all_compositions(self):
        from validate_atlas_operations import validate
        with tempfile.TemporaryDirectory() as d:
            for p in sorted(FIXTURES.glob('*.ttir')):
                with self.subTest(graph=p.stem):
                    with redirect_stdout(StringIO()):r=validate(p,Path(d)/p.stem)
                    self.assertEqual(len(r['cases']),8)
                    self.assertTrue(all(c['mismatches']==0 for c in r['cases']))


if __name__=='__main__':unittest.main()
