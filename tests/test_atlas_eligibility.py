"""Whole-kernel eligibility and numerical-policy evidence; no machine execution."""
from dataclasses import replace
from pathlib import Path
import unittest

from compiler.atlas_eligibility import (
    BF16_RTL_POLICY, BF16TileABI, TileBinding, analyze_atlas_eligibility,
)
from compiler.ttir_operation_analysis import analyze_operations
from compiler.ttir_reader import parse_ttir


def source(ops=None, elem='bf16', shape='32x32', masked=False):
    t = f'tensor<{shape}x{elem}>'
    p = f'tensor<{shape}x!tt.ptr<{elem}>>'
    ops = ops or ['%v = arith.addf %a, %b : T', '%r = arith.mulf %v, %a : T']
    lines = [f'%a = tt.load %ap : {p}', f'%b = tt.load %bp : {p}']
    if masked:
        lines += [f'%range = tt.make_range {{end = 32 : i32, start = 0 : i32}} : tensor<32xi32>',
                  '%m = arith.cmpi eq, %range, %range : tensor<32xi32>',
                  '%e = tt.expand_dims %m {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>',
                  '%mask = tt.broadcast %e : tensor<1x32xi1> -> tensor<32x32xi1>']
    lines += [o.replace('T', t) for o in ops]
    lines += [f'tt.store %yp, %r'+(', %mask' if masked else '')+f' : {p}', 'tt.return']
    return 'module {\n  tt.func public @compose(%ap: '+p+', %bp: '+p+', %yp: '+p+') {\n'+'\n'.join('    '+s for s in lines)+'\n  }\n}\n'


def analyze(text):
    return analyze_operations(parse_ttir(text).funcs[0])


def abi():
    return BF16TileABI((TileBinding(0,'input',0),TileBinding(1,'input',1),TileBinding(2,'output',0)))


def eligible(text=None, **kwargs):
    options = {'numerical_policy': BF16_RTL_POLICY, 'abi': abi()}
    options.update(kwargs)
    return analyze_atlas_eligibility(analyze(text or source()), **options)


class AtlasEligibilityTests(unittest.TestCase):
    def assert_rejected(self, result, code):
        self.assertFalse(result.eligible)
        self.assertIsNone(result.plan)
        self.assertIn(code, [d.code for d in result.diagnostics])

    def test_compositions_and_source_dependencies(self):
        for ops in [
            ['%v = arith.addf %a, %b : T', '%r = arith.subf %v, %a : T'],
            ['%v = arith.mulf %a, %b : T', '%r = arith.addf %v, %v : T'],
            ['%z = arith.constant dense<0.0> : T', '%v = arith.subf %a, %b : T', '%r = arith.maxnumf %v, %z : T'],
            ['%s = arith.constant 0.5 : bf16', '%z = tt.splat %s : bf16 -> T', '%v = arith.addf %a, %b : T', '%r = arith.mulf %v, %z : T'],
        ]:
            result = eligible(source(ops))
            self.assertTrue(result.eligible, result.diagnostics)
            self.assertEqual(len(result.plan.operations), len(result.plan.analysis.operations))
            for entry in result.plan.operations:
                self.assertTrue(any(entry.source is op for op in result.plan.analysis.operations))
            self.assertEqual(result.plan.numerical_policy, BF16_RTL_POLICY)

    def test_explicit_policy(self):
        for policy in [None, '', 'ieee', 'atlas-fp8-mlp-v1', 'future-policy']:
            self.assert_rejected(eligible(numerical_policy=policy), 'numerical-policy')

    def test_abi_required_and_contracts(self):
        for contract in [None, replace(abi(), disjoint=False), replace(abi(), layout='column-major'), replace(abi(), launch='multi-program'),
                         replace(abi(), boundary_layout='raw-row-major'), replace(abi(), alignment=16)]:
            self.assert_rejected(eligible(abi=contract), 'memory-abi')

    def test_invalid_bindings(self):
        for bindings in [None, ('invalid-entry',), abi().bindings[:-1], (TileBinding(0,'input',0),)*3,
                         (TileBinding(0,'input',0),TileBinding(1,'input',0),TileBinding(2,'output',0)),
                         (TileBinding(0,'input',64),*abi().bindings[1:]),
                         (TileBinding(True,'input',0),*abi().bindings[1:]),
                         (TileBinding(3,'input',0),*abi().bindings[1:])]:
            self.assert_rejected(eligible(abi=BF16TileABI(bindings)), 'memory-abi')

    def test_memory_roles(self):
        wrong = BF16TileABI((TileBinding(0,'output',1),TileBinding(1,'input',0),TileBinding(2,'output',0)))
        self.assert_rejected(eligible(abi=wrong), 'memory-role')

    def test_masks_not_assumed_safe(self):
        self.assert_rejected(eligible(source(masked=True)), 'masked-memory')

    def test_float_types_never_demoted(self):
        for elem in ['f16','f32']:
            r = eligible(source(elem=elem))
            self.assert_rejected(r, 'arithmetic-type')
            self.assertIn('boundary-type', [d.code for d in r.diagnostics])

    def test_wrong_tile_shape(self):
        self.assert_rejected(eligible(source(shape='16x32')), 'arithmetic-type')

    def test_non_relu_max_and_negative_zero(self):
        for literal in ['1.0','-0.0']:
            ops = [f'%z = arith.constant dense<{literal}> : T','%v = arith.addf %a, %b : T','%r = arith.maxnumf %v, %z : T']
            self.assert_rejected(eligible(source(ops)), 'relu-semantics')

    def test_relu_scalar_splat_and_operand_order(self):
        ops = ['%s = arith.constant 0.0 : bf16', '%z = tt.splat %s : bf16 -> T','%v = arith.addf %a, %b : T','%r = arith.maxnumf %z, %v : T']
        r = eligible(source(ops))
        self.assertTrue(r.eligible, r.diagnostics)
        relu = next(e for e in r.plan.operations if e.candidate.endswith(':relu'))
        self.assertIs(relu.inputs[0], relu.source.operands[1])
        self.assertEqual(len(relu.source.operands), 2)

    def test_constants_exact_rounding_and_signed_zero(self):
        for literal, bits in [('0.5',0x3f00),('-0.0',0x8000),('1.00390625',0x3f80),('1.01171875',0x3f82),('0.1',0x3dcd)]:
            ops = [f'%z = arith.constant dense<{literal}> : T','%v = arith.addf %a, %b : T','%r = arith.mulf %v, %z : T']
            r = eligible(source(ops))
            self.assertTrue(r.eligible,r.diagnostics)
            self.assertEqual(next(e.constant_bits for e in r.plan.operations if e.candidate=='atlas.virtual_vpu_splat'),bits)

    def test_nonfinite_and_hex_constants_not_qualified(self):
        ops = ['%z = arith.constant dense<0x7FC0> : T','%v = arith.addf %a, %b : T','%r = arith.mulf %v, %z : T']
        self.assert_rejected(eligible(source(ops)), 'constant-value')

    def test_conversion_rejected(self):
        ops = ['%f = arith.extf %a : T to tensor<32x32xf32>', '%v = arith.addf %a, %b : T','%r = arith.truncf %f : tensor<32x32xf32> to T']
        self.assert_rejected(eligible(source(ops)), 'unsupported-conversion')

    def test_even_identity_layout_changes_rejected(self):
        ops = ['%v = tt.reshape %a : T -> T','%r = arith.addf %v, %b : T']
        self.assert_rejected(eligible(source(ops)), 'unsupported-operation')

    def test_unknown_side_effect_rejected_on_revalidation(self):
        a = analyze(source())
        a.function.body.ops[-2].name = 'unknown.side_effect'
        self.assert_rejected(analyze_atlas_eligibility(a,numerical_policy=BF16_RTL_POLICY,abi=abi()), 'invalid-analysis')

    def test_partial_or_stale_analysis_rejected(self):
        a = analyze(source())
        for bad in [replace(a, operations=a.operations[:-1]),replace(a, dependencies=()),replace(a,memory=())]:
            self.assert_rejected(analyze_atlas_eligibility(bad,numerical_policy=BF16_RTL_POLICY,abi=abi()), 'invalid-analysis')
        a.function.body.ops[2].operands[0] = a.function.body.ops[3].results[0]
        self.assert_rejected(analyze_atlas_eligibility(a,numerical_policy=BF16_RTL_POLICY,abi=abi()), 'invalid-analysis')

    def test_duplicate_output_store_rejected(self):
        s = source().replace('    tt.return','    tt.store %yp, %r : tensor<32x32x!tt.ptr<bf16>>\n    tt.return')
        self.assert_rejected(eligible(s), 'memory-access')

    def test_multiple_outputs_and_constant_only_output(self):
        p='tensor<32x32x!tt.ptr<bf16>>'
        s=source().replace('%yp: '+p+')', '%yp: '+p+', %zp: '+p+')')
        s=s.replace('    tt.return', '    tt.store %zp, %a : '+p+'\n    tt.return')
        contract=replace(abi(),bindings=abi().bindings+(TileBinding(3,'output',1),))
        self.assertTrue(eligible(s,abi=contract).eligible)
        s='module {\n tt.func public @constant_output(%yp: '+p+') {\n %r = arith.constant dense<0.5> : tensor<32x32xbf16>\n tt.store %yp, %r : '+p+'\n tt.return\n }\n}\n'
        self.assertTrue(eligible(s,abi=BF16TileABI((TileBinding(0,'output',0),))).eligible)

    def test_missing_output_store(self):
        s=source().replace('    tt.store %yp, %r : tensor<32x32x!tt.ptr<bf16>>\n','')
        self.assert_rejected(eligible(s),'output-binding')

    def test_computed_pointer_is_not_bound_memory(self):
        s=source().replace('    %a = tt.load %ap', '    %zero = arith.constant dense<0> : tensor<32x32xi32>\n    %ptr = tt.addptr %ap, %zero : tensor<32x32x!tt.ptr<bf16>>, tensor<32x32xi32>\n    %a = tt.load %ptr')
        r=eligible(s)
        self.assert_rejected(r,'address-computation')
        self.assertIn('memory-access',[d.code for d in r.diagnostics])

    def test_index_computation_is_not_vpu_arithmetic(self):
        s = source().replace('    tt.return','    %pid = tt.get_program_id x : i32\n    tt.return')
        self.assert_rejected(eligible(s), 'address-computation')

    def test_full_graph_rejected_if_one_operation_unsupported(self):
        ops = ['%v = arith.addf %a, %b : T','%r = math.exp %v : T']
        r = eligible(source(ops))
        self.assert_rejected(r,'unsupported-operation')
        d = next(d for d in r.diagnostics if d.code=='unsupported-operation')
        self.assertEqual(d.operation_name,'math.exp')
        self.assertIsInstance(d.operation_index,int)

    def test_prior_composition_fixtures_not_misclassified(self):
        root = Path(__file__).parent/'fixtures/operation_compositions'
        for p in root.glob('*.ttir'):
            a=analyze(p.read_text())
            contract=BF16TileABI(tuple(TileBinding(i,'output' if i==len(a.function.args)-1 else 'input',i) for i in range(len(a.function.args))))
            r=analyze_atlas_eligibility(a,numerical_policy=BF16_RTL_POLICY,abi=contract)
            self.assertFalse(r.eligible,(p.name,r.diagnostics))
            self.assertIsNone(r.plan)


class AtlasNumericalPolicyEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from npu_model.configs.numerics import RtlNumerics
        except ImportError:
            raise unittest.SkipTest('existing npu-model-eval and torch needed for numerical evidence')
        cls.torch, cls.rtl = torch,RtlNumerics

    def test_relu_all_bf16_encodings_match_policy(self):
        t=self.torch
        raw=t.arange(65536,dtype=t.int32)
        result=self.rtl.unary('relu',raw.to(t.int16).view(t.bfloat16)).view(t.int16).to(t.int32)&65535
        self.assertTrue(t.equal(result,t.where(raw>=32768,0,raw)))

    def test_truncation_differs_from_bf16_rne(self):
        t=self.torch
        a=t.tensor([1.0],dtype=t.bfloat16); b=t.tensor([1.5*2**-8],dtype=t.bfloat16)
        self.assertEqual(self.rtl.add(a,b).item(),1.0)
        self.assertEqual((a+b).item(),1.0078125)
        self.assertEqual(self.rtl.sub(a,-b).item(),1.0)

    def test_exceptional_values_and_multiply(self):
        t=self.torch
        a=t.tensor([float('inf'),float('nan'),0.,-0.,1.5],dtype=t.bfloat16)
        b=t.tensor([-float('inf'),1.,float('inf'),2.,1.5],dtype=t.bfloat16)
        for fn in [self.rtl.add,self.rtl.sub,self.rtl.mul]:
            out=fn(a,b)
            self.assertTrue(out.view(t.int16)[t.isnan(out)].eq(0x7fc0).all())
        self.assertEqual(self.rtl.mul(a,b)[-1].item(),2.25)
        self.assertEqual(self.rtl.mul(a,b).view(t.int16)[3].item(),-32768)


if __name__=='__main__':
    unittest.main()
