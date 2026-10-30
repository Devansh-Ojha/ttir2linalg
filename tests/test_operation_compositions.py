"""Handwritten operation compositions; source semantic checks, no Atlas emission."""
from dataclasses import replace
from pathlib import Path
import unittest

from compiler.ttir_reader import parse_ttir
from compiler.ttir_operation_analysis import analyze_operations, UnsupportedKernel

ROOT = Path(__file__).parent / 'fixtures/operation_compositions'


def text(name):
    return (ROOT / (name + '.ttir')).read_text()


def function(name):
    return parse_ttir(text(name)).funcs[0]


class CompositionTests(unittest.TestCase):
    def test_existing_mlp_sources_need_no_pattern_dispatch(self):
        for name, depth in [('two_layer_mlp_fp8.ttir', 2), ('three_layer_mlp_fp8.ttir', 3)]:
            path = Path(__file__).parents[1] / 'examples' / name
            a = analyze_operations(parse_ttir(path.read_text()).funcs[0])
            self.assertEqual(sum(o.name == 'tt.dot' for o in a.operations), depth)
            self.assertEqual(len(a.relu_candidates), depth - 1)

    def test_all_compositions(self):
        for path in sorted(ROOT.glob('*.ttir')):
            with self.subTest(name=path.stem):
                f = function(path.stem)
                a = analyze_operations(f)
                self.assertIs(a.function, f)
                self.assertTrue(a.memory)
                self.assertEqual(a.dependencies, tuple(tuple(o.operands) for o in a.operations))
                self.assertEqual(len(a.operations), len({id(o) for o in a.operations}))

    def test_shared_inputs_and_nonzero_accumulator(self):
        a = analyze_operations(function('matmul_add_matmul'))
        dots = [o for o in a.operations if o.name == 'tt.dot']
        self.assertEqual(len(dots), 2)
        self.assertIs(dots[1].operands[1], dots[0].operands[1])
        self.assertIs(dots[1].operands[2], dots[0].results[0])
        self.assertEqual(dots[1].attrs['inputPrecision'], 'ieee')
        packing = dots[1].operands[0].defining_op
        self.assertEqual(packing.attrs['rounding'], 'rtne')
        self.assertEqual(packing.results[0].type.elem, 'f8E4M3FN')

    def test_maxnum_zero_is_preserved(self):
        for name in ('matmul_relu', 'matmul_add_relu', 'matmul_scale_relu', 'elementwise_add_relu'):
            a = analyze_operations(function(name))
            self.assertEqual(len(a.relu_candidates), 1)
            self.assertEqual(a.relu_candidates[0].name, 'arith.maxnumf')
            self.assertEqual(a.relu_candidates[0].operands[1].defining_op.attrs['value'], 'dense<0.000000e+00>')

    def test_sum_and_max_regions(self):
        for name, kind in [('reduce_scale', 'arith.addf'), ('matmul_reduce_exp', 'arith.maxnumf')]:
            a = analyze_operations(function(name))
            self.assertEqual(len(a.reductions), 1)
            r = a.reductions[0]
            self.assertEqual(r.axis, 1)
            self.assertEqual(r.combiner.name, kind)
            self.assertEqual(tuple(r.combiner.operands), tuple(r.operation.regions[0][0].args))
            self.assertIn(r.combiner, a.operations)
            self.assertEqual(r.operation.results[0].type.shape, (32,))

    def test_axis_zero_and_scalar_reduction(self):
        a = analyze_operations(parse_ttir(text('reduce_scale').replace('axis = 1', 'axis = 0')).funcs[0])
        self.assertEqual(a.reductions[0].axis, 0)
        source = text('reduce_scale').replace('tensor<32x32x', 'tensor<32x').replace('axis = 1', 'axis = 0')
        # Change vector results/output pointers to scalar, retaining vector input.
        source = source.replace('%yp: tensor<32x!tt.ptr<bf16>>', '%yp: !tt.ptr<bf16>')
        source = source.replace('-> tensor<32xbf16>', '-> bf16').replace(' : tensor<32xbf16>', ' : bf16')
        source = source.replace('arith.constant dense<0.500000e+00>', 'arith.constant 0.500000e+00')
        source = source.replace('tt.store %yp, %r : tensor<32x!tt.ptr<bf16>>', 'tt.store %yp, %r : !tt.ptr<bf16>')
        a = analyze_operations(parse_ttir(source).funcs[0])
        self.assertEqual(a.reductions[0].operation.results[0].type.kind, 'scalar')

    def test_rounding_variants_remain_distinct(self):
        for rounding in ('rtne', 'rtz'):
            a = analyze_operations(parse_ttir(text('matmul_add_matmul').replace('rounding = rtne', 'rounding = '+rounding)).funcs[0])
            self.assertEqual(next(o for o in a.operations if o.name == 'tt.fp_to_fp').attrs['rounding'], rounding)

    def test_fp8_widening(self):
        source = text('matmul_relu').replace('%h = arith.truncf %d : tensor<32x32xf32> to tensor<32x32xbf16>', '%h = tt.fp_to_fp %x : tensor<32x32xf8E4M3FN> -> tensor<32x32xbf16>')
        a = analyze_operations(parse_ttir(source).funcs[0])
        self.assertEqual(next(o for o in a.operations if o.name == 'tt.fp_to_fp').attrs, {})

    def test_unknown_flags_and_precision(self):
        for name, old, new in [
            ('matmul_relu', ', inputPrecision = ieee', ''),
            ('matmul_relu', 'arith.maxnumf %h, %zb', 'arith.maxnumf %h, %zb fastmath<nnan>'),
            ('matmul_add_relu', 'arith.addf %h, %b', 'arith.addf %h, %b fastmath<reassoc>'),
        ]:
            with self.subTest(new=new), self.assertRaises(UnsupportedKernel):
                analyze_operations(parse_ttir(text(name).replace(old, new)).funcs[0])

    def test_dot_precision_bound_preserved(self):
        source = text('matmul_relu').replace('inputPrecision = ieee', 'inputPrecision = tf32 {maxNumImpreciseAcc = 0 : i32}')
        a = analyze_operations(parse_ttir(source).funcs[0])
        dot = next(o for o in a.operations if o.name == 'tt.dot')
        self.assertEqual(dot.attrs, {'inputPrecision': 'tf32', 'maxNumImpreciseAcc': '0 : i32'})

    def test_unsupported_combiner_and_region_attributes(self):
        for source in [text('reduce_scale').replace('arith.addf %lhs, %rhs', 'arith.mulf %lhs, %rhs'),
                       text('reduce_scale').replace('}) :', '}) {extra = true} :')]:
            with self.assertRaises(UnsupportedKernel):
                analyze_operations(parse_ttir(source).funcs[0])

    def test_invalid_attributes_and_shapes(self):
        cases = [('matmul_relu', 'tt.dot', lambda o: o.attrs.update(inputPrecision='unknown')),
                 ('matmul_add_matmul', 'tt.fp_to_fp', lambda o: o.attrs.clear()),
                 ('reduce_scale', 'tt.reduce', lambda o: o.attrs.update(axis='2 : i32')),
                 ('shape_and_conversion', 'tt.expand_dims', lambda o: o.attrs.clear()),
                 ('matmul_relu', 'tt.dot', lambda o: setattr(o.results[0], 'type', replace(o.results[0].type, shape=(16,32)))),
                 ('shape_and_conversion', 'tt.broadcast', lambda o: setattr(o.results[0], 'type', replace(o.results[0].type, shape=(16,16))))]
        for name, kind, mutate in cases:
            f = function(name)
            mutate(next(o for o in f.body.ops if o.name == kind))
            with self.subTest(kind=kind), self.assertRaises(UnsupportedKernel):
                analyze_operations(f)

    def test_reduction_capture_and_escape_rejected(self):
        f = function('reduce_scale')
        r = next(o for o in f.body.ops if o.name == 'tt.reduce')
        r.regions[0][0].ops[0].operands[0] = r.operands[0]
        with self.assertRaises(UnsupportedKernel):
            analyze_operations(f)
        f = function('reduce_scale')
        r = next(o for o in f.body.ops if o.name == 'tt.reduce')
        next(o for o in f.body.ops if o.name == 'arith.mulf').operands[0] = r.regions[0][0].ops[0].results[0]
        with self.assertRaises(UnsupportedKernel):
            analyze_operations(f)

    def test_shapes_and_exp_dependencies(self):
        a = analyze_operations(function('shape_and_conversion'))
        self.assertEqual([o.name for o in a.operations].count('tt.reshape'), 2)
        a = analyze_operations(function('matmul_reduce_exp'))
        exp = next(o for o in a.operations if o.name == 'math.exp')
        self.assertIs(exp.operands[0], a.reductions[0].operation.results[0])


if __name__ == '__main__':
    unittest.main()
