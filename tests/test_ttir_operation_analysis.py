"""Real GPU fixtures are byte-exact; mutations below are rejection tests only."""
from pathlib import Path
import hashlib
import json
import unittest

from compiler.ttir_reader import parse_ttir
from compiler.ttir_operation_analysis import analyze_operations, UnsupportedKernel

ROOT = Path(__file__).parent / 'fixtures/graham_flash_attention'
MANIFEST = json.loads((ROOT / 'source.json').read_text())
LAYOUT = sorted((ROOT / 'attention_flash/inductor_ttir').glob('*.ttir'))
CUSTOM = ROOT / 'attention_custom_flash/inductor_ttir/kernel_000_flash_attention_kernel.ttir'


def parsed(text):
    return parse_ttir(text).funcs[0]


class OperationAnalysisTests(unittest.TestCase):
    def test_fixture_provenance(self):
        self.assertEqual(MANIFEST['commit'], '4ac9ccb512057a4472c7bd0e4b9864cbab99f657')
        for f in MANIFEST['files']:
            raw = (ROOT / f['fixture']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), f['sha256'])
            self.assertEqual(hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest(), f['blob'])

    def test_all_real_layout_kernels(self):
        self.assertEqual(len(LAYOUT), 6)
        for path in LAYOUT:
            with self.subTest(path=path.name):
                f = parsed(path.read_text())
                a = analyze_operations(f)
                self.assertIs(a.function, f)
                self.assertEqual(a.operations, tuple(f.body.ops))
                self.assertEqual(len(a.memory), 2)
                load, store = a.memory
                self.assertIs(load.mask, store.mask)
                self.assertIsNone(load.other)  # unspecified, not a zero fill
                self.assertIs(store.value, load.value)
                self.assertEqual(load.value.type.elem, 'f16')
                self.assertIn(load.value.type.shape, [(128,), (256,)])
                comparison = load.mask.defining_op
                self.assertEqual(comparison.attrs['predicate'], 'slt')
                for op, deps in zip(a.operations, a.dependencies):
                    self.assertEqual(deps, tuple(op.operands))

    def test_real_custom_retains_semantics_but_rejects(self):
        text = CUSTOM.read_text()
        m = parse_ttir(text)
        f = m.funcs[0]
        self.assertEqual(f.source, text)
        dots = [o for o in f.body.ops if o.name == 'tt.dot']
        self.assertEqual(len(dots), 2)
        self.assertTrue(all(o.attrs['inputPrecision'] == 'tf32' for o in dots))
        self.assertEqual(dots[0].results[0].type.shape, (16, 32))
        self.assertEqual(dots[0].results[0].type.elem, 'f32')
        self.assertTrue(all(v.type.elem == 'f16' for v in dots[0].operands[:2]))
        loads = [o for o in f.body.ops if o.name == 'tt.load']
        self.assertEqual(len(loads), 3)
        self.assertTrue(all(len(o.operands) == 3 and o.operands[1].type.elem == 'i1'
                            and o.operands[2].type.elem == 'f16' for o in loads))
        casts = [o for o in f.body.ops if o.name == 'arith.truncf']
        self.assertEqual(len(casts), 2)
        self.assertTrue(all(o.operands[0].type.elem == 'f32' and
                            o.results[0].type.elem == 'f16' for o in casts))
        self.assertIs(dots[1].operands[2].defining_op, next(o for o in f.body.ops if o.results and o.results[0].name == '%running_acc_56'))
        reductions = [o for o in f.body.ops if o.name == 'tt.reduce']
        self.assertEqual([o.attr_int('axis') for o in reductions], [1, 1])
        self.assertEqual([o.regions[0][0].ops[0].name for o in reductions], ['arith.maxnumf', 'arith.addf'])
        for op in reductions:
            self.assertEqual(len(op.regions[0][0].args), 2)
            self.assertIs(op.regions[0][0].ops[-1].operands[0].defining_op, op.regions[0][0].ops[0])
        with self.assertRaises(UnsupportedKernel):
            analyze_operations(f)

    def test_unknown_operation_rejects_whole_kernel(self):
        text = LAYOUT[0].read_text().replace('arith.muli', 'arith.subi', 1)
        with self.assertRaisesRegex(UnsupportedKernel, 'arith.subi'):
            analyze_operations(parsed(text))

    def test_unknown_memory_attribute_rejected(self):
        text = LAYOUT[0].read_text()
        text = '\n'.join(line.replace(' : ', ' {isVolatile = true} : ', 1) if ' = tt.load ' in line else line for line in text.splitlines())
        with self.assertRaises(UnsupportedKernel):
            analyze_operations(parsed(text))

    def test_unknown_function_attribute_rejected(self):
        text = LAYOUT[0].read_text().replace('noinline = false', 'noinline = false, extra = 1 : i32')
        with self.assertRaises(UnsupportedKernel):
            analyze_operations(parsed(text))

    def test_flags_cannot_be_ignored(self):
        f = parsed(LAYOUT[0].read_text())
        op = next(o for o in f.body.ops if o.name == 'arith.muli')
        op.assembly = op.assembly.replace(' : ', ' overflow<nsw> : ', 1)
        with self.assertRaises(UnsupportedKernel):
            analyze_operations(f)

    def test_bad_ssa_rejected(self):
        f = parsed(LAYOUT[0].read_text())
        op = next(o for o in f.body.ops if o.name == 'arith.muli')
        op.operands[0] = op.results[0]
        with self.assertRaisesRegex(UnsupportedKernel, 'SSA dependency'):
            analyze_operations(f)

    def test_bad_mask_rejected(self):
        f = parsed(LAYOUT[0].read_text())
        op = next(o for o in f.body.ops if o.name == 'tt.load')
        op.operands[1] = next(o.results[0] for o in f.body.ops if o.name == 'tt.make_range')
        with self.assertRaisesRegex(UnsupportedKernel, 'mask'):
            analyze_operations(f)

    def test_extra_attribute_and_regions_rejected(self):
        for region in [False, True]:
            f = parsed(LAYOUT[0].read_text())
            op = next(o for o in f.body.ops if o.name == 'tt.load')
            if region:
                op.regions = [[]]
            else:
                op.attrs['new_semantics'] = 'true'
            with self.assertRaises(UnsupportedKernel):
                analyze_operations(f)

    def test_encoded_layout_rejected(self):
        f = parsed(LAYOUT[0].read_text())
        op = next(o for o in f.body.ops if o.name == 'tt.load')
        op.assembly = op.assembly.replace('!tt.ptr<f16>>', '!tt.ptr<f16>, #blocked>')
        with self.assertRaises(UnsupportedKernel):
            analyze_operations(f)


if __name__ == '__main__':
    unittest.main()
