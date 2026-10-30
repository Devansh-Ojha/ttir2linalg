"""SSA chain analysis and real TTIR parsing; no generated output artifacts."""
from dataclasses import replace
from pathlib import Path
import unittest

from compiler.triton_analysis import analyze_mlp_chain, analyze_two_layer_mlp
from compiler.ttir_model import TBlock, TFunc, TOp, TType, TValue
from compiler.ttir_reader import _scan_text, parse_ttir


def chain_function(depth):
    """Build source dataflow directly, without the target mapper or Triton."""
    counter = iter(range(10000))

    def value(element, pointer=False):
        identity = next(counter)
        return TValue(identity, TType("tensor", element, (32, 32), pointer),
                      f"%v{identity}")

    args = [value("f8E4M3FN", True) for _ in range(depth + 1)]
    args.append(value("bf16", True))
    ops = []
    loaded = []
    for pointer in args[:-1]:
        result = value("f8E4M3FN")
        loaded.append(result)
        ops.append(TOp("tt.load", [pointer], [result]))
    zero, zero_bf = value("f32"), value("bf16")
    for result in (zero, zero_bf):
        ops.append(TOp("arith.constant", [], [result], {"value": "dense<0.0>"}))
    current = loaded[0]
    for index, weight in enumerate(loaded[1:]):
        dot, cast = value("f32"), value("bf16")
        ops.append(TOp("tt.dot", [current, weight, zero], [dot],
                       {"inputPrecision": "ieee"}))
        ops.append(TOp("arith.truncf", [dot], [cast]))
        current = cast
        if index < depth - 1:
            relu, pack = value("bf16"), value("f8E4M3FN")
            ops.append(TOp("arith.maxnumf", [cast, zero_bf], [relu]))
            ops.append(TOp("tt.fp_to_fp", [relu], [pack], {"rounding": "rtne"}))
            current = pack
    ops.extend([TOp("tt.store", [args[-1], current]), TOp("tt.return")])
    return TFunc("chain", args, TBlock(0, args=args, ops=ops))


def chain_source(depth):
    """Construct an in-memory TTIR input, independently of the Python builder."""
    fp8, bf16, f32 = ("tensor<32x32x" + kind + ">"
                      for kind in ("f8E4M3FN", "bf16", "f32"))
    pointer = "tensor<32x32x!tt.ptr<f8E4M3FN>>"
    args = [f"%p{i}: {pointer}" for i in range(depth + 1)]
    args.append("%out: tensor<32x32x!tt.ptr<bf16>>")
    lines = ["module {", "tt.func public @chain(" + ", ".join(args) + ") {"]
    lines.extend(f"%in{i} = tt.load %p{i} : {pointer}" for i in range(depth + 1))
    lines.extend([f"%z = arith.constant dense<0.0> : {f32}",
                  f"%zb = arith.constant dense<0.0> : {bf16}"])
    current = "%in0"
    for i in range(depth):
        lines.extend([
            f"%d{i} = tt.dot {current}, %in{i+1}, %z, inputPrecision = ieee : "
            f"{fp8} * {fp8} -> {f32}",
            f"%b{i} = arith.truncf %d{i} : {f32} to {bf16}",
        ])
        current = f"%b{i}"
        if i < depth - 1:
            lines.extend([
                f"%r{i} = arith.maxnumf {current}, %zb : {bf16}",
                f"%h{i} = tt.fp_to_fp %r{i}, rounding = rtne : {bf16} -> {fp8}",
            ])
            current = f"%h{i}"
    lines.extend([f"tt.store %out, {current} : tensor<32x32x!tt.ptr<bf16>>",
                  "tt.return", "}", "}"])
    return "\n".join(lines)


class MlpChainAnalysisTest(unittest.TestCase):
    def test_ordered_layers_and_source_identity(self):
        for depth in (2, 3, 4, 12):
            with self.subTest(depth=depth):
                function = chain_function(depth)
                plan = analyze_mlp_chain(function)
                dots = [op for op in function.body.ops if op.name == "tt.dot"]
                self.assertEqual(len(plan.layers), depth)
                self.assertEqual([id(layer.dot) for layer in plan.layers],
                                 [id(dot) for dot in dots])
                self.assertIs(plan.input_activation, dots[0].operands[0])
                self.assertEqual([v.id for v in plan.weights],
                                 [dot.operands[1].id for dot in dots])
                for previous, following in zip(plan.layers, plan.layers[1:]):
                    self.assertIs(following.dot.operands[0], previous.pack.results[0])
                    self.assertIs(previous.pack.operands[0], previous.relu.results[0])
                self.assertIsNone(plan.layers[-1].relu)
                self.assertIsNone(plan.layers[-1].pack)
                self.assertIs(plan.store.operands[1], plan.layers[-1].bf16_cast.results[0])

    def test_real_parser_two_three_four_layers(self):
        for depth in (2, 3, 4):
            with self.subTest(depth=depth):
                module = parse_ttir(chain_source(depth))
                plan = analyze_mlp_chain(module.funcs[0])
                self.assertEqual(len(plan.layers), depth)
                self.assertTrue(all(layer.dot.attrs["inputPrecision"] == "ieee"
                                    for layer in plan.layers))
                self.assertEqual(plan.store.operands[1].type.elem, "bf16")

    def test_two_layer_compatibility(self):
        module = parse_ttir((Path(__file__).parents[1] /
                             "examples/two_layer_mlp_fp8.ttir").read_text())
        chain = analyze_mlp_chain(module.funcs[0])
        old = analyze_two_layer_mlp(module.funcs[0])
        self.assertEqual(old.inputs, (chain.input_activation,) + chain.weights)
        self.assertIs(old.dots[0], chain.layers[0].dot)
        self.assertIs(old.bf16_casts[1], chain.layers[1].bf16_cast)
        self.assertIs(old.relu, chain.layers[0].relu)
        self.assertIs(old.pack, chain.layers[0].pack)
        self.assertIs(old.store, chain.store)
        for depth in (3, 4):
            with self.assertRaisesRegex(NotImplementedError, "exactly two"):
                analyze_two_layer_mlp(chain_function(depth))

    def test_independent_load_order_and_reversed_relu_operands(self):
        function = chain_function(4)
        loads = function.body.ops[:5]
        function.body.ops[:5] = reversed(loads)
        for op in function.body.ops:
            if op.name == "arith.maxnumf":
                op.operands.reverse()
        self.assertEqual(len(analyze_mlp_chain(function).layers), 4)

    def test_rejects_wrong_precision_at_every_layer(self):
        for index in range(4):
            for attrs in ({}, {"inputPrecision": "tf32"},
                          {"inputPrecision": "ieee", "maxNumImpreciseAcc": "1 : i32"},
                          {"inputPrecision": "ieee", "unknown": "true"}):
                function = chain_function(4)
                dots = [op for op in function.body.ops if op.name == "tt.dot"]
                dots[index].attrs = attrs
                with self.subTest(layer=index, attrs=attrs), self.assertRaisesRegex(
                        NotImplementedError, "precision"):
                    analyze_mlp_chain(function)
        function = chain_function(3)
        for op in function.body.ops:
            if op.name == "tt.dot":
                op.attrs["maxNumImpreciseAcc"] = "0 : i32"
        self.assertEqual(len(analyze_mlp_chain(function).layers), 3)

    def test_rejects_invalid_graphs(self):
        for case in ("one_layer", "undefined", "cycle", "duplicate_definition",
                     "duplicate_argument", "argument_redefinition", "inconsistent_type",
                     "broken_chain", "nonzero", "nonzero_relu", "wrong_shape", "wrong_encoding",
                     "mask", "load_attribute", "dot_arity", "cast", "cast_attribute",
                     "rounding", "missing_rounding", "relu_kind", "region",
                     "extra", "extra_store", "output", "argument_order", "return"):
            function = chain_function(1 if case == "one_layer" else 4)
            ops = function.body.ops
            dots = [op for op in ops if op.name == "tt.dot"]
            packs = [op for op in ops if op.name == "tt.fp_to_fp"]
            casts = [op for op in ops if op.name == "arith.truncf"]
            relus = [op for op in ops if op.name == "arith.maxnumf"]
            if case == "undefined":
                dots[0].operands[0] = replace(dots[0].operands[0], id=99999)
            elif case == "cycle":
                dots[0].operands[0] = packs[-1].results[0]
            elif case == "duplicate_definition":
                ops.insert(-1, TOp("arith.constant", [], [dots[0].results[0]]))
            elif case == "duplicate_argument":
                function.args.append(function.args[0])
            elif case == "argument_redefinition":
                ops.insert(0, TOp("tt.load", [], [function.args[0]]))
            elif case == "inconsistent_type":
                dots[0].operands[0] = replace(
                    dots[0].operands[0], type=replace(dots[0].operands[0].type, elem="bf16"))
            elif case == "broken_chain":
                dots[2].operands[0] = dots[0].operands[0]
            elif case == "nonzero":
                next(op for op in ops if op.name == "arith.constant").attrs["value"] = "dense<1.0>"
            elif case == "nonzero_relu":
                [op for op in ops if op.name == "arith.constant"][1].attrs["value"] = "dense<1.0>"
            elif case in ("wrong_shape", "wrong_encoding"):
                v = dots[2].operands[1]
                v.type = replace(v.type, **({"shape": (16, 32)} if case == "wrong_shape"
                                           else {"elem": "f8E4M3FNUZ"}))
            elif case == "mask":
                ops[0].operands.append(function.args[-1])
            elif case == "load_attribute":
                ops[0].attrs["isVolatile"] = "true"
            elif case == "dot_arity":
                dots[2].operands.pop()
            elif case == "cast":
                casts[1].name = "arith.extf"
            elif case == "cast_attribute":
                casts[1].attrs["fastmath"] = "true"
            elif case in ("rounding", "missing_rounding"):
                packs[1].attrs = {} if case == "missing_rounding" else {"rounding": "rtz"}
            elif case == "relu_kind":
                relus[1].name = "arith.maximumf"
            elif case == "region":
                dots[1].regions = [[TBlock(1)]]
            elif case == "extra":
                ops.insert(-1, TOp("arith.addf"))
            elif case == "extra_store":
                ops.insert(-1, TOp("tt.store", list(ops[-2].operands)))
            elif case == "output":
                ops[-2].operands[1] = relus[-1].results[0]
            elif case == "argument_order":
                function.args[1], function.args[2] = function.args[2], function.args[1]
            elif case == "return":
                ops[-1].operands = [function.args[0]]
            with self.subTest(case=case), self.assertRaises(NotImplementedError):
                analyze_mlp_chain(function)

    def test_source_dot_precision_extraction_and_rejection(self):
        source = chain_source(3)
        for setting in ("ieee", "tf32", "tf32x3"):
            text = source.replace("inputPrecision = ieee", f"inputPrecision = {setting}")
            attrs = [a for name, _, a in _scan_text(text) if name == "tt.dot"]
            self.assertEqual(attrs, [{"inputPrecision": setting}] * 3)
        for replacement in ("", ", inputPrecision = tf32"):
            module = parse_ttir(source.replace(", inputPrecision = ieee", replacement))
            with self.subTest(replacement=replacement), self.assertRaisesRegex(
                    NotImplementedError, "precision"):
                analyze_mlp_chain(module.funcs[0])
        attrs = [a for name, _, a in _scan_text(source.replace(
            ", inputPrecision = ieee", ", inputPrecision = ieee, maxNumImpreciseAcc = 1"))
                 if name == "tt.dot"]
        self.assertEqual(attrs, [{"inputPrecision": "ieee", "maxNumImpreciseAcc": "1"}] * 3)
        for replacement in (", inputPrecision = ieee, inputPrecision = ieee",
                            ", inputPrecision = ieee,\nmaxNumImpreciseAcc = 0"):
            with self.assertRaises(RuntimeError):
                _scan_text(source.replace(", inputPrecision = ieee", replacement))
        line = "%d = tt.dot %a, %b, %z : tensor<32x32xf32> // , inputPrecision = ieee"
        self.assertEqual(_scan_text(line)[0][2], {})

    def test_arith_custom_flags_are_not_lost(self):
        source = chain_source(3)
        cases = (
            source.replace("%zb :", "%zb fastmath<nnan> :"),
            source.replace(" : tensor<32x32xf32> to", " roundingmode<rtz> : tensor<32x32xf32> to"),
        )
        for text in cases:
            scanned = _scan_text(text)
            self.assertTrue(any("fastmath" in attrs or "roundingmode" in attrs
                                for _, _, attrs in scanned))
        function = chain_function(3)
        relu = next(op for op in function.body.ops if op.name == "arith.maxnumf")
        relu.attrs["fastmath"] = "nnan"
        with self.assertRaisesRegex(NotImplementedError, "ReLU"):
            analyze_mlp_chain(function)


if __name__ == "__main__":
    unittest.main()
