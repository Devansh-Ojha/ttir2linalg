import unittest

from compiler.hgir import HGraph, HOp, HValue
from compiler.hgraph_analysis import analyze_hgraph


def make_graph(op_names):
    inputs = {"x": HValue("x", "float32", (1, 4, 8))}
    values = {}
    ops = []
    previous = "x"
    for index, name in enumerate(op_names):
        output = "v%d" % index
        values[output] = HValue(output, "float32", (1, 4, 8))
        ops.append(HOp(name, [previous], [output]))
        previous = output
    return HGraph("test", inputs, values, ops, [previous])


class HGraphAnalysisTests(unittest.TestCase):
    def test_classifies_single_layer_linear_relu(self):
        result = analyze_hgraph(make_graph(["linear", "relu"]))
        self.assertEqual(result.pattern, "linear_relu")

    def test_classifies_two_layer_mlp(self):
        result = analyze_hgraph(make_graph(["linear", "relu", "linear"]))
        self.assertEqual(result.pattern, "two_layer_mlp")

    def test_classifies_sdpa_attention(self):
        result = analyze_hgraph(make_graph([
            "linear", "transpose", "scaled_dot_product_attention", "linear"
        ]))
        self.assertEqual(result.pattern, "attention")
        self.assertTrue(result.has_scaled_dot_product_attention)

    def test_classifies_attention_mlp_and_transformer_block(self):
        attention_mlp = analyze_hgraph(make_graph([
            "linear", "scaled_dot_product_attention", "linear", "linear",
            "relu", "linear", "mean",
        ]))
        transformer = analyze_hgraph(make_graph([
            "linear", "scaled_dot_product_attention", "linear", "add",
            "layer_norm", "linear", "gelu", "linear", "add", "layer_norm",
        ]))
        self.assertEqual(attention_mlp.pattern, "attention_mlp")
        self.assertEqual(transformer.pattern, "transformer_block")
        self.assertTrue(transformer.has_residual_add)
        self.assertTrue(transformer.has_normalization)

    def test_rejects_invalid_hgraph(self):
        graph = HGraph(
            "invalid", {"x": HValue("x", "float32", (1,))}, {},
            [HOp("relu", ["missing"], ["y"])], ["y"],
        )
        with self.assertRaisesRegex(ValueError, "invalid HGraph"):
            analyze_hgraph(graph)


if __name__ == "__main__":
    unittest.main()
