import unittest
from dataclasses import replace

from compiler.hgir import HGraph, HOp, HValue
from compiler.hgraph_analysis import analyze_dense_graph
from compiler.hgraph_atlas_mapping import (
    PYTORCH_DENSE_NUMERICAL_POLICY,
    map_hgraph_dense,
)


def make_mlp(two_layers):
    inputs = {
        "x": HValue("x", "float32", (1, 4)),
        "w0": HValue("w0", "float32", (8, 4)),
        "b0": HValue("b0", "float32", (8,)),
    }
    values = {"h0": HValue("h0", "float32", (1, 8)),
              "h1": HValue("h1", "float32", (1, 8))}
    ops = [HOp("linear", ["x", "w0", "b0"], ["h0"]),
           HOp("relu", ["h0"], ["h1"])]
    output = "h1"
    if two_layers:
        inputs.update({
            "w1": HValue("w1", "float32", (8, 8)),
            "b1": HValue("b1", "float32", (8,)),
        })
        values["y"] = HValue("y", "float32", (1, 8))
        ops.append(HOp("linear", ["h1", "w1", "b1"], ["y"]))
        output = "y"
    return HGraph("example", inputs, values, ops, [output])


class HGraphAtlasMappingTests(unittest.TestCase):
    def test_hgraph_round_trip_keeps_shapes_and_operations(self):
        graph = make_mlp(two_layers=True)
        restored = HGraph.from_dict(graph.to_dict())
        self.assertEqual(restored.format(), graph.format())
        self.assertEqual(restored.values["y"].shape, (1, 8))

    def test_maps_biased_linear_relu(self):
        plan = analyze_dense_graph(make_mlp(two_layers=False))
        mlir = map_hgraph_dense(
            plan, numerical_policy=PYTORCH_DENSE_NUMERICAL_POLICY
        )
        self.assertIn('"atlas.virtual_mxu_matmul"', mlir)
        self.assertIn('"atlas.virtual_vpu_binary"(%dot, %b)', mlir)
        self.assertIn('kind = "relu"', mlir)
        self.assertIn('"atlas.virtual_output_bf16"', mlir)

    def test_maps_two_layer_mlp_with_bias_and_explicit_precision(self):
        plan = analyze_dense_graph(make_mlp(two_layers=True))
        with self.assertRaisesRegex(NotImplementedError, "numerical policy"):
            map_hgraph_dense(plan)
        mlir = map_hgraph_dense(
            plan, numerical_policy=PYTORCH_DENSE_NUMERICAL_POLICY
        )
        self.assertEqual(mlir.count('"atlas.virtual_mxu_matmul"'), 2)
        self.assertEqual(mlir.count('"atlas.virtual_vpu_binary"'), 2)
        self.assertIn('"atlas.virtual_pack_fp8"', mlir)
        self.assertIn("atlas.input_dram_base = 2415919104", mlir)

    def test_rejects_unpadded_tile_sizes(self):
        graph = make_mlp(two_layers=False)
        graph.inputs["w0"] = HValue("w0", "float32", (33, 4))
        graph.values["h0"] = HValue("h0", "float32", (1, 33))
        graph.values["h1"] = HValue("h1", "float32", (1, 33))
        graph.inputs["b0"] = HValue("b0", "float32", (33,))
        plan = analyze_dense_graph(graph)
        with self.assertRaisesRegex(NotImplementedError, "dimensions"):
            map_hgraph_dense(
                plan, numerical_policy=PYTORCH_DENSE_NUMERICAL_POLICY
            )

    def test_rejects_dense_plan_with_unsupported_activation(self):
        plan = analyze_dense_graph(make_mlp(two_layers=True))
        with self.assertRaisesRegex(NotImplementedError, "ReLU activation"):
            map_hgraph_dense(
                replace(plan, activation="gelu"),
                numerical_policy=PYTORCH_DENSE_NUMERICAL_POLICY,
            )


if __name__ == "__main__":
    unittest.main()
