# Archived experiments and generated artifacts

The active path is documented in the repository [README](../../README.md) and
[FLOW](../../FLOW.md). This directory keeps earlier prototypes and captured
fixtures out of the repository root without deleting them.

- `hgir/`: earlier PyTorch/TTIR graph snapshots.
- `ttir/`: handwritten Triton captures and standalone parser/lowering samples.
- `triton_source/`: Triton source snapshots from the earlier capture path.
- `mlir/`, `atlas/`, `llvm/`: generated output from compiler experiments.

Legacy scripts live in [`tools/legacy`](../../tools/legacy). Atlas-focused
validation scripts live in [`tools/validation`](../../tools/validation). The
current checked-in Atlas fixtures remain in `examples/ttir_to_atlas/`, and the
two-layer MLP input fixture remains at `examples/two_layer_mlp_fp8.ttir`.

These files are preserved for reference. Their shape assumptions and target
policies do not necessarily match the current Inductor cross-compile flow.
