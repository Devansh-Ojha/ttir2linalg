# Bounded TTIR to Atlas prototype

The frontend recognizes four bounded patterns: a single FC + bias + ReLU,
the FP8 two-layer MLP, one single-query attention kernel, and a 32×32 matmul +
ReLU tile. `compiler/triton_analysis.py` produces target-neutral workload
plans; `compiler/atlas_mapping.py` emits existing `atlas.virtual_*` operations.
For matrix products, Atlas's N×K weight layout is handled with
`atlas.virtual_xlu_transpose` when the source tile is K×N.

[`matmul_relu.virtual.mlir`](matmul_relu.virtual.mlir) is the mapper-output
fixture for the synthetic plan used by `tests/test_triton_analysis.py`. It
checks target operation emission independently of TTIR parsing; it is not a
capture generated from a real TTIR file.

[`linear_relu.virtual.mlir`](linear_relu.virtual.mlir) and
[`attention.virtual.mlir`](attention.virtual.mlir) are likewise synthetic-plan
mapper fixtures. Tests compare the emitted operation sequence against each
fixture; real TTIR parsing still requires Triton's `libtriton`.

These are bounded compute-tile mappings. The caller stages padded tiles into
Atlas's 2 KiB-spaced input slots and applies the selected FP8/BF16 precision
policy. The frontend does not lower arbitrary TTIR pointer arithmetic, perform
runtime input movement, or implement whole-model allocation. Atlas-MLIR owns
virtual-op scheduling, register allocation, and instruction emission.

## PyTorch-exported HGraph dense path

`compiler.hgraph_to_atlas` also maps the serialized `torch.export` HGraph for
biased one-layer Linear+ReLU and two-layer MLP graphs. For example, after
`compiler.inductor_capture --model mlp` has written `workload.json`:

```bash
PYTHONPATH=. python -m compiler.hgraph_to_atlas \
  out/inductor/mlp/workload.json \
  --numerical-policy atlas-fp8-pytorch-dense-v1 \
  --out out/atlas/pytorch_mlp.virtual.mlir
```

[`pytorch_mlp.virtual.mlir`](pytorch_mlp.virtual.mlir) is the generated fixture.
The HGraph is produced by `torch.export`; this route currently maps the graph
semantics directly and does not use or associate Inductor's generated kernels.
The caller pads and stages activation/weight tiles as E4M3FN and bias tiles as
BF16. The BF16 hidden activation is packed to FP8 with scale code 127 before
the second MXU operation, so this is an explicit lossy precision policy.

With Triton's `libtriton` available, generate Atlas virtual MLIR from a
matching TTIR capture:

```bash
PYTHONPATH=. python -m compiler.ttir_to_atlas path/to/matmul_relu.ttir \
  --out out/atlas/matmul_relu.virtual.mlir
```

Then lower it from an Atlas-MLIR checkout with its built tools:

```bash
python /path/to/atlas-mlir/tools/lower_atlas_program.py \
  out/atlas/matmul_relu.virtual.mlir \
  --atlas-bin-dir /path/to/atlas-mlir/build/bin \
  --output-dir out/atlas/matmul_relu.lowered
```

The mapper is an explicit precision boundary: Atlas consumes E4M3FN tiles and
returns BF16. Numerical comparison must use those staged FP8 values and Atlas's
selected MXU/VPU semantics, rather than assuming bitwise equivalence to an
unquantized FP32 `tt.dot`.

## Workload order

The current converter patterns are `linear-relu`, `two-layer-mlp`,
`attention`, and `matmul-relu`. The first three are bounded padded-tile
conversions of the checked-in single-layer MLP, FP8 two-layer MLP, and
single-query attention TTIR samples. Each source uses an explicit Atlas
numerical policy; the caller stages the described padded input tiles, including
an additive BF16 mask tile for attention.

Examples:

```bash
PYTHONPATH=. python -m compiler.ttir_to_atlas \
  examples/archive/ttir/mlp/kernel_0_fc1_relu.ttir --pattern linear-relu \
  --numerical-policy atlas-fp8-linear-v1 --output-features 8 \
  --out out/atlas/mlp.virtual.mlir

PYTHONPATH=. python -m compiler.ttir_to_atlas \
  examples/two_layer_mlp_fp8.ttir --pattern two-layer-mlp \
  --numerical-policy atlas-fp8-mlp-v1 --out out/atlas/mlp2.virtual.mlir

PYTHONPATH=. python -m compiler.ttir_to_atlas \
  examples/archive/ttir/attention/kernel_0_attention.ttir --pattern attention \
  --numerical-policy atlas-fp8-attention-v1 \
  --out out/atlas/attention.virtual.mlir
```

The attention query tile has one active row and four active keys/features in
the checked-in sample. The mask has zero in valid key columns and a negative
value in padded columns so padding contributes zero probability. The output's
first row and eight active columns carry the sample result. The standalone
32×32 matmul + ReLU conversion remains available as `matmul-relu`.
