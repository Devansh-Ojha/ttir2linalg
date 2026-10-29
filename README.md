# PyTorch TorchInductor kernels to Atlas virtual ops

The active path starts with PyTorch models, cross-compiles them for CUDA
through TorchInductor using fake tensors, captures the real generated Triton
TTIR, and maps supported kernels to Atlas virtual operations. The default mode
does **not** need a physical GPU. It does require a CUDA-enabled PyTorch build
and compatible CUDA/Triton compiler components. The optional `--mode run`
performs actual execution and needs a GPU.

```text
PyTorch model
  -> TorchInductor AOT cross-compile (fake CUDA inputs)
  -> Inductor output_code.py + Triton compiler TTIR
  -> strict TTIR recognizers
  -> per-kernel Atlas virtual MLIR + mapped/unsupported report
  -> Atlas-MLIR verification and lowering
```

## Setup

Follow [BWRC_SETUP.md](BWRC_SETUP.md) to prepare PyTorch, Triton, and Atlas-MLIR.
Use the PyTorch install selector to install a CUDA-enabled build even when
cross-compiling on a machine without a GPU.

## Capture and lower

Capture the requested workload families. The command compiles each model with
real TorchInductor, emits compiler artifacts, then attempts Atlas mapping from
only the Inductor-generated TTIR files:

```bash
mkdir -p out/inductor
for model in linear_relu mlp attention attention_mlp transformer_block; do
  PYTHONPATH=. python -m compiler.inductor_capture \
    --model "$model" \
    --out-dir "out/inductor/$model" \
    --mode aot-cuda
done
```

Each `capture.json` records generated code, wrapper launch sites, Triton TTIR files,
FlashAttention compile status, and Atlas mapping status. `atlas/mapping.json`
lists each wrapper-referenced TTIR kernel as mapped or unsupported with analyzer
diagnostics. The existing Atlas recognizers are deliberately bounded; a
successful Inductor compile does not imply every kernel is already supported.
For a real output comparison on a GPU, rerun a workload with `--mode run`.

The workload names are:

- `linear_relu`: one projection followed by ReLU
- `mlp`: FC1 + ReLU + FC2
- `attention`: self-attention, requesting SDPA and requiring FlashAttention
- `attention_mlp`: attention followed by a feed-forward MLP
- `transformer_block`: attention, residual connections, normalization, and MLP

For the last two, the report counts kernel launch sites; one FX graph does not
mean the whole block became one kernel. Monolithic fusion is measured, not
assumed.

## Repo layout

- `model/`: active PyTorch workload definitions
- `compiler/inductor_capture.py`: AOT cross-compile and artifact capture
- `compiler/inductor_to_atlas.py`: maps exact wrapper-referenced Inductor TTIR artifacts
- `compiler/ttir_to_atlas.py`, `compiler/atlas_mapping.py`: strict analyzers
  and Atlas virtual-op emission
- `examples/ttir_to_atlas/`: current Atlas fixtures and mapping documentation
- `examples/archive/`: prior generated files and reference experiments
- `tools/legacy/`: older one-off compiler scripts
- `tools/validation/`: Atlas virtual and machine validation helpers

The separate `torch.export` HGraph-to-Atlas path and handwritten Triton kernels
are retained as experiments; they are not fallbacks in the Inductor mapping
path. See [FLOW.md](FLOW.md) and [GOAL.md](GOAL.md) for current coverage and
limitations.
