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
for model in linear_relu mlp attention_flash attention_custom_flash \
             attention_mlp transformer_block; do
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
`attention_flash` uses PyTorch's ready-made SDPA FlashAttention backend; it may
emit a native CUDA kernel instead of Triton, which the report will show.
`attention_custom_flash` uses `kernels/flash_attention.py`, our own tiled
online-softmax Triton kernel. This is the path that explicitly exposes our
Triton source and its compiler TTIR. The current Atlas attention recognizer is
limited to a different small TTIR pattern, so custom FlashAttention may appear
as unsupported in `mapping.json` until that recognizer is extended.

For numerical correctness, run on a GPU with `--mode run`: the custom Triton
result and the ready-made FlashAttention result are compared to an independent
PyTorch math-SDPA reference with identical learned projections. A successful
fake-tensor AOT compile checks code generation only; it cannot establish
numerical correctness because it does not execute the kernels.

The workload names are:

- `linear_relu`: one projection followed by ReLU
- `mlp`: FC1 + ReLU + FC2
- `attention_flash`: self-attention using PyTorch's ready-made FlashAttention
- `attention_custom_flash`: self-attention using the repository's Triton kernel
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
