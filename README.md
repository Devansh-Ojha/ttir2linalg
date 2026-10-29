# PyTorch Inductor kernels to Atlas virtual ops

The target path is **PyTorch model → real TorchInductor code and Triton TTIR
dumps → TTIR pattern recognition → Atlas virtual ops**. See [GOAL.md](GOAL.md)
for scope and current coverage.

The main path uses Inductor's generated kernel artifacts. Handwritten Triton
examples remain separate reference paths:

```text
PyTorch model -> torch.compile(backend="inductor") -> generated code + TTIR -> Atlas virtual ops
PyTorch model -> explicit Triton kernels -> Triton compiler -> verified TTIR
```

The PyTorch capture workloads are:

- `linear_relu`: one linear projection and ReLU
- `mlp`: FC1 + ReLU + FC2
- `attention`: self-attention with SDPA requested through PyTorch
- `transformer_block`: residual self-attention, normalization, and a two-layer
  feed-forward network

## Setup

For the full cluster environment, Inductor capture, and Atlas-MLIR build
sequence, follow [BWRC_SETUP.md](BWRC_SETUP.md). The concise local setup below
is suitable for a development machine and may need platform-specific package
selection.

Use the Python/PyTorch/CUDA versions supported by the target machine (see
[BWRC_SETUP.md](BWRC_SETUP.md)):

```bash
python -m pip install torch
python -m pip install triton xdsl==0.65.0
```

The Inductor-to-Atlas path needs a CUDA run when the installed Inductor backend
emits Triton kernels. CPU Inductor output is captured, but it cannot enter the
current TTIR-based mapper.

## Run the Triton-source stage

On a CPU-only machine this uses the project's explicit `@triton.jit` kernel
definitions. It does not claim that CPU Inductor generated Triton source and
does not reconstruct source from the FX graph.

```bash
PYTHONPATH=. python mlp_pipeline.py \
  --model mlp \
  --compile-triton-source \
  --triton-dir mlp-triton

PYTHONPATH=. python mlp_pipeline.py \
  --model attention \
  --compile-triton-source \
  --triton-dir attention-triton
```

Expected source files:

```text
mlp-triton/kernel_0_linear_relu_kernel.py
mlp-triton/kernel_1_linear_kernel.py
attention-triton/kernel_0_attention_kernel.py
```

The source-only stage works without Triton installed because it reads the
kernel definitions directly from `kernels/`.

## Capture actual PyTorch Inductor artifacts

For the new model-to-backend path, capture from the actual Inductor backend
instead of using the source-only fallback above:

```bash
PYTHONPATH=. python -m compiler.inductor_capture \
  --model linear_relu \
  --out-dir out/inductor/linear_relu

PYTHONPATH=. python -m compiler.inductor_capture \
  --model mlp \
  --out-dir out/inductor/mlp

PYTHONPATH=. python -m compiler.inductor_capture \
  --model attention \
  --out-dir out/inductor/attention

PYTHONPATH=. python -m compiler.inductor_capture \
  --model attention_mlp \
  --out-dir out/inductor/attention_mlp

PYTHONPATH=. python -m compiler.inductor_capture \
  --model transformer_block \
  --out-dir out/inductor/transformer_block
```

The command runs `torch.compile(..., backend="inductor")`, checks numerical
results against eager PyTorch, and writes generated code, graph context, and a
capture report. Before importing PyTorch, it enables Triton's compiler dump so
the exact Inductor-emitted `.ttir` kernels are saved under
`inductor_ttir/`. It then automatically runs `compiler.inductor_to_atlas` on
those files. Supported kernels produce per-kernel virtual MLIR under `atlas/`;
unsupported kernels and matcher diagnostics go to `atlas/mapping.json`. The
mapper does not substitute the separate `torch.export` graph or handwritten
Triton kernels. For attention workloads on CUDA, the capture requires the
FlashAttention SDPA backend; eligibility/compilation failures appear in
`capture.json`.

There is also an older graph-semantics experiment that maps a serialized
`torch.export` HGraph directly to Atlas. It is not the Inductor-kernel pipeline.
To run that separate experiment:

```bash
PYTHONPATH=. python -m compiler.hgraph_to_atlas \
  out/inductor/mlp/workload.json \
  --numerical-policy atlas-fp8-pytorch-dense-v1 \
  --out out/atlas/mlp_from_hgraph.virtual.mlir
```

This frontend admits only one- and two-layer dense graphs and is not evidence
that the generated Inductor kernels were lowered.

## Run the Triton-to-TTIR stage

These commands compile the real Triton kernels, write one `.ttir` file per
kernel, parse each file with Triton's `libtriton.ir`, and verify it before
writing HGIR.

```bash
PYTHONPATH=. python mlp_pipeline.py \
  --model mlp \
  --compile-model-triton \
  --ttir-dir mlp-ttir \
  --out mlp.hgir

PYTHONPATH=. python mlp_pipeline.py \
  --model attention \
  --compile-model-triton \
  --ttir-dir attention-ttir \
  --out attention.hgir
```

Expected TTIR files:

```text
mlp-ttir/kernel_0_fc1_relu.ttir
mlp-ttir/kernel_1_fc2.ttir
attention-ttir/kernel_0_attention.ttir
```

The checked-in source artifacts in `mlp-triton/` and `attention-triton/` are
the outputs of the source-stage commands above. Generate `.ttir` artifacts in
an environment with Triton 3.4.0 using the second set of commands.
