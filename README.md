# PyTorch to Triton/TTIR and Atlas

The updated project objective is an end-to-end PyTorch Inductor-to-SE/Linalg-
to-Atlas virtual-op path for MLP and Transformer workloads. The milestones,
scope, and completion criteria are in [GOAL.md](GOAL.md). The commands below
describe the current prototype paths; they do not yet form that complete
Inductor-to-Atlas pipeline.

This project currently has separate prototype paths for model capture and
handwritten Triton lowering:

```text
PyTorch model -> torch.compile graph capture -> Triton kernel source
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

Use Python 3.9+ and install the dependencies in the target environment:

```bash
python -m pip install torch
python -m pip install triton==3.4.0
```

Triton is required for genuine `.ttir` generation. CUDA execution is not
required; the compiler target is used only to produce compiler artifacts.

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

The command runs `torch.compile(..., backend="inductor")`, checks the result
against eager PyTorch, and writes Inductor-generated code, an exported workload
graph, any Triton kernels embedded in that code, and `capture.json`. The
generated code depends on the selected backend
(typically C++ on CPU and potentially Triton on CUDA). It does not fabricate
TTIR or Atlas operations. This capture artifact is the first explicit bridge
in the larger objective; lowering the captured graph/code to the Atlas
frontend remains under development. The attention model requests SDPA by
disabling attention weight output; `capture.json` records matching generated
code lines as backend evidence, but the actual artifact must be inspected to
confirm which fused attention kernel the installed backend selected.
If Inductor fails to compile on the host, the command still writes the exported
graph and a report containing the failure reason; it records no generated code
or successful numerical comparison in that case.

The serialized HGraph for a supported dense workload can be mapped to the
existing Atlas virtual dialect with an explicit precision policy:

```bash
PYTHONPATH=. python -m compiler.hgraph_to_atlas \
  out/inductor/mlp/workload.json \
  --numerical-policy atlas-fp8-pytorch-dense-v1 \
  --out out/atlas/mlp_from_hgraph.virtual.mlir
```

This frontend currently admits the one-layer Linear+ReLU and two-layer MLP
graphs. Its input is the separately exported `torch.export` HGraph; generated
Inductor kernels are not yet associated with or consumed by this mapping. The
caller must stage 32×32 FP8 activation/weight tiles and BF16 bias tiles.
If Inductor fails to compile on the host, the command still writes the exported
graph and a report containing the failure reason; it records no generated code
or successful numerical comparison in that case.

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
