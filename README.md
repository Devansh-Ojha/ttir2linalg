# PyTorch to Triton/TTIR

This project demonstrates two small model paths:

```text
PyTorch model -> torch.compile graph capture -> Triton kernel source
PyTorch model -> explicit Triton kernels -> Triton compiler -> verified TTIR
```

The supported models are:

- `mlp`: FC1 + ReLU + FC2
- `attention`: Q/K/V attention with an output projection

## Setup

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

## Working now

- MLP and attention models capture and numerically check with CPU
  `torch.compile`.
- MLP and attention produce explicit Triton kernel source files.
- Triton 3.4.0 produces genuine, parseable, verified TTIR for both model
  kernel families.
- TTIR can be converted into the project's hardware-agnostic HGIR.
- No CUDA kernel execution or hardware-specific backend is included.

The checked-in source artifacts in `mlp-triton/` and `attention-triton/` are
the outputs of the source-stage commands above. Generate `.ttir` artifacts in
an environment with Triton 3.4.0 using the second set of commands.
