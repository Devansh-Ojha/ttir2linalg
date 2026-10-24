# BWRC setup and first run

This guide prepares the PyTorch/Inductor capture environment and the separate
Atlas-MLIR toolchain. Run the commands on a BWRC Linux machine with an NVIDIA
GPU allocation. Use the cluster's documented module names and CUDA version;
they vary by host and account.

## 1. Inspect the machine

After logging in and requesting an interactive GPU allocation through the
cluster scheduler, inspect the available software and GPU:

```bash
module avail python cuda gcc cmake llvm 2>&1 | less
nvidia-smi
```

Load the site's supported Python, CUDA, and compiler modules. Then create a
project environment:

```bash
cd /path/to/ttir2linalg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch wheel matching the cluster CUDA runtime using the command
selected by the official [PyTorch install selector](https://pytorch.org/get-started/locally/).
Check whether that environment already provides the compatible Triton package;
install Triton only if it is missing, using a version compatible with that
PyTorch release. Then install the graph/MLIR Python dependency:

```bash
python -m pip install torch
python -c 'import triton; print(triton.__version__)' || python -m pip install triton
python -m pip install xdsl==0.65.0
```

Record the environment so results can be reproduced:

```bash
python -m pip freeze > out/bwrc-requirements.txt
python -c 'import torch, triton; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("triton", triton.__version__)'
```

If pip cannot find a compatible Triton wheel, keep the cluster's Python/CUDA
module combination and follow Triton's official installation instructions;
do not substitute a handwritten kernel and call it Inductor output.

## 2. Capture models through actual Inductor

Create output directories and capture each workload. The Transformer block
includes attention, residual paths, normalization, and an MLP. The separate
`attention_mlp` example is a simpler attention-then-MLP workload.

```bash
mkdir -p out
for model in linear_relu mlp attention attention_mlp transformer_block; do
  PYTHONPATH=. python -m compiler.inductor_capture \
    --model "$model" --out-dir "out/inductor/$model"
done
```

Each directory contains `capture.json`, `workload.hgir`, `workload.json`, and
Inductor-generated source files when compilation succeeds. Inspect the real
kernel count, backend evidence, and errors:

```bash
python - <<'PY'
import json
from pathlib import Path
for path in sorted(Path("out/inductor").glob("*/capture.json")):
    report = json.loads(path.read_text())
    print(path.parent.name, report["inductor_status"],
          "generated units:", report["code_count"],
          "Triton kernels:", report["inductor_triton_kernel_count"],
          "FlashAttention evidence:", report.get("flash_attention_kernel_detected_in_generated_code"))
    if report.get("inductor_error"):
        print("  error:", report["inductor_error"])
PY
```

SDPA requests fused attention where supported; it does not guarantee a
FlashAttention kernel. The generated code and report are the evidence. A
single captured graph also does not mean one monolithic kernel: compare the
generated unit count and inspect each artifact to see fusion boundaries.

The current HGraph-to-Atlas bridge supports only the dense MLP patterns:

```bash
PYTHONPATH=. python -m compiler.hgraph_to_atlas \
  out/inductor/mlp/workload.json \
  --numerical-policy atlas-fp8-pytorch-dense-v1 \
  --out out/atlas/mlp.virtual.mlir
```

This maps exported graph semantics, not Inductor-generated kernels. The
attention/Transformer, Inductor-kernel-to-TTIR, and SE-level bridges remain
implementation work.

## 3. Build Atlas-MLIR

Clone the Atlas-MLIR repository beside this project if it is not already
available, then use the LLVM/MLIR installation required by its README. From
the Atlas-MLIR checkout:

```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

Use Atlas-MLIR's `atlas-opt` to verify the emitted virtual program and its
`lower_atlas_program.py` script to run the existing scheduling/allocation and
machine-lowering flow:

```bash
python /path/to/atlas-mlir/tools/lower_atlas_program.py \
  out/atlas/mlp.virtual.mlir \
  --atlas-bin-dir /path/to/atlas-mlir/build/bin \
  --output-dir out/atlas/mlp.lowered
```

See [FLOW.md](FLOW.md) for which compiler boundaries are implemented and
which remain open. The acronym “SE” is not defined in this repository yet;
the current Linalg prototype is not silently treated as that target IR.
