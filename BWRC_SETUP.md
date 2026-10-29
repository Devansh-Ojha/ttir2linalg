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

Each directory contains `capture.json`, `workload.hgir`, `workload.json`,
Inductor-generated source, and actual Triton TTIR dumps when those kernels are
emitted. The capture automatically runs the strict TTIR-to-Atlas mapper; inspect
its per-kernel results and any unsupported operations:

```bash
python - <<'PY'
import json
from pathlib import Path
for path in sorted(Path("out/inductor").glob("*/capture.json")):
    report = json.loads(path.read_text())
    print(path.parent.name, report["inductor_status"],
          "generated units:", report["code_count"],
          "kernel launches:", report["inductor_kernel_launch_count"],
          "Triton kernels:", report["inductor_triton_kernel_count"],
          "TTIR kernels:", report["inductor_ttir_kernel_count"],
          "launched TTIR kernels:", report["inductor_launched_ttir_kernel_count"],
          "Atlas-mapped:", report.get("atlas_mapped_kernel_count"),
          "FlashAttention forced + successful:", report.get("flash_attention_backend_forced_and_run_succeeded"),
          "single launch:", report.get("one_kernel_fusion_observed"))
    if report.get("inductor_error"):
        print("  error:", report["inductor_error"])
PY
```

On CUDA, capture restricts SDPA to the FlashAttention backend. If the device
or shape is ineligible, the report records the failure instead of silently
claiming a fallback is FlashAttention. A single captured graph also does not
mean one monolithic kernel: compare generated unit counts and inspect each
artifact for fusion boundaries. Unsupported Inductor TTIR kernels remain in
`mapping.json` with matcher diagnostics.

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
ATLAS_VIRTUAL="$(find out/inductor/mlp/atlas -name '*.virtual.mlir' -print -quit)"
python /path/to/atlas-mlir/tools/lower_atlas_program.py \
  "$ATLAS_VIRTUAL" \
  --atlas-bin-dir /path/to/atlas-mlir/build/bin \
  --output-dir out/atlas/mlp.lowered
```

Use an `.virtual.mlir` path that is listed in `mapping.json`. See [FLOW.md](FLOW.md) for implemented
boundaries and remaining coverage work.
