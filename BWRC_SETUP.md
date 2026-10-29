# BWRC setup and first run

This guide prepares the PyTorch/Inductor CUDA cross-compile environment and
the separate Atlas-MLIR toolchain. The default path uses fake CUDA tensors, so
it can compile without allocating or running on a physical GPU. It still needs
a CUDA-enabled PyTorch build plus the CUDA/Triton compiler components. A GPU
allocation is only needed for optional runtime validation.

## 1. Inspect the machine

Inspect available software modules on BWRC:

```bash
module avail python cuda gcc cmake llvm 2>&1 | less
```

Load the site's supported Python, CUDA toolkit, and compiler modules. Then
create a project environment:

```bash
cd /path/to/ttir2linalg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch wheel using the command selected by the official
[PyTorch install selector](https://pytorch.org/get-started/locally/). A GPU is
not required to install it or run the AOT cross-compile path.
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
python -c 'import torch, triton; print("torch", torch.__version__, "CUDA build", torch.backends.cuda.is_built(), "GPU available", torch.cuda.is_available()); print("triton", triton.__version__)'
```

If pip cannot find a compatible Triton wheel, keep the cluster's Python/CUDA
module combination and follow Triton's official installation instructions;
do not substitute a handwritten kernel and call it Inductor output.

## 2. Cross-compile models through actual Inductor without a GPU

Create output directories and cross-compile each workload for CUDA. The
default mode uses fake CUDA tensors and never executes the compiled model. The
Transformer block includes attention, residual paths, normalization, and an MLP. The separate
`attention_mlp` example is a simpler attention-then-MLP workload.

```bash
mkdir -p out
for model in linear_relu mlp attention_flash attention_custom_flash \
             attention_mlp transformer_block; do
  PYTHONPATH=. python -m compiler.inductor_capture \
    --model "$model" --out-dir "out/inductor/$model" --mode aot-cuda
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
          "kernel launch sites:", report["inductor_kernel_launch_site_count"],
          "Triton kernels:", report["inductor_triton_kernel_count"],
          "TTIR kernels:", report["inductor_ttir_kernel_count"],
          "wrapper-referenced TTIR:", report["inductor_wrapper_referenced_ttir_kernel_count"],
          "Atlas-mapped:", report.get("atlas_mapped_kernel_count"),
          "FlashAttention compiled:", report.get("flash_attention_backend_compiled_under_constraint"),
          "single launch:", report.get("one_kernel_fusion_observed"))
    if report.get("inductor_error"):
        print("  error:", report["inductor_error"])
PY
```

`attention_flash` forces PyTorch's ready-made FlashAttention SDPA backend;
`attention_custom_flash` invokes the repository's tiled Triton online-softmax
kernel through `torch.library.triton_op`. The latter is expected to expose a
Triton source and TTIR file. PyTorch's built-in FlashAttention may instead
emit a native CUDA kernel, which is recorded as such. Some PyTorch versions or
cross-compile setups may still need device capability metadata to compile
these paths; failures are recorded explicitly. The existing Atlas attention
recognizer may report custom FlashAttention TTIR as unsupported. A single
captured graph also does not mean one monolithic kernel: compare generated
unit counts and inspect each artifact for fusion boundaries. Unsupported
Inductor TTIR kernels remain in `mapping.json` with matcher diagnostics.

For a numerical check, request a GPU allocation and use `--mode run`. Both
FlashAttention implementations are compared with PyTorch math SDPA using the
same learned projections. Fake-tensor `aot-cuda` verifies compilation and
artifact capture only; it cannot verify output values.

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
