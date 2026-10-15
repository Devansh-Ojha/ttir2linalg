# Triton CPU LLVM IR for the playground MLP kernel

This folder contains a standalone copy of the playground's fused `FC1 -> ReLU -> FC2 -> ReLU` Triton kernel and a compiler-only exporter. It compiles the kernel to Triton CPU LLVM IR without running the kernel. The separate third `fc3` layer in the playground model is not part of this kernel.

The default dimensions match the playground model: input size 16, hidden size 32, and a program tile of 16 rows.

## Setup

Use a Python environment with Triton built from the experimental [triton-cpu](https://github.com/triton-lang/triton-cpu) repository.

```bash
git clone https://github.com/triton-lang/triton-cpu.git
cd triton-cpu

uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install cmake ninja
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
MAX_JOBS=8 uv pip install -e . --no-build-isolation
```

On BWRC, the LLVM build may require a newer `libstdc++` and additional linker configuration.

## Running the MLP

From the `ttir2linalg` repository:

```bash
source ~/triton-cpu/.venv/bin/activate
export TRITON_DEFAULT_BACKEND=cpu

python examples/triton_cpu_mlp_export/export_llvm_ir.py
```

The default output is `out/mlp2.llir`. To select a different MLP shape, pass `--input-size` and `--hidden-size`; use `--out` to choose another output path.

## Exporting LLVM Dialect MLIR

To capture LLVM dialect MLIR before conversion to LLVM IR, add the following to `triton-cpu/third_party/cpu/backend/compiler.py`, immediately after `pm.run(mod, "make_llir")`:

```python
dump_path = os.environ.get("TRITON_CPU_LLVM_MLIR_OUT")
if dump_path:
    Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
    Path(dump_path).write_text(str(mod))
```

Then run from the `ttir2linalg` root:

```bash
export TRITON_CPU_LLVM_MLIR_OUT="$PWD/out/mlp2_llvm.mlir"

python examples/triton_cpu_mlp_export/export_llvm_ir.py --out out/mlp2.ll

unset TRITON_CPU_LLVM_MLIR_OUT
```

This generates `out/mlp2.ll` (LLVM IR) and `out/mlp2_llvm.mlir` (LLVM dialect MLIR).

## Testing

To compare the Triton CPU kernel against PyTorch:

```bash
python examples/triton_cpu_mlp_export/test_mlp.py
```

The fused MLP passes the numerical correctness test with a maximum absolute error of approximately `7.6e-6` for the tested configuration.