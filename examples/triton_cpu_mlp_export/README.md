# Triton CPU LLVM IR for the playground MLP kernel

This folder contains a standalone copy of the playground's fused
`FC1 -> ReLU -> FC2 -> ReLU` Triton kernel and a compiler-only exporter. It
compiles the kernel to Triton CPU LLVM IR without running the kernel. The
separate third `fc3` layer in the playground model is not part of this kernel.

The default dimensions match the playground model: input size 16, hidden size
32, and a program tile of 16 rows.

Use a Python environment with Triton built from the experimental
[`triton-cpu`](https://github.com/triton-lang/triton-cpu) repository. From this
directory run:

```bash
TRITON_DEFAULT_BACKEND=cpu python export_llvm_ir.py
```

The output is `out/mlp2.llir`. To select a different MLP shape, pass
`--input-size` and `--hidden-size`; use `--out` to choose another output path.
