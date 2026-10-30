Untouched compiler-generated Triton 3.6.0 SM89 TTIR, source snapshots and SHA256
metadata. Each fixture has its own source snapshot; later captures include
additional appended kernels. No original kernel was replaced.

- fp8: direct FP8 tile transpose, exact raw-byte output.
- fanout: FP8 transpose plus original producer stored to a second output.
- fp8_dot: explicit source X.T followed by FP8 matrix multiplication.
- bf16: genuine standalone BF16 transpose; deliberately retained as an
  unsupported hardware-format case, not a successful Atlas execution.

All positive machine validation uses the unedited compiler TTIR. Synthetic
negative/mutation tests only exercise rejection. Original BF16 attention TTIR
and source snapshots remain in ../triton_rowwise and are not replaced.
