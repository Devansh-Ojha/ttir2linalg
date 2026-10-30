Untouched Triton 3.6.0 SM89 compiler output for the new handwritten
`quantized_attention` kernel. `capture.json` retains the real ASTSource compiler
origin, signature/options, source hash and TTIR hash. `kernel.source.py` is the
full source snapshot. No GPU launch occurred. The original BF16 attention kernel
and all earlier fixtures remain unchanged.

The full 32x32 fused program performs Q/K/V unit-scale FP8 packing, source K
transpose, MXU0/BF16 score readout and scaling, rowwise BF16 softmax, explicit
probability scaling/FP8 packing, MXU1/BF16 readout and output compensation.
Diagnostic outputs expose scaled scores, BF16 probabilities and raw FP8 packed
probabilities in separate buffers.

Positive verifier/lowering/simulator tests use this original compiler TTIR.
Synthetic mutations in rejection/reference-control unit tests are explicitly
not compiler provenance claims. See docs/triton_quantized_attention.md for the
hardware-aware numerical contract and limits; CUDA/original BF16 equivalence
is not claimed.
