These TTIR files are untouched output of Triton 3.6.0's actual
`triton.compile(ASTSource).asm['ttir']`, CUDA SM89, four warps, one CTA,
FP fusion disabled. No GPU was launched. The handwritten source snapshot and
capture metadata retain SHA256 hashes; tests verify both. They are compiler
fixtures, unlike the existing handwritten operation fixtures.

Reproduce captures in fresh directories with `python -m compiler.jit_capture`,
module `kernels.atlas_jit_kernels`, kernel `fp8_matmul` or `fp8_two_layer`,
input signatures `*fp8e4nv` and output `*bf16`. See
`docs/triton_atlas_phase_c.md` for execution and numerical scope.
