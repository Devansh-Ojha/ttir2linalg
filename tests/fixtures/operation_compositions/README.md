These eight fixtures are handwritten TTIR for operation-analysis tests, added
in this phase. They are not GPU captures and do not assert Atlas numerical
equivalence. The genuine Graham captures remain unchanged in the adjacent
`graham_flash_attention` fixture directory.

The seven requested compositions are independent functions, with explicit
loads/stores and typed SSA values. `matmul_add_matmul.ttir` deliberately reuses
the first dot result as the second accumulator and shares a weight input; this
checks dependency analysis beyond a zero-accumulator MLP chain.
`shape_and_conversion.ttir` adds expand_dims, broadcast, order-preserving
reshape, BF16 extension, subtraction and BF16 truncation.
