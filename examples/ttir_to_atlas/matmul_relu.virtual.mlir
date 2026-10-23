// Mapper output fixture for the synthetic matmul+ReLU plan in
// tests/test_triton_analysis.py. This checks emission, not TTIR parsing.
module {
  func.func @ttir_matmul_relu_atlas() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415927296 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %lhs = "atlas.virtual_input_fp8"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2, %rhs = "atlas.virtual_input_fp8"(%io1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %rhs_nk = "atlas.virtual_xlu_transpose"(%rhs) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8
    %dot = "atlas.virtual_mxu_matmul"(%lhs, %rhs_nk) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %relu = "atlas.virtual_vpu_unary"(%dot) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %io3 = "atlas.virtual_output_bf16"(%io2, %relu) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io3 : !atlas.virtual_state
  }
}
