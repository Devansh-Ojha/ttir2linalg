module {
  func.func @ttir_two_layer_mlp_atlas() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415927296 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %x = "atlas.virtual_input_fp8"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2, %b0 = "atlas.virtual_input_fp8"(%io1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3, %b1 = "atlas.virtual_input_fp8"(%io2) {index = 2 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %w0 = "atlas.virtual_xlu_transpose"(%b0) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8
    %w1 = "atlas.virtual_xlu_transpose"(%b1) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8
    %h0 = "atlas.virtual_mxu_matmul"(%x, %w0) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1 = "atlas.virtual_vpu_unary"(%h0) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2 = "atlas.virtual_pack_fp8"(%h1) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y = "atlas.virtual_mxu_matmul"(%h2, %w1) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io4 = "atlas.virtual_output_bf16"(%io3, %y) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io4 : !atlas.virtual_state
  }
}
