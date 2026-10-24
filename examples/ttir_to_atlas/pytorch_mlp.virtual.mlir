// Source: torch.export HGraph; not a lowering of Inductor kernel code.
// Numerical policy: atlas-fp8-pytorch-dense-v1; caller stages padded/quantized Atlas tiles.
module {
  func.func @hgraph_mlp_atlas() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415935488 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %x = "atlas.virtual_input_fp8"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2, %w0 = "atlas.virtual_input_fp8"(%io1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3, %b0 = "atlas.virtual_input_bf16"(%io2) {index = 2 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io4, %w1 = "atlas.virtual_input_fp8"(%io3) {index = 3 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io5, %b1 = "atlas.virtual_input_bf16"(%io4) {index = 4 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %w0_nk = "atlas.virtual_xlu_transpose"(%w0) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8
    %w1_nk = "atlas.virtual_xlu_transpose"(%w1) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8
    %h0 = "atlas.virtual_mxu_matmul"(%x, %w0_nk) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h0b = "atlas.virtual_vpu_binary"(%h0, %b0) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h1 = "atlas.virtual_vpu_unary"(%h0b) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h1_fp8 = "atlas.virtual_pack_fp8"(%h1) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %h2 = "atlas.virtual_mxu_matmul"(%h1_fp8, %w1_nk) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %y = "atlas.virtual_vpu_binary"(%h2, %b1) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %io6 = "atlas.virtual_output_bf16"(%io5, %y) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io6 : !atlas.virtual_state
  }
}
