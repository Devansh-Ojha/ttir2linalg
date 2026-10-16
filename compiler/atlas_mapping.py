"""Atlas-specific mapping from a target-neutral TTIR operation plan."""
from __future__ import annotations

from dataclasses import dataclass

from compiler.triton_analysis import MatmulRelu


@dataclass(frozen=True)
class AtlasTileABI:
    """Runtime tile bindings and the bounded Atlas virtual-program ABI."""

    lhs_input: int = 0
    rhs_input: int = 1
    mxu_unit: int = 0
    input_base: int = 0x90000000
    output_base: int = 0x90002000
    output_index: int = 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(message)


def map_matmul_relu(plan: MatmulRelu, abi: AtlasTileABI = AtlasTileABI()) -> str:
    """Emit the existing Atlas virtual ops for one admitted 32x32 FP8 tile.

    This is an explicit precision/ABI mapping: the source inputs must already
    be represented by the caller as two row-major runtime E4M3 tiles. Atlas
    stores MXU weights as N×K, so the source K×N RHS tile is transposed with
    the existing XLU op before matmul. MXU readout and VPU ReLU produce a BF16
    output tile. This function does not allocate registers, select
    instructions, or infer runtime tensor storage.
    """
    _require(plan.shape == (32, 32), "Atlas virtual MXU currently admits only a 32x32 result tile")
    for operand, label in ((plan.lhs, "lhs"), (plan.rhs, "rhs")):
        _require(operand.type.kind == "tensor" and operand.type.shape == (32, 32),
                 f"Atlas {label} must be a static 32x32 tensor")
        _require("f8e4m3fn" in operand.type.elem.lower(),
                 f"Atlas {label} must use the selected E4M3FN FP8 encoding")
    _require(abi.lhs_input >= 0 and abi.rhs_input >= 0 and abi.lhs_input != abi.rhs_input,
             "Atlas inputs must use distinct nonnegative tile indexes")
    _require(abi.mxu_unit in (0, 1), "Atlas MXU unit must be 0 or 1")
    _require(abi.output_index >= 0, "Atlas output index must be nonnegative")
    _require(abi.input_base >= 0x80000000 and abi.output_base >= 0x80000000 and
             abi.input_base % 2048 == 0 and abi.output_base % 2048 == 0,
             "Atlas DRAM bases must be 2 KiB-aligned addresses in the selected DRAM range")
    input_end = abi.input_base + (max(abi.lhs_input, abi.rhs_input) + 1) * 2048
    output_end = abi.output_base + 2048
    _require(input_end <= 1 << 32 and output_end <= 1 << 32 and
             (input_end <= abi.output_base or output_end <= abi.input_base),
             "Atlas input and output slots overlap or exceed the 32-bit DRAM range")

    return f'''module {{
  func.func @ttir_{plan.function}_atlas() -> !atlas.virtual_state attributes {{atlas.input_dram_base = {abi.input_base} : i64, atlas.output_dram_base = {abi.output_base} : i64}} {{
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %lhs = "atlas.virtual_input_fp8"(%io0) {{index = {abi.lhs_input} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2, %rhs = "atlas.virtual_input_fp8"(%io1) {{index = {abi.rhs_input} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %rhs_nk = "atlas.virtual_xlu_transpose"(%rhs) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8
    %dot = "atlas.virtual_mxu_matmul"(%lhs, %rhs_nk) {{unit = {abi.mxu_unit} : i32}} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %relu = "atlas.virtual_vpu_unary"(%dot) {{kind = "relu"}} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %io3 = "atlas.virtual_output_bf16"(%io2, %relu) {{index = {abi.output_index} : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io3 : !atlas.virtual_state
  }}
}}
'''
