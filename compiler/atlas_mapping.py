"""Atlas-specific mapping from a target-neutral TTIR operation plan."""
from __future__ import annotations

from dataclasses import dataclass

from compiler.triton_analysis import MatmulRelu, TwoLayerMlp


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


@dataclass(frozen=True)
class AtlasMlpABI:
    """Three source [K,N] operand tiles, with fixed MXU0 then MXU1 assignment."""

    input_slots: tuple[int, int, int] = (0, 1, 2)
    input_base: int = 0x90000000
    output_base: int = 0x90002000
    output_index: int = 0


MLP_NUMERICAL_POLICY = "atlas-fp8-mlp-v1"


def map_two_layer_mlp(plan: TwoLayerMlp, abi: AtlasMlpABI = AtlasMlpABI(), *, numerical_policy=None) -> str:
    """Apply an explicitly authorized target precision transform.

    MXU0 RTL arithmetic -> BF16 readout -> RTL BF16 ReLU -> FP8Pack at
    scale_code=127 (unity, RTL saturation/underflow) -> MXU1 RTL arithmetic
    -> BF16 output. This is NOT equivalence to standard TTIR FP32 dot/cast
    arithmetic. Input E4M3FN bytes are staged by the caller without conversion.
    """
    _require(isinstance(plan, TwoLayerMlp), "expected a recognized two-layer MLP")
    _require(numerical_policy == MLP_NUMERICAL_POLICY,
             f"two-layer MLP requires numerical policy {MLP_NUMERICAL_POLICY}")
    slots = abi.input_slots
    _require(len(slots) == 3 and all(type(i) is int and 0 <= i < 64 for i in slots) and len(set(slots)) == 3,
             "MLP inputs require three distinct slot indexes below 64")
    _require(type(abi.output_index) is int and 0 <= abi.output_index < 64, "invalid output index")
    _require(all(type(a) is int and a >= 0x80000000 and a % 2048 == 0
                 for a in (abi.input_base, abi.output_base)), "DRAM bases must be 2 KiB-aligned DRAM addresses")
    input_end = abi.input_base + (max(slots) + 1) * 2048
    output_end = abi.output_base + (abi.output_index + 1) * 2048
    _require(input_end <= 1 << 32 and output_end <= 1 << 32 and
             (input_end <= abi.output_base or output_end <= abi.input_base), "MLP slots overlap or exceed 32-bit DRAM")
    for value in plan.inputs:
        _require(value.type.kind == "tensor" and value.type.shape == (32, 32) and
                 value.type.elem.lower() == "f8e4m3fn" and not value.type.elem_is_ptr,
                 "MLP inputs must be 32x32 E4M3FN tensors")
    lines = ["module {", f'  func.func @ttir_{plan.function}_atlas() -> !atlas.virtual_state attributes {{atlas.input_dram_base = {abi.input_base} : i64, atlas.output_dram_base = {abi.output_base} : i64}} {{',
             '    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state']
    for i, (name, slot) in enumerate(zip(("x", "b0", "b1"), slots)):
        lines.append(f'    %io{i+1}, %{name} = "atlas.virtual_input_fp8"(%io{i}) {{index = {slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)')
    for i in (0, 1):
        lines.append(f'    %w{i} = "atlas.virtual_xlu_transpose"(%b{i}) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8')
    lines += [
        '    %h0 = "atlas.virtual_mxu_matmul"(%x, %w0) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16',
        '    %h1 = "atlas.virtual_vpu_unary"(%h0) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16',
        '    %h2 = "atlas.virtual_pack_fp8"(%h1) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8',
        '    %y = "atlas.virtual_mxu_matmul"(%h2, %w1) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16',
        f'    %io4 = "atlas.virtual_output_bf16"(%io3, %y) {{index = {abi.output_index} : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state',
        '    return %io4 : !atlas.virtual_state', '  }', '}', '']
    return "\n".join(lines)
