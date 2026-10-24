"""Atlas virtual-SSA mapping for supported PyTorch-exported dense HGraphs."""
from __future__ import annotations

from dataclasses import dataclass
import re

from compiler.hgraph_analysis import HGraphDensePlan


PYTORCH_DENSE_NUMERICAL_POLICY = "atlas-fp8-pytorch-dense-v1"


@dataclass(frozen=True)
class AtlasHGraphDenseABI:
    """Caller-staged tile slots for one or two biased dense layers."""

    input_slots: tuple[int, ...] = (0, 1, 2, 3, 4)
    input_base: int = 0x90000000
    output_base: int = 0x90004000
    output_index: int = 0


def _validate(plan: HGraphDensePlan, abi: AtlasHGraphDenseABI,
              numerical_policy: str | None) -> tuple[str, ...]:
    if numerical_policy != PYTORCH_DENSE_NUMERICAL_POLICY:
        raise NotImplementedError(
            "PyTorch dense mapping requires numerical policy "
            + PYTORCH_DENSE_NUMERICAL_POLICY
        )
    if plan.pattern not in {"linear_relu", "two_layer_mlp"}:
        raise NotImplementedError("unsupported HGraph dense pattern %r" % plan.pattern)
    expected_layers = 1 if plan.pattern == "linear_relu" else 2
    if len(plan.layers) != expected_layers:
        raise NotImplementedError("dense plan layer count does not match its pattern")
    if plan.activation != "relu":
        raise NotImplementedError("Atlas dense mapper currently requires a ReLU activation")
    expected_inputs = 3 if plan.pattern == "linear_relu" else 5
    slots = abi.input_slots[:expected_inputs]
    if len(abi.input_slots) < expected_inputs or len(set(slots)) != expected_inputs:
        raise NotImplementedError("dense input slots must be distinct and cover every operand")
    if any(type(slot) is not int or not 0 <= slot < 64 for slot in slots):
        raise NotImplementedError("dense input slot indexes must be in [0, 63]")
    if type(abi.output_index) is not int or not 0 <= abi.output_index < 64:
        raise NotImplementedError("output index must be in [0, 63]")
    for base in (abi.input_base, abi.output_base):
        if type(base) is not int or base < 0x80000000 or base % 2048:
            raise NotImplementedError("DRAM bases must be 2 KiB-aligned Atlas DRAM addresses")
    input_end = abi.input_base + (max(slots) + 1) * 2048
    output_start = abi.output_base + abi.output_index * 2048
    output_end = output_start + 2048
    if (input_end > 1 << 32 or output_end > 1 << 32 or
            not (input_end <= output_start or output_end <= abi.input_base)):
        raise NotImplementedError("dense input/output slots overlap or exceed Atlas DRAM")
    for layer in plan.layers:
        if any(dim <= 0 or dim > 32 for dim in (
            layer.batch, layer.input_features, layer.output_features
        )):
            raise NotImplementedError("Atlas dense tile dimensions must be in [1, 32]")
    if len(plan.layers) == 2:
        first, second = plan.layers
        if first.output_features != second.input_features:
            raise NotImplementedError("two-layer MLP hidden dimensions do not match")
    return slots


def map_hgraph_dense(plan: HGraphDensePlan, abi: AtlasHGraphDenseABI = AtlasHGraphDenseABI(),
                     *, numerical_policy: str | None = None) -> str:
    """Map a strict HGraph dense plan to existing Atlas virtual operations.

    The caller stages activations and transposed K×N weight tiles as E4M3FN,
    and bias tiles as BF16, padded to 32×32. The first layer's BF16 output is
    biased and activated before `virtual_pack_fp8`; the second layer's bias is
    added before BF16 output. This mapping is lossy and requires the named
    numerical policy; it does not inherit source FP32 equivalence.
    """
    slots = _validate(plan, abi, numerical_policy)
    symbol = re.sub(r"[^A-Za-z0-9_$]", "_", plan.graph_name)
    lines = [
        "// Source: torch.export HGraph; not a lowering of Inductor kernel code.",
        "// Numerical policy: %s; caller stages padded/quantized Atlas tiles."
        % PYTORCH_DENSE_NUMERICAL_POLICY,
        "module {",
        f'  func.func @hgraph_{symbol}_atlas() -> !atlas.virtual_state attributes '
        f'{{atlas.input_dram_base = {abi.input_base} : i64, '
        f'atlas.output_dram_base = {abi.output_base} : i64}} {{',
        '    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state',
    ]

    if plan.pattern == "linear_relu":
        x_slot, w_slot, b_slot = slots
        lines.extend([
            f'    %io1, %x = "atlas.virtual_input_fp8"(%io0) {{index = {x_slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)',
            f'    %io2, %w = "atlas.virtual_input_fp8"(%io1) {{index = {w_slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)',
            f'    %io3, %b = "atlas.virtual_input_bf16"(%io2) {{index = {b_slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)',
            '    %w_nk = "atlas.virtual_xlu_transpose"(%w) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8',
            '    %dot = "atlas.virtual_mxu_matmul"(%x, %w_nk) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16',
            '    %biased = "atlas.virtual_vpu_binary"(%dot, %b) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16',
            '    %relu = "atlas.virtual_vpu_unary"(%biased) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16',
            f'    %io4 = "atlas.virtual_output_bf16"(%io3, %relu) {{index = {abi.output_index} : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state',
            '    return %io4 : !atlas.virtual_state',
        ])
    elif plan.pattern == "two_layer_mlp":
        x_slot, w0_slot, b0_slot, w1_slot, b1_slot = slots
        lines.extend([
            f'    %io1, %x = "atlas.virtual_input_fp8"(%io0) {{index = {x_slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)',
            f'    %io2, %w0 = "atlas.virtual_input_fp8"(%io1) {{index = {w0_slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)',
            f'    %io3, %b0 = "atlas.virtual_input_bf16"(%io2) {{index = {b0_slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)',
            f'    %io4, %w1 = "atlas.virtual_input_fp8"(%io3) {{index = {w1_slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)',
            f'    %io5, %b1 = "atlas.virtual_input_bf16"(%io4) {{index = {b1_slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)',
            '    %w0_nk = "atlas.virtual_xlu_transpose"(%w0) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8',
            '    %w1_nk = "atlas.virtual_xlu_transpose"(%w1) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8',
            '    %h0 = "atlas.virtual_mxu_matmul"(%x, %w0_nk) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16',
            '    %h0b = "atlas.virtual_vpu_binary"(%h0, %b0) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16',
            '    %h1 = "atlas.virtual_vpu_unary"(%h0b) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16',
            '    %h1_fp8 = "atlas.virtual_pack_fp8"(%h1) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8',
            '    %h2 = "atlas.virtual_mxu_matmul"(%h1_fp8, %w1_nk) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16',
            '    %y = "atlas.virtual_vpu_binary"(%h2, %b1) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16',
            f'    %io6 = "atlas.virtual_output_bf16"(%io5, %y) {{index = {abi.output_index} : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state',
            '    return %io6 : !atlas.virtual_state',
        ])
    else:
        raise NotImplementedError("Atlas HGraph dense mapper received %r" % plan.pattern)

    lines.extend(['  }', '}', ''])
    return "\n".join(lines)
