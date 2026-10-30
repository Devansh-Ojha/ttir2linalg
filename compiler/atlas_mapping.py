"""Atlas-specific mapping from a target-neutral TTIR operation plan."""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

from compiler.triton_analysis import (LinearRelu, MatmulRelu, MlpChain,
                                      ScaledDotProductAttention, TwoLayerMlp)
from compiler.target_mapping import TritonWorkloadPlan


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
    """Source X/[K,N] weight slots; MXUs alternate 0, 1, 0, ... in chain order."""

    input_slots: tuple[int, ...] = (0, 1, 2)
    input_base: int = 0x90000000
    output_base: int = 0x90002000
    output_index: int = 0


MLP_NUMERICAL_POLICY = "atlas-fp8-mlp-v1"
LINEAR_NUMERICAL_POLICY = "atlas-fp8-linear-v1"
ATTENTION_NUMERICAL_POLICY = "atlas-fp8-attention-v1"


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


def map_mlp_chain(plan: MlpChain, abi: AtlasMlpABI | None = None, *,
                  numerical_policy: str | None = None) -> str:
    """Emit a serial chain with just-in-time weights and alternating MXU arithmetic.

    Every hidden BF16 readout passes through RTL ReLU and FP8Pack(127).
    Unit reuse changes neither the reset/readout contract nor the arithmetic
    assigned to each layer: layer i always uses unit i % 2. The backend owns
    physical placement, waits and timing, and must still verify resource fit
    after scheduling (which may hoist weight loads).
    """
    _require(isinstance(plan, MlpChain), "expected a recognized MLP chain")
    depth = len(plan.layers)
    _require(2 <= depth < 64 and len(plan.weights) == depth,
             "MLP chain needs 2..63 layers (FP8 pack permits 64 input slots)")
    _require(numerical_policy == MLP_NUMERICAL_POLICY,
             f"MLP chain requires numerical policy {MLP_NUMERICAL_POLICY}")
    if abi is None:
        abi = AtlasMlpABI(input_slots=tuple(range(depth + 1)),
                          output_base=0x90000000 + max(4, depth + 1) * 2048)
    slots = abi.input_slots
    _require(len(slots) == depth + 1, "MLP slots must cover X and every weight")
    _require(type(abi.output_index) is int and 0 <= abi.output_index < 64,
             "invalid output index")
    # Match the backend's conservative input/output span and pack-scratch rules.
    _validate_tile_abi(slots, abi.input_base, abi.output_base)
    _require(abi.output_base + (abi.output_index + 1) * 2048 <= 1 << 32,
             "MLP output slots exceed 32-bit DRAM")
    input_end = abi.input_base + (max(slots) + 1) * 2048
    output_end = abi.output_base + (abi.output_index + 1) * 2048
    _require(input_end <= abi.output_base or output_end <= abi.input_base,
             "MLP input/output spans overlap")
    previous = plan.input_activation
    for i, layer in enumerate(plan.layers):
        _require(layer.weight is plan.weights[i] and len(layer.dot.operands) == 3 and
                 layer.dot.operands[0].id == previous.id and
                 layer.dot.operands[1].id == layer.weight.id,
                 "MLP layer dependencies do not match the analyzed plan")
        for value in (previous, layer.weight):
            _require(value.type.kind == "tensor" and value.type.shape == (32, 32) and
                     value.type.elem.lower() == "f8e4m3fn" and not value.type.elem_is_ptr,
                     "MLP inputs must be 32x32 E4M3FN tensors")
        _require(len(layer.bf16_cast.operands) == 1 and
                 layer.bf16_cast.operands[0].id == layer.dot.results[0].id,
                 "MLP BF16 readout dependency is invalid")
        if i < depth - 1:
            _require(layer.relu is not None and layer.pack is not None and
                     any(v.id == layer.bf16_cast.results[0].id for v in layer.relu.operands) and
                     len(layer.pack.operands) == 1 and
                     layer.pack.operands[0].id == layer.relu.results[0].id and
                     layer.pack.attrs == {"rounding": "rtne"},
                     "hidden MLP layer requires BF16 ReLU and RTNE FP8 conversion")
            previous = layer.pack.results[0]
        else:
            _require(layer.relu is None and layer.pack is None and
                     plan.store.operands[1].id == layer.bf16_cast.results[0].id,
                     "final MLP layer must store its BF16 output")
    if depth == 2:
        first, final = plan.layers
        legacy = TwoLayerMlp(plan.function, (plan.input_activation,) + plan.weights,
                             (first.dot, final.dot), (first.bf16_cast, final.bf16_cast),
                             first.relu, first.pack, plan.store)
        return map_two_layer_mlp(legacy, abi, numerical_policy=numerical_policy)

    lines = ["module {",
             f'  func.func @ttir_{plan.function}_atlas() -> !atlas.virtual_state attributes {{atlas.input_dram_base = {abi.input_base} : i64, atlas.output_dram_base = {abi.output_base} : i64}} {{',
             '    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state',
             f'    %io1, %x = "atlas.virtual_input_fp8"(%io0) {{index = {slots[0]} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)']
    activation = "x"
    for i, layer in enumerate(plan.layers):
        # Load weights just before use; no unbounded bank of resident weights.
        lines += [
            f'    %io{i+2}, %b{i} = "atlas.virtual_input_fp8"(%io{i+1}) {{index = {slots[i+1]} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)',
            f'    %w{i} = "atlas.virtual_xlu_transpose"(%b{i}) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8',
            f'    %d{i} = "atlas.virtual_mxu_matmul"(%{activation}, %w{i}) {{unit = {i % 2} : i32}} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16',
        ]
        if layer.pack is not None:
            lines += [
                f'    %r{i} = "atlas.virtual_vpu_unary"(%d{i}) {{kind = "relu"}} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16',
                f'    %h{i} = "atlas.virtual_pack_fp8"(%r{i}) {{scale_code = 127 : i32}} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8',
            ]
            activation = f"h{i}"
    lines += [
        f'    %done = "atlas.virtual_output_bf16"(%io{depth+1}, %d{depth-1}) {{index = {abi.output_index} : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state',
        '    return %done : !atlas.virtual_state', '  }', '}', '']
    return "\n".join(lines)


def _validate_tile_abi(slots: tuple[int, ...], input_base: int, output_base: int) -> None:
    _require(bool(slots) and all(type(i) is int and 0 <= i < 64 for i in slots) and
             len(set(slots)) == len(slots), "input slots must be distinct indexes below 64")
    _require(all(type(a) is int and a >= 0x80000000 and a % 2048 == 0
                 for a in (input_base, output_base)), "DRAM bases must be 2 KiB-aligned DRAM addresses")
    input_end = input_base + (max(slots) + 1) * 2048
    output_end = output_base + 2048
    _require(input_end <= 1 << 32 and output_end <= 1 << 32 and
             (input_end <= output_base or output_end <= input_base),
             "input and output tile slots overlap or exceed 32-bit DRAM")


@dataclass(frozen=True)
class AtlasLinearReluABI:
    """Tile slots for the per-output FC + bias + ReLU sample kernel."""

    x_slot: int = 0
    weight_slot: int = 1
    bias_slot: int = 2
    output_base: int = 0x90002000
    input_base: int = 0x90000000
    output_index: int = 0
    output_features: int = 8


def map_linear_relu(plan: LinearRelu, abi: AtlasLinearReluABI = AtlasLinearReluABI(), *, numerical_policy=None) -> str:
    """Map one source vector projection into the first row of a padded tile.

    The caller stages the activation and [K,N] weights as padded FP8 tiles,
    and the bias as a BF16 tile whose first row contains the output bias. The
    explicit precision policy records the source FP32 to Atlas FP8/BF16 change.
    """
    _require(numerical_policy == LINEAR_NUMERICAL_POLICY,
             f"single-layer MLP requires numerical policy {LINEAR_NUMERICAL_POLICY}")
    _require(0 < plan.input_features <= 32 and 0 < abi.output_features <= 32,
             "single-layer MLP dimensions must fit one padded 32x32 tile")
    _require(type(abi.output_index) is int and 0 <= abi.output_index < 64,
             "invalid output index")
    slots = (abi.x_slot, abi.weight_slot, abi.bias_slot)
    _validate_tile_abi(slots, abi.input_base, abi.output_base)
    return f'''module {{
  func.func @ttir_{plan.function}_atlas() -> !atlas.virtual_state attributes {{atlas.input_dram_base = {abi.input_base} : i64, atlas.output_dram_base = {abi.output_base} : i64}} {{
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %x = "atlas.virtual_input_fp8"(%io0) {{index = {abi.x_slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2, %weight = "atlas.virtual_input_fp8"(%io1) {{index = {abi.weight_slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3, %bias = "atlas.virtual_input_bf16"(%io2) {{index = {abi.bias_slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %weight_nk = "atlas.virtual_xlu_transpose"(%weight) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8
    %dot = "atlas.virtual_mxu_matmul"(%x, %weight_nk) {{unit = 0 : i32}} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %biased = "atlas.virtual_vpu_binary"(%dot, %bias) {{kind = "add"}} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %relu = "atlas.virtual_vpu_unary"(%biased) {{kind = "relu"}} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %io4 = "atlas.virtual_output_bf16"(%io3, %relu) {{index = {abi.output_index} : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io4 : !atlas.virtual_state
  }}
}}
'''


@dataclass(frozen=True)
class AtlasAttentionABI:
    """FP8 Q/K/V and BF16 additive-mask tile slots for one attention query."""

    input_slots: tuple[int, int, int, int] = (0, 1, 2, 3)
    input_base: int = 0x90000000
    output_base: int = 0x90004000
    output_index: int = 0


def _bf16_bits(value: float) -> int:
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((raw + 0x7FFF + ((raw >> 16) & 1)) >> 16) & 0xFFFF


def map_attention(plan: ScaledDotProductAttention, abi: AtlasAttentionABI = AtlasAttentionABI(), *, numerical_policy=None) -> str:
    """Map one query's scaled dot-product attention over a padded 32x32 tile.

    The BF16 mask input is 0 for valid keys and a sufficiently negative value
    for padded keys; this prevents padding from changing the softmax. Q/K/V
    tiles are staged in FP8. Softmax probabilities are packed to FP8 before
    the second MXU operation, matching the existing Atlas attention example.
    """
    _require(numerical_policy == ATTENTION_NUMERICAL_POLICY,
             f"attention requires numerical policy {ATTENTION_NUMERICAL_POLICY}")
    _require(plan.sequence_length <= 32 and plan.embedding_size <= 32,
             "attention dimensions must fit one padded 32x32 tile")
    _require(len(abi.input_slots) == 4 and type(abi.output_index) is int and
             0 <= abi.output_index < 64, "attention ABI requires four slots and a valid output index")
    _validate_tile_abi(abi.input_slots, abi.input_base, abi.output_base)
    q_slot, k_slot, v_slot, mask_slot = abi.input_slots
    scale = _bf16_bits(math.log2(math.e) / math.sqrt(plan.embedding_size))
    lines = [
        "module {",
        f'  func.func @ttir_{plan.function}_atlas() -> !atlas.virtual_state attributes {{atlas.input_dram_base = {abi.input_base} : i64, atlas.output_dram_base = {abi.output_base} : i64}} {{',
        '    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state',
        f'    %io1, %q = "atlas.virtual_input_fp8"(%io0) {{index = {q_slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)',
        f'    %io2, %k = "atlas.virtual_input_fp8"(%io1) {{index = {k_slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)',
        f'    %io3, %v = "atlas.virtual_input_fp8"(%io2) {{index = {v_slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)',
        f'    %io4, %mask = "atlas.virtual_input_bf16"(%io3) {{index = {mask_slot} : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)',
        '    %scores0 = "atlas.virtual_mxu_matmul"(%q, %k) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16',
        '    %scores = "atlas.virtual_vpu_binary"(%scores0, %mask) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16',
        '    %max = "atlas.virtual_vpu_reduce"(%scores) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16',
        '    %shifted = "atlas.virtual_vpu_binary"(%scores, %max) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16',
        f'    %scale = "atlas.virtual_vpu_splat"() {{bits = {scale} : i32}} : () -> !atlas.virtual_bf16',
        '    %scaled = "atlas.virtual_vpu_binary"(%shifted, %scale) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16',
        '    %e = "atlas.virtual_vpu_unary"(%scaled) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16',
        '    %sum = "atlas.virtual_vpu_reduce"(%e) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16',
        '    %inverse = "atlas.virtual_vpu_unary"(%sum) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16',
        '    %p = "atlas.virtual_vpu_binary"(%e, %inverse) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16',
        '    %p8 = "atlas.virtual_pack_fp8"(%p) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8',
        '    %v_t = "atlas.virtual_xlu_transpose"(%v) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8',
        '    %out = "atlas.virtual_mxu_matmul"(%p8, %v_t) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16',
        f'    %io5 = "atlas.virtual_output_bf16"(%io4, %out) {{index = {abi.output_index} : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state',
        '    return %io5 : !atlas.virtual_state',
        '  }', '}', '']
    return "\n".join(lines)


@dataclass(frozen=True)
class AtlasTargetMapper:
    """Atlas backend for the target-neutral workload plans in this prototype."""

    matmul_abi: AtlasTileABI = AtlasTileABI()
    mlp_abi: AtlasMlpABI | None = None
    linear_abi: AtlasLinearReluABI = AtlasLinearReluABI()
    attention_abi: AtlasAttentionABI = AtlasAttentionABI()
    numerical_policy: str | None = None

    def map(self, plan: TritonWorkloadPlan) -> str:
        if isinstance(plan, MatmulRelu):
            return map_matmul_relu(plan, self.matmul_abi)
        if isinstance(plan, TwoLayerMlp):
            return map_two_layer_mlp(plan, self.mlp_abi or AtlasMlpABI(),
                                     numerical_policy=self.numerical_policy)
        if isinstance(plan, MlpChain):
            return map_mlp_chain(plan, self.mlp_abi, numerical_policy=self.numerical_policy)
        if isinstance(plan, LinearRelu):
            return map_linear_relu(plan, self.linear_abi,
                                   numerical_policy=self.numerical_policy)
        if isinstance(plan, ScaledDotProductAttention):
            return map_attention(plan, self.attention_abi,
                                 numerical_policy=self.numerical_policy)
        raise TypeError(f"Atlas mapper does not support plan {type(plan).__name__}")


@dataclass(frozen=True)
class AtlasMatmulReluMapper:
    """Compatibility adapter for callers of the original mapper interface."""

    abi: AtlasTileABI = AtlasTileABI()

    def map_matmul_relu(self, plan: MatmulRelu) -> str:
        return map_matmul_relu(plan, self.abi)
