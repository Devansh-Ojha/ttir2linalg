"""Operation-driven Atlas virtual emission for fully qualified BF16 graphs."""
from __future__ import annotations

from dataclasses import dataclass
from compiler.atlas_eligibility import BF16TileABI, analyze_atlas_eligibility
from compiler.ttir_operation_analysis import OperationAnalysis


@dataclass(frozen=True)
class AtlasOperationMemory:
    input_base: int = 0x90000000
    output_base: int = 0x90020000


def _validate_memory(abi, memory):
    spans = []
    for role, base in [('input', memory.input_base), ('output', memory.output_base)]:
        if type(base) is not int or not 0x80000000 <= base < 1 << 32 or base % 2048:
            raise ValueError(f'{role} DRAM base must be 2048-byte-aligned in 32-bit DRAM')
        slots = [b.slot for b in abi.bindings if b.role == role]
        if slots:
            end = base + (max(slots) + 1) * 2048
            if end > 1 << 32:
                raise ValueError(f'{role} DRAM span exceeds 32-bit address range')
            spans.append((base, end))
    if len(spans) == 2 and max(spans[0][0], spans[1][0]) < min(spans[0][1], spans[1][1]):
        raise ValueError('input/output DRAM spans overlap, including reserved slot gaps')


def emit_atlas_operations(analysis: OperationAnalysis, *, numerical_policy=None,
                          abi: BF16TileABI, memory=AtlasOperationMemory()) -> str:
    """Revalidate the entire graph before constructing any output text.

    Logical row-major TTIR tiles must be staged in the ABI's column-half boundary
    encoding; this emitter does not reinterpret source pointer bytes as Atlas
    bytes. Backend verification/resource checks remain mandatory.
    """
    eligibility = analyze_atlas_eligibility(analysis, numerical_policy=numerical_policy, abi=abi)
    if not eligibility.eligible:
        raise NotImplementedError('; '.join(f'{d.code}: {d.operation_name or "kernel"}: {d.message}' for d in eligibility.diagnostics))
    if not isinstance(memory, AtlasOperationMemory):
        raise ValueError('expected AtlasOperationMemory')
    _validate_memory(abi, memory)
    plan = eligibility.plan
    tile = '!atlas.virtual_bf16'
    state = '!atlas.virtual_state'
    lines = ['module {', f'  func.func @ttir_{analysis.function.name}_atlas_ops() -> {state} attributes {{atlas.input_dram_base = {memory.input_base} : i64, atlas.output_dram_base = {memory.output_base} : i64}} {{',
             f'    %io0 = "atlas.virtual_start"() : () -> {state}']
    values = {}  # TValue.id -> emitted Atlas SSA name (never printed TTIR names)
    io = 0
    serial = 0
    for entry in plan.operations:
        op, candidate = entry.source, entry.candidate
        if candidate in {'constant-scalar', 'zero-accumulator'}:
            continue  # Eligibility resolved the literal used by each tile splat.
        if candidate in {'alias-bf16', 'alias-value'}:
            values[op.results[0].id] = values[op.operands[0].id]
            continue
        if candidate == 'return':
            lines.append(f'    return %io{io} : {state}')
            continue
        if candidate == 'atlas.virtual_dma_store_fp8':
            address = memory.output_base + entry.binding.slot * 2048
            lines.append(f'    %out_addr{io} = arith.constant {address - (1 << 32)} : i32')
            lines.append(f'    %out_size{io} = arith.constant 1024 : i32')
            lines.append(f'    %io{io+1} = "atlas.virtual_dma_store_fp8"(%io{io}, {values[op.operands[1].id]}, %out_addr{io}, %out_size{io}) : ({state}, !atlas.virtual_fp8, i32, i32) -> {state}')
            io += 1
            continue
        if candidate == 'atlas.virtual_output_bf16':
            value = values[op.operands[1].id]
            lines.append(f'    %io{io+1} = "{candidate}"(%io{io}, {value}) {{index = {entry.binding.slot} : i32}} : ({state}, {tile}) -> {state}')
            io += 1
            continue
        result = f'%v{serial}'
        serial += 1
        if candidate in {'atlas.virtual_input_bf16', 'atlas.virtual_input_fp8'}:
            result_type = '!atlas.virtual_fp8' if candidate.endswith('fp8') else tile
            lines.append(f'    %io{io+1}, {result} = "{candidate}"(%io{io}) {{index = {entry.binding.slot} : i32}} : ({state}) -> ({state}, {result_type})')
            io += 1
        elif candidate == 'atlas.virtual_xlu_transpose':
            lines.append(f'    {result} = "{candidate}"({values[entry.inputs[0].id]}) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8')
        elif candidate == 'atlas.virtual_mxu_matmul':
            lhs, rhs = (values[v.id] for v in entry.inputs[:2])
            weight = result + '_weight'
            lines.append(f'    {weight} = "atlas.virtual_xlu_transpose"({rhs}) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8')
            lines.append(f'    {result} = "{candidate}"({lhs}, {weight}) {{unit = {entry.unit} : i32}} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> {tile}')
        elif candidate == 'atlas.virtual_pack_fp8':
            lines.append(f'    {result} = "{candidate}"({values[entry.inputs[0].id]}) {{scale_code = {entry.scale_code} : i32}} : ({tile}) -> !atlas.virtual_fp8')
        elif candidate == 'atlas.virtual_vpu_splat':
            lines.append(f'    {result} = "{candidate}"() {{bits = {entry.constant_bits} : i32}} : () -> {tile}')
        elif candidate == 'bf16-div-recip-mul':
            lhs, rhs = (values[v.id] for v in entry.inputs)
            reciprocal = result + '_recip'
            lines.append(f'    {reciprocal} = "atlas.virtual_vpu_unary"({rhs}) {{kind = "recip"}} : ({tile}) -> {tile}')
            lines.append(f'    {result} = "atlas.virtual_vpu_binary"({lhs}, {reciprocal}) {{kind = "mul"}} : ({tile}, {tile}) -> {tile}')
        elif candidate.startswith(('atlas.virtual_vpu_binary:', 'atlas.virtual_vpu_unary:', 'atlas.virtual_vpu_reduce:')):
            name, kind = candidate.split(':')
            inputs = ', '.join(values[v.id] for v in entry.inputs)
            types = ', '.join(tile for _ in entry.inputs)
            lines.append(f'    {result} = "{name}"({inputs}) {{kind = "{kind}"}} : ({types}) -> {tile}')
        else:
            raise NotImplementedError(f'no emitter for qualified candidate {candidate}')
        values[op.results[0].id] = result
    lines += ['  }', '}', '']
    return '\n'.join(lines)
