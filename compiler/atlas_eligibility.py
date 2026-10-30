"""Whole-graph Atlas capability checks; no MLIR emission or resource allocation.

The named policy explicitly authorizes target arithmetic, not IEEE equivalence.
An ABI declaration is a caller contract about tensor-of-pointer arguments; their
contiguity/disjointness cannot be inferred from SSA tensor shape alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple

from compiler.ttir_model import TOp, TValue
from compiler.ttir_operation_analysis import OperationAnalysis, UnsupportedKernel, analyze_operations

BF16_RTL_POLICY = "atlas-bf16-elementwise-rtl-v1"


@dataclass(frozen=True)
class TileBinding:
    argument: int
    role: str  # input or output
    slot: int


@dataclass(frozen=True)
class BF16TileABI:
    bindings: Tuple[TileBinding, ...]
    layout: str = "row-major-contiguous"
    disjoint: bool = True
    launch: str = "single-program"
    boundary_layout: str = "atlas-bf16-column-halves"
    alignment: int = 2048


@dataclass(frozen=True)
class FP8TileABI(BF16TileABI):
    boundary_layout: str = "atlas-fp8-row-major-bf16-column-halves"


@dataclass(frozen=True)
class FP8LayoutABI(BF16TileABI):
    """Raw FP8 tiles at both boundaries; slots reserve 2048 bytes each."""
    boundary_layout: str = "atlas-fp8-row-major"


@dataclass(frozen=True)
class MixedTileABI(BF16TileABI):
    """Per-argument BF16/FP8 formats, with explicit staging and 2048-byte slots."""
    boundary_layout: str = "atlas-mixed-fp8-row-major-bf16-column-halves"
    formats: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    operation_index: Optional[int] = None
    operation_name: Optional[str] = None


@dataclass(frozen=True)
class EligibleOperation:
    source: TOp
    candidate: str
    inputs: Tuple[TValue, ...]
    constant_bits: Optional[int] = None
    binding: Optional[TileBinding] = None
    unit: Optional[int] = None
    scale_code: Optional[int] = None
    result_layout: Optional[str] = None
    precision_mode: Optional[str] = None


@dataclass(frozen=True)
class EligibilityPlan:
    analysis: OperationAnalysis
    numerical_policy: str
    abi: BF16TileABI
    operations: Tuple[EligibleOperation, ...]


@dataclass(frozen=True)
class EligibilityResult:
    diagnostics: Tuple[Diagnostic, ...]
    plan: Optional[EligibilityPlan] = None

    @property
    def eligible(self):
        return self.plan is not None and not self.diagnostics


def _tile(t, pointer=False):
    return t.kind == "tensor" and t.shape == (32, 32) and t.elem == "bf16" and t.elem_is_ptr == pointer


def _finite_bf16(code):
    exponent, mantissa = code >> 7, code & 127
    if exponent == 0:
        return Fraction(mantissa, 1 << 133)
    shift = exponent - 134
    return Fraction((128 + mantissa) * (1 << max(shift, 0)), 1 << max(-shift, 0))


def _constant_bits(op):
    """Exact decimal->BF16 RNE, without FP32 double rounding or torch imports.

    Initially only finite decimal literals are qualified. Hex/NaN/Inf constants
    are rejected (runtime BF16 inputs may contain every bit pattern).
    """
    text = op.attrs['value']
    if text.startswith('dense<') and text.endswith('>'):
        text = text[6:-1]
    if text.lower().startswith('0x'):
        raise ValueError('hexadecimal constants are not qualified')
    value = Fraction(text)
    sign = 0x8000 if text.startswith('-') else 0
    value = abs(value)
    lo, hi = 0, 0x7f7f
    if value > _finite_bf16(hi):
        raise ValueError('constant outside finite BF16 range')
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _finite_bf16(mid) <= value:
            lo = mid
        else:
            hi = mid - 1
    if lo < 0x7f7f:
        down, up = value - _finite_bf16(lo), _finite_bf16(lo + 1) - value
        if up < down or (up == down and lo & 1):
            lo += 1
    return sign | lo


def analyze_atlas_eligibility(analysis: OperationAnalysis, *, numerical_policy=None,
                              abi: Optional[BF16TileABI] = None) -> EligibilityResult:
    """Return a complete plan only when every operation and boundary qualifies.

    Plans retain source SSA dependencies and target operation candidates only;
    instruction count, registers, addresses, schedules and machine feasibility
    remain the responsibility of future emission/backend validation.
    """
    if numerical_policy == 'atlas-fp8-layout-v1':
        from compiler.atlas_layout_operations import analyze_layout_operations
        return analyze_layout_operations(analysis, numerical_policy=numerical_policy, abi=abi)
    if numerical_policy in {'atlas-bf16-rowwise-rtl-v1', 'atlas-fp8-attention-v1'}:
        from compiler.atlas_rowwise_operations import analyze_rowwise_operations
        return analyze_rowwise_operations(analysis, numerical_policy=numerical_policy, abi=abi)
    if isinstance(abi, FP8TileABI):
        from compiler.atlas_fp8_operations import analyze_fp8_operations
        return analyze_fp8_operations(analysis, numerical_policy=numerical_policy, abi=abi)
    diagnostics = []
    def reject(code, message, op=None):
        diagnostics.append(Diagnostic(code, message, op.index if op else None, op.name if op else None))

    if numerical_policy != BF16_RTL_POLICY:
        reject('numerical-policy', f'explicit authorization of {BF16_RTL_POLICY} is required; IEEE equivalence is not claimed')
    if not isinstance(analysis, OperationAnalysis):
        reject('invalid-analysis', 'expected OperationAnalysis')
        return EligibilityResult(tuple(diagnostics))
    # Never trust a hand-picked/stale list that hides unsupported source ops.
    try:
        fresh = analyze_operations(analysis.function)
    except (UnsupportedKernel, KeyError, ValueError, IndexError) as exc:
        reject('invalid-analysis', str(exc))
        return EligibilityResult(tuple(diagnostics))
    def edges(a):
        return tuple(tuple(id(v) for v in deps) for deps in a.dependencies)
    def accesses(a):
        return tuple(tuple(id(x) if x is not None else None for x in
                           (m.operation, m.pointer, m.value, m.mask, m.other)) for m in a.memory)
    if (tuple(map(id, fresh.operations)) != tuple(map(id, analysis.operations)) or
            edges(fresh) != edges(analysis) or accesses(fresh) != accesses(analysis)):
        reject('invalid-analysis', 'analysis does not cover the current complete graph and its dependencies/memory accesses')
        return EligibilityResult(tuple(diagnostics))

    args = analysis.function.args
    bindings = {}
    if not isinstance(abi, BF16TileABI):
        reject('memory-abi', 'explicit BF16TileABI required for contiguous disjoint 2048-byte tiles and a single-program launch')
    else:
        if (abi.layout != 'row-major-contiguous' or abi.disjoint is not True or
                abi.launch != 'single-program' or abi.boundary_layout != 'atlas-bf16-column-halves' or
                type(abi.alignment) is not int or abi.alignment != 2048):
            reject('memory-abi', 'requires logical row-major disjoint single-program tiles, explicitly staged as Atlas BF16 column halves with 2048-byte-aligned slots')
        slots = set()
        entries = abi.bindings
        if not isinstance(entries, tuple):
            reject('memory-abi', 'bindings must be a tuple of TileBinding entries')
            entries = ()
        for binding in entries:
            if not isinstance(binding, TileBinding):
                reject('memory-abi', 'expected TileBinding entries')
                continue
            if (type(binding.argument) is not int or not 0 <= binding.argument < len(args) or
                    binding.argument in bindings or not isinstance(binding.role, str) or binding.role not in {'input', 'output'} or
                    type(binding.slot) is not int or not 0 <= binding.slot < 64 or
                    (binding.role, binding.slot) in slots):
                reject('memory-abi', f'invalid, duplicate or out-of-range binding: {binding}')
                continue
            bindings[binding.argument] = binding
            slots.add((binding.role, binding.slot))
        if set(bindings) != set(range(len(args))):
            reject('memory-abi', 'every argument must have exactly one input/output binding')
    arg_indexes = {id(v): i for i, v in enumerate(args)}
    for i, arg in enumerate(args):
        if not _tile(arg.type, pointer=True):
            reject('boundary-type', f'argument {i} must be tensor<32x32x!tt.ptr<bf16>>; dynamic scalar pointers and index arguments are unsupported')

    operations, constants, used_bindings, writes = [], {}, set(), set()
    for op in analysis.operations:
        name = op.name
        inputs = tuple(op.operands)
        candidate, bits, binding = None, None, None
        if op.regions:
            reject('unsupported-operation', 'regions/reductions are not in the initial Atlas subset', op)
            continue
        if name in {'tt.load', 'tt.store'}:
            pointer = op.operands[0]
            i = arg_indexes.get(id(pointer))
            binding = bindings.get(i)
            value = op.results[0] if name == 'tt.load' else op.operands[1]
            if len(op.operands) != (1 if name == 'tt.load' else 2):
                reject('masked-memory', 'masks and masked-load other values are unsupported, including constant masks', op)
            if i is None or not _tile(pointer.type, True) or not _tile(value.type):
                reject('memory-access', 'memory must reference a directly bound full BF16 tile argument; gather/scatter/index calculations are unsupported', op)
            if binding is None or binding.role != ('input' if name == 'tt.load' else 'output'):
                reject('memory-role', 'loads require input bindings and stores require output bindings', op)
            else:
                used_bindings.add(i)
                if name == 'tt.store':
                    if i in writes:
                        reject('memory-access', 'each output binding may be stored only once', op)
                    writes.add(i)
            candidate = 'atlas.virtual_input_bf16' if name == 'tt.load' else 'atlas.virtual_output_bf16'
        elif name == 'arith.constant':
            t = op.results[0].type
            if not (_tile(t) or (t.kind == 'scalar' and t.elem == 'bf16')):
                reject('constant-type', 'only BF16 scalar constants or 32x32 splats are qualified', op)
            else:
                try:
                    bits = _constant_bits(op)
                    constants[id(op.results[0])] = bits
                    candidate = 'constant-scalar' if t.kind == 'scalar' else 'atlas.virtual_vpu_splat'
                except (ValueError, OverflowError) as exc:
                    reject('constant-value', str(exc), op)
        elif name == 'tt.splat':
            bits = constants.get(id(op.operands[0]))
            if bits is None or not _tile(op.results[0].type) or op.operands[0].type.kind != 'scalar':
                reject('unsupported-splat', 'only a BF16 scalar constant splatted to a full tile is qualified', op)
            else:
                constants[id(op.results[0])] = bits
                candidate = 'atlas.virtual_vpu_splat'
        elif name in {'arith.addf', 'arith.subf', 'arith.mulf'}:
            if not all(_tile(v.type) for v in op.operands + op.results):
                reject('arithmetic-type', 'arithmetic requires 32x32 BF16 tensors; FP16/FP32 conversion is never implicit', op)
            else:
                candidate = 'atlas.virtual_vpu_binary:' + {'arith.addf': 'add', 'arith.subf': 'sub', 'arith.mulf': 'mul'}[name]
        elif name == 'arith.maxnumf':
            if not all(_tile(v.type) for v in op.operands + op.results):
                reject('arithmetic-type', 'ReLU requires full BF16 tiles', op)
            else:
                zeros = [i for i, v in enumerate(op.operands) if constants.get(id(v)) == 0]
                if not zeros:
                    reject('relu-semantics', 'only maxnum with a proven positive-zero constant is authorized as RTL sign-clamp ReLU', op)
                else:
                    inputs = (op.operands[1 - zeros[0]],)
                    candidate = 'atlas.virtual_vpu_unary:relu'
        elif name == 'tt.return':
            candidate = 'return'
        elif name in {'tt.addptr', 'tt.make_range', 'tt.get_program_id', 'arith.addi', 'arith.muli', 'arith.divsi', 'arith.remsi', 'arith.cmpi'}:
            reject('address-computation', 'index/pointer calculations are not tensor VPU computations and have no qualified memory lowering', op)
        elif name in {'arith.truncf', 'arith.extf', 'tt.fp_to_fp'}:
            reject('unsupported-conversion', 'no conversion is qualified by the BF16 elementwise policy', op)
        else:
            reject('unsupported-operation', f'{name} has no mapping in the initial Atlas eligibility subset', op)
        if candidate:
            operations.append(EligibleOperation(op, candidate, inputs, bits, binding))
    if not writes:
        reject('output-binding', 'at least one full-tile output store is required')
    if set(bindings) - used_bindings:
        reject('memory-abi', 'all declared bindings must be accessed in their declared roles')
    if diagnostics:
        return EligibilityResult(tuple(diagnostics))
    return EligibilityResult((), EligibilityPlan(analysis, numerical_policy, abi, tuple(operations)))
