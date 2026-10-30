"""Conservative per-operation FP8 capability planning, no model recognition.

The existing atlas-fp8-mlp-v1 policy chooses MXU ordinal modulo two, BF16
readout, RTL ReLU, and unit-scale FP8 packing. It is not IEEE FP32 equivalence.
"""
from compiler.atlas_eligibility import (BF16TileABI, FP8TileABI, TileBinding, Diagnostic,
    EligibleOperation, EligibilityPlan, EligibilityResult, _constant_bits)
from compiler.ttir_operation_analysis import OperationAnalysis, analyze_operations, transpose_order


def analyze_fp8_operations(analysis, *, numerical_policy, abi):
    diagnostics=[];entries=[]
    def reject(code,msg,op=None):
        diagnostics.append(Diagnostic(code,msg,op.index if op else None,op.name if op else None))
    if numerical_policy!='atlas-fp8-mlp-v1':reject('numerical-policy','FP8 operations require atlas-fp8-mlp-v1')
    if not isinstance(analysis,OperationAnalysis):
        reject('invalid-analysis','expected OperationAnalysis');return EligibilityResult(tuple(diagnostics))
    try:fresh=analyze_operations(analysis.function,resolve_dot_defaults=analysis.resolved_dot_defaults)
    except (RuntimeError,KeyError,ValueError) as e:
        reject('invalid-analysis',str(e));return EligibilityResult(tuple(diagnostics))
    if ([id(o) for o in fresh.operations]!=[id(o) for o in analysis.operations] or
        [[id(v) for v in d] for d in fresh.dependencies]!=[[id(v) for v in d] for d in analysis.dependencies] or
        [[id(v) for v in (m.operation,m.pointer,m.value,m.mask,m.other)] for m in fresh.memory]!=[[id(v) for v in (m.operation,m.pointer,m.value,m.mask,m.other)] for m in analysis.memory]):
        reject('invalid-analysis','stale or partial graph');return EligibilityResult(tuple(diagnostics))
    if (abi.layout!='row-major-contiguous' or abi.disjoint is not True or abi.launch!='single-program' or
        abi.boundary_layout!='atlas-fp8-row-major-bf16-column-halves' or type(abi.alignment) is not int or abi.alignment!=2048):
        reject('memory-abi','FP8 requires disjoint single-program full tiles, explicit mixed staging and 2048-byte alignment')
    bindings={};slots=set();args=analysis.function.args
    entries_to_bind=abi.bindings if isinstance(abi.bindings,tuple) else ()
    if not isinstance(abi.bindings,tuple):reject('memory-abi','bindings must be a tuple')
    for b in entries_to_bind:
        if (not isinstance(b,TileBinding) or type(b.argument) is not int or not 0<=b.argument<len(args) or
            b.argument in bindings or b.role not in ('input','output') or type(b.slot) is not int or
            not 0<=b.slot<64 or (b.role,b.slot) in slots):
            reject('memory-abi','invalid or duplicate binding');continue
        bindings[b.argument]=b;slots.add((b.role,b.slot))
    if set(bindings)!=set(range(len(args))):reject('memory-abi','all arguments need explicit bindings')
    def tile(t,elem,ptr=False):return t.kind=='tensor' and t.shape==(32,32) and t.elem==elem and t.elem_is_ptr==ptr
    for i,v in enumerate(args):
        b=bindings.get(i)
        if not b or not tile(v.type,'f8E4M3FN' if b.role=='input' else 'bf16',True):
            reject('boundary-type','inputs must be full E4M3FN tiles and outputs full BF16 tiles')
    indexes={id(v):i for i,v in enumerate(args)};used=set();stored=set();zeros=set();bf16values=set();fp8values=set();dotcount=0
    users={}
    for op in analysis.operations:
        for v in op.operands:users.setdefault(v.id,[]).append(op)
    for op in analysis.operations:
        n=op.name;candidate=None;inputs=tuple(op.operands);bits=None;b=None;unit=None;scale=None
        if op.regions:reject('unsupported-operation','regions unsupported',op);continue
        if n in ('tt.load','tt.store'):
            idx=indexes.get(id(op.operands[0]));b=bindings.get(idx)
            if len(op.operands)!=(1 if n=='tt.load' else 2) or b is None or b.role!=('input' if n=='tt.load' else 'output'):
                reject('memory-access','only unmasked directly bound tiles are supported',op);continue
            used.add(idx)
            if n=='tt.load':
                candidate='atlas.virtual_input_fp8';fp8values.add(op.results[0].id)
            else:
                if idx in stored or op.operands[1].id not in bf16values:reject('memory-access','duplicate output or non-BF16-policy output',op)
                stored.add(idx);candidate='atlas.virtual_output_bf16'
        elif n=='arith.constant':
            if not tile(op.results[0].type,op.results[0].type.elem) or op.attrs.get('value') not in {'dense<0.000000e+00>','dense<0.0>'} or op.results[0].type.elem not in ('f32','bf16'):
                reject('constant-value','initial FP8 policy supports zero tile constants only',op);continue
            zeros.add(op.results[0].id)
            candidate='zero-accumulator' if op.results[0].type.elem=='f32' else 'atlas.virtual_vpu_splat'
            bits=0
            if op.results[0].type.elem=='bf16':bf16values.add(op.results[0].id)
        elif n=='tt.trans':
            if not all(tile(v.type,'f8E4M3FN') for v in op.operands+op.results) or op.operands[0].id not in fp8values:
                reject('transpose-type','XLU transpose requires a full FP8 value tile',op);continue
            order=transpose_order(op)
            if order not in ((0,1),(1,0)):
                reject('transpose-permutation','only rank-two identity or swap is supported',op);continue
            candidate='alias-value' if order==(0,1) else 'atlas.virtual_xlu_transpose'
            fp8values.add(op.results[0].id)
        elif n=='tt.dot':
            valid=(all(tile(v.type,'f8E4M3FN') and v.id in fp8values for v in op.operands[:2]) and op.operands[2].id in zeros and
                   op.attrs.get('inputPrecision','ieee' if analysis.resolved_dot_defaults else None)=='ieee' and
                   int(op.attrs.get('maxNumImpreciseAcc','0').split(':')[0])==0)
            consumers=users.get(op.results[0].id,[])
            if not valid or not consumers or any(c.name!='arith.truncf' or c.results[0].type.elem!='bf16' for c in consumers):
                reject('dot-policy','dot needs full FP8 operands, zero accumulator, IEEE/default precision, zero imprecision and only BF16 readouts',op);continue
            unit=dotcount%2;dotcount+=1;candidate='atlas.virtual_mxu_matmul'
        elif n=='arith.truncf':
            if op.operands[0].defining_op is None or op.operands[0].defining_op.name!='tt.dot' or not tile(op.results[0].type,'bf16'):
                reject('unsupported-conversion','only dot BF16 readout is authorized',op);continue
            candidate='alias-bf16';bf16values.add(op.results[0].id)
        elif n=='arith.maxnumf':
            z=[i for i,v in enumerate(op.operands) if v.id in zeros and v.type.elem=='bf16']
            if not z or any(v.id not in bf16values for v in op.operands):reject('relu-semantics','only BF16 maxnum with positive zero is supported',op);continue
            inputs=(op.operands[1-z[0]],);candidate='atlas.virtual_vpu_unary:relu';bf16values.add(op.results[0].id)
        elif n=='tt.fp_to_fp':
            if op.operands[0].id not in bf16values or not tile(op.results[0].type,'f8E4M3FN') or op.attrs!={'rounding':'rtne'}:
                reject('unsupported-conversion','only BF16->E4M3FN RTNE policy packing is supported',op);continue
            candidate='atlas.virtual_pack_fp8';scale=127;fp8values.add(op.results[0].id)
        elif n=='tt.return':candidate='return'
        else:reject('unsupported-operation',f'{n} not qualified by FP8 operation policy',op)
        if candidate:entries.append(EligibleOperation(op,candidate,inputs,bits,b,unit,scale))
    if not stored or used!=set(bindings):reject('memory-abi','every input/output must be used and at least one output stored')
    if diagnostics:return EligibilityResult(tuple(diagnostics))
    return EligibilityResult((),EligibilityPlan(analysis,numerical_policy,abi,tuple(entries)))
