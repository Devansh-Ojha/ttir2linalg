"""Reusable BF16 row arithmetic with opt-in FP8 contraction/pack composition.

FP32 arithmetic is never implicitly demoted: only supported operations with
exclusively explicit BF16 readouts are admitted under the named target policy.
"""
from compiler.atlas_eligibility import (BF16TileABI, FP8TileABI, TileBinding,
    Diagnostic, EligibleOperation, EligibilityPlan, EligibilityResult, _tile,
    _constant_bits, MixedTileABI)
from compiler.ttir_operation_analysis import OperationAnalysis, analyze_operations, transpose_order

ROWWISE_POLICY = 'atlas-bf16-rowwise-rtl-v1'
FP8_ROWWISE_POLICY = 'atlas-fp8-attention-v1'


def analyze_rowwise_operations(analysis, *, numerical_policy, abi):
    diagnostics = []; entries = []
    def reject(code, message, op=None):
        diagnostics.append(Diagnostic(code,message,op.index if op else None,op.name if op else None))
    mixed = numerical_policy == FP8_ROWWISE_POLICY
    if numerical_policy not in (ROWWISE_POLICY, FP8_ROWWISE_POLICY):
        reject('numerical-policy',f'explicit {ROWWISE_POLICY} or {FP8_ROWWISE_POLICY} required')
    if not isinstance(analysis,OperationAnalysis):
        return EligibilityResult((Diagnostic('invalid-analysis','expected OperationAnalysis'),))
    try:
        fresh = analyze_operations(analysis.function,resolve_dot_defaults=analysis.resolved_dot_defaults)
    except (RuntimeError,KeyError,ValueError,IndexError) as exc:
        return EligibilityResult((Diagnostic('invalid-analysis',str(exc)),))
    # Compare identities as well as complete edges/accesses/region facts.
    def inventory(a):
        return ([id(o) for o in a.operations],[[id(v) for v in d] for d in a.dependencies],
                [(id(m.operation),id(m.pointer),id(m.value),id(m.mask),id(m.other)) for m in a.memory],
                [(id(r.operation),r.axis,id(r.combiner)) for r in a.reductions])
    if inventory(fresh) != inventory(analysis):
        return EligibilityResult((Diagnostic('invalid-analysis','stale or partial source graph'),))
    expected_layout = 'atlas-mixed-fp8-row-major-bf16-column-halves' if mixed else 'atlas-bf16-column-halves'
    valid_abi = isinstance(abi,MixedTileABI) if mixed else (isinstance(abi,BF16TileABI) and not isinstance(abi,(FP8TileABI,MixedTileABI)))
    if (not valid_abi or abi.layout!='row-major-contiguous' or abi.disjoint is not True or abi.launch!='single-program' or
        abi.boundary_layout!=expected_layout or type(abi.alignment) is not int or abi.alignment!=2048):
        return EligibilityResult((Diagnostic('memory-abi','explicit disjoint single-program tile ABI with declared formats required'),))
    if mixed and (not isinstance(abi.formats,tuple) or len(abi.formats)!=len(analysis.function.args) or
                  any(fmt not in ('bf16','f8E4M3FN') for fmt in abi.formats)):
        return EligibilityResult((Diagnostic('memory-abi','one explicit supported format per argument is required'),))
    bindings = {}; slots = set(); args = analysis.function.args
    if not isinstance(abi.bindings,tuple):
        return EligibilityResult((Diagnostic('memory-abi','bindings must be a tuple'),))
    for b in abi.bindings:
        if (not isinstance(b,TileBinding) or type(b.argument) is not int or not 0<=b.argument<len(args) or
            b.argument in bindings or b.role not in ('input','output') or type(b.slot) is not int or
            not 0<=b.slot<64 or (b.role,b.slot) in slots):
            reject('memory-abi','invalid or duplicate binding'); continue
        bindings[b.argument]=b; slots.add((b.role,b.slot))
    if set(bindings)!=set(range(len(args))): reject('memory-abi','every argument requires a binding')
    for i,a in enumerate(args):
        fmt=abi.formats[i] if mixed else 'bf16'
        t=a.type
        if not (t.kind=='tensor' and t.shape==(32,32) and t.elem==fmt and t.elem_is_ptr):
            reject('boundary-type','argument must match its explicitly declared full-tile format')
    arg_indexes={id(v):i for i,v in enumerate(args)}
    users={}
    for op in analysis.operations:
        for v in op.operands: users.setdefault(v.id,[]).append(op)
    layouts={}; precision={}; used=set(); stored=set()
    reductions={id(r.operation):r for r in analysis.reductions}
    zeros=set(); dotcount=0
    for op in analysis.function.body.ops:
        n=op.name; candidate=None; inputs=tuple(op.operands); bits=None; binding=None
        shape=None; mode='bf16'; unit=None; scale=None
        tags=[layouts.get(v.id) for v in op.operands]
        modes=[precision.get(v.id) for v in op.operands]
        t=op.results[0].type if op.results else None
        def fail(message,code='unsupported-operation'):
            reject(code,message,op)
        if n in ('tt.load','tt.store'):
            i=arg_indexes.get(id(op.operands[0])); binding=bindings.get(i)
            if (len(op.operands)!=(1 if n=='tt.load' else 2) or binding is None or
                binding.role!=('input' if n=='tt.load' else 'output')):
                fail('only unmasked directly bound tiles in declared roles are supported','memory-access'); continue
            used.add(i)
            if n=='tt.load':
                fp8=op.results[0].type.elem=='f8E4M3FN'
                candidate='atlas.virtual_input_fp8' if fp8 else 'atlas.virtual_input_bf16'
                shape='tile'; mode='fp8' if fp8 else 'bf16'
            else:
                fp8=op.operands[1].type.elem=='f8E4M3FN'
                if i in stored or tags[1]!='tile' or modes[1]!=('fp8' if fp8 else 'bf16'):
                    fail('output must be a unique full tile matching its declared format and precision','memory-access'); continue
                stored.add(i); candidate='atlas.virtual_dma_store_fp8' if fp8 else 'atlas.virtual_output_bf16'
        elif n=='arith.constant':
            if mixed and t.elem=='f32' and t.shape==(32,32) and op.attrs.get('value') in ('dense<0.000000e+00>','dense<0.0>'):
                zeros.add(op.results[0].id)
                entries.append(EligibleOperation(op,'zero-accumulator',inputs,result_layout='tile',precision_mode='zero-accumulator'))
                layouts[op.results[0].id]='tile'; precision[op.results[0].id]='zero-accumulator'
                continue
            if not _tile(t): fail('only full BF16 splat constants are supported'); continue
            try: bits=_constant_bits(op)
            except ValueError as exc: fail(str(exc)); continue
            candidate='atlas.virtual_vpu_splat'; shape='tile'
        elif n=='tt.fp_to_fp' and mixed:
            if modes!=['bf16'] or tags!=['tile'] or t.elem!='f8E4M3FN' or t.shape!=(32,32) or op.attrs!={'rounding':'rtne'}:
                fail('only explicit full BF16->E4M3FN RTNE unit-scale packing is qualified','unsupported-conversion'); continue
            candidate='atlas.virtual_pack_fp8'; shape='tile'; mode='fp8'; scale=127
        elif n=='tt.dot' and mixed:
            valid=(tags[:2]==['tile','tile'] and modes[:2]==['fp8','fp8'] and op.operands[2].id in zeros and
                   op.attrs.get('inputPrecision','ieee' if analysis.resolved_dot_defaults else None)=='ieee' and
                   ('maxNumImpreciseAcc' not in op.attrs or op.attr_int('maxNumImpreciseAcc')==0))
            if not valid:
                fail('dot requires full E4M3 operands, zero accumulator and qualified IEEE/zero-imprecision attributes','dot-policy'); continue
            candidate='atlas.virtual_mxu_matmul'; shape='tile'; mode='readout'
            unit=dotcount%2; dotcount+=1
        elif n=='tt.trans' and mixed and modes==['fp8'] and tags==['tile']:
            order=transpose_order(op)
            if order not in ((0,1),(1,0)) or t.shape!=(32,32):
                fail('only full FP8 rank-two identity or swap is qualified','unsupported-layout'); continue
            candidate='alias-value' if order==(0,1) else 'atlas.virtual_xlu_transpose'
            shape='tile'; mode='fp8'
        elif n=='arith.extf':
            if op.operands[0].type.elem!='bf16' or t.elem!='f32' or modes!=['bf16'] or tags[0] is None:
                fail('only exact BF16 promotion of a represented value is supported','unsupported-conversion'); continue
            candidate='alias-bf16'; shape=tags[0]; mode='extended'
        elif n=='arith.truncf':
            if t.elem!='bf16' or modes[0] not in ('readout','extended') or tags[0] is None:
                fail('only explicit qualified BF16 readout or exact promotion roundtrip is supported','unsupported-conversion'); continue
            candidate='alias-bf16'; shape=tags[0]
        elif n in ('tt.expand_dims','tt.broadcast','tt.reshape'):
            a=op.operands[0].type
            if n=='tt.expand_dims' and tags[0]=='row' and op.attr_int('axis')==1 and t.shape==(32,1):
                shape='row-column'
            elif n=='tt.broadcast' and tags[0]=='row-column' and t.shape==(32,32):
                shape='tile'
            elif n=='tt.reshape' and tags[0] in ('row','row-column') and (a.shape,t.shape) in (((32,),(32,1)),((32,1),(32,))):
                shape='row-column' if t.shape==(32,1) else 'row'
            else:
                fail('only reduction-row [32] <-> [32,1] views and [32,1] -> [32,32] replication are supported','unsupported-layout'); continue
            candidate='alias-bf16'; mode=modes[0]
        elif n=='tt.reduce':
            reduction=reductions[id(op)]
            if (reduction.axis!=1 or tags!=['tile'] or modes!=['extended'] or
                op.operands[0].type.elem!='f32' or t.shape!=(32,)):
                fail('row reductions require axis 1 of a full BF16 tile exactly promoted to FP32','reduction-policy'); continue
            kind={'arith.addf':'row_sum','arith.maxnumf':'row_max'}[reduction.combiner.name]
            candidate='atlas.virtual_vpu_reduce:'+kind; shape='row'; mode='readout'
        elif n in ('arith.addf','arith.subf','arith.mulf','arith.divf','math.exp'):
            if any(tag!='tile' for tag in tags):
                fail('arithmetic requires full logical tiles (broadcast rows explicitly)','unsupported-layout'); continue
            if t.elem=='f32':
                if any(m!='extended' for m in modes):
                    fail('FP32 operations require exactly promoted BF16 inputs, with no intervening FP32 computation','precision-policy'); continue
                mode='readout'
            elif t.elem!='bf16' or any(m!='bf16' for m in modes):
                fail('only BF16 arithmetic or qualified FP32 operation/readout islands are supported','precision-policy'); continue
            candidate={'arith.addf':'atlas.virtual_vpu_binary:add','arith.subf':'atlas.virtual_vpu_binary:sub',
                       'arith.mulf':'atlas.virtual_vpu_binary:mul','arith.divf':'bf16-div-recip-mul',
                       'math.exp':'atlas.virtual_vpu_unary:exp'}[n]
            shape='tile'
        elif n=='tt.trans':
            fail('XLU transposes 32x32 bytes; BF16 tile transpose needs a verified movement decomposition','transpose-hardware-format'); continue
        elif n=='tt.dot':
            fail('configured MXU multiplicands are FP8 E4M3; BF16 accumulator load/readout does not provide BF16 multiplication','dot-hardware-format'); continue
        elif n=='tt.return': candidate='return'
        else: fail(f'{n} has no rowwise policy lowering'); continue
        if mode=='readout':
            consumers=users.get(op.results[0].id,[])
            if not consumers or any(c.name!='arith.truncf' or c.results[0].type.elem!='bf16' for c in consumers):
                fail('FP32 result must have only explicit BF16 readout consumers','precision-policy'); continue
        if candidate:
            entries.append(EligibleOperation(op,candidate,inputs,bits,binding,unit=unit,scale_code=scale,result_layout=shape,precision_mode=mode))
            if t is not None: layouts[op.results[0].id]=shape; precision[op.results[0].id]=mode
    if not stored or used!=set(bindings): reject('memory-abi','every binding must be used and at least one output stored')
    if diagnostics: return EligibilityResult(tuple(diagnostics))
    return EligibilityResult((),EligibilityPlan(analysis,numerical_policy,abi,tuple(entries)))
