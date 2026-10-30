"""Exact FP8 bit-layout operations on full static tiles, independent of models.

The XLU moves bytes. BF16 is intentionally rejected rather than reinterpreted
as bytes or quantized. Store uses the backend's real virtual DMA FP8 operation.
"""
from compiler.atlas_eligibility import (FP8LayoutABI, TileBinding, Diagnostic,
    EligibleOperation, EligibilityPlan, EligibilityResult)
from compiler.ttir_operation_analysis import OperationAnalysis, analyze_operations, transpose_order

LAYOUT_POLICY = 'atlas-fp8-layout-v1'


def analyze_layout_operations(analysis, *, numerical_policy, abi):
    diagnostics=[]; entries=[]
    def reject(code,message,op=None):
        diagnostics.append(Diagnostic(code,message,op.index if op else None,op.name if op else None))
    if numerical_policy!=LAYOUT_POLICY: reject('numerical-policy',f'explicit {LAYOUT_POLICY} required')
    if not isinstance(analysis,OperationAnalysis):
        return EligibilityResult((Diagnostic('invalid-analysis','expected OperationAnalysis'),))
    try: fresh=analyze_operations(analysis.function,resolve_dot_defaults=analysis.resolved_dot_defaults)
    except (RuntimeError,KeyError,ValueError,IndexError) as exc:
        return EligibilityResult((Diagnostic('invalid-analysis',str(exc)),))
    def inventory(a):
        return ([id(o) for o in a.operations],[[id(v) for v in d] for d in a.dependencies],
                [(id(m.operation),id(m.pointer),id(m.value),id(m.mask),id(m.other)) for m in a.memory],
                [(id(r.operation),r.axis,id(r.combiner)) for r in a.reductions])
    if inventory(fresh)!=inventory(analysis):
        return EligibilityResult((Diagnostic('invalid-analysis','stale or partial source graph'),))
    if (not isinstance(abi,FP8LayoutABI) or abi.layout!='row-major-contiguous' or abi.disjoint is not True or
        abi.launch!='single-program' or abi.boundary_layout!='atlas-fp8-row-major' or
        type(abi.alignment) is not int or abi.alignment!=2048 or not isinstance(abi.bindings,tuple)):
        return EligibilityResult((Diagnostic('memory-abi','explicit FP8 row-major disjoint single-program ABI required'),))
    args=analysis.function.args; bindings={}; slots=set()
    for b in abi.bindings:
        if (not isinstance(b,TileBinding) or type(b.argument) is not int or not 0<=b.argument<len(args) or
            b.argument in bindings or b.role not in ('input','output') or type(b.slot) is not int or
            not 0<=b.slot<64 or (b.role,b.slot) in slots):
            reject('memory-abi','invalid or duplicate binding');continue
        bindings[b.argument]=b; slots.add((b.role,b.slot))
    if set(bindings)!=set(range(len(args))):reject('memory-abi','every argument needs an explicit binding')
    def tile(t,pointer=False):
        return t.kind=='tensor' and t.elem=='f8E4M3FN' and t.shape==(32,32) and t.elem_is_ptr==pointer
    for arg in args:
        if not tile(arg.type,True):reject('boundary-type','XLU layout path requires full FP8 input/output tiles')
    indexes={id(v):i for i,v in enumerate(args)}; values=set(); used=set(); stored=set()
    for op in analysis.function.body.ops:
        n=op.name; candidate=None; b=None
        if op.regions:reject('unsupported-operation','layout path has no region execution',op);continue
        if n in ('tt.load','tt.store'):
            i=indexes.get(id(op.operands[0])); b=bindings.get(i)
            if (len(op.operands)!=(1 if n=='tt.load' else 2) or b is None or
                b.role!=('input' if n=='tt.load' else 'output') or not tile(op.operands[0].type,True)):
                reject('memory-access','only unmasked directly bound FP8 tiles are supported',op);continue
            used.add(i)
            if n=='tt.load': candidate='atlas.virtual_input_fp8'; values.add(op.results[0].id)
            else:
                if i in stored or op.operands[1].id not in values:
                    reject('memory-access','store needs one defined full FP8 value, once per output',op);continue
                stored.add(i);candidate='atlas.virtual_dma_store_fp8'
        elif n=='tt.trans':
            if not all(tile(v.type) for v in op.operands+op.results) or op.operands[0].id not in values:
                reject('transpose-type','XLU requires full FP8 input/output value tiles',op);continue
            order=transpose_order(op)
            if order not in ((0,1),(1,0)):
                reject('transpose-permutation','only rank-two identity or swap supported',op);continue
            candidate='alias-value' if order==(0,1) else 'atlas.virtual_xlu_transpose'
            values.add(op.results[0].id)
        elif n=='tt.return':candidate='return'
        else:reject('unsupported-operation',f'{n} has no exact FP8 layout lowering',op)
        if candidate: entries.append(EligibleOperation(op,candidate,tuple(op.operands),binding=b,
                                                      result_layout='fp8-row-major',precision_mode='raw-bits'))
    if not stored or used!=set(bindings):reject('memory-abi','all bindings must be used and at least one output stored')
    if diagnostics:return EligibilityResult(tuple(diagnostics))
    return EligibilityResult((),EligibilityPlan(analysis,numerical_policy,abi,tuple(entries)))
