"""Prove static single-program addresses and derive a boundary TFunc graph.

Original compiler TTIR is never edited. Derived TOp objects carry derived
assembly; their provenance remains the original TFunc and operation indices.
No normalized TTIR is presented as compiler output.
"""
from dataclasses import dataclass
import numpy as np
from compiler.ttir_model import TType, TValue, TOp, TBlock, TFunc
from compiler.ttir_operation_analysis import analyze_operations, UnsupportedKernel


@dataclass(frozen=True)
class TileNormalization:
    original: object
    analysis: object
    address_operations: tuple
    precision_operations: tuple
    grid: tuple


def normalize_static_tiles(function, *, grid, numerical_policy, resolve_dot_defaults=False):
    if tuple(grid) != (1,1,1):
        raise UnsupportedKernel('only an explicitly single-program grid (1,1,1) is supported')
    original = analyze_operations(function, resolve_dot_defaults=resolve_dot_defaults)
    if numerical_policy not in {'atlas-bf16-elementwise-rtl-v1','atlas-fp8-mlp-v1','atlas-bf16-rowwise-rtl-v1','atlas-fp8-layout-v1','atlas-fp8-attention-v1'}:
        raise UnsupportedKernel('static tile normalization requires an explicit supported numerical policy')
    if numerical_policy not in {'atlas-bf16-rowwise-rtl-v1','atlas-fp8-attention-v1'} and any(op.name == 'arith.divf' for op in original.operations):
        raise UnsupportedKernel('arith.divf requires the explicit rowwise reciprocal/multiply policy')
    args=[]; mapped={}; static={}; bases={}; address=[]; precision=[]; operations=[]
    for i,a in enumerate(function.args):
        if a.type.kind!='ptr' or a.type.elem not in {'bf16','f8E4M3FN'}:
            raise UnsupportedKernel('only scalar BF16/FP8 pointer arguments are supported')
        raw=f'tensor<32x32x!tt.ptr<{a.type.elem}>>'
        t=TType('tensor',a.type.elem,(32,32),True,raw)
        v=TValue(a.id,t,a.name,True)
        args.append(v);mapped[a.id]=v
        static[a.id]=np.array(0,dtype=np.int64);bases[a.id]=i
    users={}
    for op in original.operations:
        for v in op.operands:users.setdefault(v.id,[]).append(op)
    # A BF16 value promoted only for maxnum(+0) and immediately narrowed back
    # can remain a BF16 maxnum under the explicitly authorized target policy.
    promoted={}; demoted={}; skipped=set()
    for op in original.operations:
        if op.name!='arith.truncf' or op.results[0].type.elem!='bf16':continue
        maximum=op.operands[0].defining_op
        if maximum is None or maximum.name!='arith.maxnumf' or maximum.results[0].type.elem!='f32':continue
        ext=None;zero=None
        for v in maximum.operands:
            p=v.defining_op
            if p and p.name=='arith.extf' and p.operands[0].type.elem=='bf16':ext=p
            if p and p.name=='arith.constant' and p.attrs.get('value')=='dense<0.000000e+00>':zero=p
        if ext is None or zero is None:continue
        if (len(users[ext.results[0].id])!=1 or users[ext.results[0].id][0] is not maximum or
            len(users[maximum.results[0].id])!=1 or users[maximum.results[0].id][0] is not op):
            raise UnsupportedKernel('promoted maxnum has extra FP32 consumers')
        promoted[maximum.index]=(ext,zero)
        demoted[op.index]=maximum
        skipped.add(ext.index)
    for op in function.body.ops:
        name=op.name
        if op.regions and (numerical_policy not in {'atlas-bf16-rowwise-rtl-v1','atlas-fp8-attention-v1'} or name != 'tt.reduce'):
            raise UnsupportedKernel('regions are not supported by static tile normalization')
        if name == 'tt.reduce':
            # The analyzer already proved this pure scalar combiner. Clone its
            # block arguments and edges without changing types, axis or order.
            new = TOp(name,[mapped[v.id] for v in op.operands],[],dict(op.attrs),[],op.index,op.assembly)
            region = op.regions[0][0]
            local = {}
            block_args = []
            for value in region.args:
                cloned = TValue(value.id,value.type,value.name,True)
                local[value.id] = cloned
                block_args.append(cloned)
            inner_ops = []
            for inner in region.ops:
                cloned = TOp(inner.name,[local[v.id] for v in inner.operands],[],dict(inner.attrs),[],inner.index,inner.assembly)
                for value in inner.results:
                    result = TValue(value.id,value.type,value.name,False,cloned)
                    cloned.results.append(result)
                    local[value.id] = result
                inner_ops.append(cloned)
            new.regions = [[TBlock(region.id,block_args,inner_ops)]]
            for value in op.results:
                result = TValue(value.id,value.type,value.name,False,new)
                new.results.append(result)
                mapped[value.id] = result
            operations.append(new)
            continue
        if name in {'tt.load','tt.store'}:
            ptr=op.operands[0]
            offsets=static.get(ptr.id)
            if len(op.operands)!=(1 if name=='tt.load' else 2) or offsets is None or ptr.id not in bases:
                raise UnsupportedKernel('memory must be an unmasked proven static tile')
            if offsets.shape!=(32,32) or not np.array_equal(offsets,np.arange(1024).reshape(32,32)):
                raise UnsupportedKernel('tile pointers must enumerate row-major offsets 0..1023 exactly')
            operands=[args[bases[ptr.id]]]+[mapped[v.id] for v in op.operands[1:]]
        elif name=='tt.return':operands=[]
        elif name=='arith.constant' and op.results[0].type.elem=='i32':
            literal=op.attrs['value'];literal=literal[6:-1] if literal.startswith('dense<') else literal
            static[op.results[0].id]=np.full(op.results[0].type.shape,int(literal),dtype=np.int64)
            address.append(op);continue
        elif name in {'tt.make_range','tt.expand_dims','tt.broadcast','tt.splat','tt.addptr','arith.addi','arith.muli'} and (name=='tt.make_range' or all(v.id in static for v in op.operands)):
            ins=[static[v.id] for v in op.operands]
            if name=='tt.make_range':out=np.arange(op.attr_int('start'),op.attr_int('end'),dtype=np.int64)
            elif name=='tt.expand_dims':out=np.expand_dims(ins[0],op.attr_int('axis'))
            elif name in {'tt.broadcast','tt.splat'}:out=np.broadcast_to(ins[0],op.results[0].type.shape)
            elif name in {'tt.addptr','arith.addi'}:out=ins[0]+ins[1]
            else:out=ins[0]*ins[1]
            if out.size>1024 or np.any(out<-(1<<31)) or np.any(out>=(1<<31)):
                raise UnsupportedKernel('index expression exceeds supported size/i32 range')
            static[op.results[0].id]=out
            pointer_inputs=[v for v in op.operands if v.id in bases]
            if pointer_inputs:
                if len(pointer_inputs)!=1:raise UnsupportedKernel('ambiguous pointer base')
                bases[op.results[0].id]=bases[pointer_inputs[0].id]
            address.append(op);continue
        elif op.index in skipped:
            precision.append(op);continue
        elif op.index in demoted:
            mapped[op.results[0].id]=mapped[demoted[op.index].results[0].id]
            precision.append(op);continue
        elif op.index in promoted:
            operands=[]
        else:
            try:operands=[mapped[v.id] for v in op.operands]
            except KeyError as exc:raise UnsupportedKernel(f'{name}: unproved data/address dependency') from exc
        new=TOp(name,operands,[],dict(op.attrs),[],op.index,op.assembly)
        if op.index in promoted:
            ext,zero=promoted[op.index]
            # Create a distinct BF16 zero; the original FP32 zero may still be
            # a dot accumulator and must not change type.
            zid=-100000-op.index
            zt=TType('tensor','bf16',(32,32),False,'tensor<32x32xbf16>')
            z=TOp('arith.constant',[],[],{'value':'dense<0.000000e+00>','type':zt.raw},[],zid,
                  f'%nz{op.index} = arith.constant dense<0.000000e+00> : {zt.raw}')
            zv=TValue(zid,zt,f'%nz{op.index}',False,z);z.results=[zv];operations.append(z)
            new.operands=[mapped[ext.operands[0].id],zv]
            precision.append(op)
        for v in op.results:
            t=v.type
            if op.index in promoted:t=TType('tensor','bf16',(32,32),False,'tensor<32x32xbf16>')
            nv=TValue(v.id,t,v.name,False,new);new.results.append(nv);mapped[v.id]=nv
        if name in {'tt.load','tt.store'}:
            prefix=(new.results[0].name+' = ') if new.results else ''
            new.assembly=prefix+name+' '+', '.join(v.name for v in new.operands)+' : '+new.operands[0].type.raw
        elif op.index in promoted:
            new.assembly=f'{new.results[0].name} = arith.maxnumf '+', '.join(v.name for v in new.operands)+' : tensor<32x32xbf16>'
        operations.append(new)
    # Remove now-unused FP32 constants (only constants, never side effects).
    used={v.id for op in operations for v in op.operands}
    operations=[op for op in operations if not(op.name=='arith.constant' and op.results[0].id not in used)]
    block=TBlock(function.body.id,args,operations)
    derived=TFunc(function.name,args,block,function.source)
    result=analyze_operations(derived,resolve_dot_defaults=resolve_dot_defaults)
    return TileNormalization(original,result,tuple(address),tuple(precision),tuple(grid))
