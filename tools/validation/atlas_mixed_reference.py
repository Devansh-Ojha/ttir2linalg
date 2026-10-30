"""Independent scalar SSA evaluation for composed FP8/BF16 numerical policies.

Reuses independent integer/Fraction dot/pack, BF16 arithmetic and Decimal exp
references. No torch, NPU arithmetic, machine code or virtual MLIR interpretation.
Qualified FP32 islands are represented by their explicit BF16 policy readouts.
"""
from fractions import Fraction
from atlas_bf16_reference import round_binary,binary
from atlas_fp8_reference import dot,pack
from atlas_rowwise_reference import reduce_rows,exponential,reciprocal
from compiler.ttir_operation_analysis import transpose_order


def source_reference(analysis,inputs,*,with_trace=False):
    args={v.id:i for i,v in enumerate(analysis.function.args)}
    reductions={id(r.operation):r.combiner.name for r in analysis.reductions}
    values={};outputs={};ordinal=0
    for op in analysis.function.body.ops:
        n=op.name;ins=[values[v.id] for v in op.operands if v.id in values]
        if n=='tt.load':out=list(inputs[args[op.operands[0].id]])
        elif n=='arith.constant':
            literal=op.attrs['value'].removeprefix('dense<').removesuffix('>')
            if op.results[0].type.elem=='f32' and Fraction(literal)!=0:
                raise ValueError('unqualified FP32 constant')
            out=[round_binary(Fraction(literal),negative_zero=literal.startswith('-'))]*1024
        elif n=='tt.fp_to_fp':out=[pack(b) for b in ins[0]]
        elif n=='tt.trans':out=ins[0][:] if transpose_order(op)==(0,1) else [ins[0][c*32+r] for r in range(32) for c in range(32)]
        elif n=='tt.dot':
            if any(ins[2]):raise ValueError('nonzero accumulator is not authorized')
            out=dot(ins[0],ins[1],ordinal%2);ordinal+=1
        elif n in ('arith.extf','arith.truncf','tt.expand_dims','tt.reshape'):out=ins[0]
        elif n=='tt.broadcast':out=[b for b in ins[0] for _ in range(32)]
        elif n=='tt.reduce':out=reduce_rows(ins[0],reductions[id(op)])
        elif n=='math.exp':out=[exponential(b) for b in ins[0]]
        elif n=='arith.divf':out=[binary('mul',a,reciprocal(b)) for a,b in zip(*ins)]
        elif n in ('arith.addf','arith.subf','arith.mulf'):out=[binary(n[6:-1],a,b) for a,b in zip(*ins)]
        elif n=='tt.store':outputs[args[op.operands[0].id]]=list(ins[0]);continue
        elif n=='tt.return':continue
        else:raise ValueError(f'unsupported mixed reference operation {n}')
        values[op.results[0].id]=out
    return (outputs,values) if with_trace else outputs
