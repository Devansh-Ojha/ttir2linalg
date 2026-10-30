"""Independent scalar reference for atlas-bf16-rowwise-rtl-v1.

No torch, NPU numerical primitives, evaluator or runtime lookup tables. Exp's
Q9.12 conversion, Q2.12 log2(e), 32-entry Q1.16 interpolation and BF16 RNE
follow ExpLane.scala/common.scala/Qmn.scala/ExLUT.scala. LUT entries are derived
from Decimal mathematics. Reciprocal follows Rcp.scala/RcpLUT.scala/LUTParams.scala: rounded Q1.16
lookup followed by truncated BF16 mantissa and wrapped exponent.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from atlas_bf16_reference import _value, round_binary, binary



@lru_cache(maxsize=32)
def exp_lut(i):
    with localcontext() as ctx:
        ctx.prec = 80
        return int(((Decimal(2).ln() * Decimal(i)/32).exp()*65536 + Decimal('0.5')).to_integral_value(rounding='ROUND_FLOOR'))


@lru_cache(maxsize=65536)
def exponential(bits):
    magnitude=bits & 0x7fff
    if magnitude > 0x7f80:
        return (bits & 0x8000) | 0x7f80 | (0x3f if bits & 0x40 else 0x7f)
    if magnitude == 0: return 0x3f80
    if magnitude == 0x7f80: return 0 if bits & 0x8000 else 0x7f80
    if not bits & 0x8000 and magnitude > 0x42b1: return 0x7f80
    # The fixed-point input register truncates to 21 bits, including its sign.
    q = int(abs(_value(bits))*4096)
    if bits & 0x8000: q = -q
    q = (q + 2**20) % 2**21 - 2**20
    scaled = (q*5909)//4096
    k, fraction = divmod(scaled,4096)
    index, low = divmod(fraction,128)
    y0=exp_lut(index); y1=exp_lut(index+1) if index<31 else 131071
    interpolated=y0+(y1-y0)*low//128
    # RawFloat.sExp is a signed ten-bit wire, including its recoded bias.
    k = (k + 256 + 512) % 1024 - 512 - 256
    return round_binary(Fraction(interpolated,65536)*Fraction(2)**k)


@lru_cache(maxsize=65536)
def reciprocal(bits):
    sign=bits & 0x8000; exponent=(bits>>7)&255; fraction=bits&127
    if exponent==255: return sign  # Both infinity and NaN flush to signed zero.
    if exponent==0: return sign | 0x7f80  # Includes subnormal inputs.
    exact=Fraction(8388608,128+fraction)
    fixed=(2*exact.numerator+exact.denominator)//(2*exact.denominator)
    highest=fixed.bit_length()-1
    out_exponent=(254+highest-16-exponent)&255
    if out_exponent==255: return sign | 0x7f80
    return sign | (out_exponent<<7) | ((fixed>>max(0,highest-7))&127)



def _f32_add(a,b):
    # Inputs are raw FP32; nonfinite behavior is HardFloat canonical NaN.
    ea,eb=(a>>23)&255,(b>>23)&255
    if (ea==255 and a&0x7fffff) or (eb==255 and b&0x7fffff): return 0x7fc00000
    if ea==255 or eb==255:
        if ea==eb==255 and (a^b)&0x80000000: return 0x7fc00000
        return a if ea==255 else b
    return round_binary(_value(a,23)+_value(b,23),23,negative_zero=bool(a&b&0x80000000))


def reduce_rows(bits,kind):
    out=[]
    for r in range(32):
        row=bits[r*32:r*32+32]
        if kind=='arith.maxnumf':
            def order(b): return b^65535 if b&32768 else b^32768
            out.append(max(row,key=order))
        elif kind=='arith.addf':
            tree=[b<<16 for b in row]
            while len(tree)>1:
                tree=[_f32_add(a,b) for a,b in zip(tree[::2],tree[1::2])]
            raw=tree[0]
            if raw&0x7f800000==0x7f800000:
                out.append(0x7fc0 if raw&0x7fffff else raw>>16)
            else:
                out.append(round_binary(_value(raw,23),negative_zero=bool(raw&0x80000000)))
        else: raise ValueError(kind)
    return out


def source_reference(analysis,inputs):
    """Evaluate source operations with logical shapes, including reduction rows.

    Qualified FP32 operations use the policy's BF16 hardware result only at
    their explicit BF16 readout. Eligibility proves no intervening FP32 user.
    """
    values={}; outputs={}; args={v.id:i for i,v in enumerate(analysis.function.args)}
    reductions={id(r.operation):r.combiner.name for r in analysis.reductions}
    for op in analysis.function.body.ops:
        n=op.name
        ins=[values[v.id] for v in op.operands if v.id in values]
        if n=='tt.load': out=list(inputs[args[op.operands[0].id]])
        elif n in ('arith.extf','arith.truncf','tt.expand_dims','tt.reshape'): out=ins[0]
        elif n=='tt.broadcast': out=[x for x in ins[0] for _ in range(32)]
        elif n=='tt.reduce': out=reduce_rows(ins[0],reductions[id(op)])
        elif n=='math.exp': out=[exponential(x) for x in ins[0]]
        elif n=='arith.divf': out=[binary('mul',a,reciprocal(b)) for a,b in zip(*ins)]
        elif n in ('arith.addf','arith.subf','arith.mulf'): out=[binary(n[6:-1],a,b) for a,b in zip(*ins)]
        elif n=='arith.constant':
            literal=op.attrs['value'].removeprefix('dense<').removesuffix('>')
            out=[round_binary(Fraction(literal),negative_zero=literal.startswith('-'))]*1024
        elif n=='tt.store': outputs[args[op.operands[0].id]]=ins[0]; continue
        elif n=='tt.return': continue
        else: raise ValueError(f'unsupported reference operation {n}')
        values[op.results[0].id]=out
    return outputs
