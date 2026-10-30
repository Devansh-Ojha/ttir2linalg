"""Scalar integer reference for atlas-fp8-mlp-v1, independent of NPU model.

Executes the qualified source SSA graph, never emitted MLIR. Exponent-zero
FP8 operands flush; exponent-15 encodings use the MXU's finite bit datapath.
SA uses ordered BF16 MACs with jammed alignment; IPT uses a 32-bit anchor sum.
"""
from fractions import Fraction
from atlas_bf16_reference import _value, round_binary


def product(code):
    e = (code >> 3) & 15
    return (0 if not e else (-1 if code & 128 else 1) *
            Fraction(8 + (code & 7)) * Fraction(2) ** (e - 10))


def sa_mac(a, b, acc):
    exact = product(a) * product(b)
    if not exact:
        return acc
    p = round_binary(exact)
    if not acc & 0x7f80:
        return p
    # Both significands carry three alignment/rounding bits. Keep the
    # discarded tail as a sticky bit through subtraction and normalization.
    operands = sorted((p, acc), key=lambda x: x & 0x7fff, reverse=True)
    high, low = operands
    exponent = (high >> 7) & 255
    delta = min(11, exponent - ((low >> 7) & 255))
    small = (128 + (low & 127)) * 8
    tail = bool(small % (1 << delta))
    small = (small >> delta) | int(tail)
    large = (128 + (high & 127)) * 8
    total = large + small if (high ^ low) < 32768 else large - small
    if total == 0:
        return 0
    if total >= 2048:
        tail = tail or bool(total & 1)
        total >>= 1
        exponent += 1
    else:
        shift = min(max(0, 11 - total.bit_length()), exponent)
        total <<= shift
        exponent -= shift
    significand, remainder = divmod(total, 8)
    significand += int(remainder > 4 or (remainder == 4 and (tail or significand & 1)))
    if significand == 256:
        significand = 128
        exponent += 1
    return (high & 32768) | (min(exponent, 254) << 7) | (127 if exponent > 254 else significand & 127)


def dot(a, b, unit):
    result = []
    for r in range(32):
        for c in range(32):
            if unit == 0:
                acc = 0
                for k in range(32):
                    acc = sa_mac(a[r*32+k], b[k*32+c], acc)
            else:
                products = [product(a[r*32+k]) * product(b[k*32+c]) for k in range(32)]
                nonzero = [abs(x) for x in products if x]
                if not nonzero:
                    acc = 0
                else:
                    # Anchor seven above the largest product's binary exponent.
                    largest = max(nonzero)
                    exponent = largest.numerator.bit_length() - largest.denominator.bit_length()
                    if largest < Fraction(2) ** exponent:
                        exponent -= 1
                    quantum = Fraction(2) ** (exponent + 7 - 31)
                    total = sum(int(x / quantum) for x in products)
                    total = (total + 2**31) % 2**32 - 2**31
                    acc = round_binary(total * quantum)
            result.append(acc)
    return result


def pack(bits):
    if not bits & 0x7f80 or (bits & 0x7fff) > 0x7f80:
        return 0
    if bits & 0x7fff == 0x7f80:
        return (bits >> 8 & 128) | 126
    v = _value(bits)
    sign = (bits >> 8) & 128
    if abs(v) >= 448:
        return sign | 126
    # Round a NORMAL significand to three fraction bits using a wide exponent
    # FIRST, then flush target subnormals. Generic IEEE FP8 subnormal rounding
    # would incorrectly promote BF16 0x3c70..0x3c77 (and negatives) to 0x08.
    wide = round_binary(v, mantissa=3, exponent_bits=8)
    exponent = ((wide >> 3) & 255) - 127
    if exponent < -6:
        return 0
    return sign | ((exponent + 7) << 3) | (wide & 7)



def source_reference(analysis, inputs):
    args = {v.id:i for i,v in enumerate(analysis.function.args)}
    values = {}; outputs = {}; unit = 0
    for op in analysis.operations:
        ins = [values[v.id] for v in op.operands if v.id in values]
        if op.name == 'tt.load': out = inputs[args[op.operands[0].id]]
        elif op.name == 'tt.trans':
            from compiler.ttir_operation_analysis import transpose_order
            out = ins[0] if transpose_order(op) == (0,1) else [ins[0][c*32+r] for r in range(32) for c in range(32)]
        elif op.name == 'arith.constant': out = [0]*1024
        elif op.name == 'tt.dot':
            if any(ins[2]): raise ValueError('nonzero accumulator')
            out = dot(ins[0], ins[1], unit % 2); unit += 1
        elif op.name == 'arith.truncf': out = ins[0]
        elif op.name == 'arith.maxnumf':
            zero = next(i for i,v in enumerate(op.operands) if v.defining_op and v.defining_op.name == 'arith.constant')
            out = [0 if b & 32768 else b for b in ins[1-zero]]
        elif op.name == 'tt.fp_to_fp': out = [pack(b) for b in ins[0]]
        elif op.name == 'tt.store':
            outputs[args[op.operands[0].id]] = ins[0]; continue
        elif op.name == 'tt.return': continue
        else: raise ValueError(f'unsupported reference operation {op.name}')
        values[op.results[0].id] = out
    return outputs
