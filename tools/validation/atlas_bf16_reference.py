"""Independent integer/rational reference for atlas-bf16-elementwise-rtl-v1.

Does not import torch, RtlNumerics, the virtual evaluator, or emitted MLIR.
IEEE finite arithmetic is exact Fraction arithmetic, then explicitly rounded.
"""
from fractions import Fraction


def _value(bits, mantissa=7, exponent_bits=8):
    sign = -1 if bits >> (mantissa + exponent_bits) else 1
    exponent = (bits >> mantissa) & ((1 << exponent_bits) - 1)
    fraction = bits & ((1 << mantissa) - 1)
    bias = (1 << (exponent_bits - 1)) - 1
    power = (exponent - bias if exponent else 1 - bias) - mantissa
    significant = fraction + ((1 << mantissa) if exponent else 0)
    return sign * Fraction(significant * (1 << max(power, 0)), 1 << max(-power, 0))


def round_binary(value, mantissa=7, exponent_bits=8, negative_zero=False):
    sign = int(value < 0 or (value == 0 and negative_zero)) << (mantissa + exponent_bits)
    value = abs(value)
    if not value:
        return sign
    bias = (1 << (exponent_bits - 1)) - 1
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(1 << max(exponent, 0), 1 << max(-exponent, 0)):
        exponent -= 1
    quantum = max(exponent, 1 - bias) - mantissa
    scaled = value * Fraction(1 << max(-quantum, 0), 1 << max(quantum, 0))
    q, remainder = divmod(scaled.numerator, scaled.denominator)
    if 2 * remainder > scaled.denominator or (2 * remainder == scaled.denominator and q & 1):
        q += 1
    if exponent < 1 - bias:
        return sign | q
    if q == 1 << (mantissa + 1):
        q >>= 1
        exponent += 1
    if exponent > bias:
        return sign | (((1 << exponent_bits) - 1) << mantissa)
    return sign | ((exponent + bias) << mantissa) | (q - (1 << mantissa))


def binary(kind, a, b):
    ea, eb = (a >> 7) & 255, (b >> 7) & 255
    if (ea == 255 and a & 127) or (eb == 255 and b & 127):
        return 0x7fc0
    if kind == 'sub':
        b ^= 0x8000
    if kind in ('add', 'sub'):
        if ea == 255 or eb == 255:
            if ea == eb == 255 and (a ^ b) & 0x8000:
                return 0x7fc0
            return a if ea == 255 else b
        exact = _value(a) + _value(b)
        # IEEE round-nearest exact cancellation is +0; -0+-0 is -0.
        return round_binary(exact, mantissa=23, negative_zero=bool(a & b & 0x8000)) >> 16
    if kind != 'mul':
        raise ValueError(kind)
    sign = (a ^ b) & 0x8000
    if ea == 255 or eb == 255:
        if (a & 0x7fff) == 0 or (b & 0x7fff) == 0:
            return 0x7fc0
        return sign | 0x7f80
    return round_binary(_value(a) * _value(b), negative_zero=bool(sign))


def source_reference(analysis, inputs):
    """Input/output keys are TTIR argument indices, values are raw BF16 lists."""
    args = {v.id: i for i, v in enumerate(analysis.function.args)}
    values, outputs, zero_constants = {}, {}, set()
    for op in analysis.operations:
        name = op.name
        if name == 'tt.load':
            result = list(inputs[args[op.operands[0].id]])
        elif name == 'arith.constant':
            literal = op.attrs['value']
            if literal.startswith('dense<'):
                literal = literal[6:-1]
            bits = round_binary(Fraction(literal), negative_zero=literal.startswith('-'))
            result = bits if op.results[0].type.kind == 'scalar' else [bits] * 1024
            if bits == 0:
                zero_constants.add(op.results[0].id)
        elif name == 'tt.splat':
            result = [values[op.operands[0].id]] * 1024
            if result[0] == 0:
                zero_constants.add(op.results[0].id)
        elif name in ('arith.addf', 'arith.subf', 'arith.mulf'):
            result = [binary(name[6:-1], a, b) for a, b in zip(*(values[v.id] for v in op.operands))]
        elif name == 'arith.maxnumf':
            zero = next(i for i, v in enumerate(op.operands) if v.id in zero_constants)
            result = [0 if b & 0x8000 else b for b in values[op.operands[1-zero].id]]
        elif name == 'tt.store':
            outputs[args[op.operands[0].id]] = list(values[op.operands[1].id])
            continue
        elif name == 'tt.return':
            continue
        else:
            raise ValueError(f'unqualified source reference operation: {name}')
        values[op.results[0].id] = result
    return outputs
