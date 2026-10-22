#!/usr/bin/env python3
"""Compare generated and handwritten Atlas virtual MLPs without artifacts.

Both execute the shared RTL-aware numerical model. This checks virtual-program
semantic equivalence, not independent RTL or hardware correctness.
"""
import sys
sys.dont_write_bytecode = True

from collections import Counter
from pathlib import Path

import torch
from atlas_virtual_evaluator import RuntimeInputs, Tile, evaluate, operation_name, parse_program
from npu_model.configs.numerics import RtlNumerics

ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "out/atlas/two_layer_mlp_generated.virtual.mlir"
REFERENCE = Path.home() / "atlas-mlir/test/examples/virtual_fp8_two_layer_mlp.mlir"
# Independent launch ABI: X, B0[K,N], B1[K,N], and BF16 output slot zero.
# The reference uses the same role slots, with weights supplied as W[N,K].
EXPECTED_INPUT_SLOTS = (0, 1, 2)
EXPECTED_OUTPUT_INDEX = 0


def fp8_tile(values):
    """Pack at unity using RTL semantics and retain raw row-major encodings."""
    encoded = RtlNumerics.to_fp8(values, 127).contiguous()
    return encoded, Tile("fp8", tuple(encoded.view(torch.uint8).flatten().tolist()))


def raw_tile(codes, offset=0):
    """Supply exact row-major E4M3FN encodings, bypassing packer flushing."""
    bits = tuple(codes[(i + offset) % len(codes)] for i in range(1024))
    encoded = torch.tensor(bits, dtype=torch.uint8).reshape(32, 32)
    return encoded.view(torch.float8_e4m3fn), Tile("fp8", bits)


def _first_layer_cases():
    for seed in (0, 1, 42, 20261021):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        lhs = (torch.randint(-96, 97, (32, 32), generator=generator).float() / 32).to(torch.bfloat16)
        rhs = (torch.randint(-80, 81, (32, 32), generator=generator).float() / 32).to(torch.bfloat16)
        yield f"random_seed_{seed}", fp8_tile(lhs), fp8_tile(rhs)

    index = torch.arange(1024).reshape(32, 32)
    positive = ((index % 63 + 1).float() / 16).to(torch.bfloat16)
    other = (((index * 17 + index // 32) % 79 + 1).float() / 32).to(torch.bfloat16)
    zero = fp8_tile(torch.zeros((32, 32), dtype=torch.bfloat16))
    yield "zero_both", zero, zero
    yield "zero_lhs", zero, fp8_tile(other)
    yield "zero_rhs", fp8_tile(positive), zero
    for left_sign, right_sign in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        yield f"signs_{left_sign:+d}_{right_sign:+d}", fp8_tile(positive * left_sign), fp8_tile(other * right_sign)

    # Min normal, adjacent normals, values around one, and max finite +/-448.
    normal = (0x08, 0x09, 0x37, 0x38, 0x39, 0x7D, 0x7E,
              0x88, 0x89, 0xB7, 0xB8, 0xB9, 0xFD, 0xFE)
    yield "normal_boundaries", raw_tile(normal), raw_tile(normal, 5)
    # MXU products flush exponent-zero inputs; Torch diagnostic retains subnormals.
    tiny = (0x00, 0x80, 0x01, 0x07, 0x81, 0x87)
    yield "signed_zero_subnormal", raw_tile(tiny), raw_tile(normal, 3)
    finite = tuple(code for code in range(256) if code not in (0x7F, 0xFF))
    yield "all_finite_fp8_encodings", raw_tile(finite), raw_tile(finite, 73)


def test_cases():
    generator = torch.Generator().manual_seed(20261022)
    w1 = fp8_tile((torch.randint(-64, 65, (32, 32), generator=generator).float() / 32).to(torch.bfloat16))
    old_cases = list(_first_layer_cases())
    baseline = next(c for c in old_cases if c[0] == "random_seed_20261021")
    ordered = [baseline] + [c for c in old_cases if c is not baseline]
    for name, x, w0 in ordered:
        yield name, (x, w0, w1)
    identity = fp8_tile(torch.eye(32, dtype=torch.bfloat16))
    for name, major, minor, scale in (("pack_tie_even_low", 1.0, 0.0625, 1.0),
                                      ("pack_tie_even_high", 1.125, 0.0625, 1.0),
                                      ("pack_underflow", 0.015625, 0.0, 0.5),
                                      ("pack_saturation", 448.0, 0.0, 2.0)):
        x = torch.zeros((32, 32), dtype=torch.bfloat16)
        x[:, 0], x[:, 1] = major, minor
        w = torch.zeros_like(x)
        w[:, 0], w[:, 1] = scale, scale
        yield name, (fp8_tile(x), fp8_tile(w), identity)
    zero = fp8_tile(torch.zeros((32, 32), dtype=torch.bfloat16))
    yield "zero_second_weights", (baseline[1], baseline[2], zero)
    yield "negative_second_weights", (baseline[1], baseline[2], fp8_tile(-torch.ones((32, 32), dtype=torch.bfloat16)))
    # Exercise all finite encodings in both weight orientations, not only X/W0.
    finite = tuple(v for v in range(256) if v not in (0x7F, 0xFF))
    yield "boundary_second_weights", (baseline[1], baseline[2], raw_tile(finite, 73))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(op):
    result = {k: v for k, v in op.attributes.items() if k != "op_name__"}
    require(not result.keys() & op.properties.keys(), "duplicate operation fields")
    return result | op.properties


def integer(op, name):
    return fields(op)[name].value.data


def inspect(path, transposed, *, expected_slots=EXPECTED_INPUT_SLOTS, source=None):
    """Check the graph against the caller's ABI, never infer the expected ABI."""
    program = parse_program(path.read_text() if source is None else source)
    require(program.function is not None and len(program.blocks) == 1 and not program.control_widths,
            f"{path}: expected a single function/block with no controls")
    operations = program.operations
    counts = Counter(operation_name(op) for op in operations)
    expected = {"atlas.virtual_start": 1, "atlas.virtual_input_fp8": 3,
                "atlas.virtual_mxu_matmul": 2, "atlas.virtual_vpu_unary": 1,
                "atlas.virtual_pack_fp8": 1, "atlas.virtual_output_bf16": 1, "func.return": 1}
    if transposed:
        expected["atlas.virtual_xlu_transpose"] = 2
    require(counts == Counter(expected), f"{path}: unexpected operation inventory {dict(counts)}")
    definitions = {value: op for op in operations for value in op.results}
    visited = set()

    def producer(value, name):
        op = definitions.get(value)
        require(op is not None and operation_name(op) == name, f"{path}: expected {name} producer")
        visited.add(op)
        return op

    output = next(op for op in operations if operation_name(op) == "atlas.virtual_output_bf16")
    dot1 = producer(output.operands[1], "atlas.virtual_mxu_matmul")
    require(integer(dot1, "unit") == 1, "second dot must use MXU1")
    pack = producer(dot1.operands[0], "atlas.virtual_pack_fp8")
    require(integer(pack, "scale_code") == 127, "pack must use scale_code=127")
    relu = producer(pack.operands[0], "atlas.virtual_vpu_unary")
    require(fields(relu)["kind"].data == "relu", "expected BF16 ReLU")
    dot0 = producer(relu.operands[0], "atlas.virtual_mxu_matmul")
    require(dot0 is not dot1 and integer(dot0, "unit") == 0, "first dot must use MXU0")

    def input_slot(value, weight=False):
        if weight and transposed:
            transpose = producer(value, "atlas.virtual_xlu_transpose")
            value = transpose.operands[0]
        load = producer(value, "atlas.virtual_input_fp8")
        require(value is load.results[1], "input must use the FP8 tensor result")
        return integer(load, "index")

    slots = (input_slot(dot0.operands[0]), input_slot(dot0.operands[1], True),
             input_slot(dot1.operands[1], True))
    require(len(set(slots)) == 3 and dict(program.input_formats) == {i: "fp8" for i in slots},
            f"{path}: input slot mismatch {slots}")
    require(slots == expected_slots,
            f"{path}: role-to-slot ABI mismatch: expected X/B0/B1={expected_slots}, got {slots}")
    index = integer(output, "index")
    require(index == EXPECTED_OUTPUT_INDEX, f"{path}: output ABI requires slot zero")
    require(tuple(program.output_indices) == (index,), "output slot mismatch")
    computational = {op for op in operations if operation_name(op) not in
                     ("atlas.virtual_start", "atlas.virtual_output_bf16", "func.return")}
    require(visited == computational, "unexpected or reused computational nodes")
    attrs = program.function.attributes
    require(set(attrs) == {"atlas.input_dram_base", "atlas.output_dram_base"},
            "unsupported function attributes/layout")
    input_base = attrs["atlas.input_dram_base"].value.data
    output_base = attrs["atlas.output_dram_base"].value.data
    require(input_base >= 0x80000000 and output_base >= 0x80000000 and
            input_base % 2048 == output_base % 2048 == 0, "unsupported DRAM bases")
    spans = [(input_base + slot * 2048, 1024) for slot in slots] + [(output_base + index * 2048, 2048)]
    require(all(a + n <= 1 << 32 for a, n in spans), "boundary exceeds 32-bit DRAM")
    ordered = sorted(spans)
    require(all(a + n <= b for (a, n), (b, _) in zip(ordered, ordered[1:])), "overlapping boundary slots")
    print(f"{path.name}: X/weight0/weight1 slots={slots}; addresses={[hex(a) for a, _ in spans[:3]]}; "
          f"output index={index}, address={spans[-1][0]:#x}; RHS transposes={2 if transposed else 0}")
    return program, slots, index


def transpose(tile):
    require(tile.format == "fp8" and tile.shape == (32, 32), "expected 32x32 FP8 input")
    return Tile("fp8", tuple(tile.bits[c * 32 + r] for r in range(32) for c in range(32)))


def execute(spec, tiles):
    program, slots, output_index = spec
    result = evaluate(program, RuntimeInputs(tiles=dict(zip(slots, tiles))))
    require(set(result.outputs) == {output_index}, "missing or unexpected outputs")
    tile = result.outputs[output_index]
    require(tile.format == "bf16" and tile.shape == (32, 32) and len(tile.bits) == 1024,
            "unsupported output layout")
    # EvaluationResult exposes logical row-major encodings after boundary decoding.
    bits = torch.tensor(tile.bits, dtype=torch.uint16).reshape(32, 32)
    values = bits.view(torch.bfloat16).float()
    require(bool(torch.isfinite(values).all()), "nonfinite BF16 result in finite-input test")
    return bits, values


def main():
    torch.set_num_threads(1)
    generated = inspect(GENERATED, True)
    reference = inspect(REFERENCE, False)
    print("Output layout: evaluator Tile.bits is row-major 32x32 BF16 (physical boundary stores two 32x16 column halves).")
    print("Reference: shared RTL-aware MXU0 -> BF16 ReLU -> FP8Pack(127) -> MXU1; NOT independent hardware verification.")
    print("Criterion: zero differing BF16 encodings; all outputs must be finite.")
    print("Reused cases define N,K weights: generated B=W.T, reference W=B.T; transposition preserves raw FP8 bits.")
    planned = list(test_cases())
    passed = 0
    asymmetric = 0
    for name, encoded in planned:
        try:
            x, w0, w1 = (pair[1] for pair in encoded)
            b0, b1 = transpose(w0), transpose(w1)
            asymmetric += b0.bits != w0.bits and b1.bits != w1.bits
            actual, actual_values = execute(generated, (x, b0, b1))
            wanted, wanted_values = execute(reference, (x, transpose(b0), transpose(b1)))
            mismatches = int((actual != wanted).sum())
            error = float((actual_values - wanted_values).abs().max())
            ok = mismatches == 0
            print(f"{name}: {'PASS' if ok else 'FAIL'}; mismatches={mismatches}/1024; max_abs_error={error:.9g}; finite=True", flush=True)
            if mismatches:
                row, col = (actual != wanted).nonzero()[0].tolist()
                print(f"  First mismatch [{row},{col}]: generated=0x{int(actual[row,col]):04x}, reference=0x{int(wanted[row,col]):04x}")
            passed += ok
        except Exception as error:
            print(f"{name}: FAIL; {type(error).__name__}: {error}", flush=True)
    require(asymmetric > 0, "no case exercises asymmetric weights in both layers")
    failed = len(planned) - passed
    print(f"{'PASS' if failed == 0 else 'FAIL'}: {passed}/{len(planned)} cases passed; {failed} failed; "
          f"{asymmetric} cases have asymmetric weights in both layers")
    return int(failed != 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
