#!/usr/bin/env python3
"""Extended deterministic checks of existing Atlas virtual matmul/ReLU.

Uses the passing validator's public setup without writing bytecode caches.
The reference shares evaluator primitives: this is not independent RTL proof.
"""
import sys

sys.dont_write_bytecode = True

from validate_matmul_relu import (
    MLIR, SCALE_CODE, RtlNumerics, RuntimeInputs, Tile, evaluate, fp8_tile,
    parse_program, torch,
)


def raw_tile(codes, offset=0):
    """Supply exact row-major E4M3FN encodings, bypassing packer flushing."""
    bits = tuple(codes[(i + offset) % len(codes)] for i in range(1024))
    encoded = torch.tensor(bits, dtype=torch.uint8).reshape(32, 32)
    return encoded.view(torch.float8_e4m3fn), Tile("fp8", bits)


def cases():
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


def check_case(program, name, left, right):
    lhs, lhs_tile = left
    rhs, rhs_tile = right
    result = evaluate(program, RuntimeInputs(tiles={0: lhs_tile, 1: rhs_tile}))
    if set(result.outputs) != {0} or result.outputs[0].format != "bf16":
        raise ValueError("expected exactly one BF16 output at index 0")
    actual_bits = torch.tensor(result.outputs[0].bits, dtype=torch.uint16).reshape(32, 32)
    actual = actual_bits.view(torch.bfloat16)
    # Virtual transpose makes W[N,K]; evaluator passes W.T, so reference is A@RHS.
    dot = RtlNumerics.systolic_matmul(lhs, rhs, torch.zeros((32, 32), dtype=torch.bfloat16))
    reference = RtlNumerics.unary("relu", dot).contiguous()
    mismatches = int((actual_bits != reference.view(torch.uint16)).sum().item())
    max_error = float((actual.float() - reference.float()).abs().max().item())
    ordinary = torch.relu((lhs.float() @ rhs.float()).to(torch.bfloat16)).contiguous()
    diagnostic_mismatches = int((actual_bits != ordinary.view(torch.uint16)).sum().item())
    diagnostic_error = float((actual.float() - ordinary.float()).abs().max().item())
    passed = mismatches == 0
    print(f"{name}: {'PASS' if passed else 'FAIL'}; mismatches={mismatches}/1024; max_abs_error={max_error:.9g}; "
          f"PyTorch diagnostic: mismatches={diagnostic_mismatches}/1024, max_abs_error={diagnostic_error:.9g}")
    if mismatches:
        row, col = (actual_bits != reference.view(torch.uint16)).nonzero()[0].tolist()
        print(f"  First mismatch [{row},{col}]: actual=0x{int(actual_bits[row, col]):04x}, "
              f"reference=0x{int(reference.view(torch.uint16)[row, col]):04x}")
    return passed


def main():
    torch.set_num_threads(1)
    program = parse_program(MLIR.read_text(), function="ttir_matmul_relu_kernel_atlas")
    if dict(program.input_formats) != {0: "fp8", 1: "fp8"} or tuple(program.output_indices) != (0,):
        raise ValueError("expected two FP8 inputs and one BF16 output at index 0")
    print(f"MLIR: {MLIR}")
    print(f"CPU; 32x32 row-major E4M3FN; scale_code={SCALE_CODE} (unity); unit 0")
    print("Criterion: bitwise BF16 equality with RTL systolic matmul + RTL-table ReLU; zero mismatches required.")
    print("Reference shares numerical primitives with evaluator; NOT independent RTL or hardware verification.")
    print("PyTorch is diagnostic only; it lacks per-MAC RTL rounding and MXU subnormal flushing.")
    print("Raw boundary cases bypass packer flushing; NaN encodings 0x7f/0xff are excluded.")
    total = failures = 0
    for name, left, right in cases():
        total += 1
        try:
            passed = check_case(program, name, left, right)
        except Exception as error:
            print(f"{name}: FAIL; {type(error).__name__}: {error}")
            passed = False
        failures += not passed
    print(f"{'PASS' if failures == 0 else 'FAIL'}: {total - failures}/{total} cases passed; {failures} failed")
    return int(failures != 0)


if __name__ == "__main__":
    sys.exit(main())
