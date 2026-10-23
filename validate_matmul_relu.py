#!/usr/bin/env python3
"""Validate existing Atlas virtual matmul/ReLU using the installed RTL model.

Run from ttir2linalg with atlas-eval-venv and the model/tools on PYTHONPATH.
This checks evaluator/model consistency, not independent hardware correctness.
"""
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import torch
from atlas_virtual_evaluator import RuntimeInputs, Tile, evaluate, parse_program
from npu_model.configs.numerics import RtlNumerics


SEED = 20261021
SCALE_CODE = 127  # E8M0 unity: 2**(127 - 127); no extra MXU scale.
MLIR = Path(__file__).resolve().parent / "out/atlas/matmul_relu_generated.virtual.mlir"


def fp8_tile(values):
    # Pack with the actual RTL converter, then reinterpret rather than cast bits.
    encoded = RtlNumerics.to_fp8(values, SCALE_CODE).contiguous()
    return encoded, Tile("fp8", tuple(encoded.view(torch.uint8).flatten().tolist()))


def main():
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    # Signed, asymmetric, noninteger values exercise rounding and RHS layout.
    lhs_source = (torch.randint(-96, 97, (32, 32), generator=generator).float() / 32).to(torch.bfloat16)
    rhs_source = (torch.randint(-80, 81, (32, 32), generator=generator).float() / 32).to(torch.bfloat16)
    lhs, lhs_tile = fp8_tile(lhs_source)
    rhs, rhs_tile = fp8_tile(rhs_source)
    program = parse_program(MLIR.read_text(), function="ttir_matmul_relu_kernel_atlas")
    if dict(program.input_formats) != {0: "fp8", 1: "fp8"} or tuple(program.output_indices) != (0,):
        raise ValueError("expected two FP8 inputs and one BF16 output at index 0")
    result = evaluate(program, RuntimeInputs(tiles={0: lhs_tile, 1: rhs_tile}))
    if set(result.outputs) != {0} or result.outputs[0].format != "bf16":
        raise ValueError("expected exactly one BF16 output at index 0")
    actual_bits = torch.tensor(result.outputs[0].bits, dtype=torch.uint16).reshape(32, 32)
    actual = actual_bits.view(torch.bfloat16)

    # MLIR transposes RHS to W[N,K]; MXU consumes W.T, hence A @ RHS.
    # Unit 0 rounds each custom FMA in K order, starting from positive zero.
    accumulator = torch.zeros((32, 32), dtype=torch.bfloat16)
    dot = RtlNumerics.systolic_matmul(lhs, rhs, accumulator)
    reference = RtlNumerics.unary("relu", dot).contiguous()
    reference_bits = reference.view(torch.uint16)
    mismatches = int((actual_bits != reference_bits).sum().item())
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    max_error = float((actual.float() - reference.float()).abs().max().item())

    # Diagnostic only: ordinary FP32 matmul lacks per-MAC RTL rounding.
    ordinary = torch.relu((lhs.float() @ rhs.float()).to(torch.bfloat16))
    ordinary_error = float((actual.float() - ordinary.float()).abs().max().item())
    ordinary_mismatches = int((actual_bits != ordinary.contiguous().view(torch.uint16)).sum().item())
    passed = finite and mismatches == 0
    print(f"MLIR: {MLIR}")
    print(f"Seed: {SEED}; CPU; two row-major 32x32 E4M3FN tiles; scale_code={SCALE_CODE} (unity)")
    print("Reference: RtlNumerics unit-0 systolic matmul + RTL-table ReLU")
    print("Comparison: bitwise BF16; threshold = 0 differing encodings (shared deterministic RTL model)")
    print("Output sample [0:2, 0:8]:", actual[:2, :8].float().tolist())
    print("Reference sample [0:2, 0:8]:", reference[:2, :8].float().tolist())
    print(f"Maximum absolute error: {max_error:.9g}")
    print(f"Mismatch count: {mismatches}/1024; all finite: {finite}")
    print(f"Ordinary PyTorch diagnostic only: max absolute error={ordinary_error:.9g}; BF16 mismatches={ordinary_mismatches}/1024")
    print("Limitation: shared model primitives; one finite-input case, not independent RTL or hardware validation.")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        raise
