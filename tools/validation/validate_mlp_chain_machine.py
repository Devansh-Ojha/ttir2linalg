#!/usr/bin/env python3
"""Validate a three-layer program against direct Atlas-policy source arithmetic.

The reference never interprets emitted MLIR. Numerical primitives are shared
with the software model; this is not independent RTL/silicon verification.
Artifacts must already exist; this command never rewrites them.
"""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import torch
from atlas_virtual_evaluator import operation_name, parse_program
from npu_model.configs.numerics import RtlNumerics

from validate_two_layer_mlp_frontend import fp8_tile, test_cases, transpose
from validate_two_layer_mlp_generated_machine import (
    derive_layout, load_instructions, require, run_case,
)


def reference(tiles):
    """Source [K,N] weights, MXU0/1/0, BF16 readout at every layer."""
    current = tiles[0]
    hidden = []
    for i, weight in enumerate(tiles[1:]):
        multiply = (RtlNumerics.systolic_matmul if i % 2 == 0 else
                    RtlNumerics.inner_product_matmul)
        current = multiply(current, weight, torch.zeros((32, 32), dtype=torch.bfloat16))
        require(current.dtype == torch.bfloat16, "reference readout must be BF16")
        if i < len(tiles) - 2:
            activated = RtlNumerics.unary("relu", current)
            current = RtlNumerics.to_fp8(activated, 127)
            hidden.append((activated, current))
    return current.view(torch.uint16).contiguous(), hidden


def cases():
    """Retain all 21 previous cases, adding an asymmetric third source weight."""
    generator = torch.Generator().manual_seed(20261029)
    third = fp8_tile((torch.randint(-16, 17, (32, 32), generator=generator).float() / 16)
                     .to(torch.bfloat16))
    for name, encoded in test_cases():
        # The old suite's weights are [N,K]; stage source [K,N] for this suite.
        weights = []
        for tensor, tile in encoded[1:]:
            weights.append((tensor.T.contiguous(), transpose(tile)))
        yield name, (encoded[0], *weights, third), None
    identity = fp8_tile(torch.eye(32, dtype=torch.bfloat16))
    for name, major, minor, scale, code in (
            ("second_pack_tie_even_low", 1.0, 0.0625, 1.0, 0x38),
            ("second_pack_tie_even_high", 1.125, 0.0625, 1.0, 0x3A),
            ("second_pack_underflow", 0.015625, 0.0, 0.5, 0x00),
            ("second_pack_saturation", 448.0, 0.0, 2.0, 0x7E)):
        x = torch.zeros((32, 32), dtype=torch.bfloat16)
        x[:, 0], x[:, 1] = major, minor
        weight = torch.zeros_like(x)
        weight[0, :], weight[1, :] = scale, scale
        yield name, (fp8_tile(x), identity, fp8_tile(weight), identity), code
    # Independent layout check: three permutations compose to a known shift.
    x = torch.tensor([[(r * 3 + c) % 5 - 2 for c in range(32)] for r in range(32)],
                     dtype=torch.bfloat16)
    weights = []
    for shift in (3, 5, 7):
        w = torch.zeros_like(x)
        for c in range(32):
            w[(c + shift) % 32, c] = 1
        weights.append(fp8_tile(w))
    yield "three_permutations", (fp8_tile(x), *weights), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("virtual", type=Path)
    parser.add_argument("lowered", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    program = parse_program(args.virtual.read_text())
    slots = tuple(sorted(program.input_formats))
    require(slots == (0, 1, 2, 3) and tuple(program.output_indices) == (0,),
            "validation suite requires default three-layer role slots")
    ops = program.operations
    units = [op.properties.get("unit", op.attributes.get("unit")).value.data
             for op in ops if operation_name(op) == "atlas.virtual_mxu_matmul"]
    require(units == [0, 1, 0], "expected MXU0 -> MXU1 -> MXU0")
    instructions, rows = load_instructions(args.lowered, expected_word_count=None)
    spec = program, slots, 0
    addresses, halves = derive_layout(rows, spec, depth=3)
    passed = total = 0
    for name, encoded, code in cases():
        expected, hidden = reference(tuple(pair[0] for pair in encoded))
        if code is not None:
            require(bool((hidden[1][1].view(torch.uint8) == code).all()),
                    f"{name}: case did not exercise the expected second-pack rounding")
        if name == "three_permutations":
            x = encoded[0][0].to(torch.bfloat16)
            wanted = torch.stack([torch.stack([x[r, (c + 15) % 32].clamp_min(0)
                                              for c in range(32)]) for r in range(32)])
            require(torch.equal(expected, wanted.view(torch.uint16)),
                    "reference disagrees with known permutation result")
        ok, _ = run_case(instructions, spec, addresses, halves, name, encoded,
                         source_weights=True, reference_bits=expected)
        total += 1
        passed += ok
    print(f"Three-layer machine/reference: {passed}/{total} passed; words={len(instructions)}")
    return int(passed != total)


if __name__ == "__main__":
    sys.exit(main())
