#!/usr/bin/env python3
"""Execute encoding-verified compiled Atlas instructions against virtual MLIR.

No trace artifacts: logging goes to /dev/null. Numerical primitives are shared
with the virtual evaluator; this is not independent RTL/hardware verification.
"""
import sys
sys.dont_write_bytecode = True

import json
import re
from dataclasses import replace
from pathlib import Path

from validate_matmul_relu_extended import cases
from validate_matmul_relu import MLIR, RuntimeInputs, evaluate, parse_program, torch
from npu_model.configs import isa_definition as isa
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.software.program import InstantiableProgram

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "out/atlas/matmul_relu_generated.lowered"
MAX_CYCLES = 100000
LHS, RHS, OUTPUT = 0x90000000, 0x90000800, 0x90002000
METADATA = {
    "atlas.virtual_cfg_block", "atlas.virtual_cfg_source",
    "atlas.virtual_cfg_operation", "atlas.virtual_tensor_result",
    "atlas.virtual_tile_command", "atlas.virtual_mxu_command",
    "atlas.reason", "atlas.complete",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def adapt(row):
    """Accept only the exact physical instruction variants this workload uses."""
    fields = {key: value for key, value in row["fields"].items() if key not in METADATA}
    operation = row["operation"].removeprefix("atlas.")
    # Class, fixed variant fields, constructor-argument -> JSON-field mapping.
    specs = {
        "alu_imm": (isa.ADDI, {"kind": "addi"}, {"rd": "dst", "rs1": "src", "imm": "immediate"}),
        "upper": (isa.LUI, {"kind": "lui"}, {"rd": "dst", "imm": "immediate"}),
        "dma_config": (isa.DMA_CONFIG, {}, {"rs1": "base_reg", "ch": "channel"}),
        "dma_wait": (isa.DMA_WAIT, {}, {"ch": "channel"}),
        "vload": (isa.VLOAD, {"format": "raw"}, {"vd": "dst", "rs1": "base", "imm": "offset"}),
        "vstore": (isa.VSTORE, {"format": "raw"}, {"vd": "src", "rs1": "base", "imm": "offset"}),
        "delay": (isa.DELAY, {}, {"imm": "cycles"}),
        "xlu_transpose": (isa.VTRPOSE_XLU, {}, {"vd": "dst", "vs1": "src"}),
        "mxu_push": (isa.VMATPUSH_WEIGHT_MXU0, {"kind": "weight_fp8", "unit": 0}, {"vd": "slot", "vs1": "src"}),
        "mxu_matmul": (isa.VMATMUL_MXU0, {"unit": 0, "accumulate": False}, {"vd": "acc_slot", "vs1": "src", "vs2": "weight_slot"}),
        "mxu_pop": (isa.VMATPOP_BF16_ACC_MXU0, {"unit": 0, "format": "bf16", "scale_reg": 0}, {"vd": "dst", "vs2": "slot"}),
        "vpu_unary": (isa.VRELU_BF16, {"kind": "relu"}, {"vd": "dst", "vs1": "src"}),
        "csr": (isa.CSRRW, {"kind": "rrw"}, {"rd": "dst", "rs1": "source", "imm": "address"}),
        "trap": (isa.ECALL, {"kind": "ecall"}, {}),
    }
    if operation == "dma":
        direction = fields.get("direction")
        require(direction in ("load", "store"), "unsupported DMA direction")
        cls = isa.DMA_LOAD if direction == "load" else isa.DMA_STORE
        mapping = {"rd": "reg", "rs1": "dram", "rs2": "size", "ch": "channel"}
        if direction == "store":
            mapping.update(rd="dram", rs1="reg")
        spec = cls, {"direction": direction}, mapping
    else:
        require(row["operation"].startswith("atlas.") and operation in specs,
                f"unsupported operation {row['operation']}")
        spec = specs[operation]
    cls, fixed, mapping = spec
    require(set(fields) == set(fixed) | set(mapping.values()),
            f"unexpected/missing fields for {operation}: {fields}")
    for key, expected in fixed.items():
        require(type(fields[key]) is type(expected) and fields[key] == expected,
                f"unsupported {operation} variant {key}={fields[key]!r}")
    args = {}
    for argument, field in mapping.items():
        value = fields[field]
        require(type(value) is int, f"{field} must be an integer")
        if argument == "imm":
            low, high = (-2048, 2047) if operation == "alu_imm" else (0, 4095)
            if operation == "upper":
                high = (1 << 20) - 1
            require(low <= value <= high, f"out-of-range {operation} immediate {value}")
            if operation == "alu_imm":
                value &= 0xFFF  # Constructor retains raw encoded immediate bits.
        else:
            limit = 7 if argument == "ch" else 31
            if argument in ("vd", "vs1", "vs2"):
                limit = 63
            if field in ("slot", "acc_slot", "weight_slot"):
                limit = 1
            require(0 <= value <= limit, f"out-of-range {field}: {value}")
        args[argument] = value
    return cls(**args)


def load_instructions():
    document = json.loads((ARTIFACTS / "5-program.json").read_text())
    require(document.get("schema") == "atlas.physical_program.v1", "unsupported program schema")
    require(document.get("entry_word_index") == 0, "entry must be word zero")
    require(document.get("pc_unit") == "instruction_word", "unsupported PC unit")
    require(document.get("word_endianness") == "little", "unsupported word endianness")
    lines = (ARTIFACTS / "5-words.hex").read_text().splitlines()
    require(all(re.fullmatch(r"[0-9a-fA-F]{8}", line) for line in lines), "invalid hex word")
    words = [int(line, 16) for line in lines]
    rows = document["instructions"]
    require(document.get("word_count") == len(rows) == len(words) == 41, "expected exactly 41 words")
    instructions = []
    for index, (row, word) in enumerate(zip(rows, words)):
        try:
            require(type(row["word_index"]) is int and row["word_index"] == index, "noncontiguous word index")
            require(row["control"] == {"kind": "trap" if row["operation"] == "atlas.trap" else "sequential"},
                    "unsupported control variant")
            if row["operation"] == "atlas.delay":
                require(row["explicit_delay_cycles"] == row["fields"]["cycles"], "delay metadata mismatch")
            instruction = adapt(row)
            require(type(row["word_u32"]) is int and
                    instruction.to_bytecode() == row["word_u32"] == word == int(row["word_hex"], 16),
                    f"encoding mismatch: model=0x{instruction.to_bytecode():08x}, JSON={row['word_u32']:#x}, hex={word:#x}")
            instructions.append(instruction)
        except Exception as error:
            raise ValueError(f"word {index} ({row.get('operation')}): {error}") from error
    print("Encoding verification: PASS, 41/41 model words == JSON words == hex words")
    return instructions


def run_case(instructions, virtual, name, left, right):
    inputs = {0: left[1], 1: right[1]}
    expected = evaluate(virtual, RuntimeInputs(tiles=inputs)).outputs[0]
    require(expected.format == "bf16", "virtual output must be BF16")
    expected_bits = torch.tensor(expected.bits, dtype=torch.uint16).reshape(32, 32)
    # Explicit little-endian serialization, with the two 32x16 column halves.
    expected_bytes = b"".join(int(expected_bits[r, c]).to_bytes(2, "little")
                              for half in (0, 16) for r in range(32) for c in range(half, half + 16))
    # Each sentinel byte differs from its expected result, proving full overwrite.
    sentinel = bytes(byte ^ 0xFF for byte in expected_bytes)
    preserved = [(LHS, bytes(inputs[0].bits)), (RHS, bytes(inputs[1].bits))]
    for address, length in ((LHS, 1024), (RHS, 1024), (OUTPUT, 2048)):
        preserved.extend(((address - 32, bytes([0x5A]) * 32),
                          (address + length, bytes([0xA5]) * 32)))
    program = InstantiableProgram(instructions)
    program.dram_base = 0
    program.memory_regions = [(address, torch.tensor(list(data), dtype=torch.uint8))
                              for address, data in preserved + [(OUTPUT, sentinel[:1024]), (OUTPUT + 1024, sentinel[1024:])]]
    config = DefaultHardwareConfig()
    config.arch_state_config = replace(config.arch_state_config, numerics="rtl", randomize_init=False)
    sim = None
    try:
        sim = Simulation(config, LoggerConfig(filename="/dev/null"), program,
                         verbose=False, ignore_runtime_errors=False)
        sim.run(max_cycles=MAX_CYCLES)
        state = sim.core.arch_state
        complete = state.halted and state.halt_reason == "ecall" and state.read_csrf(0xC10) == 1
        halves = [state.read_dram(OUTPUT + offset, 1024).clone() for offset in (0, 1024)]
        # Decode explicitly: host endianness must not change the boundary layout.
        decoded = []
        for raw in halves:
            pairs = raw.to(torch.int32).reshape(-1, 2)
            decoded.append((pairs[:, 0] | (pairs[:, 1] << 8)).reshape(32, 16))
        actual_bits = torch.cat(decoded, dim=1).to(torch.uint16).contiguous()
        mismatches = int((actual_bits != expected_bits).sum().item())
        max_error = float((actual_bits.view(torch.bfloat16).float() - expected_bits.view(torch.bfloat16).float()).abs().max())
        actual_bytes = bytes(torch.cat(halves).tolist())
        overwritten = sum(a != b for a, b in zip(actual_bytes, sentinel))
        memory_ok = all(bytes(state.read_dram(address, len(data)).tolist()) == data
                        for address, data in preserved)
        passed = complete and mismatches == 0 and overwritten == 2048 and memory_ok and not sim.runtime_errors
        print(f"{name}: {'PASS' if passed else 'FAIL'}; cycles={sim.cycle_count}; mismatches={mismatches}/1024; "
              f"max_abs_error={max_error:.9g}; halted={state.halted}; halt_reason={state.halt_reason}; "
              f"CSR[0xC10]={state.read_csrf(0xC10)}; overwritten={overwritten}/2048; inputs_guards_preserved={memory_ok}", flush=True)
        if mismatches:
            row, col = (actual_bits != expected_bits).nonzero()[0].tolist()
            print(f"  First mismatch [{row},{col}]: machine=0x{int(actual_bits[row,col]):04x}, virtual=0x{int(expected_bits[row,col]):04x}")
        return passed
    except Exception as error:
        details = ""
        if sim is not None:
            state = sim.core.arch_state
            details = (f"; cycle={sim.core.cycle_count}; execute_pc={state.execute_pc}; "
                       f"halted={state.halted}; halt_reason={state.halt_reason}; CSR[0xC10]={state.read_csrf(0xC10)}")
        print(f"{name}: FAIL; {type(error).__name__}: {error}{details}", flush=True)
        return False
    finally:
        if sim is not None:
            sim.close()


def main():
    torch.set_num_threads(1)
    instructions = load_instructions()
    virtual = parse_program(MLIR.read_text(), function="ttir_matmul_relu_kernel_atlas")
    require(dict(virtual.input_formats) == {0: "fp8", 1: "fp8"} and tuple(virtual.output_indices) == (0,),
            "unexpected virtual interface")
    print("Compiled physical instruction execution vs virtual MLIR; criterion: zero BF16 bit mismatches.")
    print("Shared numerical primitives: NOT independent RTL/hardware verification. Cycle budget=100000.")
    all_cases = list(cases())
    baseline = next(case for case in all_cases if case[0] == "random_seed_20261021")
    ordered = [baseline] + [case for case in all_cases if case[0] != baseline[0]]
    failures = 0
    for index, (name, left, right) in enumerate(ordered):
        passed = run_case(instructions, virtual, name, left, right)
        failures += not passed
        if index == 0 and not passed:
            print("FAIL: baseline failed; extended cases not run.")
            return 1
    print(f"{'PASS' if failures == 0 else 'FAIL'}: {len(ordered) - failures}/{len(ordered)} cases passed; {failures} failed")
    return int(failures != 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
