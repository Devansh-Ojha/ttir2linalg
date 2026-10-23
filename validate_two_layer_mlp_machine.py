#!/usr/bin/env python3
"""Validate the existing two-layer Atlas MLP physical program against virtual MLIR.

No trace artifacts: logging goes to /dev/null. Numerical primitives are shared
with the virtual evaluator; this is not independent RTL/hardware verification.
"""
import sys
sys.dont_write_bytecode = True

import json
import re
from dataclasses import replace
from pathlib import Path

from validate_matmul_relu_extended import cases as previous_cases, raw_tile
from validate_matmul_relu import RuntimeInputs, evaluate, parse_program, torch, fp8_tile
from npu_model.configs import isa_definition as isa
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.software.program import InstantiableProgram

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "out/atlas/two_layer_mlp.lowered"
MAX_CYCLES = 100000
MLIR = Path.home() / "atlas-mlir/test/examples/virtual_fp8_two_layer_mlp.mlir"
INPUT_ADDRESSES = (0x90000000, 0x90000800, 0x90001000)
OUTPUT = 0x90002000
METADATA = {
    "atlas.virtual_cfg_block", "atlas.virtual_cfg_source",
    "atlas.virtual_cfg_operation", "atlas.virtual_tensor_result",
    "atlas.virtual_tile_command", "atlas.virtual_mxu_command",
    "atlas.reason", "atlas.complete", "atlas.virtual_cfg_helper",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def adapt_common(row):
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


def adapt(row):
    fields = {k: v for k, v in row["fields"].items() if k not in METADATA}
    op = row["operation"]
    if op in ("atlas.mxu_push", "atlas.mxu_matmul", "atlas.mxu_pop"):
        unit = fields.get("unit")
        require(type(unit) is int and unit in (0, 1), "unsupported MXU unit")
        normalized = dict(row, fields=dict(row["fields"], unit=0))
        original = adapt_common(normalized)
        if unit == 0:
            return original
        if op == "atlas.mxu_push":
            return isa.VMATPUSH_WEIGHT_MXU1(vd=original.vd, vs1=original.vs1)
        if op == "atlas.mxu_matmul":
            return isa.VMATMUL_MXU1(vd=original.vd, vs1=original.vs1, vs2=original.vs2)
        return isa.VMATPOP_BF16_ACC_MXU1(vd=original.vd, vs2=original.vs2)
    if op == "atlas.scalar_load" and fields.get("kind") == "seli":
        require(set(fields) == {"kind", "base", "dst", "offset"} and fields["base"] == 0,
                "unsupported SELI fields/base")
        require(type(fields["dst"]) is int and 0 <= fields["dst"] < 32 and
                type(fields["offset"]) is int and 0 <= fields["offset"] < 4096, "invalid SELI operands")
        return isa.SELI(rd=fields["dst"], imm=fields["offset"])
    specs = {
        "atlas.scalar_load": (isa.LW, {"kind": "lw"}, {"rd": "dst", "rs1": "base", "imm": "offset"}),
        "atlas.scalar_store": (isa.SW, {"kind": "sw"}, {"rs2": "src", "rs1": "base", "imm": "offset"}),
        "atlas.vpu_pack": (isa.VPACK_BF16_FP8, {"direction": "bf16_to_fp8"}, {"vd": "dst", "vs2": "src", "es1": "scale_reg"}),
        "atlas.branch": (isa.BLT, {"kind": "blt"}, {"rs1": "lhs", "rs2": "rhs", "imm": "offset_bytes"}),
    }
    if op not in specs:
        return adapt_common(row)
    cls, fixed, mapping = specs[op]
    require(set(fields) == set(fixed) | set(mapping.values()), f"unexpected {op} fields")
    for key, value in fixed.items():
        require(type(fields[key]) is type(value) and fields[key] == value, f"unsupported {op} variant")
    args = {}
    for arg, field in mapping.items():
        value = fields[field]
        require(type(value) is int, f"{field} must be integer")
        low, high = (0, 63 if arg in ("vd", "vs2") else 31)
        if arg == "imm":
            low, high = (-4096, 4094) if op == "atlas.branch" else (-2048, 2047)
        require(low <= value <= high, f"invalid {op} operand {field}={value}")
        args[arg] = value
    if op == "atlas.branch":
        require(args["imm"] % 2 == 0, "branch encoded immediate must be even")
    return cls(**args)


def load_instructions():
    document = json.loads((ARTIFACTS / "5-program.json").read_text())
    require(document.get("schema") == "atlas.physical_program.v1", "unsupported schema")
    require(document.get("entry_word_index") == 0 and document.get("pc_unit") == "instruction_word", "unsupported entry/PC")
    require(document.get("word_endianness") == "little", "unsupported endianness")
    lines = (ARTIFACTS / "5-words.hex").read_text().splitlines()
    require(all(re.fullmatch(r"[0-9a-fA-F]{8}", line) for line in lines), "invalid hex")
    words = [int(line, 16) for line in lines]
    rows = document["instructions"]
    require(document.get("word_count") == len(rows) == len(words) == 97, "expected 97 words")
    instructions = []
    for index, (row, word) in enumerate(zip(rows, words)):
        try:
            require(type(row["word_index"]) is int and row["word_index"] == index, "noncontiguous indices")
            control = {"kind": "trap" if row["operation"] == "atlas.trap" else "sequential"}
            if row["operation"] == "atlas.branch":
                target = index + row["fields"]["offset_bytes"] // 2
                require(0 <= target < len(rows), "branch target out of range")
                control = {"kind": "conditional_direct", "target_word_index": target,
                           "delay_slot_word_index": index + 1, "fallthrough_word_index": index + 2}
            require(row["control"] == control, "control metadata mismatch")
            if row["operation"] == "atlas.delay":
                require(row["explicit_delay_cycles"] == row["fields"]["cycles"], "delay metadata mismatch")
            instruction = adapt(row)
            require(type(row["word_u32"]) is int and instruction.to_bytecode() == row["word_u32"] == word == int(row["word_hex"], 16),
                    f"encoding mismatch: model={instruction.to_bytecode():08x}, JSON={row['word_u32']:08x}, hex={word:08x}")
            instructions.append(instruction)
        except Exception as error:
            raise ValueError(f"word {index} ({row['operation']}): {error}") from error
    verify_layout(rows)
    print(f"Encoding verification: PASS; {len(instructions)}/{len(words)} model words == JSON == hex", flush=True)
    return instructions


def verify_layout(rows):
    """Follow straight-line scalar setup to confirm all physical DMA spans.

    The sole helper loop updates only scratch pointers/counter; none of those
    values feed DMA, and all output pointers are subsequently initialized.
    """
    x = [0] * 32
    base = 0
    transfers, stores = [], {}
    for row in rows:
        f, op = row["fields"], row["operation"]
        if op == "atlas.upper":
            x[f["dst"]] = (f["immediate"] << 12) & 0xFFFFFFFF
        elif op == "atlas.alu_imm":
            if f["dst"]:
                x[f["dst"]] = (x[f["src"]] + f["immediate"]) & 0xFFFFFFFF
        elif op == "atlas.dma_config":
            base = x[f["base_reg"]]
        elif op == "atlas.vstore":
            require(f["offset"] == 0, "unexpected store offset")
            stores[x[f["base"]] * 4] = f["src"]
        elif op == "atlas.dma":
            address = (base << 32) | x[f["dram"]]
            vmem, size = x[f["reg"]] * 4, x[f["size"]]
            transfers.append((f["direction"], address, size, vmem))
            if f["direction"] == "store":
                expected_reg = 32 if address == OUTPUT else 33
                require(stores.get(vmem) == expected_reg, "unexpected output register/store layout")
        elif op == "atlas.branch":
            require(f["kind"] == "blt" and f["lhs"] == 13 and f["rhs"] == 14 and
                    row["word_index"] == 69 and f["offset_bytes"] == -56,
                    "layout analysis only supports the inspected pack helper loop")
            require(all(r["operation"] in {"atlas.scalar_load", "atlas.scalar_store", "atlas.delay", "atlas.alu_imm"}
                        for r in rows[41:69]), "unexpected effects in pack loop")
            require(all(r["fields"]["dst"] in (10, 11, 12, 13)
                        for r in rows[41:69] if r["operation"] == "atlas.alu_imm"), "unexpected loop scalar writes")
    expected = [("load", a, 1024, i * 2048) for i, a in enumerate(INPUT_ADDRESSES)]
    expected += [("store", OUTPUT, 1024, 0x40000), ("store", OUTPUT + 1024, 1024, 0x40400)]
    require(transfers == expected, f"unexpected physical DMA layout: {transfers}")
    require(rows[-1]["operation"] == "atlas.trap" and rows[-1]["fields"]["kind"] == "ecall", "missing final ecall")
    require({k: v for k, v in rows[-2]["fields"].items() if k not in METADATA} ==
            {"address": 0xC10, "dst": 0, "kind": "rrw", "source": 1} and x[1] == 1,
            "unexpected completion convention")
    print("Physical layout: X=0x90000000 W0=0x90000800 W1=0x90001000; output halves=0x90002000/0x90002400")


def test_cases():
    generator = torch.Generator().manual_seed(20261022)
    w1 = fp8_tile((torch.randint(-64, 65, (32, 32), generator=generator).float() / 32).to(torch.bfloat16))
    old_cases = list(previous_cases())
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


def run_case(instructions, virtual, name, tiles):
    sim = None
    try:
        inputs = {i: tile[1] for i, tile in enumerate(tiles)}
        result = evaluate(virtual, RuntimeInputs(tiles=inputs))
        require(set(result.outputs) == {0} and result.outputs[0].format == "bf16", "unexpected virtual output")
        bits = result.outputs[0].bits
        expected = torch.tensor(bits, dtype=torch.uint16).reshape(32, 32)
        expected_bytes = b"".join(bits[r * 32 + c].to_bytes(2, "little")
                                  for half in (0, 16) for r in range(32) for c in range(half, half + 16))
        sentinel = bytes(b ^ 255 for b in expected_bytes)
        preserved = [(a, bytes(inputs[i].bits)) for i, a in enumerate(INPUT_ADDRESSES)]
        for address, size in [(a, 1024) for a in INPUT_ADDRESSES] + [(OUTPUT, 2048)]:
            preserved += [(address - 32, b"\x5a" * 32), (address + size, b"\xa5" * 32)]
        program = InstantiableProgram(instructions)
        program.dram_base = 0
        program.memory_regions = [(a, torch.tensor(list(data), dtype=torch.uint8))
                                  for a, data in preserved + [(OUTPUT, sentinel)]]
        config = DefaultHardwareConfig()
        config.arch_state_config = replace(config.arch_state_config, numerics="rtl", randomize_init=False)
        sim = Simulation(config, LoggerConfig(filename="/dev/null"), program,
                         verbose=False, ignore_runtime_errors=False)
        sim.run(max_cycles=MAX_CYCLES)
        state = sim.core.arch_state
        halves = [state.read_dram(OUTPUT + offset, 1024).clone() for offset in (0, 1024)]
        decoded = []
        for raw in halves:
            pairs = raw.to(torch.int32).reshape(-1, 2)
            decoded.append((pairs[:, 0] | (pairs[:, 1] << 8)).reshape(32, 16))
        actual = torch.cat(decoded, dim=1).to(torch.uint16).contiguous()
        mismatches = int((actual != expected).sum())
        error = float((actual.view(torch.bfloat16).float() - expected.view(torch.bfloat16).float()).abs().max())
        overwritten = sum(a != b for a, b in zip(bytes(torch.cat(halves).tolist()), sentinel))
        memory_ok = all(bytes(state.read_dram(a, len(data)).tolist()) == data for a, data in preserved)
        complete = state.halted and state.halt_reason == "ecall" and state.read_csrf(0xC10) == 1
        passed = complete and mismatches == 0 and overwritten == 2048 and memory_ok and not sim.runtime_errors
        print(f"{name}: {'PASS' if passed else 'FAIL'}; cycles={sim.cycle_count}; mismatches={mismatches}/1024; "
              f"max_abs_error={error:.9g}; halted={state.halted}; halt_reason={state.halt_reason}; "
              f"CSR[0xC10]={state.read_csrf(0xC10)}; overwritten={overwritten}/2048; inputs_guards_preserved={memory_ok}", flush=True)
        if mismatches:
            r, c = (actual != expected).nonzero()[0].tolist()
            print(f"  First mismatch [{r},{c}]: machine=0x{int(actual[r,c]):04x}, virtual=0x{int(expected[r,c]):04x}")
        return passed
    except Exception as error:
        details = ""
        if sim is not None:
            state = sim.core.arch_state
            details = f"; cycle={sim.core.cycle_count}; execute_pc={state.execute_pc}; halted={state.halted}; halt_reason={state.halt_reason}; CSR={state.read_csrf(0xC10)}"
        print(f"{name}: FAIL; {type(error).__name__}: {error}{details}", flush=True)
        print("  Timing/runtime errors may indicate compiler/model incompatibility; instructions were not altered.")
        return False
    finally:
        if sim is not None:
            sim.close()


def main():
    torch.set_num_threads(1)
    instructions = load_instructions()
    virtual = parse_program(MLIR.read_text(), function="two_layer_mlp")
    require(dict(virtual.input_formats) == {0: "fp8", 1: "fp8", 2: "fp8"} and tuple(virtual.output_indices) == (0,), "unexpected virtual interface")
    print("Reference: virtual X @ W0.T -> MXU0 BF16 -> ReLU -> FP8 pack(scale_code=127) -> MXU1 with W1.T.")
    print("Criterion: bitwise BF16 equality. Shared numerical primitives; NOT independent RTL/silicon verification.")
    planned = list(test_cases())
    passed = attempted = 0
    for name, tiles in planned:
        attempted += 1
        ok = run_case(instructions, virtual, name, tiles)
        passed += ok
        if attempted == 1 and not ok:
            break
    failed = attempted - passed
    print(f"{'PASS' if failed == 0 else 'FAIL'}: {passed}/{attempted} cases passed; {failed} failed; "
          f"{len(planned)-attempted} skipped; verified machine words={len(instructions)}")
    return int(failed != 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
