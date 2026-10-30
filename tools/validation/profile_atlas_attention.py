#!/usr/bin/env python3
"""Profile preserved machine artifacts while reusing all numerical/guard checks.

Busy-cycle counters overlap; they are not an exclusive cycle attribution.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import torch
import validate_transpose_operations as harness
from validate_quantized_attention import cases, BINDINGS, POLICY, instruction_adapter
from atlas_mixed_reference import source_reference
from compiler.ttir_reader import parse_ttir
from compiler.triton_tile_normalization import normalize_static_tiles
from compiler.atlas_eligibility import MixedTileABI
from compiler.atlas_operation_mapping import AtlasOperationMemory
from validate_two_layer_mlp_generated_machine import load_instructions


def profile(capture, lowered, output, selected=('bounded_random_0',), repeats=2):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    text = (capture/'kernel.ttir').read_bytes()
    norm = normalize_static_tiles(parse_ttir(text.decode()).funcs[0], grid=(1,1,1), numerical_policy=POLICY, resolve_dot_defaults=True)
    abi = MixedTileABI(BINDINGS, formats=tuple(a.type.elem for a in norm.analysis.function.args))
    instructions, rows = load_instructions(lowered, expected_word_count=None, adapter=instruction_adapter)
    report = {'ttir_sha256': hashlib.sha256(text).hexdigest(),
        'words_sha256': hashlib.sha256((lowered/'5-words.hex').read_bytes()).hexdigest(),
        'words': len(rows), 'static_instructions': dict(Counter(r['operation'] for r in rows)),
        'runs': [], 'counter_semantics': 'per-unit overlapping busy cycles; not exclusive stalls'}
    original = harness.Simulation
    class MeasuredSimulation(original):
        def __init__(self, *args, **kwargs):
            begin = time.perf_counter()
            kwargs['record_timeline'] = True
            args = list(args)
            trace = output/f'trace-{len(report["runs"])}.json'
            args[1] = replace(args[1], filename=str(trace))
            report['runs'][-1]['trace'] = trace.name
            super().__init__(*args, **kwargs)
            self.setup_seconds = time.perf_counter()-begin
        def run(self, *args, **kwargs):
            begin = time.perf_counter()
            super().run(*args, **kwargs)
            report['runs'][-1].update(setup_seconds=self.setup_seconds,
                run_wall_seconds=time.perf_counter()-begin, statistics=asdict(self.get_stats()))
            timeline_path = output/f'timeline-{len(report["runs"])}.json'
            timeline_path.write_text(json.dumps(self.timeline))
            report['runs'][-1]['timeline'] = timeline_path.name
            events = json.loads((output/report['runs'][-1]['trace']).read_text())
            execution = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'E']
            report['runs'][-1]['trace_execution_mnemonics'] = dict(Counter(
                e['name'].split(': ', 1)[1].split()[0] for e in execution))
            report['runs'][-1]['all_units_idle_cycles'] = sum(not any(row.values()) for row in self.timeline)
    try:
        harness.Simulation = MeasuredSimulation
        available = {name: tiles for name, tiles, _ in cases()}
        for name in selected:
            tiles = available[name]
            for repeat in range(repeats):
                begin = time.perf_counter()
                expected = source_reference(norm.analysis, {b.argument: t for b,t in zip(BINDINGS[:3], tiles)})
                reference_seconds = time.perf_counter()-begin
                run = {'case': name, 'repeat': repeat, 'reference_seconds': reference_seconds,
                       'input_sha256': hashlib.sha256(json.dumps(tiles).encode()).hexdigest()}
                report['runs'].append(run)
                run.update(harness.run_case(norm.analysis, abi, AtlasOperationMemory(), instructions,
                    name, tiles, lambda analysis, inputs: expected))
    finally:
        harness.Simulation = original
        (output/'profile.json').write_text(json.dumps(report, indent=2)+'\n')
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--lowered', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--case', action='append')
    p.add_argument('--repeats', type=int, default=2)
    a = p.parse_args()
    if a.repeats < 1: p.error('--repeats must be positive')
    profile(a.capture, a.lowered, a.out_dir, a.case or ('bounded_random_0',), a.repeats)
