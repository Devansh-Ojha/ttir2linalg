#!/usr/bin/env python3
"""Validate preserved genuine compiler TTIR through operation-based Atlas.

Capture first with compiler.jit_capture; this command never replaces captures.
Atlas-policy comparison is explicit, not a claim of CUDA/IEEE equivalence.
"""
from compiler.toolchain_paths import atlas_root as default_atlas_root
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from compiler.ttir_reader import parse_ttir
from compiler.triton_tile_normalization import normalize_static_tiles
from compiler.atlas_eligibility import FP8TileABI, TileBinding
from compiler.atlas_operation_mapping import emit_atlas_operations
from atlas_fp8_reference import source_reference
from validate_atlas_operations import instruction_adapter
from atlas_virtual_evaluator import parse_program
from validate_two_layer_mlp_frontend import test_cases, raw_tile, transpose
from validate_two_layer_mlp_generated_machine import load_instructions, derive_layout, run_case

POLICY = 'atlas-fp8-mlp-v1'


def validate(capture, output):
    metadata = json.loads((capture/'capture.json').read_text())
    text = (capture/'kernel.ttir').read_text()
    assert metadata['origin'] == 'triton.compile(ASTSource).asm[ttir]'
    assert metadata['target'] == {'backend':'cuda','arch':89,'warp_size':32}
    assert hashlib.sha256(text.encode()).hexdigest() == metadata['ttir_sha256']
    assert hashlib.sha256((capture/'kernel.source.py').read_bytes()).hexdigest() == metadata['source_sha256']
    normalized = normalize_static_tiles(parse_ttir(text).funcs[0], grid=metadata['grid'],
                                       numerical_policy=POLICY, resolve_dot_defaults=True)
    analysis = normalized.analysis
    count = len(analysis.function.args)-1
    abi = FP8TileABI(tuple(TileBinding(i,'input',i) for i in range(count)) +
                     (TileBinding(count,'output',0),))
    virtual = emit_atlas_operations(analysis,numerical_policy=POLICY,abi=abi)
    output.mkdir(parents=True,exist_ok=False)
    path = output/'program.virtual.mlir'; path.write_text(virtual)
    atlas = default_atlas_root()
    for cmd, log in [([str(atlas/'build/bin/atlas-opt'),'--verify-atlas-virtual-stream',str(path),'-o',str(output/'verified.mlir')],'verify'),
                     ([sys.executable,str(atlas/'tools/lower_atlas_program.py'),str(path),'--output-dir',str(output/'lowered')],'lower')]:
        result = subprocess.run(cmd,text=True,capture_output=True)
        (output/f'{log}.log').write_text(result.stdout+result.stderr)
        result.check_returncode()
    instructions, rows = load_instructions(output/'lowered',expected_word_count=None,adapter=instruction_adapter)
    program = parse_program(virtual); spec = program, tuple(range(count)), 0
    depth = sum(op.name=='tt.dot' for op in analysis.operations)
    addresses, halves = derive_layout(rows,spec,depth=depth,strict_legacy_layout=False)
    planned = []
    for name, encoded in test_cases():
        # Existing fixtures describe weights [N,K]; source Triton uses [K,N].
        source = (encoded[0],) + tuple((v.T.contiguous(),transpose(t)) for v,t in encoded[1:count])
        planned.append((name,source))
    planned.append(('all_raw_fp8_including_nan',tuple(raw_tile(tuple(range(256)),i*73) for i in range(count))))
    results = []
    for name, encoded in planned:
        expected = source_reference(analysis,{i:list(pair[1].bits) for i,pair in enumerate(encoded)})[count]
        expected = torch.tensor(expected,dtype=torch.uint16).reshape(32,32)
        ok, cycles = run_case(instructions,spec,addresses,halves,name,encoded,
                              source_weights=True,reference_bits=expected)
        results.append({'case':name,'passed':ok,'cycles':cycles})
    report = {'kernel':metadata['kernel'],'triton_version':metadata['triton_version'],
              'ttir_sha256':metadata['ttir_sha256'],'policy':POLICY,'words':len(rows),
              'address_operations':[o.index for o in normalized.address_operations],
              'precision_operations':[o.index for o in normalized.precision_operations],
              'bindings':[asdict(b) for b in abi.bindings], 'cases':results,
              'reference':'independent scalar integer/Fraction source SSA evaluation; no NPU arithmetic imports',
              'cuda_equivalence':False}
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return all(r['passed'] for r in results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path);p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    return int(not validate(args.capture,args.out_dir))

if __name__=='__main__':sys.exit(main())
