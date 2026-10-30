#!/usr/bin/env python3
"""Verify/lower/simulate eligible graphs against an independent source reference.

Fresh output directories only. Backend/model code is read-only. Machine words
are compared with simulator instruction encodings before executing that stream.
"""
from __future__ import annotations
from compiler.toolchain_paths import atlas_root as default_atlas_root
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import torch
from compiler.ttir_reader import parse_ttir
from compiler.ttir_operation_analysis import analyze_operations
from compiler.atlas_eligibility import BF16_RTL_POLICY, BF16TileABI, TileBinding
from compiler.atlas_operation_mapping import emit_atlas_operations, AtlasOperationMemory
from atlas_bf16_reference import source_reference
from validate_two_layer_mlp_generated_machine import (
    adapt, load_instructions, METADATA, require,
)
from npu_model.configs import isa_definition as isa
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.software.program import InstantiableProgram

ROOT = Path(__file__).resolve().parents[2]


def instruction_adapter(row):
    if row['operation'] == 'atlas.vli':
        f={k:v for k,v in row['fields'].items() if k not in METADATA}
        require(set(f)=={'mode','dst','immediate'} and f['mode']=='all', 'unsupported VLI variant')
        require(type(f['dst']) is int and 0<=f['dst']<64 and type(f['immediate']) is int and 0<=f['immediate']<65536,'invalid VLI operands')
        return isa.VLI_ALL(vd=f['dst'],imm=f['immediate'])
    if row['operation'] != 'atlas.vpu_binary':
        return adapt(row)
    f={k:v for k,v in row['fields'].items() if k not in METADATA}
    require(set(f)=={'kind','dst','lhs','rhs'}, 'unexpected binary instruction fields')
    cls={'add':isa.VADD_BF16,'sub':isa.VSUB_BF16,'mul':isa.VMUL_BF16}.get(f['kind'])
    require(cls is not None, 'unsupported binary instruction')
    require(all(type(f[k]) is int and 0<=f[k]<64 for k in ('dst','lhs','rhs')), 'invalid vector register')
    return cls(vd=f['dst'],vs1=f['lhs'],vs2=f['rhs'])


def encode(bits):
    return b''.join(bits[r*32+c].to_bytes(2,'little') for half in (0,16) for r in range(32) for c in range(half,half+16))


def decode(data):
    words=[int.from_bytes(data[i:i+2],'little') for i in range(0,2048,2)]
    return [words[(c//16)*512+r*16+c%16] for r in range(32) for c in range(32)]


def cases(count):
    """Each lane pattern repeats across asymmetric row/column positions."""
    patterns=[
        ('signed_zero', [0,0x8000,0x3f80,0xbf80]),
        ('subnormals', [1,2,0x7f,0x80,0x8001,0x807f,0x8080]),
        ('overflow_underflow', [0x7f7f,0xff7f,0x0080,0x8080,0x3f00,0x4000,1,0x8001]),
        ('nan_infinity', [0x7f80,0xff80,0x7fc0,0xffc0,0x7f81,0xff81,0x7fff,0xffff,0,0x8000,0x3f80]),
        ('truncation', [0x3f80,0x3bc0,0xbf80,0xbbc0,0x3f81,0xbf81]),
        ('asymmetric', [0x3f80,0xc000,0x3f00,0x4040,0xbf00,0x3e80,0x4080]),
    ]
    for name, pattern in patterns:
        yield name, [[pattern[(r*7+c*3+i*5)%len(pattern)] for r in range(32) for c in range(32)] for i in range(count)]
    generator=torch.Generator().manual_seed(20261029)
    for i in range(2):
        # Includes arbitrary NaN payloads, signs, subnormals and finite extremes.
        yield f'raw_random_{i}', [torch.randint(0,65536,(1024,),generator=generator).tolist() for _ in range(count)]


def validation_abi(analysis):
    """Test harness bindings, explicitly staged; never used by the emitter."""
    args={v.id:i for i,v in enumerate(analysis.function.args)}
    roles={args[m.pointer.id]:('input' if m.operation.name=='tt.load' else 'output') for m in analysis.memory}
    counts={'input':0,'output':0}; bindings=[]
    for i in range(len(args)):
        role=roles[i]
        slot=2*counts[role]+(1 if role=='output' else 0)
        counts[role]+=1
        bindings.append(TileBinding(i,role,slot))
    return BF16TileABI(tuple(bindings))


def run_case(analysis,abi,memory,instructions,name,tiles,*,reference=source_reference):
    inputs=[b for b in abi.bindings if b.role=='input']
    outputs=[b for b in abi.bindings if b.role=='output']
    arguments={b.argument:bits for b,bits in zip(inputs,tiles)}
    expected=reference(analysis,arguments)
    regions=[]; preserved=[]; sentinels={}
    for b,bits in zip(inputs,tiles):
        address=memory.input_base+b.slot*2048
        data=encode(bits)
        regions.append((address,data)); preserved.append((address,data))
    for b in outputs:
        address=memory.output_base+b.slot*2048
        sentinels[b.argument]=bytes(v^255 for v in encode(expected[b.argument]))
        regions.append((address,sentinels[b.argument]))
    for b in abi.bindings:
        address=(memory.input_base if b.role=='input' else memory.output_base)+b.slot*2048
        for a,data in [(address-32,b'\x5a'*32),(address+2048,b'\xa5'*32)]:
            regions.append((a,data));preserved.append((a,data))
    ordered=sorted((a,len(d)) for a,d in regions)
    require(all(a+n<=b for (a,n),(b,_) in zip(ordered,ordered[1:])), 'overlapping validation memory')
    program=InstantiableProgram(instructions)
    program.dram_base=0
    program.memory_regions=[(a,torch.tensor(list(d),dtype=torch.uint8)) for a,d in regions]
    config=DefaultHardwareConfig()
    config.arch_state_config=replace(config.arch_state_config,numerics='rtl',randomize_init=False)
    sim=None
    try:
        sim=Simulation(config,LoggerConfig(filename='/dev/null'),program,verbose=False,ignore_runtime_errors=False)
        sim.run(max_cycles=100000)
        state=sim.core.arch_state
        require(state.halted and state.halt_reason=='ecall' and state.read_csrf(0xc10)==1,'machine did not complete')
        require(not sim.runtime_errors,'machine runtime/timing errors')
        mismatches=0
        for b in outputs:
            data=bytes(state.read_dram(memory.output_base+b.slot*2048,2048).tolist())
            actual=decode(data)
            mismatch=sum(a!=e for a,e in zip(actual,expected[b.argument]))
            if mismatch:
                at=next(i for i,(a,e) in enumerate(zip(actual,expected[b.argument])) if a!=e)
                raise ValueError(f'{name}: output arg {b.argument}, lane {at}, machine={actual[at]:04x}, reference={expected[b.argument][at]:04x}; {mismatch} mismatches')
            require(all(a!=b for a,b in zip(data,sentinels[b.argument])),'output bytes not all overwritten')
            mismatches+=mismatch
        require(all(bytes(state.read_dram(a,len(d)).tolist())==d for a,d in preserved),'input or guard modified')
        print(f'{analysis.function.name}/{name}: PASS; mismatches={mismatches}; cycles={sim.cycle_count}',flush=True)
        return {'case':name,'cycles':sim.cycle_count,'mismatches':mismatches}
    finally:
        if sim is not None:
            sim.close()


def validate(source,output,atlas_root=None):
    atlas_root = Path(atlas_root) if atlas_root is not None else default_atlas_root()
    analysis=analyze_operations(parse_ttir(source.read_text()).funcs[0])
    abi=validation_abi(analysis); memory=AtlasOperationMemory()
    virtual=emit_atlas_operations(analysis,numerical_policy=BF16_RTL_POLICY,abi=abi,memory=memory)
    # Unsupported graphs cannot create even an empty output directory.
    output.mkdir(parents=True,exist_ok=False)
    path=output/'program.virtual.mlir'
    path.write_text(virtual)
    subprocess.run([str(atlas_root/'build/bin/atlas-opt'),'--verify-atlas-virtual-stream',str(path),'-o',str(output/'verified.mlir')],check=True,capture_output=True,text=True)
    subprocess.run([sys.executable,str(atlas_root/'tools/lower_atlas_program.py'),str(path),'--output-dir',str(output/'lowered')],check=True,capture_output=True,text=True)
    instructions,rows=load_instructions(output/'lowered',expected_word_count=None,adapter=instruction_adapter)
    results=[run_case(analysis,abi,memory,instructions,name,tiles) for name,tiles in cases(sum(b.role=='input' for b in abi.bindings))]
    report={'source':str(source.resolve()),'policy':BF16_RTL_POLICY,'words':len(rows),'cases':results,'reference':'exact integer/Fraction source graph, no RtlNumerics or emitted-MLIR interpretation'}
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources',type=Path,nargs='+')
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.out_dir.exists():
        parser.error('choose a fresh output directory')
    # Preflight every source before creating the run directory.
    for p in args.sources:
        a=analyze_operations(parse_ttir(p.read_text()).funcs[0])
        emit_atlas_operations(a,numerical_policy=BF16_RTL_POLICY,abi=validation_abi(a))
    torch.set_num_threads(1)
    for i,p in enumerate(args.sources):
        report=validate(p,args.out_dir/f'{i}-{p.stem}')
        print(f'{p.name}: {len(report["cases"])}/{len(report["cases"])} passed; words={report["words"]}',flush=True)
    return 0


if __name__=='__main__':
    sys.exit(main())
