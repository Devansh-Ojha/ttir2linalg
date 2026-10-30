#!/usr/bin/env python3
"""Genuine TTIR layout/composition validation with bit-preserving boundaries.

No source function-name recognition. Independent source SSA reference, actual
backend verification/lowering, word re-encoding, RTL simulator and guards.
"""
from compiler.toolchain_paths import atlas_root as default_atlas_root
import argparse
from dataclasses import replace,asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import torch
from compiler.ttir_reader import parse_ttir
from compiler.ttir_operation_analysis import transpose_order
from compiler.triton_tile_normalization import normalize_static_tiles
from compiler.atlas_eligibility import FP8LayoutABI,FP8TileABI,TileBinding,analyze_atlas_eligibility
from compiler.atlas_layout_operations import LAYOUT_POLICY
from compiler.atlas_operation_mapping import emit_atlas_operations,AtlasOperationMemory
from atlas_fp8_reference import source_reference as fp8_reference
from validate_atlas_operations import instruction_adapter as existing_adapter,encode,decode
from validate_two_layer_mlp_frontend import test_cases,transpose,raw_tile
from validate_two_layer_mlp_generated_machine import load_instructions,require
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.software.program import InstantiableProgram


def instruction_adapter(row):
    # Explicit virtual DMA addresses add backend scalar-provenance metadata.
    # It is not an ISA field; keep legacy adapters strict and normalize only
    # this documented non-encoding attribute in this additive adapter.
    fields=dict(row['fields'])
    for key in ('atlas.virtual_scalar_result','atlas.virtual_dma_transfer'):
        if key in fields:
            value=fields.pop(key)
            require(type(value) is int and value>=0,f'invalid provenance ID: {key}')
    return existing_adapter(dict(row,fields=fields))


def layout_reference(analysis,inputs):
    args={v.id:i for i,v in enumerate(analysis.function.args)};values={};outputs={}
    for op in analysis.function.body.ops:
        if op.name=='tt.load': out=list(inputs[args[op.operands[0].id]])
        elif op.name=='tt.trans':
            source=values[op.operands[0].id]
            out=source[:] if transpose_order(op)==(0,1) else [source[c*32+r] for r in range(32) for c in range(32)]
        elif op.name=='tt.store': outputs[args[op.operands[0].id]]=values[op.operands[1].id];continue
        elif op.name=='tt.return':continue
        else:raise ValueError(f'unqualified reference operation {op.name}')
        values[op.results[0].id]=out
    return outputs


def cases(layout):
    if not layout:
        for name,encoded in test_cases():
            source=(encoded[0],)+tuple((v.T.contiguous(),transpose(t)) for v,t in encoded[1:2])
            yield name,[list(t.bits) for _,t in source]
        yield 'all_raw_fp8_including_nan',[list(raw_tile(tuple(range(256)),i*73)[1].bits) for i in range(2)]
        return
    yield 'one_hot_3_17',[[0x38 if (r,c)==(3,17) else 0 for r in range(32) for c in range(32)]]
    yield 'asymmetric',[[((r*11+c*7)^(r>>1))&255 for r in range(32) for c in range(32)]]
    yield 'all_256_raw_encodings',[[i%256 for i in range(1024)]]
    yield 'row_indices',[[r for r in range(32) for _ in range(32)]]
    yield 'column_indices',[[c for _ in range(32) for c in range(32)]]
    yield 'signed_zeros',[[0x80 if (r*3+c)%5 else 0 for r in range(32) for c in range(32)]]
    yield 'nan_subnormal_bits',[[[0x7f,0xff,1,7,0x81,0x87,0x7e,0xfe][(r*5+c)%8] for r in range(32) for c in range(32)]]
    rng=random.Random(20261029)
    for i in range(3):yield f'raw_random_{i}',[[rng.randrange(256) for _ in range(1024)]]


def run_case(analysis,abi,memory,instructions,name,tiles,reference):
    inputs=[b for b in abi.bindings if b.role=='input'];outputs=[b for b in abi.bindings if b.role=='output']
    require(len(inputs)==len(tiles),'case input count does not match explicit bindings')
    expected=reference(analysis,{b.argument:t for b,t in zip(inputs,tiles)})
    require(set(expected)=={b.argument for b in outputs},'reference output arguments do not match bindings')
    regions=[];preserved=[];sentinels={}
    for b,t in zip(inputs,tiles):
        address=memory.input_base+b.slot*2048
        data=bytes(t) if analysis.function.args[b.argument].type.elem=='f8E4M3FN' else encode(t)
        regions.append((address,data));preserved.append((address,data))
    for b in outputs:
        fp8=analysis.function.args[b.argument].type.elem=='f8E4M3FN'
        target=bytes(expected[b.argument]) if fp8 else encode(expected[b.argument])
        sentinel=bytes(v^255 for v in target);sentinels[b.argument]=sentinel
        regions.append((memory.output_base+b.slot*2048,sentinel))
    for b in abi.bindings:
        address=(memory.input_base if b.role=='input' else memory.output_base)+b.slot*2048
        size=1024 if analysis.function.args[b.argument].type.elem=='f8E4M3FN' else 2048
        for a,data in [(address-32,b'\x5a'*32),(address+size,b'\xa5'*32)]:
            regions.append((a,data));preserved.append((a,data))
    ordered=sorted((a,len(d)) for a,d in regions)
    require(all(a+n<=b for (a,n),(b,_) in zip(ordered,ordered[1:])),'overlapping test memory')
    program=InstantiableProgram(instructions);program.dram_base=0
    program.memory_regions=[(a,torch.tensor(list(d),dtype=torch.uint8)) for a,d in regions]
    config=DefaultHardwareConfig();config.arch_state_config=replace(config.arch_state_config,numerics='rtl',randomize_init=False)
    sim=None
    try:
        sim=Simulation(config,LoggerConfig(filename='/dev/null'),program,verbose=False,ignore_runtime_errors=False)
        sim.run(max_cycles=100000);state=sim.core.arch_state
        require(state.halted and state.halt_reason=='ecall' and state.read_csrf(0xc10)==1,'incomplete machine execution')
        require(not sim.runtime_errors,'runtime/timing errors')
        mismatches=0
        for b in outputs:
            size=len(sentinels[b.argument]);data=bytes(state.read_dram(memory.output_base+b.slot*2048,size).tolist())
            actual=list(data) if size==1024 else decode(data)
            diff=sum(a!=e for a,e in zip(actual,expected[b.argument]));mismatches+=diff
            if diff:
                lane=next(i for i,(a,e) in enumerate(zip(actual,expected[b.argument])) if a!=e)
                raise ValueError(f'{name}: {diff} mismatches, lane {lane}, machine={actual[lane]:04x}, reference={expected[b.argument][lane]:04x}')
            require(all(a!=b for a,b in zip(data,sentinels[b.argument])),'incomplete overwrite')
        require(all(bytes(state.read_dram(a,len(d)).tolist())==d for a,d in preserved),'input/guard corruption')
        print(f'{name}: PASS; cycles={sim.cycle_count}; mismatches={mismatches}',flush=True)
        return {'case':name,'passed':True,'cycles':sim.cycle_count,'mismatches':mismatches,'completion':True,'inputs_guards_preserved':True}
    finally:
        if sim is not None:sim.close()


def validate(capture,output,*,numerical_policy,bindings=None):
    require(numerical_policy in (LAYOUT_POLICY,'atlas-fp8-mlp-v1'),'explicit supported policy required')
    metadata=json.loads((capture/'capture.json').read_text());text=(capture/'kernel.ttir').read_text()
    require(metadata['origin']=='triton.compile(ASTSource).asm[ttir]','not compiler TTIR')
    require(metadata['target']=={'backend':'cuda','arch':89,'warp_size':32},'expected offline SM89 target')
    require(hashlib.sha256(text.encode()).hexdigest()==metadata['ttir_sha256'],'TTIR hash mismatch')
    require(hashlib.sha256((capture/'kernel.source.py').read_bytes()).hexdigest()==metadata['source_sha256'],'source hash mismatch')
    layout=numerical_policy==LAYOUT_POLICY
    norm=normalize_static_tiles(parse_ttir(text).funcs[0],grid=metadata['grid'],numerical_policy=numerical_policy,resolve_dot_defaults=not layout)
    a=norm.analysis;count=len(a.function.args)-1
    declared=tuple(bindings) if bindings is not None else tuple(TileBinding(i,'input',i) for i in range(count))+(TileBinding(count,'output',0),)
    abi=(FP8LayoutABI if layout else FP8TileABI)(declared)
    memory=AtlasOperationMemory();virtual=emit_atlas_operations(a,numerical_policy=numerical_policy,abi=abi,memory=memory)
    output.mkdir(parents=True,exist_ok=False);path=output/'program.virtual.mlir';path.write_text(virtual)
    atlas=default_atlas_root()
    for cmd,name in [([str(atlas/'build/bin/atlas-opt'),'--verify-atlas-virtual-stream',str(path),'-o',str(output/'verified.mlir')],'verify'),
                     ([sys.executable,str(atlas/'tools/lower_atlas_program.py'),str(path),'--output-dir',str(output/'lowered')],'lower')]:
        result=subprocess.run(cmd,text=True,capture_output=True)
        (output/f'{name}.log').write_text(result.stdout+result.stderr);result.check_returncode()
    instructions,rows=load_instructions(output/'lowered',expected_word_count=None,adapter=instruction_adapter)
    results=[]
    for name,tiles in cases(layout):
        try: result=run_case(a,abi,memory,instructions,name,tiles,layout_reference if layout else fp8_reference)
        except Exception as exc:
            print(f'{name}: FAIL {exc}',flush=True);result={'case':name,'passed':False,'error':str(exc)}
        results.append(result)
    plan=analyze_atlas_eligibility(a,numerical_policy=numerical_policy,abi=abi).plan
    report={'capture':metadata,'policy':numerical_policy,'words':len(rows),'bindings':[asdict(b) for b in abi.bindings],
            'reference':'source SSA raw permutation' if layout else 'independent scalar integer/Fraction FP8 source SSA',
            'operation_plan':[{'source_index':e.source.index,'operation':e.source.name,'candidate':e.candidate,'order':list(transpose_order(e.source)) if e.source.name=='tt.trans' else None} for e in plan.operations],
            'cases':results}
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{metadata["kernel"]}: {sum(r["passed"] for r in results)}/{len(results)}; words={len(rows)}',flush=True)
    return all(r['passed'] for r in results)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--numerical-policy',required=True)
    p.add_argument('--tile-binding',action='append',metavar='ARG:ROLE:SLOT')
    args=p.parse_args();torch.set_num_threads(1)
    bindings=None if args.tile_binding is None else tuple(TileBinding(int(a),r,int(s)) for a,r,s in (b.split(':') for b in args.tile_binding))
    return int(not validate(args.capture,args.out_dir,numerical_policy=args.numerical_policy,bindings=bindings))

if __name__=='__main__':sys.exit(main())
