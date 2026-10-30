#!/usr/bin/env python3
"""Validate genuine mixed-operation TTIR, including every diagnostic output.

Compiler eligibility/emission is operation-based. Only this workload harness
selects attention inputs, diagnostic roles and mathematical accuracy metrics.
"""
from compiler.toolchain_paths import atlas_root as default_atlas_root
import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import torch
from compiler.ttir_reader import parse_ttir
from compiler.triton_tile_normalization import normalize_static_tiles
from compiler.atlas_eligibility import MixedTileABI,TileBinding,analyze_atlas_eligibility
from compiler.atlas_rowwise_operations import FP8_ROWWISE_POLICY
from compiler.atlas_operation_mapping import emit_atlas_operations,AtlasOperationMemory
from atlas_bf16_reference import round_binary,_value
from atlas_fp8_reference import pack,product
from atlas_mixed_reference import source_reference
from validate_transpose_operations import run_case,instruction_adapter as transpose_adapter
from validate_rowwise_operations import instruction_adapter as row_adapter
from validate_two_layer_mlp_generated_machine import load_instructions,require

POLICY=FP8_ROWWISE_POLICY
# Deliberate gaps exercise explicit binding and reserved span handling.
BINDINGS=tuple(TileBinding(i,'input',2*i) for i in range(3))+tuple(TileBinding(i,'output',2*(i-3)+1) for i in range(3,7))


def instruction_adapter(row):
    if row['operation'] in ('atlas.vpu_unary','atlas.vpu_reduce'):
        fields=dict(row['fields'])
        for key in ('atlas.virtual_scalar_result','atlas.virtual_dma_transfer'):
            if key in fields:
                value=fields.pop(key);require(type(value) is int and value>=0,'invalid provenance metadata')
        return row_adapter(dict(row,fields=fields))
    return transpose_adapter(row)


def cases():
    def tile(fn):return [round_binary(Fraction(fn(r,c))) for r in range(32) for c in range(32)]
    zero=[0]*1024
    yield 'zero_all',[zero,zero,zero],'core'
    constant=tile(lambda r,c:2)
    yield 'uniform_constant_value',[zero,zero,constant],'core'
    asym=tile(lambda r,c:Fraction((r*7+c*3)%31-15,8))
    yield 'uniform_asymmetric_values',[zero,zero,asym],'core'
    rng=random.Random(20261029)
    for i in range(3):
        inputs=[tile(lambda r,c:Fraction(rng.randint(-32,32),32)) for _ in range(3)]
        yield f'bounded_random_{i}',inputs,'core'
    identity=tile(lambda r,c:1 if r==c else 0)
    yield 'asymmetric_permutation',[identity,tile(lambda r,c:2 if c==(r+7)%32 else 0),asym],'core'
    yield 'dominant_diagonal',[tile(lambda r,c:8 if r==c else 0)]*2+[asym],'core'
    yield 'two_equal_maxima',[tile(lambda r,c:8 if c==r//2 else 0),tile(lambda r,c:8 if c==r//2 else 0),asym],'core'
    yield 'probability_underflow_tails',[tile(lambda r,c:1 if c==0 else 0),tile(lambda r,c:2*(r-16) if c==0 else 0),asym],'boundary'
    ties=[Fraction(17,16),Fraction(19,16),Fraction(-17,16),Fraction(-19,16),Fraction(31,32),Fraction(33,32)]
    yield 'input_pack_ties',[[round_binary(ties[(r*3+c+i)%len(ties)]) for r in range(32) for c in range(32)] for i in range(3)],'boundary'
    tiny=[0,0x8000,1,0x8001,0x3c70,0xbc70,0x3c80,0xbc80]
    yield 'pack_underflow_and_subnormals',[[tiny[(r*3+c+i)%len(tiny)] for r in range(32) for c in range(32)] for i in range(3)],'boundary'
    yield 'value_saturation',[identity,identity,tile(lambda r,c:512 if (r+c)%2 else -512)],'saturation'
    yield 'constant_value_saturation',[zero,zero,tile(lambda r,c:512)],'saturation'
    yield 'large_score_span',[tile(lambda r,c:448 if r==c else 0)]*2+[asym],'stress'
    special=[0x7fc0,0xffc0,0x7f81,0xff81,0x7f80,0xff80,0,0x8000]
    yield 'nonfinite_inputs',[[special[(r*5+c+i)%len(special)] for r in range(32) for c in range(32)] for i in range(3)],'nonfinite'
    yield 'raw_bf16_random',[[rng.randrange(65536) for _ in range(1024)] for _ in range(3)],'stress'


def mathematical_attention(tiles,*,packed_inputs=False):
    if packed_inputs:
        q,k,v=[[float(product(pack(b))) for b in t] for t in tiles]
    else:
        q,k,v=[[float(_value(b)) for b in t] for t in tiles]
    result=[]
    for r in range(32):
        scores=[math.fsum(q[r*32+d]*k[c*32+d] for d in range(32))/math.sqrt(32) for c in range(32)]
        maximum=max(scores);exps=[math.exp(s-maximum) for s in scores];denominator=math.fsum(exps)
        p=[e/denominator for e in exps]
        result.extend(math.fsum(p[j]*v[j*32+c] for j in range(32)) for c in range(32))
    return result


def accuracy(tiles,expected):
    if any(b&0x7f80==0x7f80 for t in tiles for b in t):
        return {'mathematical_attention':'undefined for nonfinite inputs'}
    if any(b&0x7f80==0x7f80 for b in expected):
        return {'mathematical_attention':'nonfinite Atlas-policy output; no finite error metric'}
    actual=[float(_value(b)) for b in expected]
    original=mathematical_attention(tiles);quantized=mathematical_attention(tiles,packed_inputs=True)
    return {'max_absolute_error':max(abs(a-b) for a,b in zip(actual,original)),
            'rms_error':math.sqrt(math.fsum((a-b)**2 for a,b in zip(actual,original))/1024),
            'max_absolute_error_packed_qkv_baseline':max(abs(a-b) for a,b in zip(actual,quantized)),
            'baseline':'float64 stable mathematical attention on original BF16 input values'}


def validate(capture,output,*,numerical_policy,selected=None):
    require(numerical_policy==POLICY,'explicit composed numerical policy required')
    metadata=json.loads((capture/'capture.json').read_text());text=(capture/'kernel.ttir').read_text()
    require(metadata['origin']=='triton.compile(ASTSource).asm[ttir]','not compiler-generated TTIR')
    require(metadata['target']=={'backend':'cuda','arch':89,'warp_size':32},'expected offline SM89')
    require(hashlib.sha256(text.encode()).hexdigest()==metadata['ttir_sha256'],'TTIR hash mismatch')
    require(hashlib.sha256((capture/'kernel.source.py').read_bytes()).hexdigest()==metadata['source_sha256'],'source hash mismatch')
    norm=normalize_static_tiles(parse_ttir(text).funcs[0],grid=metadata['grid'],numerical_policy=POLICY,resolve_dot_defaults=True)
    a=norm.analysis
    formats=tuple(arg.type.elem for arg in a.function.args)
    require(formats==('bf16',)*6+('f8E4M3FN',),'harness diagnostic signature mismatch')
    abi=MixedTileABI(BINDINGS,formats=formats);memory=AtlasOperationMemory()
    virtual=emit_atlas_operations(a,numerical_policy=POLICY,abi=abi,memory=memory)
    output.mkdir(parents=True,exist_ok=False);path=output/'program.virtual.mlir';path.write_text(virtual)
    atlas=default_atlas_root()
    for cmd,name in [([str(atlas/'build/bin/atlas-opt'),'--verify-atlas-virtual-stream',str(path),'-o',str(output/'verified.mlir')],'verify'),
                     ([sys.executable,str(atlas/'tools/lower_atlas_program.py'),str(path),'--output-dir',str(output/'lowered')],'lower')]:
        result=subprocess.run(cmd,text=True,capture_output=True)
        (output/f'{name}.log').write_text(result.stdout+result.stderr);result.check_returncode()
    instructions,rows=load_instructions(output/'lowered',expected_word_count=None,adapter=instruction_adapter)
    results=[]
    for name,tiles,profile in cases():
        if selected is not None and name not in selected:continue
        try:
            expected,trace=source_reference(a,dict(enumerate(tiles)),with_trace=True)
            # Compare all four actual memory outputs, including raw FP8 bytes.
            result=run_case(a,abi,memory,instructions,name,tiles,lambda analysis,inputs:expected)
            result['profile']=profile;result['accuracy']=accuracy(tiles,expected[3])
            result['packed_probability_zero_count']=sum(b==0 for b in expected[6])
            result['probability_row_sum_error']=max(abs(sum(float(_value(b)) for b in expected[5][r*32:r*32+32])-1) for r in range(32)) if all(b&0x7f80!=0x7f80 for b in expected[5]) else None
            if name=='zero_all':require(expected[3]==[0]*1024,'zero known result failed')
            if name=='constant_value_saturation':require(expected[3]==[0x43e0]*1024,'known saturation-to-448 result failed')
            if name=='uniform_constant_value':
                require(expected[3]==[0x4000]*1024,'constant-value known result failed')
                require(expected[5]==[0x3d00]*1024 and expected[6]==[0x38]*1024,'uniform probability known result failed')
            # Record independently evaluated source intermediates, including all
            # pack/dot results; these are reference traces, not simulator taps.
            trace_data={str(op.index):{'operation':op.name,'results':[{'id':v.id,'source_type':str(v.type),'bits':trace[v.id]} for v in op.results]} for op in a.function.body.ops if op.results}
            (output/f'{name}.reference_trace.json').write_text(json.dumps(trace_data)+'\n')
        except Exception as exc:
            print(f'{name}: FAIL {type(exc).__name__}: {exc}',flush=True)
            result={'case':name,'profile':profile,'passed':False,'error':str(exc)}
        results.append(result)
    require(bool(results),'no cases selected')
    plan=analyze_atlas_eligibility(a,numerical_policy=POLICY,abi=abi).plan
    report={'kernel':metadata['kernel'],'capture':metadata,'policy':POLICY,'fused_kernel':True,'launches_per_case':1,
            'words':len(rows),'bindings':[asdict(b) for b in abi.bindings],'formats':list(formats),
            'operation_plan':[{'source_index':e.source.index,'operation':e.source.name,'candidate':e.candidate,'unit':e.unit,'scale_code':e.scale_code,'layout':e.result_layout,'precision_mode':e.precision_mode} for e in plan.operations],
            'reference':'independent scalar integer/Fraction/Decimal source SSA; no NPU numerical imports','cases':results}
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'quantized attention: {sum(r["passed"] for r in results)}/{len(results)}; words={len(rows)}',flush=True)
    return all(r['passed'] for r in results)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--numerical-policy',required=True)
    p.add_argument('--case',action='append');args=p.parse_args();torch.set_num_threads(1)
    return int(not validate(args.capture,args.out_dir,numerical_policy=args.numerical_policy,selected=args.case))

if __name__=='__main__':sys.exit(main())
