#!/usr/bin/env python3
"""Verify, lower and simulate qualified genuine TTIR against a scalar reference.

Accepts any supported source graph; no softmax/function-name recognition.
"""
from compiler.toolchain_paths import atlas_root as default_atlas_root
import argparse
from dataclasses import asdict
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
from compiler.atlas_rowwise_operations import ROWWISE_POLICY
from compiler.atlas_eligibility import BF16TileABI,TileBinding,analyze_atlas_eligibility
from compiler.atlas_operation_mapping import emit_atlas_operations,AtlasOperationMemory
from atlas_bf16_reference import round_binary,_value
from atlas_rowwise_reference import source_reference
from validate_atlas_operations import instruction_adapter as bf16_adapter,run_case,cases as raw_cases
from validate_two_layer_mlp_generated_machine import load_instructions,require,METADATA
from npu_model.configs import isa_definition as isa


def instruction_adapter(row):
    n=row['operation']; f={k:v for k,v in row['fields'].items() if k not in METADATA}
    if n in ('atlas.vpu_unary','atlas.vpu_reduce'):
        require(set(f)=={'kind','dst','src'},'unexpected unary/reduce instruction fields')
        classes={'exp':isa.VEXP_BF16,'recip':isa.VRECIP_BF16,
                 'row_max':isa.VREDMAX_ROW_BF16,'row_sum':isa.VREDSUM_ROW_BF16}
        cls=classes.get(f['kind'])
        if cls is None: return bf16_adapter(row)
        require((n=='atlas.vpu_reduce')==(f['kind'] in ('row_max','row_sum')),'invalid kind/category')
        require(all(type(f[k]) is int and 0<=f[k]<64 for k in ('dst','src')),'invalid register')
        return cls(vd=f['dst'],vs1=f['src'])
    return bf16_adapter(row)


def softmax_cases():
    def encode(xs): return [round_binary(x) for x in xs]
    from fractions import Fraction
    yield 'uniform_zero',[0]*1024
    yield 'uniform_rows',encode([r-16 for r in range(32) for c in range(32)])
    yield 'signed_zero',[0x8000 if (r+c)%2 else 0 for r in range(32) for c in range(32)]
    yield 'increasing_columns',encode([Fraction(c-16,4) for r in range(32) for c in range(32)])
    yield 'asymmetric_rows',encode([Fraction((r*13+c*7)%97-48,8) for r in range(32) for c in range(32)])
    yield 'large_negative',encode([-64-c*2-r for r in range(32) for c in range(32)])
    yield 'large_common_offset',encode([256+r*4+c*2 for r in range(32) for c in range(32)])
    yield 'one_dominant',encode([32 if c==(r*7)%32 else -32 for r in range(32) for c in range(32)])
    yield 'two_equal_maxima',encode([4 if c in (r,(r+13)%32) else -8 for r in range(32) for c in range(32)])
    yield 'subnormal_inputs',[1 if (r+c)%3 else 0x8001 for r in range(32) for c in range(32)]
    rng=random.Random(20261029)
    for i in range(3):
        yield f'finite_random_{i}',encode([Fraction(rng.randint(-256,256),32) for _ in range(1024)])
    yield 'nan_inf_raw',[ [0x7fc0,0x7f81,0xffc0,0xff81,0x7f80,0xff80,0,0x8000][(r*3+c)%8] for r in range(32) for c in range(32)]


def softmax_accuracy(analysis,tiles):
    actual=source_reference(analysis,{0:tiles})[1]
    max_error=0.; max_sum_error=0.
    for r in range(32):
        row=[float(_value(b)) for b in tiles[r*32:r*32+32]]
        exps=[math.exp(x-max(row)) for x in row]; total=sum(exps)
        got=[float(_value(b)) for b in actual[r*32:r*32+32]]
        max_error=max(max_error,max(abs(a-b/total) for a,b in zip(got,exps)))
        max_sum_error=max(max_sum_error,abs(sum(got)-1))
    return {'max_absolute_error_vs_real_softmax':max_error,'max_row_sum_error':max_sum_error}


def validate(capture,output,*,numerical_policy,check_softmax=False):
    if numerical_policy!=ROWWISE_POLICY: raise ValueError(f'explicit {ROWWISE_POLICY} required')
    metadata=json.loads((capture/'capture.json').read_text())
    text=(capture/'kernel.ttir').read_text()
    require(metadata['origin']=='triton.compile(ASTSource).asm[ttir]','not compiler TTIR')
    require(metadata['target']=={'backend':'cuda','arch':89,'warp_size':32},'expected offline SM89 capture')
    require(hashlib.sha256(text.encode()).hexdigest()==metadata['ttir_sha256'],'capture hash mismatch')
    require(hashlib.sha256((capture/'kernel.source.py').read_bytes()).hexdigest()==metadata['source_sha256'],'source hash mismatch')
    norm=normalize_static_tiles(parse_ttir(text).funcs[0],grid=metadata['grid'],numerical_policy=numerical_policy)
    a=norm.analysis
    # This harness binds one input and one output; generalized compiler plans
    # support multiple inputs/outputs through explicit caller-supplied bindings.
    require(len(a.function.args)==2,'harness needs one input and one output')
    abi=BF16TileABI((TileBinding(0,'input',0),TileBinding(1,'output',0)))
    memory=AtlasOperationMemory()
    virtual=emit_atlas_operations(a,numerical_policy=numerical_policy,abi=abi,memory=memory)
    output.mkdir(parents=True,exist_ok=False)
    path=output/'program.virtual.mlir';path.write_text(virtual)
    atlas=default_atlas_root()
    for cmd,name in [([str(atlas/'build/bin/atlas-opt'),'--verify-atlas-virtual-stream',str(path),'-o',str(output/'verified.mlir')],'verify'),
                     ([sys.executable,str(atlas/'tools/lower_atlas_program.py'),str(path),'--output-dir',str(output/'lowered')],'lower')]:
        result=subprocess.run(cmd,text=True,capture_output=True)
        (output/f'{name}.log').write_text(result.stdout+result.stderr);result.check_returncode()
    instructions,rows=load_instructions(output/'lowered',expected_word_count=None,adapter=instruction_adapter)
    results=[]
    planned=softmax_cases() if check_softmax else ((n,t[0]) for n,t in raw_cases(1))
    for name,tile in planned:
        try:
            result=run_case(a,abi,memory,instructions,name,[tile],reference=source_reference)
            result['passed']=True
            if check_softmax and all(b&0x7f80!=0x7f80 for b in tile):
                result.update(softmax_accuracy(a,tile))
                require(result['max_absolute_error_vs_real_softmax']<0.02,'softmax accuracy exceeded 0.02')
                require(result['max_row_sum_error']<0.03,'row sum error exceeded 0.03')
        except Exception as exc:
            print(f'{name}: FAIL {type(exc).__name__}: {exc}',flush=True)
            result={'case':name,'passed':False,'error':str(exc)}
        results.append(result)
    report={'kernel':metadata['kernel'],'policy':numerical_policy,'capture':metadata,
            'words':len(rows),'bindings':[asdict(b) for b in abi.bindings],
            'address_operations':[{'index':o.index,'name':o.name} for o in norm.address_operations],
            'reference':'independent scalar source graph, Decimal/Fraction exp interpolation, FP32 tree and independent Q1.16 reciprocal LUT/truncation',
            'operation_plan':[{'source_index':e.source.index,'operation':e.source.name,'candidate':e.candidate,'layout':e.result_layout,'precision':e.precision_mode} for e in analyze_atlas_eligibility(a,numerical_policy=numerical_policy,abi=abi).plan.operations],
            'cases':results}
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{metadata["kernel"]}: {sum(r["passed"] for r in results)}/{len(results)}; words={len(rows)}',flush=True)
    return all(r['passed'] for r in results)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--numerical-policy',required=True)
    p.add_argument('--check-softmax',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    return int(not validate(args.capture,args.out_dir,numerical_policy=args.numerical_policy,check_softmax=args.check_softmax))

if __name__=='__main__':sys.exit(main())
