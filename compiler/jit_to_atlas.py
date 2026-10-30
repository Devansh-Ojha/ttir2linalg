"""Generic handwritten Triton Python -> genuine TTIR -> Atlas compilation.

The module/kernel/signature are user supplied, not selected by a recognizer.
Runtime buffers must satisfy the explicit single-tile staging/nonaliasing ABI.
"""
from __future__ import annotations
from compiler.toolchain_paths import atlas_root as default_atlas_root
import argparse
from dataclasses import dataclass,asdict
import json
from pathlib import Path
import subprocess
import sys
from compiler.jit_capture import capture_jit_kernel
from compiler.ttir_reader import parse_ttir
from compiler.triton_tile_normalization import normalize_static_tiles
from compiler.atlas_eligibility import BF16TileABI,FP8TileABI,FP8LayoutABI,MixedTileABI,TileBinding
from compiler.atlas_operation_mapping import emit_atlas_operations,AtlasOperationMemory


@dataclass(frozen=True)
class JitAtlasCompilation:
    captured_ttir: Path
    virtual_mlir: Path
    normalization: object
    abi: object
    memory: AtlasOperationMemory
    lowered: Path | None


def compile_jit_to_atlas(module,kernel,signature,*,out_dir,bindings,numerical_policy,
                         grid=(1,1,1),arch=89,constexprs=None,
                         memory=AtlasOperationMemory(),lower=False,atlas_root=None):
    if numerical_policy not in {'atlas-bf16-elementwise-rtl-v1','atlas-fp8-mlp-v1','atlas-bf16-rowwise-rtl-v1','atlas-fp8-layout-v1','atlas-fp8-attention-v1'}:
        raise ValueError('explicit supported Atlas numerical policy required')
    if tuple(grid)!=(1,1,1):raise ValueError('only grid (1,1,1) is supported')
    output=Path(out_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    captured=capture_jit_kernel(module,kernel,signature,out_dir=output/'capture',constexprs=constexprs,arch=arch,grid=grid)
    metadata=json.loads((captured.parent/'capture.json').read_text())
    fp8=numerical_policy in {'atlas-fp8-mlp-v1','atlas-fp8-attention-v1'}
    function=parse_ttir(captured.read_text()).funcs[0]
    norm=normalize_static_tiles(function,grid=metadata['grid'],numerical_policy=numerical_policy,resolve_dot_defaults=fp8)
    abi_type = FP8LayoutABI if numerical_policy == 'atlas-fp8-layout-v1' else (FP8TileABI if fp8 else BF16TileABI)
    abi=MixedTileABI(tuple(bindings),formats=tuple(a.type.elem for a in function.args)) if numerical_policy=='atlas-fp8-attention-v1' else abi_type(tuple(bindings))
    virtual=emit_atlas_operations(norm.analysis,numerical_policy=numerical_policy,abi=abi,memory=memory)
    path=output/'program.virtual.mlir';path.write_text(virtual)
    proof={'grid':metadata['grid'],'original_ttir':str(captured),'original_sha256':metadata['ttir_sha256'],
           'address_operations':[{'index':o.index,'name':o.name} for o in norm.address_operations],
           'precision_operations':[{'index':o.index,'name':o.name} for o in norm.precision_operations],
           'resolved_triton_dot_defaults':fp8,'abi':asdict(abi),'memory':asdict(memory),
           'derived_graph_only':True,'normalized_ttir_generated':False}
    (output/'normalization.json').write_text(json.dumps(proof,indent=2)+'\n')
    lowered=None
    if lower:
        atlas=Path(atlas_root) if atlas_root is not None else default_atlas_root()
        lowered=output/'lowered'
        for command in [[str(atlas/'build/bin/atlas-opt'),'--verify-atlas-virtual-stream',str(path),'-o',str(output/'verified.mlir')],
                        [sys.executable,str(atlas/'tools/lower_atlas_program.py'),str(path),'--output-dir',str(lowered)]]:
            result=subprocess.run(command,capture_output=True,text=True)
            if result.returncode:raise RuntimeError(f'backend failed: {command}: {result.stderr}\n{result.stdout}')
    return JitAtlasCompilation(captured,path,norm,abi,memory,lowered)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module',required=True);parser.add_argument('--kernel',required=True)
    parser.add_argument('--signature',type=json.loads,required=True)
    parser.add_argument('--constexprs',type=json.loads,default={})
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--numerical-policy',required=True)
    parser.add_argument('--tile-binding',action='append',required=True,metavar='ARG:ROLE:SLOT')
    parser.add_argument('--arch',type=int,default=89);parser.add_argument('--grid',default='1,1,1')
    parser.add_argument('--input-base',type=lambda s:int(s,0),default=0x90000000)
    parser.add_argument('--output-base',type=lambda s:int(s,0),default=0x90020000)
    parser.add_argument('--lower',action='store_true')
    args=parser.parse_args()
    bindings=[]
    for b in args.tile_binding:
        argument,role,slot=b.split(':');bindings.append(TileBinding(int(argument),role,int(slot)))
    result=compile_jit_to_atlas(args.module,args.kernel,args.signature,out_dir=args.out_dir,
        bindings=bindings,numerical_policy=args.numerical_policy,grid=tuple(map(int,args.grid.split(','))),
        arch=args.arch,constexprs=args.constexprs,memory=AtlasOperationMemory(args.input_base,args.output_base),lower=args.lower)
    print(result.virtual_mlir)
    if result.lowered:print(result.lowered/'5-words.hex')


if __name__=='__main__':main()
