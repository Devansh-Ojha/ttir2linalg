#!/usr/bin/env python3
"""Compile genuine @triton.jit examples offline and validate actual Atlas execution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch

from compiler.jit_to_atlas import compile_jit_to_atlas
from compiler.atlas_eligibility import TileBinding
from validate_atlas_operations import cases,run_case,instruction_adapter
from validate_two_layer_mlp_generated_machine import load_instructions,derive_layout,run_case as run_fp8_case
from validate_two_layer_mlp_frontend import test_cases,transpose
from validate_mlp_chain_machine import reference as fp8_reference
from atlas_virtual_evaluator import parse_program

# Examples select only source signatures and references. Analysis/emission never
# dispatch on this catalog, model names, or number of layers.
EXAMPLES={
 'bf16-add-relu':('bf16_add_relu',{'X':'*bf16','Y':'*bf16','Out':'*bf16'},'atlas-bf16-elementwise-rtl-v1'),
 'fp8-matmul':('fp8_matmul',{'X':'*fp8e4nv','W':'*fp8e4nv','Out':'*bf16'},'atlas-fp8-mlp-v1'),
 'fp8-two-layer':('fp8_two_layer',{'X':'*fp8e4nv','W0':'*fp8e4nv','W1':'*fp8e4nv','Out':'*bf16'},'atlas-fp8-mlp-v1'),
}


def validate_example(example, output, *, numerical_policy, arch=89):
    kernel,signature,policy=EXAMPLES[example]
    if numerical_policy!=policy:raise ValueError(f'example requires explicit {policy}')
    fp8=any(t=='*fp8e4nv' for t in signature.values())
    n=len(signature)-1
    # Explicit test ABI. The reusable compiler never infers roles from names.
    bindings=tuple([TileBinding(i,'input',i if fp8 else 2*i) for i in range(n)]+[TileBinding(n,'output',0 if fp8 else 1)])
    compiled=compile_jit_to_atlas('kernels.atlas_jit_kernels',kernel,signature,
        out_dir=output,bindings=bindings,numerical_policy=policy,arch=arch,lower=True)
    output=compiled.virtual_mlir.parent;captured=compiled.captured_ttir
    analysis=compiled.normalization.analysis;abi=compiled.abi;memory=compiled.memory
    path=compiled.virtual_mlir;virtual=path.read_text()
    instructions,rows=load_instructions(output/'lowered',expected_word_count=None,adapter=instruction_adapter)
    results=[]
    if not fp8:
        for name,tiles in cases(2):results.append(run_case(analysis,abi,memory,instructions,name,tiles))
        reference='independent exact integer/Fraction Atlas-policy arithmetic on the proven boundary graph'
    else:
        depth=sum(o.name=='tt.dot' for o in analysis.operations)
        program=parse_program(virtual);spec=(program,tuple(range(depth+1)),0)
        addresses,halves=derive_layout(rows,spec,depth=depth,strict_legacy_layout=False)
        for name,encoded in test_cases():
            source=(encoded[0],)+tuple((a.T.contiguous(),transpose(t)) for a,t in encoded[1:depth+1])
            expected,_=fp8_reference(tuple(pair[0] for pair in source))
            ok,cycles=run_fp8_case(instructions,spec,addresses,halves,name,source,source_weights=True,reference_bits=expected)
            if not ok:raise RuntimeError(f'FP8 machine/reference mismatch: {name}')
            results.append({'case':name,'cycles':cycles,'mismatches':0})
        reference='existing validated direct Atlas FP8/BF16 numerical reference; shared RTL primitives, no GPU/IEEE equivalence claim'
    report={'example':example,'kernel':kernel,'policy':policy,'words':len(rows),'cases':results,
            'gpu_launched':False,'reference':reference,'capture':str(captured),'virtual':str(path)}
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{example}: {len(results)}/{len(results)} passed; words={len(rows)}; output={output}',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example',choices=EXAMPLES,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--numerical-policy',required=True)
    parser.add_argument('--arch',type=int,default=89)
    args=parser.parse_args();torch.set_num_threads(1)
    validate_example(args.example,args.out_dir,numerical_policy=args.numerical_policy,arch=args.arch)


if __name__=='__main__':main()
