"""Official Triton ASTSource/compile capture in a fresh isolated worker.

No device launch, CUDA driver, TorchInductor, patched Triton stage or TTIR text
synthesis is used. The CUDA target is an offline compiler target, not Atlas.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


def capture_jit_kernel(module, kernel, signature, *, out_dir, constexprs=None,
                       arch=89, grid=(1,1,1)):
    output=Path(out_dir).resolve()
    output.mkdir(parents=True,exist_ok=False)
    request={'module':module,'kernel':kernel,'signature':signature,
             'constexprs':constexprs if constexprs is not None else {},
             'arch':arch,'grid':list(grid),'options':{'num_warps':4,'num_ctas':1,'enable_fp_fusion':False}}
    job=output/'request.json';job.write_text(json.dumps(request,indent=2)+'\n')
    env=dict(os.environ,TRITON_CACHE_DIR=str(output/'triton-cache'),
             TRITON_DUMP_DIR=str(output/'triton-dump'),TRITON_KERNEL_DUMP='0',
             TRITON_KERNEL_OVERRIDE='0',PYTHONDONTWRITEBYTECODE='1')
    repo=Path(__file__).resolve().parents[1]
    env['PYTHONPATH']=str(repo)+os.pathsep+env.get('PYTHONPATH','')
    result=subprocess.run([sys.executable,'-m','compiler.jit_capture','--worker',str(job)],
                          env=env,text=True,capture_output=True)
    (output/'compiler.stdout.log').write_text(result.stdout)
    (output/'compiler.stderr.log').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f'Triton compilation failed; see {output}/compiler.stderr.log: {result.stderr[-2000:]}')
    return output/'kernel.ttir'


def _worker(job):
    from triton.backends.compiler import GPUTarget
    import triton
    from compiler.triton_capture import _compile_kernel
    request=json.loads(job.read_text());output=job.parent
    fn=getattr(importlib.import_module(request['module']),request['kernel'])
    source_path=Path(inspect.getsourcefile(fn.fn))
    source=source_path.read_bytes();(output/'kernel.source.py').write_bytes(source)
    ttir=_compile_kernel(fn,signature=request['signature'],constexprs=request['constexprs'],
                         target=GPUTarget('cuda',request['arch'],32),options=request['options'])
    (output/'kernel.ttir').write_text(ttir)
    metadata=dict(request,triton_version=triton.__version__,target={'backend':'cuda','arch':request['arch'],'warp_size':32},
                  source_path=str(source_path),source_sha256=hashlib.sha256(source).hexdigest(),
                  ttir_sha256=hashlib.sha256(ttir.encode()).hexdigest(),
                  origin='triton.compile(ASTSource).asm[ttir]',gpu_launched=False)
    (output/'capture.json').write_text(json.dumps(metadata,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--module');parser.add_argument('--kernel')
    parser.add_argument('--signature',type=json.loads);parser.add_argument('--constexprs',type=json.loads,default={})
    parser.add_argument('--out-dir',type=Path);parser.add_argument('--arch',type=int,default=89)
    parser.add_argument('--grid',default='1,1,1')
    args=parser.parse_args()
    if args.worker:_worker(args.worker);return
    if not all((args.module,args.kernel,args.signature,args.out_dir)):
        parser.error('--module, --kernel, --signature and --out-dir are required')
    print(capture_jit_kernel(args.module,args.kernel,args.signature,out_dir=args.out_dir,
                            constexprs=args.constexprs,arch=args.arch,grid=tuple(map(int,args.grid.split(',')))))


if __name__=='__main__':main()
