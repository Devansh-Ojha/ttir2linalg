"""Compile preserved Triton TTIR through existing operation-level Atlas lowering.

Input authenticity is a caller responsibility; the manifest hashes the bytes
consumed. This API never synthesizes TTIR or invokes CUDA code generation.
"""
from compiler.toolchain_paths import atlas_root as default_atlas_root
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from compiler.ttir_reader import parse_ttir
from compiler.triton_tile_normalization import normalize_static_tiles
from compiler.atlas_eligibility import (BF16TileABI, FP8TileABI, FP8LayoutABI,
    MixedTileABI, TileBinding, analyze_atlas_eligibility)
from compiler.atlas_operation_mapping import AtlasOperationMemory, emit_atlas_operations

POLICIES = ('atlas-bf16-elementwise-rtl-v1', 'atlas-fp8-mlp-v1',
    'atlas-bf16-rowwise-rtl-v1', 'atlas-fp8-layout-v1', 'atlas-fp8-attention-v1')

@dataclass(frozen=True)
class AtlasTarget:
    """Supported fixed hardware profile, not arbitrary backend configuration."""
    profile: str = 'atlas-rtl-v1'
    tile: tuple = (32, 32)
    grid: tuple = (1, 1, 1)

    def validate(self):
        if self.profile != 'atlas-rtl-v1' or tuple(self.tile) != (32, 32) or tuple(self.grid) != (1, 1, 1):
            raise ValueError('supported target: atlas-rtl-v1, tile 32x32, grid 1,1,1')


def compile_ttir_to_atlas(ttir_path, *, out_dir, numerical_policy, bindings,
                          target=AtlasTarget(), memory=AtlasOperationMemory(),
                          lower=False, atlas_root=None):
    target.validate()
    if numerical_policy not in POLICIES:
        raise ValueError('explicit supported numerical policy required')
    source = Path(ttir_path).resolve()
    raw = source.read_bytes()
    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'input': str(source), 'sha256': hashlib.sha256(raw).hexdigest(),
              'target': asdict(target), 'policy': numerical_policy,
              'memory': asdict(memory), 'timings_seconds': {}, 'success': False}
    stage = 'parse'
    started = time.perf_counter()
    try:
        module = parse_ttir(raw.decode())
        if len(module.funcs) != 1:
            raise ValueError('exactly one TTIR function required')
        function = module.funcs[0]
        fp8 = numerical_policy in ('atlas-fp8-mlp-v1', 'atlas-fp8-attention-v1')
        stage = 'normalize'
        norm = normalize_static_tiles(function, grid=target.grid,
            numerical_policy=numerical_policy, resolve_dot_defaults=fp8)
        abi_type = FP8LayoutABI if numerical_policy == 'atlas-fp8-layout-v1' else (FP8TileABI if fp8 else BF16TileABI)
        abi = (MixedTileABI(tuple(bindings), formats=tuple(a.type.elem for a in function.args))
               if numerical_policy == 'atlas-fp8-attention-v1' else abi_type(tuple(bindings)))
        report['abi'] = asdict(abi)
        stage = 'eligibility'
        eligibility = analyze_atlas_eligibility(norm.analysis, numerical_policy=numerical_policy, abi=abi)
        report['diagnostics'] = [asdict(d) for d in eligibility.diagnostics]
        if not eligibility.eligible:
            raise NotImplementedError('; '.join(f'{d.code}: {d.message}' for d in eligibility.diagnostics))
        stage = 'emit'
        virtual = emit_atlas_operations(norm.analysis, numerical_policy=numerical_policy, abi=abi, memory=memory)
        (output/'input.ttir').write_bytes(raw)
        path = output/'program.virtual.mlir'
        path.write_text(virtual)
        report['normalization'] = {k: [{'index': o.index, 'name': o.name} for o in getattr(norm, k)]
                                    for k in ('address_operations', 'precision_operations')}
        report['timings_seconds']['frontend'] = time.perf_counter()-started
        if lower:
            atlas = Path(atlas_root) if atlas_root else default_atlas_root()
            for stage, command in (
                ('verify', [str(atlas/'build/bin/atlas-opt'), '--verify-atlas-virtual-stream', str(path), '-o', str(output/'verified.mlir')]),
                ('lower', [sys.executable, str(atlas/'tools/lower_atlas_program.py'), str(path), '--output-dir', str(output/'lowered')])):
                begin = time.perf_counter()
                result = subprocess.run(command, capture_output=True, text=True)
                report['timings_seconds'][stage] = time.perf_counter()-begin
                (output/f'{stage}.log').write_text(result.stdout+result.stderr)
                result.check_returncode()
        report['success'] = True
        return path
    except Exception as exc:
        report['error'] = {'stage': stage, 'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        (output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ttir', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--numerical-policy', choices=POLICIES, required=True)
    p.add_argument('--target-config', type=Path, help='JSON AtlasTarget fields; only fixed supported profile accepted')
    p.add_argument('--tile-binding', action='append', required=True, metavar='ARG:ROLE:SLOT')
    p.add_argument('--lower', action='store_true')
    p.add_argument('--atlas-root', type=Path)
    args = p.parse_args()
    try:
        target = AtlasTarget(**json.loads(args.target_config.read_text())) if args.target_config else AtlasTarget()
        bindings = tuple(TileBinding(int(a), r, int(s)) for a,r,s in (v.split(':') for v in args.tile_binding))
        print(compile_ttir_to_atlas(args.ttir, out_dir=args.out_dir, numerical_policy=args.numerical_policy,
              bindings=bindings, target=target, lower=args.lower, atlas_root=args.atlas_root))
    except Exception as exc:
        p.exit(1, f'{type(exc).__name__}: {exc}\n')

if __name__ == '__main__': main()
