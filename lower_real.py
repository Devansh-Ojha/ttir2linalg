"""TTIR -> parsed Triton IR -> real MLIR module -> printed + verified MLIR.

usage: PYTHONPATH=. python lower_real.py [kernel.ttir] [--limit N] [--out linalg.mlir]
  --limit N   lower only the first N top-level ops (then `return`) -- milestone 1
"""
import argparse
from compiler.real_mlir import lower_ttir

ap = argparse.ArgumentParser()
ap.add_argument("ttir", nargs="?", default="kernel.ttir")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default="linalg.mlir")
args = ap.parse_args()

tmod, mod, ssa = lower_ttir(
    open(args.ttir).read(),
    limit=args.limit,
    arithmetic_only=True,
)
print("[1] Triton parsed + verified: %d func(s), %d ops" % (len(tmod.funcs), len(tmod.all_ops)))
print("[2] real MLIR module verified: OK")
text = mod.str()
open(args.out, "w").write(text)
print("[3] wrote %s (%d SSA value(s))\n" % (args.out, len(ssa)))
print(text)
