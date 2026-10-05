"""TTIR -> parsed Triton IR -> structured MLIR module -> printed + verified MLIR.

usage: PYTHONPATH=. python lower_real.py [kernel.ttir] [--limit N] [--out linalg.mlir]
  --limit N   lower only the first N top-level ops (then `return`) -- milestone 1
"""
import argparse
import sys

from compiler.ttir_reader import parse_ttir
from compiler.linalg_lowering import Lowerer
from compiler.mlir_ir import print_module, verify_ssa
from compiler.verify import verify_external

ap = argparse.ArgumentParser()
ap.add_argument("ttir", nargs="?", default="kernel.ttir")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default="linalg.mlir")
args = ap.parse_args()

tmod = parse_ttir(open(args.ttir).read())          # real Triton IR, verified by libtriton
print("[1] Triton parsed + verified: %d func(s), %d ops" % (len(tmod.funcs), len(tmod.all_ops)))

mod = Lowerer(limit=args.limit).lower_module(tmod)  # structured MLIR module
errs = verify_ssa(mod)
print("[2] internal SSA check:", "OK" if not errs else errs)

text = print_module(mod)
open(args.out, "w").write(text)
print("[3] wrote %s\n" % args.out)
print(text)

status, detail = verify_external(text)
print("[4] MLIR verifier: %s -- %s" % (status.upper(), detail))
sys.exit(1 if (errs or status == "failed") else 0)
