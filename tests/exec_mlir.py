"""Executor: runs under the *MLIR* python (not the triton env).

usage: <mlir-python> tests/exec_mlir.py kernel.mlir inputs.npz out.npy

Lowers the Linalg module to LLVM and JIT-executes `linear_kernel` once per
program id (the grid), with signature (pid:i32, x, w, b, out).
"""
import ctypes
import re
import sys

import numpy as np
from mlir import ir, passmanager
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import get_ranked_memref_descriptor

PIPELINE = ("builtin.module("
            "one-shot-bufferize{bufferize-function-boundaries=1},"
            "convert-linalg-to-loops,expand-strided-metadata,lower-affine,"
            "convert-scf-to-cf,convert-to-llvm,reconcile-unrealized-casts)")

mlir_path, npz_path, out_path = sys.argv[1:4]
text = open(mlir_path).read()
# the JIT needs a C-callable wrapper; harness-only change, lowering output is untouched
text = re.sub(r"(func\.func @\w+\([^)]*\)) \{", r"\1 attributes {llvm.emit_c_interface} {",
              text, count=1)

d = np.load(npz_path)
x, w, b = d["x"], d["w"], d["b"]
out = np.zeros(b.shape[0], dtype=np.float32)


def memref(a):
    return ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(a)))


with ir.Context():
    module = ir.Module.parse(text)
    passmanager.PassManager.parse(PIPELINE).run(module.operation)
    engine = ExecutionEngine(module, opt_level=2)
    for pid in range(out.shape[0]):
        engine.invoke("linear_kernel", ctypes.pointer(ctypes.c_int32(pid)),
                      memref(x), memref(w.reshape(-1)), memref(b), memref(out))
np.save(out_path, out)
