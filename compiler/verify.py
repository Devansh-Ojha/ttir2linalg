"""Verify generated MLIR text with a REAL MLIR parser/verifier.

libtriton cannot do this (it has no linalg/tensor/memref/func dialects), so we use,
in order:
  1. a python interpreter that has the `mlir` package, given by the env var
     TTIR2LINALG_MLIR_PYTHON (run in a subprocess so it never shares a process
     with libtriton), or
  2. `mlir-opt` on PATH.
Returns (status, detail) with status in {"verified", "failed", "unavailable"}.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
import tempfile

_CHECK = r"""
import sys
from mlir import ir
text = open(sys.argv[1]).read()
with ir.Context():
    m = ir.Module.parse(text)
    ok = m.operation.verify()
print("VERIFIED" if ok else "INVALID")
"""


def verify_external(mlir_text: str):
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(mlir_text)
        path = f.name
    try:
        py = os.environ.get("TTIR2LINALG_MLIR_PYTHON")
        if py:
            p = subprocess.run([py, "-c", _CHECK, path], capture_output=True, text=True)
            if p.returncode == 0 and "VERIFIED" in p.stdout:
                return "verified", "python mlir bindings (%s)" % py
            return "failed", (p.stderr or p.stdout).strip()
        opt = shutil.which("mlir-opt")
        if opt:
            p = subprocess.run([opt, path, "-o", os.devnull], capture_output=True, text=True)
            if p.returncode == 0:
                return "verified", "mlir-opt"
            return "failed", p.stderr.strip()
        return "unavailable", ("no MLIR verifier found: set TTIR2LINALG_MLIR_PYTHON to a "
                               "python that can `import mlir`, or put mlir-opt on PATH")
    finally:
        os.unlink(path)
