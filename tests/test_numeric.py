"""Numerical test: TTIR -> Linalg (JIT-executed) vs a PyTorch/NumPy reference.

Run:  PYTHONPATH=. python tests/test_numeric.py [kernel.ttir]
Needs TTIR2LINALG_MLIR_PYTHON (python with the `mlir` package). Without it the
test reports SKIPPED (exit 2) -- it never reports PASS without actually running.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

from compiler.ttir_reader import parse_ttir
from compiler.linalg_lowering import Lowerer
from compiler.mlir_ir import print_module, verify_ssa

HERE = os.path.dirname(os.path.abspath(__file__))
K, N = 4, 5            # kernel is specialised for K=4; N = grid size (output features)


def reference(x, w, b):
    try:
        import torch
        lin = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w),
                                         torch.from_numpy(b))
        return lin.numpy(), "torch"
    except ImportError:
        return w @ x + b, "numpy"


def execute(mlir_text, x, w, b, py):
    with tempfile.TemporaryDirectory() as td:
        mp, ip, op = (os.path.join(td, n) for n in ("k.mlir", "in.npz", "out.npy"))
        open(mp, "w").write(mlir_text)
        np.savez(ip, x=x, w=w, b=b)
        p = subprocess.run([py, os.path.join(HERE, "exec_mlir.py"), mp, ip, op],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError("execution failed:\n" + p.stderr)
        return np.load(op)


def main():
    py = os.environ.get("TTIR2LINALG_MLIR_PYTHON")
    if not py:
        print("SKIPPED: set TTIR2LINALG_MLIR_PYTHON to a python with the `mlir` package")
        return 2
    ttir_path = sys.argv[1] if len(sys.argv) > 1 else "kernel.ttir"
    mod = Lowerer().lower_module(parse_ttir(open(ttir_path).read()))
    assert not verify_ssa(mod), "internal SSA check failed"
    text = print_module(mod)

    rng = np.random.default_rng(0)
    x = rng.standard_normal(K).astype(np.float32)
    w = rng.standard_normal((N, K)).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    ref, ref_name = reference(x, w, b)

    got = execute(text, x, w, b, py)
    err = float(np.abs(got - ref).max())
    print("reference: %s | max abs error: %.3g" % (ref_name, err))
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-6), "MISMATCH\n got %s\n ref %s" % (got, ref)

    # negative control: a corrupted lowering MUST be caught, otherwise this test proves nothing
    bad = text.replace("arith.constant 0.000000e+00 : f32", "arith.constant 1.000000e+00 : f32", 1)
    assert bad != text, "negative control did not modify the IR"
    bad_out = execute(bad, x, w, b, py)
    assert not np.allclose(bad_out, ref, rtol=1e-5, atol=1e-6), \
        "negative control NOT detected -- test is not sensitive"
    print("negative control: corrupted reduce init correctly detected")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
