"""Probe the compiler-only path from a PyTorch MLP toward Triton/TTIR.

This intentionally does not synthesize TTIR.  On CPU, PyTorch Inductor uses
its CPU backend (normally C++), so the genuine representation available from
``torch.compile`` is the captured FX graph.  In an environment with Triton,
the existing ``triton_capture`` module can compile a Triton source to TTIR
without launching it, but that is a direct Triton-kernel path rather than a
CPU ``torch.compile`` path.
"""
from __future__ import annotations

import importlib.util
import json

import torch

from compiler.hgir_from_fx import from_fx_graph
from model.mlp import SimpleMLP


def probe() -> dict:
    model = SimpleMLP(4, 8, 8).eval()
    sample = torch.randn(1, 4)
    captured = []

    def capture_backend(graph_module, example_inputs):
        captured.append(graph_module)
        return graph_module

    compiled_output_matches = False
    compile_error = None
    try:
        compiled = torch.compile(model, backend=capture_backend)
        actual = compiled(sample)
        torch.testing.assert_close(actual, model(sample))
        compiled_output_matches = True
    except Exception as exc:
        compile_error = "%s: %s" % (type(exc).__name__, exc)

    triton_available = importlib.util.find_spec("triton") is not None
    try:
        import torch._inductor.config as inductor_config

        cpu_backend = str(inductor_config.cpu_backend)
    except Exception as exc:
        cpu_backend = "unavailable: %s: %s" % (type(exc).__name__, exc)

    report = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "triton_available": triton_available,
        "cpu_inductor_backend": cpu_backend,
        "torch_compile_graph_captured": bool(captured),
        "compiled_output_matches": compiled_output_matches,
        "compile_error": compile_error,
        "representation": "fx_graph" if captured else None,
        "ttir_obtained": False,
        "ttir_reason": (
            "CPU Inductor selected %s; it does not emit Triton/TTIR"
            % cpu_backend
            if not triton_available
            else (
                "torch.compile used the CPU path; use a Triton backend or an "
                "explicit Triton source compilation to obtain TTIR"
            )
        ),
    }
    if captured:
        report["hgir"] = from_fx_graph(captured[0].graph).format()
    return report


def main() -> None:
    print(json.dumps(probe(), indent=2))


if __name__ == "__main__":
    main()
