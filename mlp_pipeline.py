"""Capture a tiny PyTorch MLP and print the project's hardware-agnostic IR.

CPU/local path:
    PYTHONPATH=. python mlp_pipeline.py

On CUDA with Triton installed, ``torch.compile`` is also exercised.  Inductor's
generated Triton source/cache is implementation detail, so the stable graph
input to this project is the exported FX graph.  A TTIR file can be inspected
separately with ``--ttir`` using the existing parser.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compiler.hgir_from_fx import from_exported_graph
from model.mlp import SimpleMLP


def capture(model, sample):
    exported = torch.export.export(model, (sample,))
    graph = from_exported_graph(exported)
    compiled = False
    numerical_match = None
    compile_status = "not attempted: CUDA/Triton unavailable"
    if sample.is_cuda:
        try:
            reference = model(sample)
            compiled_model = torch.compile(model, backend="inductor")
            actual = compiled_model(sample)
            torch.testing.assert_close(actual, reference)
            compiled = True
            numerical_match = True
            compile_status = "torch.compile backend=inductor succeeded"
        except Exception as exc:
            numerical_match = False
            compile_status = "torch.compile failed: %s: %s" % (
                type(exc).__name__, exc
            )
    return graph, compiled, numerical_match, compile_status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--output-size", type=int, default=8)
    parser.add_argument("--ttir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleMLP(args.input_size, args.hidden_size, args.output_size).to(device)
    sample = torch.randn(1, args.input_size, device=device)
    graph, compiled, numerical_match, status = capture(model, sample)
    if args.ttir:
        from compiler.hgir_from_ttir import from_ttir_module
        from compiler.ttir_reader import parse_ttir

        graph = from_ttir_module(parse_ttir(args.ttir.read_text()))

    report = {
        "torch": torch.__version__,
        "device": device,
        "torch_compile": compiled,
        "numerical_match": numerical_match,
        "compile_status": status,
        "ttir_input": str(args.ttir) if args.ttir else None,
    }
    print(json.dumps(report, indent=2))
    print(graph.format())
    if args.out:
        args.out.write_text(graph.format())


if __name__ == "__main__":
    main()
