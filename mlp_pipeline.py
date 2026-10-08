"""Capture a tiny PyTorch MLP and print the project's hardware-agnostic IR.

CPU/local path:
    PYTHONPATH=. python mlp_pipeline.py

The script uses a compiler callback with ``torch.compile`` on every device.
This captures the generated FX graph and checks its numerical result without
requiring CUDA, an NVIDIA driver, or Triton execution.  A TTIR file can be
inspected separately with ``--ttir`` using the existing parser.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compiler.hgir_from_fx import from_exported_graph, from_fx_graph
from model.mlp import SimpleMLP


def capture(model, sample):
    exported = torch.export.export(model, (sample,))
    graph = from_exported_graph(exported)
    compiled = False
    compile_mode = "torch.export"
    numerical_match = None
    compile_status = "torch.compile not attempted"
    captured = []

    def capture_backend(graph_module, example_inputs):
        captured.append(graph_module)
        return graph_module

    try:
        compiled_model = torch.compile(model, backend=capture_backend)
        actual = compiled_model(sample)
        reference = model(sample)
        torch.testing.assert_close(actual, reference)
        compiled = True
        compile_mode = "torch.compile graph capture"
        numerical_match = True
        compile_status = "captured FX graph without a device backend"
        if captured:
            graph = from_fx_graph(captured[0].graph)
    except Exception as exc:
        compile_status = "torch.compile capture failed: %s: %s" % (
            type(exc).__name__, exc
        )
    return graph, compiled, numerical_match, compile_status, compile_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--output-size", type=int, default=8)
    parser.add_argument(
        "--ttir",
        type=Path,
        action="append",
        default=[],
        help="captured TTIR kernel; repeat for every generated kernel",
    )
    parser.add_argument(
        "--ttir-dir",
        type=Path,
        default=None,
        help="directory containing captured *.ttir kernels",
    )
    parser.add_argument(
        "--compile-triton",
        action="store_true",
        help="compile a representative Triton linear+ReLU kernel to TTIR",
    )
    parser.add_argument(
        "--compile-mlp-triton",
        action="store_true",
        help="compile FC1+ReLU and FC2 to separate Triton TTIR files",
    )
    parser.add_argument(
        "--compile-mlp-triton-source",
        action="store_true",
        help="capture Triton source emitted by PyTorch Inductor",
    )
    parser.add_argument(
        "--triton-dir",
        type=Path,
        default=None,
        help="directory for Inductor-generated Triton source files",
    )
    parser.add_argument("--ttir-out", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleMLP(args.input_size, args.hidden_size, args.output_size).to(device)
    sample = torch.randn(1, args.input_size, device=device)
    graph, compiled, numerical_match, status, compile_mode = capture(model, sample)
    if args.compile_mlp_triton_source:
        from compiler.triton_capture import write_mlp_triton_source

        source_dir = args.triton_dir or Path("mlp-triton")
        files = write_mlp_triton_source(
            source_dir,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            output_size=args.output_size,
        )
        print(json.dumps({
            "triton_source_dir": str(source_dir),
            "triton_source_count": len(files),
            "triton_source_files": [str(path) for path in files],
        }, indent=2))
        return
    ttir_paths = list(args.ttir)
    if args.compile_triton:
        from compiler.triton_capture import write_verified_ttir

        ttir_path = args.ttir_out or Path("mlp_linear.ttir")
        write_verified_ttir(
            ttir_path,
            input_size=args.input_size,
            output_size=args.hidden_size,
        )
        ttir_paths.append(ttir_path)
    compiled_mlp = False
    if args.compile_mlp_triton:
        from compiler.triton_capture import write_mlp_ttir

        ttir_dir = args.ttir_dir or Path("mlp-ttir")
        ttir_paths.extend(write_mlp_ttir(
            ttir_dir,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            output_size=args.output_size,
        ))
        compiled_mlp = True
    if args.ttir_dir and not compiled_mlp:
        ttir_paths.extend(sorted(args.ttir_dir.glob("*.ttir")))

    graphs = None
    if ttir_paths:
        from compiler.hgir_from_ttir import from_ttir_files

        graphs = from_ttir_files(ttir_paths)

    report = {
        "torch": torch.__version__,
        "device": device,
        "torch_compile": compiled,
        "compile_mode": compile_mode,
        "numerical_match": numerical_match,
        "compile_status": status,
        "ttir_input": [str(path) for path in ttir_paths] or None,
        "ttir_kernel_count": len(graphs) if graphs else 0,
    }
    print(json.dumps(report, indent=2))
    if graphs:
        rendered = "\n".join(
            "=== HGIR kernel %d/%d ===\n%s" % (index, len(graphs), item.format())
            for index, item in enumerate(graphs, 1)
        )
    else:
        rendered = graph.format()
    print(rendered)
    if args.out:
        args.out.write_text(rendered)


if __name__ == "__main__":
    main()
