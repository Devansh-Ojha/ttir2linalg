"""Capture genuine PyTorch Inductor code and a typed model graph.

This module is intentionally an observation boundary: it records whatever
the selected Inductor backend generated (for example C++ on CPU or Triton on
CUDA) and never substitutes hand-written Triton when Inductor did not emit it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from compiler.hgir_from_fx import from_exported_graph
from compiler.hgraph_analysis import analyze_hgraph
from compiler.triton_capture import extract_inductor_triton_sources


def _model_and_input(name: str, *, input_size: int, hidden_size: int,
                     output_size: int, sequence_length: int,
                     embed_size: int, num_heads: int):
    if name == "mlp":
        from model.mlp import SimpleMLP

        return SimpleMLP(input_size, hidden_size, output_size).eval(), (1, input_size)
    if name == "linear_relu":
        from model.linear_relu import LinearRelu

        return LinearRelu(input_size, output_size).eval(), (1, input_size)
    if name == "attention":
        from model.attention import SimpleAttention

        return SimpleAttention(embed_size, num_heads).eval(), (1, sequence_length, embed_size)
    if name == "attention_mlp":
        from model.attention_mlp import AttentionMLP

        return AttentionMLP(embed_size, num_heads, hidden_size).eval(), (1, sequence_length, embed_size)
    if name == "transformer_block":
        from model.transformer_block import TransformerBlock

        return TransformerBlock(embed_size, num_heads, hidden_size).eval(), (1, sequence_length, embed_size)
    raise ValueError("unsupported model %r" % name)


def _artifact_suffix(code: str) -> str:
    if "#include" in code or "extern \"C\"" in code:
        return ".cpp"
    if "triton_" in code or "@triton.jit" in code:
        return ".py"
    return ".txt"


def capture_inductor(model_name: str = "mlp", out_dir: str | Path = "inductor-artifacts",
                     *, input_size: int = 4, hidden_size: int = 8,
                     output_size: int = 8, sequence_length: int = 4,
                     embed_size: int = 8, num_heads: int = 2) -> dict[str, Any]:
    """Run the real Inductor backend and persist its emitted code and graph."""
    model, shape = _model_and_input(
        model_name, input_size=input_size, hidden_size=hidden_size,
        output_size=output_size, sequence_length=sequence_length,
        embed_size=embed_size, num_heads=num_heads,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    sample = torch.randn(*shape, device=device)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    exported = torch.export.export(model, (sample,))
    graph = from_exported_graph(exported)
    graph.name = model_name
    graph_path = output / "workload.hgir"
    graph_path.write_text(graph.format())
    graph_json_path = output / "workload.json"
    graph_json_path.write_text(json.dumps(graph.to_dict(), indent=2) + "\n")
    workload = analyze_hgraph(graph)

    compile_error = None
    code_capture_error = None
    numerical_match = None
    compile_succeeded = False
    code_units = []
    try:
        expected = model(sample)
        compiled = torch.compile(model, backend="inductor")
        actual = compiled(sample)
        torch.testing.assert_close(actual, expected)
        numerical_match = True
        compile_succeeded = True
    except Exception as exc:
        compile_error = "%s: %s" % (type(exc).__name__, exc)

    if compile_succeeded:
        try:
            from torch._inductor.utils import get_code

            code_units = get_code(compiled, sample)
            if not code_units:
                raise RuntimeError(
                    "Inductor completed but its code capture returned no artifacts"
                )
        except Exception as exc:
            code_capture_error = "%s: %s" % (type(exc).__name__, exc)

    code_paths: list[str] = []
    for index, code in enumerate(code_units):
        suffix = _artifact_suffix(code)
        path = output / ("inductor_%02d%s" % (index, suffix))
        path.write_text(code)
        code_paths.append(str(path))

    # The check above and source capture have distinct outcomes: successful
    # execution is useful evidence even if a PyTorch version lacks get_code.
    if not code_units and compile_succeeded and code_capture_error is None:
        code_capture_error = "Inductor returned no generated code artifacts"

    try:
        triton_sources = extract_inductor_triton_sources(code_units) if code_units else []
    except RuntimeError:
        triton_sources = []
    triton_paths = []
    for index, source in enumerate(triton_sources):
        path = output / ("inductor_triton_%02d.py" % index)
        path.write_text(source)
        triton_paths.append(str(path))

    report: dict[str, Any] = {
        "model": model_name,
        "torch_version": torch.__version__,
        "device": device,
        "compiler": "torch.compile(backend='inductor')",
        "inductor_status": (
            "captured" if code_units else
            "code_capture_failed" if compile_succeeded else "failed"
        ),
        "code_origin": "pytorch_inductor" if code_units else None,
        "code_artifacts": code_paths,
        "code_count": len(code_paths),
        "generated_code_kinds": sorted({Path(p).suffix for p in code_paths}),
        "inductor_triton_kernel_artifacts": triton_paths,
        "inductor_triton_kernel_count": len(triton_paths),
        "numerical_match": numerical_match,
        "inductor_error": compile_error,
        "code_capture_error": code_capture_error,
        "hgraph_origin": "torch.export",
        "hgraph_artifact": str(graph_path),
        "hgraph_json_artifact": str(graph_json_path),
        "hgraph_op_count": len(graph.ops),
        "workload_analysis": workload.to_dict(),
        "atlas_virtual_ops_emitted": False,
    }
    if model_name in {"attention", "attention_mlp", "transformer_block"}:
        report["attention_request"] = "nn.MultiheadAttention(need_weights=False) / SDPA"
        report["attention_backend_evidence"] = [
            line.strip()
            for code in code_units
            for line in code.splitlines()
            if any(token in line.lower() for token in (
                "flash", "fmha", "efficient_attention", "scaled_dot_product", "cudnn_attention"
            ))
        ]
        report["flash_attention_kernel_detected_in_generated_code"] = any(
            any(token in line.lower() for token in ("flash", "fmha"))
            for line in report["attention_backend_evidence"]
        )
    report_path = output / "capture.json"
    report["report_artifact"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(
        "linear_relu", "mlp", "attention", "attention_mlp", "transformer_block"
    ), default="mlp")
    parser.add_argument("--out-dir", type=Path, default=Path("inductor-artifacts"))
    parser.add_argument("--input-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--output-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--embed-size", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(capture_inductor(
        args.model, args.out_dir,
        input_size=args.input_size, hidden_size=args.hidden_size,
        output_size=args.output_size, sequence_length=args.sequence_length,
        embed_size=args.embed_size, num_heads=args.num_heads,
    ), indent=2))


if __name__ == "__main__":
    main()
