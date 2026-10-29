"""Capture genuine PyTorch Inductor code and a typed model graph.

This module is intentionally an observation boundary: it records whatever
the selected Inductor backend generated (for example C++ on CPU or Triton on
CUDA) and never substitutes hand-written Triton when Inductor did not emit it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any
from contextlib import nullcontext

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
    if name == "attention_flash":
        from model.attention import SimpleAttention

        return SimpleAttention(embed_size, num_heads).eval(), (1, sequence_length, embed_size)
    if name == "attention_custom_flash":
        from model.custom_flash_attention import CustomFlashAttention

        return CustomFlashAttention(embed_size, num_heads).eval(), (1, sequence_length, embed_size)
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


def _launch_sites(code_units: list[str]) -> list[dict[str, Any]]:
    """Collect generated wrapper call sites for Triton kernel launches."""
    sites = []
    for unit_index, code in enumerate(code_units):
        for line_number, line in enumerate(code.splitlines(), 1):
            match = re.search(r"\b([A-Za-z_]\w*)\.run\(", line)
            if match:
                sites.append({
                    "code_unit": unit_index,
                    "line": line_number,
                    "kernel_symbol": match.group(1),
                })
    return sites


def _ttir_function_symbols(path: Path) -> set[str]:
    text = path.read_text()
    return set(re.findall(
        r"\btt\.func\s+(?:public\s+)?@([A-Za-z_][A-Za-z0-9_.$-]*)",
        text,
    ))


def _make_fake_cuda_model(torch, model_name: str, *, input_size: int,
                          hidden_size: int, output_size: int,
                          sequence_length: int, embed_size: int,
                          num_heads: int):
    """Construct model parameters and inputs as fake CUDA tensors, no device use."""
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError as exc:
        raise RuntimeError(
            "CUDA cross-compilation requires FakeTensorMode from this PyTorch build"
        ) from exc
    mode = FakeTensorMode()
    try:
        with mode, torch.device("cuda"):
            model, shape = _model_and_input(
                model_name, input_size=input_size, hidden_size=hidden_size,
                output_size=output_size, sequence_length=sequence_length,
                embed_size=embed_size, num_heads=num_heads,
            )
            sample = torch.randn(*shape, device="cuda")
    except Exception as exc:
        raise RuntimeError(
            "could not create fake CUDA model/input; install a CUDA-enabled "
            "PyTorch build and check its fake-tensor support: %s: %s"
            % (type(exc).__name__, exc)
        ) from exc
    return model, sample


def capture_inductor(model_name: str = "mlp", out_dir: str | Path = "inductor-artifacts",
                     *, input_size: int = 4, hidden_size: int = 8,
                     output_size: int = 8, sequence_length: int = 128,
                     embed_size: int = 64, num_heads: int = 2,
                     mode: str = "aot-cuda") -> dict[str, Any]:
    """Compile through Inductor; CUDA AOT mode does not execute on a device."""
    if mode not in {"aot-cuda", "run"}:
        raise ValueError("mode must be 'aot-cuda' or 'run'")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Triton reads these settings during import. Set them before importing
    # torch/model modules so Triton dumps TTIR while Inductor compiles its
    # actual kernels. Force compilation so cache hits cannot hide the IR dump.
    triton_dump_dir = output / "inductor_triton_dump" / ("run-%d" % time.time_ns())
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["TRITON_DUMP_DIR"] = str(triton_dump_dir.resolve())
    if mode == "aot-cuda":
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    import torch

    if mode == "aot-cuda":
        if not torch.backends.cuda.is_built():
            raise RuntimeError(
                "CUDA-target cross-compilation needs a CUDA-enabled PyTorch build; "
                "it does not need a physical GPU"
            )
        model, sample = _make_fake_cuda_model(
            torch, model_name, input_size=input_size, hidden_size=hidden_size,
            output_size=output_size, sequence_length=sequence_length,
            embed_size=embed_size, num_heads=num_heads,
        )
        device = "cuda (fake tensors; compile only)"
    else:
        model, shape = _model_and_input(
            model_name, input_size=input_size, hidden_size=hidden_size,
            output_size=output_size, sequence_length=sequence_length,
            embed_size=embed_size, num_heads=num_heads,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        sample = torch.randn(*shape, device=device)

    graph_path = output / "workload.hgir"
    graph_json_path = output / "workload.json"
    graph_error = None
    workload = None
    try:
        exported = torch.export.export(model, (sample,))
        graph = from_exported_graph(exported)
        graph.name = model_name
        graph_path.write_text(graph.format())
        graph_json_path.write_text(json.dumps(graph.to_dict(), indent=2) + "\n")
        workload = analyze_hgraph(graph)
    except Exception as exc:
        graph_error = "%s: %s" % (type(exc).__name__, exc)

    compile_error = None
    code_capture_error = None
    numerical_match = None
    compile_succeeded = False
    code_units = []
    attention_backend_constraint = "not_applicable"
    attention_context = nullcontext()
    ready_made_flash = model_name in {"attention", "attention_flash", "attention_mlp", "transformer_block"}
    if ready_made_flash and device.startswith("cuda"):
        # Make FlashAttention a required backend for this experiment, rather
        # than inferring it from the fact that the model requested SDPA.
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            attention_context = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
            attention_backend_constraint = "flash_attention_required"
        except ImportError:
            attention_backend_constraint = "flash_attention_constraint_unavailable"
    try:
        if mode == "aot-cuda":
            import torch._inductor.config as inductor_config

            inductor_config.trace.enabled = True
            inductor_config.trace.debug_dir = str((output / "torch_compile_debug").resolve())
            compiled = torch.compile(model, backend="inductor", fullgraph=True)
            with attention_context:
                compiled.forward.aot_compile(((sample,), {}))
        else:
            compiled = torch.compile(model, backend="inductor")
            with attention_context:
                reference = getattr(model, "reference_forward", model)
                expected = reference(sample)
                actual = compiled(sample)
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
            numerical_match = True
        compile_succeeded = True
    except Exception as exc:
        compile_error = "%s: %s" % (type(exc).__name__, exc)

    if compile_succeeded:
        if mode == "aot-cuda":
            debug_root = output / "torch_compile_debug"
            for source_path in sorted(debug_root.rglob("output_code.py")):
                code_units.append(source_path.read_text())
            if not code_units:
                code_capture_error = (
                    "AOT compile succeeded but no output_code.py appeared in "
                    "the configured Inductor debug directory"
                )
        else:
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
    launch_sites = _launch_sites(code_units)

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

    # These are compiler-emitted TTIR dumps, not TTIR recompiled from our
    # handwritten reference Triton kernels. Copy the dump artifacts into the
    # capture directory so downstream translation has a stable input.
    dumped_ttir_paths = []
    referenced_ttir_paths = []
    unassociated_ttir_paths = []
    ttir_launch_associations = []
    launch_symbols = {site["kernel_symbol"] for site in launch_sites}
    if triton_dump_dir.exists():
        emitted_ttir_dir = output / "inductor_ttir"
        for index, source_path in enumerate(sorted(triton_dump_dir.rglob("*.ttir"))):
            target = emitted_ttir_dir / ("kernel_%03d_%s" % (index, source_path.name))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            dumped_ttir_paths.append(str(target))
            ttir_symbols = sorted(_ttir_function_symbols(target))
            matched_symbols = sorted(launch_symbols.intersection(ttir_symbols))
            ttir_launch_associations.append({
                "ttir_artifact": str(target),
                "ttir_symbols": ttir_symbols,
                "matched_launch_symbols": matched_symbols,
            })
            if matched_symbols:
                referenced_ttir_paths.append(str(target))
            else:
                unassociated_ttir_paths.append(str(target))

    report: dict[str, Any] = {
        "model": model_name,
        "torch_version": torch.__version__,
        "device": device,
        "compile_mode": mode,
        "physical_gpu_used": mode == "run" and device == "cuda",
        "runtime_validated": numerical_match is True,
        "compiler": "torch.compile(backend='inductor')",
        "inductor_status": (
            "captured" if code_units else
            "code_capture_failed" if compile_succeeded else "failed"
        ),
        "code_origin": "pytorch_inductor" if code_units else None,
        "code_artifacts": code_paths,
        "code_count": len(code_paths),
        "inductor_kernel_launch_site_count": len(launch_sites),
        "inductor_kernel_launch_sites": launch_sites,
        "generated_code_kinds": sorted({Path(p).suffix for p in code_paths}),
        "inductor_triton_kernel_artifacts": triton_paths,
        "inductor_triton_kernel_count": len(triton_paths),
        "inductor_ttir_artifacts": dumped_ttir_paths,
        "inductor_ttir_kernel_count": len(dumped_ttir_paths),
        "inductor_wrapper_referenced_ttir_artifacts": referenced_ttir_paths,
        "inductor_wrapper_referenced_ttir_kernel_count": len(referenced_ttir_paths),
        "inductor_unassociated_ttir_artifacts": unassociated_ttir_paths,
        "inductor_ttir_launch_associations": ttir_launch_associations,
        "inductor_kernel_launch_symbols": sorted(launch_symbols),
        "inductor_ttir_origin": "triton_compiler_dump" if dumped_ttir_paths else None,
        "numerical_match": numerical_match,
        "inductor_error": compile_error,
        "code_capture_error": code_capture_error,
        "hgraph_origin": "torch.export",
        "hgraph_artifact": str(graph_path),
        "hgraph_json_artifact": str(graph_json_path),
        "hgraph_op_count": len(graph.ops) if workload is not None else None,
        "hgraph_error": graph_error,
        "workload_analysis": workload.to_dict() if workload is not None else None,
        "atlas_virtual_ops_emitted": False,
    }
    if model_name == "attention_custom_flash":
        report["custom_triton_kernel_source"] = str(
            (Path(__file__).resolve().parents[1] / "kernels/flash_attention.py")
        )
        report["custom_triton_kernel"] = "flash_attention_kernel (online softmax)"
        report["custom_kernel_numerical_validation"] = (
            "passed" if numerical_match is True else
            "not_run" if numerical_match is None else "failed"
        )
    if model_name in {"attention", "attention_flash", "attention_custom_flash", "attention_mlp", "transformer_block"}:
        report["attention_request"] = (
            "custom Triton online-softmax FlashAttention"
            if model_name == "attention_custom_flash"
            else "nn.MultiheadAttention(need_weights=False) / SDPA"
        )
        report["attention_implementation"] = (
            "repository_triton_flash_attention"
            if model_name == "attention_custom_flash"
            else "pytorch_ready_made_sdpa_flash_attention"
        )
        report["correctness_reference"] = (
            "PyTorch math SDPA with identical learned projections"
            if hasattr(model, "reference_forward") else "eager PyTorch model"
        )
        report["correctness_check_passed"] = numerical_match
        report["correctness_check_scope"] = (
            "compiled output compared to independent PyTorch math-SDPA reference"
            if hasattr(model, "reference_forward") else
            "runtime output compared to eager model; AOT compile alone does not validate numerics"
        )
        report["attention_backend_constraint"] = attention_backend_constraint
        report["flash_attention_backend_compiled_under_constraint"] = (
            attention_backend_constraint == "flash_attention_required"
            and compile_succeeded
        )
        report["flash_attention_backend_forced_and_run_succeeded"] = (
            attention_backend_constraint == "flash_attention_required"
            and numerical_match is True
        )
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
        report["flash_attention_kernel_detected_in_ttir"] = any(
            any(token in (Path(path).name + Path(path).read_text()).lower()
                for token in ("flash", "fmha", "fused_attention"))
            for path in dumped_ttir_paths
        )
    if model_name in {"attention_mlp", "transformer_block"}:
        report["one_kernel_fusion_observed"] = (
            len(launch_sites) == 1 if device.startswith("cuda") and code_units else None
        )
    report_path = output / "capture.json"
    report["report_artifact"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        from compiler.inductor_to_atlas import map_capture

        mapping = map_capture(output)
        report["atlas_mapping_status"] = (
            "completed_with_supported_kernels"
            if mapping["mapped_kernel_count"] else "no_supported_kernels"
        )
        report["atlas_mapping_report"] = mapping["report_artifact"]
        report["atlas_virtual_ops_emitted"] = mapping["mapped_kernel_count"] > 0
        report["atlas_mapped_kernel_count"] = mapping["mapped_kernel_count"]
    except Exception as exc:
        report["atlas_mapping_status"] = "not_available"
        report["atlas_mapping_error"] = "%s: %s" % (type(exc).__name__, exc)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(
        "linear_relu", "mlp", "attention", "attention_flash",
        "attention_custom_flash", "attention_mlp", "transformer_block"
    ), default="mlp")
    parser.add_argument("--mode", choices=("aot-cuda", "run"), default="aot-cuda",
                        help="aot-cuda cross-compiles with fake CUDA tensors and no GPU")
    parser.add_argument("--out-dir", type=Path, default=Path("inductor-artifacts"))
    parser.add_argument("--input-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--output-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(capture_inductor(
        args.model, args.out_dir,
        input_size=args.input_size, hidden_size=args.hidden_size,
        output_size=args.output_size, sequence_length=args.sequence_length,
        embed_size=args.embed_size, num_heads=args.num_heads, mode=args.mode,
    ), indent=2))


if __name__ == "__main__":
    main()
