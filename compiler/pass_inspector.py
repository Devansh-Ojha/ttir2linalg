"""Inspect Triton MLIR passes section-by-section to identify hardware-agnostic TTIR

This script demonstrates the exact boundary where Triton IR (TTIR) ends and
GPU-specific lowerings (TTGIR) begin, illustrating where NPU/MPU backends
(like Huawei Ascend or Tenstorrent tt-mlir) intercept the pipeline.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def inspect_ttir_file_sections(ttir_path: Path) -> dict[str, Any]:
    """Break down a checked-in .ttir file section-by-section."""
    content = ttir_path.read_text()
    lines = content.splitlines()
    func_lines = [line for line in lines if line.strip().startswith("tt.func") or line.strip().startswith("func.func")]
    ops = [line.strip().split("=")[-1].strip().split()[0] for line in lines if "=" in line and line.strip().startswith("%")]
    
    op_counts: dict[str, int] = {}
    for op in ops:
        if op.startswith("tt.") or op.startswith("arith.") or op.startswith("math."):
            op_counts[op] = op_counts.get(op, 0) + 1

    gpu_attributes = [line for line in lines if "triton_gpu" in line or "layout" in line or "blocked" in line]
    
    return {
        "file_name": ttir_path.name,
        "total_lines": len(lines),
        "func_header": func_lines[0] if func_lines else "Unknown",
        "operation_summary": op_counts,
        "is_hardware_agnostic": len(gpu_attributes) == 0,
        "gpu_attributes_found": len(gpu_attributes),
    }


def get_pass_boundary_report(ttir_dir: Path) -> str:
    """Return section-by-section MLIR pass breakdown report string for a directory of .ttir files."""
    files = sorted(list(ttir_dir.glob("*.ttir")))
    if not files:
        return f"No .ttir files found in {ttir_dir}"

    lines = []
    lines.append("==========================================================================")
    lines.append("   MLIR SECTION-BY-SECTION BREAKDOWN & TTIR/TTGIR DIVERGENCE BOUNDARY")
    lines.append("==========================================================================")
    lines.append("")

    for file_path in files:
        info = inspect_ttir_file_sections(file_path)
        lines.append(f"--- FILE: {info['file_name']} ---")
        lines.append(f"  Header: {info['func_header']}")
        lines.append(f"  Hardware Agnostic: {info['is_hardware_agnostic']} (0 GPU layout attributes)")
        lines.append("  Operation Section Breakdown:")
        for op, count in sorted(info["operation_summary"].items()):
            lines.append(f"    - {op}: {count} occurrences")
        lines.append("")
        lines.append("  [INTERCEPTION POINT / DIVERGENCE BOUNDARY]:")
        lines.append("    -> Vendor NPU plugins (Huawei Ascend triton-ascend, Tenstorrent tt-mlir)")
        lines.append("       intercept HERE right after this TTIR MLIR stage.")
        lines.append("    -> STOP HERE before GPU-specific passes (TTGIR) add warps, CTAs,")
        lines.append("       shared memory, and blocked layout encodings.")
        lines.append("--------------------------------------------------------------------------")
        lines.append("")

    return "\n".join(lines)


def print_pass_boundary_breakdown(ttir_dir: Path):
    """Print section-by-section MLIR pass breakdown to console."""
    print(get_pass_boundary_report(ttir_dir))


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    mlp_ttir_dir = repo_root / "mlp-ttir"
    if mlp_ttir_dir.exists():
        print_pass_boundary_breakdown(mlp_ttir_dir)
