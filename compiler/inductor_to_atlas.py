"""Translate Atlas-admissible TTIR dumped from real Inductor Triton kernels.

This deliberately does not fall back to the separately exported HGraph or to
the repository's handwritten Triton examples. Each emitted Atlas function is
therefore backed by one exact Inductor TTIR artifact; unsupported kernels are
reported with the analyzer failures that prevented mapping.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compiler.atlas_mapping import (
    ATTENTION_NUMERICAL_POLICY,
    LINEAR_NUMERICAL_POLICY,
    MLP_NUMERICAL_POLICY,
)
from compiler.ttir_to_atlas import convert_file


_PATTERNS = (
    ("linear-relu", LINEAR_NUMERICAL_POLICY),
    ("attention", ATTENTION_NUMERICAL_POLICY),
    ("matmul-relu", None),
    ("two-layer-mlp", MLP_NUMERICAL_POLICY),
)


def map_capture(capture_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    capture_dir = capture_dir.resolve()
    report_path = capture_dir / "capture.json"
    if not report_path.is_file():
        raise FileNotFoundError("missing Inductor capture report: %s" % report_path)
    capture = json.loads(report_path.read_text())
    raw_artifacts = capture.get("inductor_launched_ttir_artifacts", [])
    artifacts = [Path(p) for p in raw_artifacts]
    if not artifacts:
        raise RuntimeError(
            "no TTIR dump was associated with an Inductor wrapper launch; check "
            "inductor_kernel_launch_symbols and inductor_unassociated_ttir_artifacts"
        )
    output_dir = (output_dir or capture_dir / "atlas").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for artifact in artifacts:
        if not artifact.is_absolute():
            artifact = (capture_dir / artifact).resolve()
        item: dict[str, Any] = {
            "source_ttir": str(artifact),
            "source_origin": "pytorch_inductor_triton_dump",
            "status": "unsupported",
            "attempts": [],
        }
        if not artifact.is_file():
            item["error"] = "TTIR artifact listed in capture report is missing"
            results.append(item)
            continue
        for pattern, policy in _PATTERNS:
            target = output_dir / (artifact.stem + "." + pattern + ".virtual.mlir")
            try:
                virtual_mlir = convert_file(
                    artifact, pattern=pattern, numerical_policy=policy
                )
            except Exception as exc:
                item["attempts"].append({
                    "pattern": pattern,
                    "reason": "%s: %s" % (type(exc).__name__, exc),
                })
                continue
            target.write_text(virtual_mlir)
            item.update({
                "status": "mapped",
                "pattern": pattern,
                "numerical_policy": policy,
                "atlas_artifact": str(target),
            })
            break
        results.append(item)

    summary = {
        "model": capture.get("model"),
        "input_capture": str(report_path),
        "source": "compiler-dumped TTIR from PyTorch Inductor",
        "inductor_launched_ttir_kernel_count": len(artifacts),
        "mapped_kernel_count": sum(r["status"] == "mapped" for r in results),
        "unsupported_kernel_count": sum(r["status"] != "mapped" for r in results),
        "kernels": results,
    }
    summary_path = output_dir / "mapping.json"
    summary["report_artifact"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    capture["atlas_mapping_report"] = str(summary_path)
    capture["atlas_virtual_ops_emitted"] = summary["mapped_kernel_count"] > 0
    capture["atlas_mapped_kernel_count"] = summary["mapped_kernel_count"]
    report_path.write_text(json.dumps(capture, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path,
                        help="directory written by compiler.inductor_capture")
    parser.add_argument("--out-dir", type=Path,
                        help="defaults to <capture_dir>/atlas")
    args = parser.parse_args()
    print(json.dumps(map_capture(args.capture_dir, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
