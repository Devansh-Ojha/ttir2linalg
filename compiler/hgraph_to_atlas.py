"""Map a serialized PyTorch-export HGraph dense workload to Atlas virtual SSA.

This frontend consumes the normalized graph artifact, not Inductor's generated
machine/code artifact. It is a bridge from torch.export semantics to the
existing Atlas virtual dialect; code-to-graph kernel association remains a
separate unfinished compiler stage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compiler.hgir import HGraph
from compiler.hgraph_analysis import analyze_dense_graph
from compiler.hgraph_atlas_mapping import (
    PYTORCH_DENSE_NUMERICAL_POLICY,
    AtlasHGraphDenseABI,
    map_hgraph_dense,
)


def convert_file(source: Path, output: Path, *,
                 numerical_policy: str | None = None,
                 abi: AtlasHGraphDenseABI = AtlasHGraphDenseABI()) -> str:
    graph = HGraph.from_dict(json.loads(source.read_text()))
    plan = analyze_dense_graph(graph)
    virtual_mlir = map_hgraph_dense(
        plan, abi, numerical_policy=numerical_policy
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(virtual_mlir)
    return plan.pattern


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="serialized workload.json HGraph")
    parser.add_argument("--out", type=Path, required=True,
                        help="Atlas virtual SSA MLIR output")
    parser.add_argument(
        "--numerical-policy", required=True,
        choices=(PYTORCH_DENSE_NUMERICAL_POLICY,),
        help="explicitly authorize the FP8/BF16 dense mapping",
    )
    args = parser.parse_args()
    pattern = convert_file(
        args.input, args.out, numerical_policy=args.numerical_policy
    )
    print("Mapped torch.export HGraph pattern %s to %s" % (pattern, args.out))


if __name__ == "__main__":
    main()
