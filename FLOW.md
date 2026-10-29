# Current PyTorch Inductor → Atlas flow

The main path starts from a PyTorch model and translates artifacts produced by
the actual Inductor/Triton compiler. The source `torch.export` HGraph is saved
for context; it does not provide operations to the Inductor-to-Atlas mapper.

```text
model/*.py + sample tensors
       │
       ▼
torch.compile(..., backend="inductor")
       ├── generated wrapper / Triton source ──► inductor_*.py/.cpp
       ├── actual Triton compilation ──────────► inductor_ttir/*.ttir
       ├── torch.export (context only) ─────────► workload.hgir/.json
       └── eager comparison + evidence ─────────► capture.json
                                                    │
                                                    ▼
                                    compiler.inductor_to_atlas
                                      ├── parse exact dumped TTIR
                                      ├── strict supported-pattern analysis
                                      ├── per-kernel Atlas mapping
                                      └── mapping.json (mapped + unsupported)
                                                    │
                                                    ▼
                                      Atlas-MLIR virtual verification
                                      scheduling / allocation / lowering
```

`compiler/inductor_capture.py` sets `TRITON_KERNEL_DUMP` and the dump directory
before importing PyTorch, so TTIR comes from Triton compiling Inductor's own
kernels. It never recompiles the project's handwritten Triton reference
kernels for this route. CUDA is needed for the Triton kernel path. For attention
workloads the capture restricts SDPA to FlashAttention on CUDA; if the shape,
device, or installed stack cannot select it, the capture reports the failure.

`compiler/inductor_to_atlas.py` reads the TTIR artifact list in `capture.json`
and attempts the existing strict analyzers for linear+ReLU, attention,
matmul+ReLU, or the recognized two-layer MLP. An Atlas file is written only
when an exact TTIR artifact matches a supported pattern. Every other kernel is
reported with diagnostics; no graph-side or handwritten-kernel fallback is
used. Atlas files state an explicit numerical policy and caller-staged tile
ABI, and represent compute operations rather than performing runtime tensor
movement.

## Workloads

| Model flag | Model | Current expectation |
| --- | --- | --- |
| `linear_relu` | One projection + ReLU | First Inductor kernel matching target. |
| `mlp` | FC1 + ReLU + FC2 | Kernels matched independently; complete coverage depends on emitted TTIR. |
| `attention` | Self-attention using SDPA | FlashAttention is required on CUDA; current Atlas attention matcher is bounded and may report unsupported. |
| `attention_mlp` | Attention followed by MLP | Captures generated kernels and reports per-kernel Atlas coverage. |
| `transformer_block` | Attention, residual, normalization, MLP | Same; monolithic fusion is measured, not assumed. |

Inductor may emit C++ or external library calls instead of Triton for an
operation. Those are retained as real capture artifacts but do not enter the
TTIR mapper. The capture report distinguishes code units from Triton kernel
count and TTIR count. One FX graph does not imply one device kernel.

## Important independent experiments

`torch.export → HGIR → hgraph_to_atlas.py` is a graph-semantics prototype for
small dense MLPs. It does not consume Inductor kernel output. Similarly,
`kernels/*.py → triton_capture.py` is a handwritten Triton reference path, and
`ttir_to_linalg.py` is an older Linalg experiment. Neither is inserted into the
main flow above.

## Atlas backend ownership

This repository recognizes supported TTIR semantics and emits Atlas virtual
SSA programs. The Atlas-MLIR checkout owns virtual-op verification, scheduling,
allocation, and machine lowering. A generated virtual file is not executable
until the caller stages the tensor tiles required by its explicit ABI.
