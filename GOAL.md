# Project goal: TorchInductor kernels to Atlas

## Objective

For representative PyTorch workloads, run the real TorchInductor backend,
capture its generated kernels and compiler IR, and translate supported emitted
operations into Atlas virtual SSA. The workloads are a linear+ReLU, a two-layer
MLP, FlashAttention/SDPA, and a small attention-plus-MLP Transformer block.
Generated Atlas operations must name their exact Inductor TTIR source; the
separate `torch.export` HGraph is optional semantic context and is never a
substitute for consuming Inductor output.

## Intended path

```text
PyTorch module + example inputs
  -> torch.compile(..., backend="inductor")
  -> Inductor generated source + Triton compiler TTIR dumps (when emitted)
  -> parse and recognize supported operations in those exact TTIR files
  -> Atlas virtual SSA with explicit tile/precision/ABI policy
  -> Atlas-MLIR verification, scheduling, allocation, and lowering
```

Some Inductor backends emit C++ or library calls instead of Triton kernels.
Those artifacts must be reported as unsupported inputs to this Atlas mapper;
never synthesize a Triton kernel and label it Inductor output. The existing
handwritten Triton examples remain reference fixtures only.

## Workloads and fusion

Use a single linear+ReLU and two-layer MLP as the first end-to-end cases, then
force the PyTorch SDPA FlashAttention backend for attention workloads on CUDA.
The attention+MLP block combines attention, residual/normalization work, and a
feed-forward MLP. Record how many real generated kernels exist and which exact
ones map. Pursue a monolithic Atlas kernel as an optimization goal; do not
claim that separate kernels or a single FX graph have achieved that goal.
Keep precise unsupported-operation diagnostics and a multi-kernel result when
whole-block fusion is not supported.

## Completion criteria

- One command captures actual Inductor code and Triton TTIR dumps for each
  workload, with numerical-check and backend-selection results recorded.
- Atlas conversion consumes those exact TTIR artifacts and writes one output
  per mapped kernel. Every unmapped kernel has a diagnostic; no HGraph or
  handwritten-kernel fallback is mislabeled as an Inductor lowering.
- FlashAttention is explicitly selected or its failure/fallback is reported.
- Atlas virtual programs pass the Atlas-MLIR verifier and downstream lowering.
- The attention+MLP report states actual kernel/launch boundaries and whether
  a one-kernel implementation was achieved.

## Current implementation state

`compiler.inductor_capture` invokes real Inductor, enables Triton's compiler
IR dump before importing PyTorch, saves generated code and exact `.ttir` dumps,
and forces Flash SDPA for CUDA attention workloads. It then invokes
`compiler.inductor_to_atlas`, which consumes only those dumped TTIR files and
tries the repository's strict Atlas workload recognizers. Successful matches
produce per-kernel virtual MLIR with source provenance; unmatched kernels are
listed with the attempted-pattern diagnostics. This path is structurally
implemented but still needs to be run on BWRC to confirm the installed
PyTorch/Triton versions, actual FlashAttention selection, and which Inductor
TTIR shapes match the bounded Atlas recognizers.

The existing HGraph-to-Atlas mapper remains a separate graph-semantics
experiment. It does not feed the Inductor-kernel path. The repository's older
Linalg prototype is also separate and is not an intermediate stage in this
goal. Atlas mapping still has bounded shape/precision support; full attention,
normalization, residual and monolithic-fusion coverage remain work items.
