# Project goal: model-to-Atlas compilation

## Objective

Turn this repository into a reproducible, end-to-end compiler experiment that
starts from ordinary PyTorch Transformer workloads, follows PyTorch Inductor's
actual compiler outputs, and lowers supported computations to both the
repository's Linalg/SE-level representation and Atlas virtual SSA operations.
The first workload family is a single linear+activation, a two-layer MLP,
self-attention using FlashAttention-style online softmax where supported, and
a residual Transformer block combining attention and a two-layer MLP.

The purpose is to demonstrate and inspect each compiler boundary, not to claim
that arbitrary PyTorch models can already run on Atlas. Unsupported shapes,
operations, or layouts must fail with a useful diagnostic.

## Intended compilation path

```text
PyTorch module + example inputs
  -> torch.compile / PyTorch Inductor capture and generated artifacts
  -> normalized workload graph with shape, layout, dtype, and fusion metadata
  -> Linalg/SCF/arith (the SE-level boundary is to be named precisely)
  -> workload analysis and Atlas tile/precision/ABI selection
  -> Atlas virtual SSA operations
  -> existing Atlas-MLIR verification, scheduling, allocation, and lowering
```

Keep the framework capture independent from Atlas mapping. Inductor is the
source of compiler evidence; a handwritten Triton kernel may remain as a
focused reference, but must not be presented as Inductor output. Retain
intermediate artifacts and explain when a stage is a capture, a translation,
or a target-specific mapping.

## Kernel fusion target

For the micro-Transformer (attention plus feed-forward MLP), target a
monolithic fused kernel that keeps intermediate tensors on-chip and packs as
much supported computation into one launch as practical. Treat this as a
measured optimization goal, not an assumption: one Dynamo/FX graph does not
imply one Inductor kernel, and FlashAttention may remain a separate fused
attention kernel. Record graph-region count, generated kernel count, launch
boundaries, and the reason for each required split. Preserve a correct
multi-kernel fallback when a whole-block fusion is not legal or profitable.

## Workload sequence

1. **Baseline MLP:** one linear + activation and the existing two-layer MLP.
   Establish genuine Inductor capture and bridge its emitted kernel/IR into the
   current TTIR analysis and Atlas mapper.
2. **Attention:** use a PyTorch SDPA workload that selects an Inductor
   FlashAttention-style path where the installed backend supports it. Record
   the selected kernel and fallback behavior; do not call naive materialized
   attention “FlashAttention.”
3. **Transformer block:** compose self-attention, residual connections,
   normalization, and the two-layer feed-forward MLP. Track fused and unfused
   graph regions, pursue the monolithic fusion target, and lower supported
   subgraphs in dependency order.
4. **SE/Linalg and Atlas coverage:** define the SE-level IR and its relationship
   to the existing Linalg prototype, then implement and validate explicit
   mappings for the operations reachable in the selected workloads.
5. **Integrated example:** provide one command that emits inspectable artifacts
   at each boundary and a summary of mapped, approximated, and unsupported
   operations.

## Completion criteria

- Each workload can be invoked from PyTorch with fixed documented shapes and
  produces captured Inductor artifacts in a supported environment.
- The capture path identifies the actual generated kernels and preserves
  provenance; no handwritten-kernel artifact is mislabeled as Inductor output.
- The Transformer report distinguishes captured graph regions from actual
  generated kernels and records whether the target monolithic fusion succeeded.
- A normalized graph records operation semantics and the shape/layout/dtype
  information needed by downstream lowering.
- Supported graph regions lower to the declared SE/Linalg representation and
  then Atlas virtual SSA. Atlas operations pass Atlas-MLIR virtual verification.
- Numerical policy, tile staging, precision changes, and unsupported graph
  regions are explicit in generated output or diagnostics.
- Documentation gives exact setup and run commands and distinguishes local
  capture from Atlas backend execution.

## Current state and first milestone

The repository currently has PyTorch FX capture, a limited Inductor Triton
probe, handwritten Triton-to-TTIR examples, TTIR-to-Linalg prototypes, and
bounded TTIR-to-Atlas mappings. `compiler.inductor_capture` now invokes the
real Inductor backend and saves its generated code when compilation succeeds,
alongside an exported typed graph and a status report; it does not synthesize
TTIR. The attention model now requests SDPA with
`need_weights=False`, and the capture report records generated-code evidence
for the selected attention implementation. HGraph now normalizes overloaded
FX targets, records schemas plus result dtypes/shapes, and classifies the
linear/MLP/SDPA/Transformer graph shape. The graph and generated code are
reported by one command, though the graph is independently produced by
`torch.export` and is not yet mapped to specific Inductor kernels. Neither is
yet lowered from this new path to Atlas. The next boundary is to recover
Inductor-produced Triton kernels (when
the selected backend emits them), compile those exact kernels to TTIR, and map
their supported regions through the existing analyzers.

The HGraph serialization and strict dense-pattern analyzer now feed a separate
HGraph-to-Atlas mapper for one- and two-layer biased MLPs. The mapping spells
out FP8 activation/weight inputs, BF16 bias inputs, intermediate packing, and
the existing virtual Atlas operations. It is explicitly sourced from
`torch.export` HGraph and does not yet consume or associate Inductor kernels.
The generated two-layer MLP fixture parses through Atlas-MLIR's virtual SSA
evaluator interface and contains 16 valid operations; the local Atlas compiler
binaries are not built, so its own `atlas-opt` verifier has not run.

The MLP capture was exercised with PyTorch 2.14.1 in an isolated temporary
environment. The host's system C++ compiler is unusable until its Xcode
license is accepted, so Inductor produced no code; the command saved a valid
HGraph and a failed-Inductor report. CUDA is unavailable on this host, so
FlashAttention kernel selection cannot be established here.

“SE-level” needs a precise expansion and artifact contract before that stage is
implemented. Until clarified, treat the existing Linalg/MLIR prototype as the
nearest current representation and do not silently equate Linalg with SE.
