# Architecture decisions

This file records the main choices in the repository's current compiler
prototype. It describes implemented behavior, not a claim that every path is
complete or production-ready.

## Current architecture

The repository has two front ends and multiple downstream experiments:

- The PyTorch/FX path exports a model with `torch.export`, converts its FX
  graph into the target-neutral `HGraph`, and writes a readable `.hgir`
  artifact.
- The Triton path obtains explicit `@triton.jit` kernels, compiles them to
  TTIR, parses/verifies the TTIR, and can convert supported operations into
  HGraph or the existing Linalg-style MLIR prototype.
- The Atlas path is a separate bounded TTIR-to-Atlas virtual-op conversion.
  It recognizes a single FC + ReLU kernel, a fixed FP8 two-layer MLP, one
  single-query attention kernel, and a 32×32 matmul + ReLU tile. Atlas-MLIR
  owns scheduling, allocation, machine lowering, and instruction emission.
- Target-independent TTIR patterns are represented as semantic plans; target
  backends implement `TritonTargetMapper` to emit their own IR.
- `examples/triton_cpu_mlp_export/` is an independent experiment for
  exporting LLVM IR from Triton CPU. It is not a stage in the Atlas path.

The FX-to-HGraph and TTIR-to-HGraph paths are useful for inspecting model and
kernel structure, but the current Atlas converter starts from TTIR. It does
not yet translate FX or HGraph directly into Atlas operations.

## Decisions

### Keep Triton analysis target neutral

The parsed Triton object model lives in `compiler/ttir_model.py`, and pattern
recognition lives in `compiler/triton_analysis.py`. Analysis returns semantic
plans for supported workload patterns without constructing Atlas operations.
This keeps those patterns reusable by other target mappers.

### Keep Atlas mapping in its own layer

`compiler/atlas_mapping.py` consumes the neutral plan and emits only existing
`atlas.virtual_*` operations. It does not implement virtual-op allocation,
instruction selection, scheduling, or machine code generation. Those remain
the responsibility of Atlas-MLIR and its lowering tools.

`compiler/target_mapping.py` defines the mapper boundary. Another dialect can
consume the same workload plans by implementing `map`; the TTIR reader and
analysis do not need target-specific branches. This is an in-process
interface, not a new intermediate dialect.

### Keep each workload contract deliberately narrow

Each admitted pattern is static and fails closed when its operation graph,
shape, or encoding differs from the checked-in sample. Atlas inputs are staged
in caller-provided padded tiles; the converter does not infer arbitrary
runtime pointer arithmetic or memory movement.

The mapper assumes the caller stages those exact operand tiles in the Atlas
runtime input slots. It does not interpret arbitrary TTIR pointer arithmetic,
program IDs, masks, tails, or host memory layouts.

### Make data layout and precision boundaries explicit

The source RHS is represented as logical K×N, while the Atlas MXU weight tile
uses N×K. The mapper inserts the existing `atlas.virtual_xlu_transpose` for
that layout change. Atlas MXU output and ReLU are BF16, while TTIR dot
accumulation commonly has a wider type. Therefore this mapping is an explicit
FP8-input/BF16-output target mapping; it does not promise bitwise equivalence
to unquantized FP32 computation.

### Reuse Atlas-MLIR for backend work

The converter stops at Atlas virtual SSA. The generated program is intended
to pass through `tools/lower_atlas_program.py`, which invokes Atlas-MLIR's
existing verification, scheduling, allocation, timing, and emission stages.
The project does not duplicate those backend responsibilities.

### Keep hardware-independent and backend-specific artifacts separate

HGraph and Triton analysis describe operations and types independently of
Atlas. Atlas tile addresses, slot indexes, MXU selection, Atlas virtual
operation names, and E4M3FN/BF16 behavior are isolated in the Atlas mapper.
This boundary is intended to let other targets reuse the Triton analysis
without inheriting Atlas ABI assumptions.

## Current limitations

- Input staging remains a caller responsibility, including the attention
  padding mask and FP32-to-FP8/BF16 conversions selected by each policy.
- The FC and attention mappings cover the repository's small static samples;
  arbitrary dimensions, masks, tails, multi-tile loops, and model launch
  orchestration are not generalized.
- Atlas verification, numerical evaluation, and machine lowering require a
  built Atlas-MLIR toolchain; the converter alone does not perform them.
- The FX graph path is not connected to Atlas conversion.
