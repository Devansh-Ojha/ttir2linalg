# Current compiler flow

This document follows the checked-in files from source model or kernel to
their current outputs and explains how the new Atlas slice fits alongside
the existing paths.

## Repository areas

| Path | Role |
| --- | --- |
| `model/` | Small PyTorch example models used by the pipeline. |
| `kernels/` | Explicit Triton kernels used for source capture and TTIR compilation. |
| `mlp_pipeline.py` | Top-level driver for model export, Triton source/TTIR capture, and HGraph output. |
| `compiler/ttir_model.py` | Target-neutral data classes used by the parsed TTIR representation. |
| `compiler/ttir_reader.py` | Uses Triton's `libtriton` to parse/verify TTIR and expose functions, blocks, operations, values, types, and attributes. |
| `compiler/hgir.py` | Small hardware-agnostic graph representation and verifier. |
| `compiler/hgir_from_fx.py` | Converts an exported PyTorch FX graph into HGraph. |
| `compiler/hgir_from_ttir.py` | Converts parsed TTIR into HGraph for inspection and downstream experiments. |
| `compiler/linalg_lowering.py`, `compiler/real_mlir.py` | Existing TTIR-to-Linalg/MLIR prototype path. |
| `compiler/triton_analysis.py` | Recognizes bounded linear + ReLU, two-layer MLP, attention, and matmul + ReLU plans without Atlas-specific emission. |
| `compiler/target_mapping.py` | Defines the target-neutral workload mapper interface. |
| `compiler/atlas_mapping.py` | Maps those plans to existing Atlas virtual SSA and applies Atlas tile ABI and precision policies. |
| `compiler/ttir_to_atlas.py` | CLI joining TTIR parsing, pattern analysis, and selected target mapping. |
| `examples/ttir_to_atlas/` | Atlas conversion scope and handoff commands. |
| `examples/triton_cpu_mlp_export/` | Separate Triton CPU LLVM IR export experiment. |
| `tests/` | Unit and pipeline checks, including the focused analyzer/mapper tests. |

## Model and FX inspection path

```text
model/*.py
    │
    ▼
mlp_pipeline.py
    │ torch.export / optional torch.compile graph capture
    ▼
compiler/hgir_from_fx.py ──► compiler/hgir.py ──► .hgir
```

This path describes the exported PyTorch graph in the project's neutral HGraph
format. It does not preserve or create an Atlas virtual kernel today.

## Triton kernel to TTIR path

```text
kernels/*.py (@triton.jit)
    │
    ├── source capture ──► mlp_pipeline.py --compile-triton-source
    │                       └──► *-triton/*.py
    │
    └── Triton compiler ──► compiler/triton_capture.py
                             └──► *-ttir/*.ttir
                                  │
                                  ├── compiler/ttir_reader.py parses and verifies
                                  ├── compiler/hgir_from_ttir.py ──► .hgir
                                  └── compiler/linalg_lowering.py / real_mlir.py
                                      ──► Linalg/MLIR prototype output
```

The checked-in single-layer MLP and attention TTIR use small FP32 reductions;
their Atlas mappings explicitly stage padded FP8/BF16 tiles under target
numerical policies. The two-layer FP8 MLP sample has its own strict analyzer.

## TTIR to Atlas virtual SSA path

```text
matching *.ttir
    │
    ▼
compiler/ttir_to_atlas.py
    │
    ├── compiler/ttir_reader.py
    │     Triton's libtriton parser creates TFunc / TOp / TValue / TType
    │     instances defined in compiler/ttir_model.py
    │
    ├── compiler/triton_analysis.py
    │     recognizes linear-ReLU, two-layer MLP, attention, or matmul-ReLU
    │     and returns a target-neutral workload plan
    │
    └── compiler/target_mapping.py → AtlasTargetMapper
          selects the Atlas virtual-op mapping and explicit precision policy
          emits existing Atlas operations; no allocation or instruction
          emission is implemented in this repository
                    │
                    ▼
             *.virtual.mlir
                    │
                    ▼
Atlas-MLIR/tools/lower_atlas_program.py
    │
    ├── Atlas virtual verification and scheduling
    ├── allocation and machine lowering
    ├── timing pass
    └── atlas-emit program artifacts
```

The converter owns the semantic recognition and Atlas virtual operation
construction. Atlas-MLIR owns allocation and backend instruction emission. The
current converter assigns runtime input slot indexes and emits aligned DRAM
bases; it does not copy a Triton tensor from its original address into those
slots.

## Atlas conversion commands

From this repository, with Triton and its `libtriton` Python bindings
available, select a workload pattern and explicit precision policy:

```bash
PYTHONPATH=. python -m compiler.ttir_to_atlas \
  mlp-ttir-test/kernel_0_fc1_relu.ttir --pattern linear-relu \
  --numerical-policy atlas-fp8-linear-v1 \
  --out out/atlas/mlp.virtual.mlir

PYTHONPATH=. python -m compiler.ttir_to_atlas \
  attention-ttir/kernel_0_attention.ttir --pattern attention \
  --numerical-policy atlas-fp8-attention-v1 \
  --out out/atlas/attention.virtual.mlir
```

Then, from an Atlas-MLIR checkout with its tools built:

```bash
python /path/to/atlas-mlir/tools/lower_atlas_program.py \
  out/atlas/matmul_relu.virtual.mlir \
  --atlas-bin-dir /path/to/atlas-mlir/build/bin \
  --output-dir out/atlas/matmul_relu.lowered
```

The converter's accepted input pattern and numerical precision caveat are
described in [examples/ttir_to_atlas/README.md](examples/ttir_to_atlas/README.md).

## Separate Triton CPU export experiment

`examples/triton_cpu_mlp_export/` is independent of the Atlas path:

```text
examples/triton_cpu_mlp_export/mlp_kernel.py
    └── @triton.jit kernel
         └── export_llvm_ir.py + Triton CPU backend ──► LLVM IR text
```

It does not use Atlas virtual operations, `compiler/atlas_mapping.py`, or the
Atlas-MLIR lowering pipeline.
