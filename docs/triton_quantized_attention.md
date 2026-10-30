# Genuine quantized Triton attention on Atlas

## Numerical contract: atlas-fp8-attention-v1

This is a new workload and numerical policy. The original `bf16_simple_attention`
function and its genuine compiler TTIR are preserved. CUDA numerical equivalence
and original BF16 attention equivalence are not claimed.

The workload uses a single 32x32 program, d=32, no causal mask, no online softmax
and no FlashAttention. Each Q/K/V input is a logical row-major BF16 tile. Four
separate output tiles hold final BF16 output, scaled BF16 scores, BF16
probabilities and raw FP8 packed probabilities. All buffers are disjoint and
explicitly bound. BF16 staging uses column halves; FP8 staging is row-major.
Each slot reserves 2048 bytes; actual transfers are 2048/1024 bytes respectively.

The source explicitly specifies the following computation:

1. Convert Q, K and V from BF16 to E4M3FN, RTNE, unit scale (Atlas scale code
   127). Hardware packing rounds normals to nearest-even, saturates magnitudes
   above the finite limit to 448, flushes subnormal outputs and BF16 zero/
   subnormal inputs to positive zero, maps NaNs to positive zero and infinities
   to signed 448. This target policy differs from general CUDA FP8 conversion.
2. Transpose packed K with a typed [1,0] permutation on the actual byte XLU.
3. First dot uses MXU0, a zero accumulator, source IEEE/default input precision
   and zero imprecision allowance. Atlas's ordered custom MAC rounds each step
   to BF16; the source has an explicit BF16 readout. No FP32 matrix result is
   claimed to be available between operations.
4. Multiply that BF16 readout by **0.1767578125**, raw BF16 **0x3e35**. This is
   the BF16 approximation of 1/sqrt(32), at an explicitly chosen source boundary.
5. Apply the validated BF16 rowwise policy: total ordered-bit row maximum;
   BF16 subtraction using FP32 round then upper-16-bit truncation; hardware
   natural exponential approximation; adjacent-pair FP32 RNE sum tree with one
   BF16 RNE readout; hardware reciprocal LUT followed by BF16 RNE multiplication.
   FP32 extensions/reductions/exp/div in TTIR are admitted only at qualified
   BF16 readout boundaries. They are not arbitrary exposed FP32 computations.
6. Multiply BF16 probabilities by **32** in BF16, then explicitly RTNE-pack
   to E4M3FN at unit scale. Thus stored probability codes represent 32P.
   The nominal smallest normal effective probability is 1/2048 instead of
   1/64; values can round up across that boundary. Smaller rounded subnormals
   still flush. A valid probability at most one cannot saturate this pack.
7. Multiply packed 32P by packed V on MXU1 with a zero accumulator. MXU1 uses
   its actual 32-bit anchor accumulation and BF16 readout policy, distinct from
   MXU0's ordered MAC arithmetic.
8. Multiply that BF16 result by **1/32** (raw BF16 **0x3d00**) in BF16 and store
   the final result. Q/K/V have unit scales, so no other compensation is needed.

Every source scale multiplication, conversion and BF16 readout remains an SSA
operation. Zero accumulators, dot precision attributes, transpose orientation,
source memory roles and store ordering are checked. The policy authorizes the
primitive arithmetic, not a function name, graph topology or fixed layer count:
other supported operation compositions may use the same policy.

Accuracy is a measured property of supplied inputs. Inputs beyond E4M3 range
clip; very small values flush. Extreme score spans can expose the exponential
lane's fixed-width behavior. Nonfinite source inputs have explicit packing
semantics but no ordinary mathematical-attention error metric. No universal
mathematical accuracy bound is asserted.

## Hardware evidence and reused implementations

The actual RTL was inspected read-only: `common/MxuParams.scala` fixes E4M3
activation/weight inputs and BF16 accumulation/readout; MXU0's default custom
FMA and MXU1's anchor tree agree with the simulator ISA. `xlu/XLU.scala` performs
32x32 byte transposes. The VPU exposes BF16 multiply, max/sum reduction,
exponential, reciprocal and FP8 packing. Its internal FP32 reduction tree does
not expose a general FP32 tensor pipeline.

The inspected pack converter (`FP8Pack` behavior, mirrored in the simulator's
`rtl_math.pack_row`) divides by its E8M0 scale. This workload keeps code 127 and
implements probability scaling/compensation explicitly with already-supported
BF16 multiply. No new variable-scale converter, unpack instruction, arithmetic
mode or invented hardware instruction is required.

The existing `atlas.virtual_pack_fp8` lowering includes the verified VMEM
reordering helper needed to turn the VPU's register-order pack into logical
row-major FP8. Reusing only the raw pack instruction without that helper would
be a layout error. The existing emitter also emits the MXU RHS weight transpose
in addition to the source K transpose; both remain explicit in this version.

Compiler emission reuses existing virtual input/output, XLU, MXU, BF16 binary,
row-reduction, unary and FP8 pack operations. Mixed FP8 output uses the existing
virtual DMA store. Scheduling, allocation, waits and timing come from the
actual external Atlas backend, which was not modified.

## Generalized compiler composition

`MixedTileABI` declares one BF16/FP8 format per argument, in addition to the
existing slot, role, alignment and nonaliasing contract. The generic JIT API
records those formats from the supplied source signature in its normalization
proof. Eligibility checks each actual argument against the declared format.

The rowwise planner now composes qualified FP8 pack, transpose and dot operations
under the new policy, reusing its existing row-layout and precision tracking.
FP32 zero dot accumulators are metadata; nonzero accumulators are rejected.
Dots alternate MXU0/1 by operation ordinal and require only explicit BF16
readout users. Exact BF16 promotions, row views, reduction combiners and
readout-only islands keep their existing checks. BF16 dots, unsupported casts,
extra FP32 consumers, masked accesses and arbitrary data layouts remain rejected.

The implementation dispatches on operation names, source types, dependencies,
layout and precision. It contains no attention-specific recognizer. All old
policies keep their original capabilities and numerical rules. The original
BF16 attention still fails its required hardware/precision checks.

## Validation method

The handwritten `quantized_attention` source compiles with actual Triton 3.6.0,
ASTSource, offline SM89, four warps, one CTA, fusion disabled and grid (1,1,1).
No GPU is launched and no TTIR is synthesized or edited. Captures retain the
full source snapshot, original TTIR, compile settings and SHA256 provenance.

The independent mixed-operation reference evaluates the normalized source SSA,
using the already independent integer/Fraction FP8 dots/pack, exact BF16
arithmetic, Decimal exponential interpolation, reciprocal LUT construction and
FP32 reduction tree. It imports no NPU numerical implementation and interprets
neither emitted MLIR nor machine code. Qualified FP32 source islands are
represented by their policy BF16 readout values. Saved traces label source
operations/types and are reference traces, not additional simulator taps.

Actual Atlas verification and all machine lowering stages run before simulation.
Each instruction re-encodes exactly to the emitted JSON and hex words. The
simulator runs the unmodified schedule with RTL numerics and timing errors
enabled. Every case compares all four physical output buffers bit-for-bit,
including raw FP8 probability bytes. Inputs and surrounding guards must remain
unchanged, every output byte must be overwritten, and completion requires ECALL,
CSR[0xc10]=1 and no runtime errors. Deliberate slot gaps exercise actual memory
bindings and buffer liveness.

Mathematical error is measured separately against float64 stable attention on
the original BF16 input values and, additionally, on decoded packed Q/K/V. This
separates the quantized input effect from other target approximations. Those
metrics do not replace the bitwise hardware-policy comparison.

## Machine and numerical results

**The genuine handwritten quantized attention kernel executes correctly in the
NPU simulator under the declared policy.** It is one fused computation, one
compiler/kernel invocation per case. The backend's pack relayout helper loops
are internal instructions, not additional kernel launches.

| Validation stage | Result |
| --- | --- |
| Real Triton TTIR capture | Passed, Triton 3.6.0 / SM89 with source and TTIR hashes |
| Parsing, SSA/region analysis and static address proof | Passed |
| Mixed-format operation eligibility | Passed |
| Generated Atlas virtual MLIR and actual verifier | Passed |
| Scheduling, allocation, waits, timing and machine generation | Passed |
| Machine re-encoding against JSON/hex | 328/328 words match |
| Simulator / independent source reference | **17/17 cases**, zero bit mismatches across all four outputs |
| Completion, input/guard preservation, complete output overwrite | Passed for every case |
| Cycles | **17,158 per case** |

The 17 cases comprise the full 16-case report plus one additional independently
checked constant-clipping case. Both use identical virtual MLIR and machine
words. The aggregation is saved as `out/atlas_quantized_attention/summary.json`;
individual reports remain preserved separately.

Numerical cases include all-zero and uniform known results, asymmetric values,
three bounded random seeds, permutation-sensitive inputs, dominant diagonal and
tied maxima, probability underflow tails, input packing ties, subnormal inputs,
V clipping, large score spans and raw/nonfinite BF16 inputs. Known-result checks
require uniform probabilities 1/32 (BF16 0x3d00), packed 32P=1 (FP8 0x38),
constant V=2 yielding output 2, and constant V=512 yielding clipped output 448.

Mathematical error is separate from bitwise policy agreement:

| Input group | Cases | Largest absolute output error vs float64 mathematical attention |
| --- | ---: | ---: |
| Bounded/core: zeros, uniform, bounded random, permutation, dominant/tied cases | 9 | **0.018713784948946086** |
| Packing ties, underflow/subnormals and probability tails | 3 | **0.09074423606231896** |
| Saturating V cases | 2 | **64.0** |
| Large finite score-span probe | 1 | 0.0 for this particular probe |
| Inputs containing NaN/Inf (including raw-random input) | 2 | Undefined mathematical baseline; policy bit comparison still passed |

Across the three bounded random cases, RMS errors were approximately 0.00413 to
0.00438. The largest error against the packed-Q/K/V mathematical baseline among
bounded/core cases was 0.012154917096686771. These are measured test results,
not universal bounds or an assertion that all approximation errors decompose
additively. The clipping case's error of 64 is intentional: original BF16 V=512
versus a packed value/output of 448. It must not be hidden behind the smaller
bounded-input result. All 15 finite-baseline cases together have maximum error
64.0. Stress cases do not establish reliable accuracy for arbitrary score spans.

### Independent reference correction, preserved failures

The initial full attention run passed 15/16; the subnormal-boundary case exposed
an error in the earlier independent FP8 pack reference. The reference had used
IEEE FP8 subnormal rounding before flushing, incorrectly promoting BF16
0x3c70..0x3c77 and the corresponding negative encodings to minimum-normal FP8.
Atlas actually rounds the normalized significand first, then flushes target
subnormals. The corrected reference uses a wide exponent for significand
rounding before that flush. BF16 0x3c78, for example, does carry up to FP8 0x08;
0x3c70..0x3c77 flush to positive zero.

The corrected independent reference matches the model packer for **all 65,536
BF16 input encodings**, with zero mismatches. The comparison imports the model
only in the test, never in reference evaluation. The old reference failure and
16 mismatching encodings are retained in `pack_exhaustive.json`; the corrected
report is `pack_exhaustive_v2.json`.

The genuine Triton source, original TTIR, numerical policy, emitted virtual
program and machine words were unchanged by this correction. Byte-for-byte
comparison confirms the failed and passing full-run virtual MLIR and hex are
identical. Failed and successful reports are kept in separate directories.
The existing numerical policies were not changed; this fixes an independently
implemented reference that previously missed those boundary cases.

## Resources, buffers and performance limits

The validated program contains four pack conversions and their four internal
relayout loops, three XLU transposes, two MXU pushes/matmuls/readouts, two row
reductions, exp/reciprocal and BF16 arithmetic. The source K transpose and
backend weight-orientation transpose are intentionally both retained. Removing
redundant movement would require a separately verified optimization.

The actual allocation report fits, with no failures: peak six BF16 values
(capacity 15), four FP8 values (capacity 32), two scalar values (capacity nine),
and one weight/accumulator slot on each MXU. VMEM scratch for the pack and
relayout helpers is explicitly allocated. Diagnostic outputs add DMA traffic
and constrain lifetimes. There are 13 actual 1024-byte DMA transfers, totaling
13,312 bytes, per case. This implementation prioritizes correctness; packing,
relayout and DMA dominate the small computation. No optimized-attention or
FlashAttention performance claim is made.

Argument bindings used for validation are explicit and deliberately spaced:

| Argument | Role / slot | Format |
| --- | --- | --- |
| Q | input 0 | BF16 |
| K | input 2 | BF16 |
| V | input 4 | BF16 |
| Out | output 1 | BF16 |
| Scores | output 3 | BF16 |
| Probability | output 5 | BF16 |
| PackedProbability | output 7 | E4M3FN |

The default input/output DRAM bases are 0x90000000 and 0x90020000. The program
has one start/state chain; stores preserve their original source order and all
buffers are disjoint. Internal temporaries remain compiler-managed registers/
VMEM scratch. Materialized diagnostics are compared to reference values in
actual DRAM; other saved source traces are independent reference observations.

## Regression results

- **196 unit tests passed**, comprising all previous 178 and 18 new tests.
- Preserved genuine FP8 matmul machine/reference regression: **22/22**, 39 words.
- Preserved genuine two-layer FP8 MLP machine/reference regression: **22/22**,
  103 words. Both were rerun with the corrected independent pack reference.
- Legacy two-layer machine: **21/21**, 101 words, 5883 cycles/case.
- Legacy three-layer machine: **26/26**, 163 words, 8497 cycles/case.
- Existing BF16 elementwise, reduction, softmax and transpose machine integrations
  remain in the full unit suite. New tests also compile and simulate an existing
  transpose graph and a row-sum graph under the composed policy, demonstrating
  that eligibility does not require an attention-shaped graph.

Additional function-style tests were rerun in the established environment:
**seven passed, one known failure, two unavailable**, matching the prior audit.
The known `test_linear.test_linear_lowering` assertion expects the old text
`linalg.constant()` while the unchanged legacy printer emits `linalg.constant`.
`compiler/lower.py`, that test, and `kernels/mlp_kernel.py` are unchanged relative
to HEAD. The unavailable tests require `MLP_TTIR_DIR` and a separate
`TTIR2LINALG_MLIR_PYTHON` interpreter. They are not counted as passes. No package
was installed, no existing test was weakened, and no new regression failure
was observed. `git diff --check` passed.

## Limitations and next capabilities

This validates the new quantized workload only. The original BF16 attention's
BF16 contractions and FP32 score-scaling requirements remain unsupported.
The unit-scale input pack clips/flushes; probability packing also loses small
mass. These precision changes are explicit in the new source and policy.

The frontend remains limited to single-program, disjoint, static full 32x32
tiles. It does not support masked/partial tiles, arbitrary gathers, dynamic
shapes, general data layouts, BF16 MXU multiplicands, nonzero generalized dot
accumulators, arbitrary FP32 tensor chains or unqualified conversions. Scale
code 127 is retained for each pack; the source's 32 and 1/32 factors are ordinary
BF16 operations, not newly supported dynamic converter scales.

Useful independently testable next steps are per-tensor configurable scale
contracts with source-visible compensation, wider accuracy characterization,
removing diagnostic outputs through a separate source specialization/ABI,
and verified elimination of redundant transposes. Larger attention requires
separate tiling/memory work. None of these should be inferred to have passed
from the current static test. This work implements neither online softmax nor
FlashAttention and does not overlap that algorithm investigation.

## Changed files and preserved artifacts

Added:

- `tools/validation/atlas_mixed_reference.py`: independent mixed SSA evaluation.
- `tools/validation/validate_quantized_attention.py`: genuine-capture validation,
  mixed-output simulation, reference traces and separate mathematical metrics.
- `tests/test_atlas_quantized_attention.py`: provenance, policy/format/precision
  rejection, source-scale controls, exhaustive packing, generic compositions,
  original-kernel preservation and actual fused execution tests.
- `tests/fixtures/triton_quantized_attention/`: untouched compiler TTIR, source
  snapshot, capture metadata and README.
- This document.

Modified additively:

- `compiler/atlas_eligibility.py`: per-argument `MixedTileABI` and new-policy
  dispatch.
- `compiler/atlas_rowwise_operations.py`: opt-in composition of row operations
  with FP8 pack/transpose/dot and mixed-format memory, retaining old-policy gates.
- `compiler/triton_tile_normalization.py`: allow the new policy's existing
  reduction/division/address proof path.
- `compiler/jit_to_atlas.py`: expose the policy and mixed ABI through the generic
  source-driven API/CLI.
- `kernels/atlas_jit_kernels.py`: append the separately named quantized kernel;
  original BF16 attention remains unchanged.
- `tools/validation/atlas_fp8_reference.py`: correct independent normal-rounding/
  subnormal-flush order, with exhaustive validation.
- `tools/validation/validate_transpose_operations.py`: shared simulator harness
  stages inputs by their actual declared FP8/BF16 format.
- `docs/triton_attention_transpose_blockers.md`: link to this new-workload result.

The existing operation emitter required no attention-specific rewrite or new
instruction. All unrelated local changes and earlier artifacts remain intact.
Nothing was staged, committed, pushed, reset, discarded or switched to another
branch.

All artifacts are local under Git-ignored `out/atlas_quantized_attention/`:

- `initial/capture/`: genuine source, TTIR, metadata and compiler logs.
- `initial/`: first generic compile/verify/lower with compact slot bindings.
- `full-validation/`: preserved 15/16 failed-reference run.
- `full-validation-v2/`: corrected full 16/16 run with the spaced validation ABI.
- `saturation-validation/`: the additional known constant-clipping case.
- `summary.json`: aggregate 17-case results and hardware evidence hashes.
- Validation directories retain virtual/verified MLIR, scheduled/allocated/timed
  stages, physical JSON/hex, source reference traces and per-case results.
- `regression-fp8-matmul/`, `regression-fp8-mlp/`, exhaustive-pack reports,
  function-regression report and archived logs retain the remaining evidence.

## Reproduction

Use the existing `~/atlas-eval-venv/bin/python`; set
`PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`PYTHONPATH=.:tests:tools/validation:$HOME/atlas-mlir/tools:$HOME/npu-model-eval`.
Choose fresh output paths; existing paths are refused.

```
python -m compiler.jit_capture --module kernels.atlas_jit_kernels \
  --kernel quantized_attention \
  --signature '{"Q":"*bf16","K":"*bf16","V":"*bf16","Out":"*bf16","Scores":"*bf16","Probability":"*bf16","PackedProbability":"*fp8e4nv"}' \
  --out-dir /tmp/my-quantized-attention-capture
python tools/validation/validate_quantized_attention.py \
  /tmp/my-quantized-attention-capture \
  --numerical-policy atlas-fp8-attention-v1 \
  --out-dir /tmp/my-quantized-attention-validation
python -m unittest discover -s tests -p 'test_*.py'
```

A fresh current validator run includes all 17 cases. The preserved aggregate
report combines the earlier 16-case full run and the separately added clipping
case to avoid rewriting or repeating working results unnecessarily.

The generic `python -m compiler.jit_to_atlas` entry point also accepts this
module/kernel/signature with `--numerical-policy atlas-fp8-attention-v1`,
`--tile-binding 0:input:0 --tile-binding 1:input:2 --tile-binding 2:input:4`,
`--tile-binding 3:output:1 --tile-binding 4:output:3`,
`--tile-binding 5:output:5 --tile-binding 6:output:7`, `--lower`, and a fresh
`--out-dir`. It is not an attention-only entry point.
