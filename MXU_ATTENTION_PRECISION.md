# Atlas MXU precision notes for attention

## Short answer

Atlas MXU matmul operands are FP8 tiles in the current virtual interface.
Results can be read out as BF16 or FP8; the existing attention mapping chooses
BF16. “FP8 matmul” describes the input format; it does not mean the running
sum or output must be FP8. The two MXU units also have different observed
accumulation behavior, so one generic accumulator model is not safe.

## What this repository's mapping currently assumes

`compiler/atlas_mapping.py::map_attention` stages Q, K, and V as FP8. The first
MXU matmul produces BF16 scores. The VPU then applies the mask, score scaling,
softmax max/subtract/exp/sum/reciprocal operations in BF16. The probabilities
are packed to FP8 before the second MXU matmul with V, which produces a BF16
output. This is an explicit, lossy precision policy, not a claim that the
source PyTorch/TTIR FP32 or FP16 computation is bit-identical to Atlas.

The current convenience operation `atlas.virtual_mxu_matmul` represents a
reset FP8-by-FP8 contraction with BF16 result. The typed virtual interface also
has explicit FP8/BF16 accumulator seeding, continuation, and BF16/FP8 readout
operations. See the adjacent Atlas-MLIR checkout's
[`dialect-reference.md`](https://github.com/ucb-bar/atlas-mlir/blob/aea28c5f7a2e9c64e8684b59bc83da31c718b948/docs/dialect-reference.md)
and [`virtual_mxu_accumulation.mlir`](https://github.com/ucb-bar/atlas-mlir/blob/aea28c5f7a2e9c64e8684b59bc83da31c718b948/test/examples/virtual_mxu_accumulation.mlir).

The MXU's optional FP8 readout is a separate operation with an explicit scale.
Do not assume it is the same conversion as the VPU's BF16-to-FP8 pack; the
Atlas-MLIR docs say the general MXU FP8 readout scale policy is still open.

## What the selected-core evidence says about accumulation

The adjacent Atlas-MLIR checkout documents bounded runs against a selected
standalone AtlasCore at RTL revision
`0079c0541111197741a231c002e3843fa6f545b2`:

- **MXU0:** the systolic processing elements use an E4M3 FMA path. In the
  tested finite-normal witnesses, the result matched a BF16 round after each
  product-add step.
- **MXU1:** the inner-product unit uses an exponent-anchor tree. In the tested
  witnesses, it matched one final BF16 round after the reduction. Its integer
  alignment can discard small terms in other input ranges, so this must not be
  described as exact accumulation for every input.
- The units differ. For FP8 operands representing `1, 1/16, 1/16`, MXU0
  returned BF16 bits `0x3f80`; MXU1 returned `0x3f81`.

These are limited standalone-core observations, not full-domain or integrated
SoC qualification. The evidence and test limits are recorded in
[`mxu-arithmetic-discriminator-observation.md`](https://github.com/ucb-bar/atlas-mlir/blob/aea28c5f7a2e9c64e8684b59bc83da31c718b948/docs/mxu-arithmetic-discriminator-observation.md)
and [`mxu1-anchor-observation.md`](https://github.com/ucb-bar/atlas-mlir/blob/aea28c5f7a2e9c64e8684b59bc83da31c718b948/docs/mxu1-anchor-observation.md).

The inspected `npu_model` implementation instead converts both units' FP8
operands to FP16, computes a matrix product, then converts the result to BF16.
That model expression disagreed with the selected-core results on the cited
precision witnesses. It should not be treated as the hardware arithmetic
oracle for these cases.

## Implication for attention mapping

The existing attention mapper's FP8 Q/K/V inputs, BF16 score/output values,
and BF16-to-FP8 probability packing are plausible Atlas virtual operations.
Before claiming numerical equivalence, the mapper must specify which MXU unit
performs each contraction and validate the MXU0/MXU1 rounding behavior and the
FP8 probability packing against Atlas. The current mapper uses MXU0 for QK and
MXU1 for PV, but its numerical policy does not yet model those distinct
accumulation paths or prove PyTorch equivalence.
