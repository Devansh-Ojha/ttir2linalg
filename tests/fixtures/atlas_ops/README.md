These seven TTIR files are handwritten semantic composition tests for the
operation-driven Atlas emitter. They are not TorchInductor/GPU captures.

Requested graphs: add, add_relu, mul_add, mul_add_relu, sub_add.
Additional probes: constants (scalar BF16 -> tile splat used by arithmetic)
and direct_relu (raw sign/NaN behavior without arithmetic canonicalization).
Every file uses the same operation analyzer, eligibility pass and emitter.

The numerical contract is explicitly atlas-bf16-elementwise-rtl-v1. It does not
claim IEEE maxnum/add/sub equivalence. Each runtime input is a static logical
row-major 32x32 BF16 tile, staged into Atlas column-half byte order. Test-runner
bindings use non-consecutive input slots and a nonzero output slot to catch
implicit argument-order/address assumptions.
