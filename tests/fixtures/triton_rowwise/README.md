These are untouched outputs of Triton 3.6.0's actual offline SM89 compiler.
Each .capture.json retains source/TTIR hashes, compile options, signature and
constexprs. No GPU was launched. `softmax.source.py` is the source snapshot for
softmax; `probes.source.py` is the snapshot for row_max, row_sum and attention.
Tests verify all hashes. Earlier BF16/FP8 capture fixtures remain unchanged.

Softmax, row_max and row_sum pass operation eligibility, Atlas virtual verifier,
machine lowering and numerical simulation. Attention compiles to genuine TTIR
but is unsupported by the generalized frontend; it is preserved as a negative
feasibility fixture, not a successful Atlas execution.

Synthetic source variations in negative unit tests are explicitly tests of
rejection and are never presented as genuine compiler captures.
