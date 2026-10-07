# TTIR to Linalg

A compiler experiment for lowering Triton IR (TTIR) into a Linalg-based intermediate representation.

The project starts from Triton kernels, extracts TTIR using Triton's compiler infrastructure, parses the resulting MLIR, and walks the TTIR operations to build a Linalg-oriented representation. The long-term goal is to lower Triton GPU kernels into standard MLIR/Linalg operations that can be further transformed and lowered through the MLIR ecosystem.

## Current Status

The current pipeline successfully:

- Compiles a Triton kernel with Triton 3.4.0
- Extracts the generated TTIR from `CompiledKernel.asm["ttir"]`
- Parses TTIR using Triton's `libtriton` bindings
- Walks the parsed TTIR module and operations
- Maps the current set of TTIR operations to a Linalg-oriented representation
- Handles nested `tt.reduce` regions and `tt.reduce.return`
- Writes the resulting representation to `linalg.mlir`
- Includes a basic correctness test for the Triton linear kernel

`compiler/lower.py` and `compiler/linalg_lowering.py` remain the textual and
structured reference paths. `lower_real.py` now also exercises the first real
MLIR milestone through Triton 3.4.0's `libtriton.ir` bindings: it parses TTIR,
creates a real `func.func`, builds an SSA map keyed by Triton `Value.id()`,
prints the module, and verifies it. The real path materializes arithmetic,
pointer, memory, range, reshape, and reduction operations using Triton
3.4.0's operation builders; no operation is emitted as a placeholder string.

Example:

```text
%c4_i32 = linalg.constant

%0 = linalg.program_id

%1 = linalg.index_range

%2 = linalg.broadcast

%3 = linalg.pointer_add

%4 = linalg.load

...

%14 = linalg.reduce

%17 = linalg.add

    linalg.reduce_yield

%15 = linalg.add

%16 = linalg.pointer_add

linalg.store

linalg.return
```

## Project Structure

```text
ttir2linalg/
├── compiler/
│   ├── lower.py
│   └── mlir_builder.py
├── kernels/
│   └── mlp_kernel.py
├── tests/
│   └── test_linear.py
├── dump_ttir.py
├── linalg.mlir
└── README.md
```

## Triton Kernel

The initial kernel is a small linear layer implemented in Triton. It computes the dot product between an input vector and a row of weights, adds a bias, and writes the result.

The kernel is used as a small but representative test case for exercising:

- program IDs
- pointer arithmetic
- tensor construction
- loads and stores
- elementwise arithmetic
- reshapes
- reductions
- nested TTIR regions

The reference implementation is compared against the equivalent PyTorch `nn.Linear` computation.

## Environment

Development and testing has been done on:

- Berkeley EECS EDA machines
- RHEL 9
- x86-64
- Python 3.9
- PyTorch 2.8.0
- Triton 3.4.0

## Setup

```bash
ssh eda-*

python3 -m venv ~/ttir-env
source ~/ttir-env/bin/activate

python -m pip install --upgrade pip
python -m pip install torch triton numpy
```

Verify the environment:

```bash
python -c "import torch, triton; print(torch.__version__); print(triton.__version__)"
```

## Running the Kernel

Run the Triton kernel in interpreter mode:

```bash
PYTHONPATH=. TRITON_INTERPRET=1 python kernels/mlp_kernel.py
```

## Generating TTIR

Generate the TTIR for the kernel:

```bash
PYTHONPATH=. python dump_ttir.py
```

The Triton compiler also exposes TTIR directly through:

```python
result.asm["ttir"]
```

The lowering pipeline uses this representation rather than reparsing a dumped file.

## TTIR → Linalg

Run the current lowering pass with:

```bash
PYTHONPATH=. python compiler/lower.py
```

This prints the lowered operations and writes:

```text
linalg.mlir
```

The current output is an intermediate textual representation. It is intentionally separate from actual MLIR construction at this stage.

## Tests

The current test can be run directly with:

```bash
PYTHONPATH=. python tests/test_linear.py
```

Expected output:

```text
PASS
```

`pytest` is not currently required by the test suite.

## Roadmap

### 1. Preserve TTIR semantics

Replace the current operation-name-only lowering with structured operations that preserve:

- operands
- results
- types
- attributes
- tensor shapes
- reduction axes
- nested regions

For example, move from:

```text
%12 = linalg.mul
```

toward a representation containing the actual operands and type information.

### 2. Build real MLIR

Replace the textual Linalg representation with actual MLIR operations. The
first plumbing step is available via:

```bash
PYTHONPATH=. python lower_real.py kernel.ttir --limit 1
```

This uses Triton's bundled MLIR bindings rather than assuming a standalone
`mlir` Python package. The generated module is currently Triton-dialect MLIR;
the existing textual Linalg lowering remains available as a semantic reference
until equivalent Linalg dialect construction is exposed by the target runtime.
The current real-IR milestone lowers `arith.muli`, `arith.mulf`, `arith.addf`,
`tt.make_range`, and `tt.reshape` through typed builder APIs while preserving
the Triton-value-to-MLIR-value SSA map.

The generated module should be parseable and verifiable by MLIR rather than simply being a printed list of operations.

### 3. Lower Core TTIR Operations

Implement structured lowering for the core operations used by Triton kernels:

- `tt.make_range`
- `tt.splat`
- `tt.addptr`
- `tt.load`
- `tt.store`
- `arith.*`
- `tt.reshape`
- `tt.reduce`
- `tt.get_program_id`

### 4. Handle Tensor Semantics

Map Triton's tensor semantics onto appropriate MLIR/Linalg representations, including:

- shaped tensors
- elementwise operations
- broadcasting
- indexing
- reductions
- memory accesses

### 5. Verification

Add structural and numerical tests that compare:

```text
Triton kernel
      vs.
Lowered representation
      vs.
PyTorch reference
```

The goal is to catch semantic mismatches rather than only checking that lowering succeeds.

### 6. Expand Kernel Coverage

Move beyond the initial linear kernel and add progressively more representative kernels:

- elementwise kernels
- reductions
- matrix multiplication
- softmax
- MLP layers
- fused operations

### 7. MLIR Lowering Pipeline

Once valid Linalg IR is produced, connect the output to standard MLIR transformations and lower it toward lower-level dialects.

The eventual pipeline is:

```text
Triton Python
     ↓
TTIR
     ↓
TTIR → Linalg
     ↓
MLIR / Linalg
     ↓
MLIR transformations
     ↓
Lower-level IR
```

## Design Goal
