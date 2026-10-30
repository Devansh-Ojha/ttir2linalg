# Reproducing the generalized Triton → Atlas compiler

Run commands from the repository root. This is a source checkout, not an
installed Python package. `PYTHONPATH` below supplies the compiler, validation
helpers, external Atlas Python dialect/evaluator, and simulator imports.

## Validated dependencies

The verified Linux environment uses Python 3.14.8, CUDA-enabled PyTorch 2.11.0,
Triton 3.6.0, xdsl 0.65.0, NumPy 2.4.4, and typing_extensions 4.16.0. The simulator requires Python >=3.14.
`requirements.txt` now starts at torch>=2.11: the former >=2.14 bound came from
the initial BWRC setup documentation, not a demonstrated API requirement.
The 2.11 environment passes the capture API probes and offline compiler suite.
This is not a claim that every later PyTorch/Triton combination is compatible.
`constraints/validated.txt` records the tested Python package versions; it is
not a full transitive or cross-platform lockfile.

For a new environment (do not reinstall an already validated environment):

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# First install CUDA-enabled torch==2.11.0 using your platform's wheel index.
# Select the matching command at https://pytorch.org/get-started/locally/.
python -m pip install -r requirements.txt -c constraints/validated.txt
python -m pip install triton==3.6.0 -c constraints/validated.txt
```

Do not substitute a CPU-only PyTorch wheel for the tested compiler toolchain.
Offline ASTSource compilation requires a working Triton CUDA toolchain but
launches no GPU kernel. Inductor AOT capture is a separate route and may still
query a CUDA driver/device. Numerical-policy simulator results do not establish
CUDA/IEEE numerical equivalence or physical hardware correctness.

Obtain the team's external repositories at these verified revisions:

| Checkout | Revision | Required content |
|---|---|---|
| Atlas-MLIR | `aea28c5f7a2e9c64e8684b59bc83da31c718b948` | `build/bin/atlas-opt`, `atlas-emit`, Python tools |
| npu-model-eval | `6eb3894032b4ac0c50a7fe4eef667015e9dfc73b` | `npu_model`, ISA/numerics, `configs/data/*.bin.gz` |

Use the team's remotes/access instructions; neither checkout is vendored here.
Build Atlas with the LLVM/MLIR version required by that revision's README
(see also [BWRC setup](../BWRC_SETUP.md)). Follow the simulator's own dependency
setup. These commands use its source tree directly; its full development/IREE
stack is not required by this repository's compiler-only path. The full tests
and machine validators do require the simulator and its numerical data.

## Paths and environment

The compiler and generalized validators default to an `atlas-mlir` checkout
beside this repository. Set `ATLAS_ROOT` to use any other location. The
exhaustive unary test similarly uses a sibling `npu-model-eval` or
`NPU_MODEL_ROOT`. Public compiler API `atlas_root` arguments and the preserved
TTIR CLI's `--atlas-root` override `ATLAS_ROOT` for that call.

```bash
export REPO_ROOT="$PWD"
export ATLAS_ROOT="$(cd ../atlas-mlir && pwd)"
export NPU_MODEL_ROOT="$(cd ../npu-model-eval && pwd)"
export ATLAS_OOT_BIN_DIR="$ATLAS_ROOT/build/bin"
# Replace those two locations if your checkouts are elsewhere.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/tests:$REPO_ROOT/tools/validation:$ATLAS_ROOT/tools:$NPU_MODEL_ROOT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c 'import torch, triton, xdsl, numpy, atlas_virtual_evaluator, npu_model; print(torch.__version__, triton.__version__, numpy.__version__)'
test -x "$ATLAS_ROOT/build/bin/atlas-opt"
test -x "$ATLAS_ROOT/build/bin/atlas-emit"
RUN_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/triton-atlas.XXXXXXXX")
export TRITON_CACHE_DIR="$RUN_ROOT/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/inductor-cache"
```

All compiler/validation output paths below must be new. Keep generated lowering
stages, logs and caches under `RUN_ROOT` or ignored `out/`, never in fixtures.
Fixture source snapshots, capture JSON and TTIR intentionally preserve original
compiler paths and hashes. Those paths are provenance, not runtime inputs.

## Fresh-checkout invocation

Compile preserved genuine TTIR without requiring any ignored capture directory:

```bash
python -m compiler.ttir_atlas_pipeline \
  tests/fixtures/triton_quantized_attention/kernel.ttir \
  --out-dir "$RUN_ROOT/preserved" --numerical-policy atlas-fp8-attention-v1 \
  --tile-binding 0:input:0 --tile-binding 1:input:2 --tile-binding 2:input:4 \
  --tile-binding 3:output:1 --tile-binding 4:output:3 \
  --tile-binding 5:output:5 --tile-binding 6:output:7 --lower
```

Or compile Python through genuine Triton and verify/lower/simulate it freshly:

```bash
python -m compiler.jit_capture --module kernels.atlas_jit_kernels \
  --kernel quantized_attention \
  --signature '{"Q":"*bf16","K":"*bf16","V":"*bf16","Out":"*bf16","Scores":"*bf16","Probability":"*bf16","PackedProbability":"*fp8e4nv"}' \
  --out-dir "$RUN_ROOT/attention-capture"
python tools/validation/validate_quantized_attention.py \
  "$RUN_ROOT/attention-capture" --numerical-policy atlas-fp8-attention-v1 \
  --out-dir "$RUN_ROOT/attention-validation"
python tools/validation/validate_jit_atlas.py \
  --example fp8-matmul --numerical-policy atlas-fp8-mlp-v1 \
  --out-dir "$RUN_ROOT/matmul"
python tools/validation/validate_jit_atlas.py \
  --example fp8-two-layer --numerical-policy atlas-fp8-mlp-v1 \
  --out-dir "$RUN_ROOT/mlp"
python -m compiler.jit_capture --module kernels.atlas_jit_kernels \
  --kernel tile_transpose --signature '{"X":"*fp8e4nv","Out":"*fp8e4nv"}' \
  --out-dir "$RUN_ROOT/transpose-capture"
python tools/validation/validate_transpose_operations.py \
  "$RUN_ROOT/transpose-capture" --numerical-policy atlas-fp8-layout-v1 \
  --out-dir "$RUN_ROOT/transpose-validation"
python -m compiler.jit_capture --module kernels.atlas_jit_kernels \
  --kernel bf16_row_softmax --signature '{"X":"*bf16","Out":"*bf16"}' \
  --out-dir "$RUN_ROOT/softmax-capture"
python tools/validation/validate_rowwise_operations.py \
  "$RUN_ROOT/softmax-capture" --numerical-policy atlas-bf16-rowwise-rtl-v1 \
  --check-softmax --out-dir "$RUN_ROOT/softmax-validation"
python -m unittest discover -s tests -p 'test_*.py' -v
```

Unittest discovery runs 199 tests. On a checkout without `out/`, one optional
comparison to a *previously saved* two-layer MLIR file is skipped; all committed
fixture-based compiler/machine integration tests still run. Do not describe a
newly regenerated file as an independent historical baseline. To rerun the
legacy two-layer machine validator from source instead of relying on ignored
artifacts, choose a checkout with no existing output at the paths below:

```bash
python -m compiler.ttir_to_atlas examples/two_layer_mlp_fp8.ttir \
  --pattern two-layer-mlp --numerical-policy atlas-fp8-mlp-v1 \
  --out out/atlas/two_layer_mlp_generated.virtual.mlir
python "$ATLAS_ROOT/tools/lower_atlas_program.py" \
  out/atlas/two_layer_mlp_generated.virtual.mlir \
  --output-dir out/atlas/two_layer_mlp_generated.lowered
python tools/validation/validate_two_layer_mlp_generated_machine.py
```

For three-layer generation/lowering/validation use the complete recipe in
[the bounded frontend README](../examples/ttir_to_atlas/README.md).

## Coverage and remaining setup limits

`unittest` does not discover the function-style tests in `test_linear.py`,
`test_mlp_pipeline.py`, or `test_inductor_triton_probe.py`. Install pytest separately if needed (`python -m pip install 'pytest>=8.4'`).
Then run `python -m pytest tests` (not repository-wide collection of example scripts).
The unchanged linear test has a known printer-string expectation failure.
`MLP_TTIR_DIR` is required to exercise the artifact-based MLP test;
`TTIR2LINALG_MLIR_PYTHON` must name a separate MLIR-enabled Python for
`python tests/test_numeric.py`. These are legacy paths, not generalized Atlas
compiler prerequisites. CUDA/CPU Triton example scripts require an appropriate
execution backend; offline compilation does not provide one.

Neither fresh package installation nor an external toolchain build is implied
by a passing run in an existing environment. Platform wheel availability,
external repository access and LLVM compatibility remain setup prerequisites.
The full simulator environment has not been reduced to a transitive lockfile.
