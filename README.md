# TTIR to Linalg Experiments

## Triton Linear Kernel

This repo currently tests a simple Linear layer implemented in Triton and compares
its output against the equivalent PyTorch `nn.Linear`.

The current test covers the first layer of `SimpleMLP`.

## Running on EDA

Tested on:

- `eda-4.eecs.berkeley.edu`
- RHEL 9 / x86-64
- PyTorch 2.8.0
- Triton 3.4.0

### Environment setup

```bash
ssh eda-4

python3 -m venv ~/ttir-env
source ~/ttir-env/bin/activate

python -m pip install --upgrade pip
python -m pip install torch triton numpy
```

### Verify

```bash
python -c "import torch, triton; print(torch.__version__); print(triton.__version__)"
```

### Run the Triton kernel

```bash
PYTHONPATH=. TRITON_INTERPRET=1 python kernels/mlp_kernel.py
```

### Generate TTIR

```bash
PYTHONPATH=. python dump_ttir.py
```

## Generated TTIR

```mlir
module {
  ...
}
```