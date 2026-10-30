# FlashAttention GPU results — 2026-10-29

Both attention workloads executed successfully through TorchInductor and matched
their independent PyTorch math-SDPA reference (`rtol=1e-3`, `atol=1e-3`).
Input: batch 1, sequence length 32, embedding size 32, one head; FP16 parameters
and inputs, with gradient tracking disabled.

Environment: NVIDIA GeForce RTX 5060 Laptop GPU (compute capability 12.0),
Python 3.12.3, PyTorch 2.15.0+cu132, CUDA 13.2, Triton 3.9.0, xdsl 0.65.0.
See [environment.json](environment.json) and the
[installed package versions](requirements.freeze.txt).
The source base was `a5fdcd2eefc7c05a6cd288a792b7dd641886d8bc`, with the fixes
committed on this results branch.

## Results

| Workload | Numerical check | Triton launch sites | TTIR dump files | Atlas kernels mapped |
| --- | --- | ---: | ---: | ---: |
| [attention_flash](attention_flash/capture.json) | Passed | 3 | 6 | 0 |
| [attention_custom_flash](attention_custom_flash/capture.json) | Passed | 1 | 1 | 0 |

The built-in workload uses native `aten._scaled_dot_product_flash_attention`;
its three Triton kernels handle Q/K/V layout operations. Its six TTIR files
contain three distinct function symbols, with two compiler dumps per symbol.
The custom workload emits the repository's tiled online-softmax
[`flash_attention_kernel` TTIR](attention_custom_flash/inductor_ttir/kernel_000_flash_attention_kernel.ttir).
Learned projections remain separate external matrix-multiply calls in both
wrappers; these launch counts do not mean the whole model became one kernel.

Atlas mapping ran but supports none of these emitted kernels. The custom
attention matcher currently accepts only one 8-feature query and 4×8 key/value
tiles, and does not lower this online-softmax kernel. Full diagnostics are in
the [built-in](attention_flash/atlas/mapping.json) and
[custom](attention_custom_flash/atlas/mapping.json) mapping reports.

## Reproduce

From this repository on the results branch, with a compatible NVIDIA GPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r results/flash-attention-2026-10-29/requirements.freeze.txt
python -c 'import torch; assert torch.cuda.is_available(), "PyTorch cannot see the GPU"; print(torch.cuda.get_device_name(0))'

for model in attention_flash attention_custom_flash; do
  PYTHONPATH=. python -m compiler.inductor_capture \
    --model "$model" --mode run \
    --sequence-length 32 --embed-size 32 --num-heads 1 \
    --out-dir "out/run/$model"
done

python -m unittest discover -s tests -p test_inductor_capture_artifacts.py
```

The capture exports the model graph, executes `torch.compile(backend="inductor")`,
captures that execution's generated wrapper/source and compiler-dumped TTIR,
compares the output with math SDPA using the same learned projections, then
attempts Atlas mapping. Check each regenerated `capture.json` for
`physical_gpu_used`, `numerical_match`, and `correctness_check_passed` all `true`.
Both artifact-link regression tests passed; `pip check` also passed.

Runs initialize fresh random weights and inputs; these commands reproduce the
procedure rather than byte-identical artifacts. Saved reports/source retain
the original machine paths. Only files referenced by the final captures are
saved here; fresh captures go under ignored `out/run/`.

## Fixes included

CUDA attention uses FP16, the custom Triton operator has required tensor/return
annotations, and its probability/value dot operands have matching dtypes.
Inference runs disable gradients to avoid requesting an unregistered backward.
Runtime source capture observes the actual execution; absolute artifact paths
and explicit wrapper-to-TTIR bindings keep mapping inputs associated correctly.
NumPy is included in the dependencies.
