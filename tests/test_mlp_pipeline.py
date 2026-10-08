import os
import subprocess
import sys


def test_mlp_pipeline():
    result = subprocess.run(
        [sys.executable, "mlp_pipeline.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout
    assert "graph exported_mlp {" in out
    assert "linear(" in out
    assert "relu(" in out
    assert "return" in out
    assert '"torch_compile": true' in out
    assert '"compile_mode": "torch.compile graph capture"' in out
    assert '"numerical_match": true' in out


def test_ttir_mlp_hgir():
    """Validate the artifact-driven path when EDA TTIR captures are present."""
    ttir_dir = os.environ.get("MLP_TTIR_DIR")
    if not ttir_dir:
        return
    result = subprocess.run(
        [sys.executable, "mlp_pipeline.py", "--ttir-dir", ttir_dir],
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout
    assert '"ttir_kernel_count":' in out
    assert "=== HGIR kernel" in out
    assert "linear" in out
    assert any(name in out for name in ("maximum", "compare", "select", "relu"))
