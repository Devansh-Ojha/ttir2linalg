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
