import json
import subprocess
import sys


def test_inductor_triton_probe_does_not_fabricate_ttir():
    result = subprocess.run(
        [sys.executable, "-m", "compiler.inductor_triton_probe"],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["torch_compile_graph_captured"] is True
    assert report["compiled_output_matches"] is True
    assert report["representation"] == "fx_graph"
    assert report["ttir_obtained"] is False
    assert "linear" in report["hgir"]
    assert "relu" in report["hgir"]
