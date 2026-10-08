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


def test_direct_triton_capture_requires_or_uses_triton():
    from compiler.triton_capture import compile_linear_relu_ttir

    try:
        text = compile_linear_relu_ttir()
    except RuntimeError as exc:
        assert "Triton" in str(exc)
    else:
        assert "tt.return" in text
        assert "tt.store" in text


def test_mlp_triton_capture_requires_or_uses_triton(tmp_path):
    from compiler.triton_capture import write_mlp_ttir

    try:
        files = write_mlp_ttir(tmp_path)
    except RuntimeError as exc:
        assert "Triton" in str(exc)
    else:
        assert [path.name for path in files] == [
            "kernel_0_fc1_relu.ttir",
            "kernel_1_fc2.ttir",
        ]
        for path in files:
            text = path.read_text()
            assert "tt.return" in text
            assert "tt.store" in text


def test_mlp_triton_source_capture_requires_cuda_or_writes_sources(tmp_path):
    from compiler.triton_capture import write_mlp_triton_source

    files = write_mlp_triton_source(tmp_path)
    assert len(files) == 2
    for path in files:
        source = path.read_text()
        assert "@triton.jit" in source
        assert "def " in source


def test_attention_triton_source_capture_requires_cuda_or_writes_sources(tmp_path):
    from compiler.triton_capture import write_model_triton_source

    files = write_model_triton_source(tmp_path, model_name="attention")
    assert len(files) == 1
    assert all("@triton.jit" in path.read_text() for path in files)
