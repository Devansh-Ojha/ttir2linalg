import subprocess
import sys


def test_linear_lowering():
    result = subprocess.run(
        [sys.executable, "compiler/lower.py"],
        capture_output=True,
        text=True,
        check=True,
    )

    out = result.stdout

    assert "=== TTIR → LINALG ===" in out
    assert "linalg.constant()" in out
    assert "linalg.program_id()" in out
    assert "linalg.index_range()" in out
    assert "linalg.broadcast" in out
    assert "linalg.pointer_add" in out
    assert "linalg.load" in out
    assert "linalg.mul" in out
    assert "linalg.reshape" in out
    assert "linalg.reduce" in out
    assert "linalg.add(%14, %11)" in out
    assert "linalg.store(%16, %15)" in out
    assert "linalg.add(%arg4, %arg5)" not in out
if __name__ == "__main__":
    test_linear_lowering()
    print("PASS")
