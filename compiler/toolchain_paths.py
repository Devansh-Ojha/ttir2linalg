"""Locations of separately installed Atlas tools; no compiler semantics."""
import os
from pathlib import Path


def atlas_root():
    """Explicit ATLAS_ROOT, or an atlas-mlir checkout beside this repository."""
    return Path(os.environ.get("ATLAS_ROOT", Path(__file__).resolve().parents[2] / "atlas-mlir")).expanduser().resolve()


def npu_model_root():
    """Explicit NPU_MODEL_ROOT, or a sibling npu-model-eval checkout."""
    return Path(os.environ.get("NPU_MODEL_ROOT", Path(__file__).resolve().parents[2] / "npu-model-eval")).expanduser().resolve()
