"""Target mapping boundary for target-neutral Triton workload plans."""
from __future__ import annotations

from typing import Protocol, TypeAlias

from compiler.triton_analysis import (LinearRelu, MatmulRelu,
                                      ScaledDotProductAttention, TwoLayerMlp)


TritonWorkloadPlan: TypeAlias = MatmulRelu | TwoLayerMlp | LinearRelu | ScaledDotProductAttention


class TritonTargetMapper(Protocol):
    """Map a recognized Triton workload plan to a target IR artifact."""

    def map(self, plan: TritonWorkloadPlan) -> str:
        """Return target IR implementing the plan or reject it explicitly."""


class MatmulReluMapper(Protocol):
    """Compatibility interface for the original one-pattern prototype."""

    def map_matmul_relu(self, plan: MatmulRelu) -> str:
        """Map one matmul + ReLU plan to target IR."""
