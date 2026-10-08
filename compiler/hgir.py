"""Small hardware-agnostic graph IR used by the model pipeline.

The IR deliberately describes tensor dataflow, not Triton syntax or a target
memory hierarchy.  Backends can later choose layouts, tiling, and address
spaces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HValue:
    name: str
    type: str
    shape: tuple[int | str, ...] = ()


@dataclass
class HOp:
    name: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    regions: list[list["HOp"]] = field(default_factory=list)


@dataclass
class HGraph:
    name: str
    inputs: dict[str, HValue]
    values: dict[str, HValue]
    ops: list[HOp]
    outputs: list[str]

    def verify(self) -> list[str]:
        errors: list[str] = []
        defined = set(self.inputs)
        for op in self.ops:
            missing = [name for name in op.inputs if name not in defined]
            if missing:
                errors.append("%s uses undefined values %s" % (op.name, missing))
            for name in op.outputs:
                if name in defined:
                    errors.append("value %s is defined twice" % name)
                if name not in self.values:
                    errors.append("missing type for value %s" % name)
                defined.add(name)
        errors.extend(
            "graph output %s is undefined" % name
            for name in self.outputs
            if name not in defined
        )
        return errors

    def format(self) -> str:
        lines = ["graph %s {" % self.name]
        for name, value in self.inputs.items():
            lines.append("  %s: %s%s" % (
                name, value.type,
                (" " + str(value.shape)) if value.shape else "",
            ))
        for op in self.ops:
            attrs = ""
            if op.attrs:
                attrs = " " + ", ".join(
                    "%s=%r" % (key, value) for key, value in op.attrs.items()
                )
            lines.append(
                "  %s = %s(%s)%s" %
                (", ".join(op.outputs), op.name, ", ".join(op.inputs), attrs)
            )
            for region in op.regions:
                lines.append("    region {")
                for nested in region:
                    lines.append(
                        "      %s = %s(%s)" % (
                            ", ".join(nested.outputs),
                            nested.name,
                            ", ".join(nested.inputs),
                        )
                    )
                lines.append("    }")
        lines.append("  return %s" % ", ".join(self.outputs))
        lines.append("}")
        return "\n".join(lines) + "\n"
