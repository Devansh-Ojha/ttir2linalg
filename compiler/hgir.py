"""Small hardware-agnostic graph IR used by the model pipeline.

The IR deliberately describes tensor dataflow, not Triton syntax or a target
memory hierarchy.  Backends can later choose layouts, tiling, and address
spaces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return str(value)


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

    def to_dict(self) -> dict[str, Any]:
        def value_dict(value: HValue):
            return {
                "name": value.name,
                "type": value.type,
                "shape": _json_value(value.shape),
            }

        def op_dict(op: HOp):
            return {
                "name": op.name,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
                "attrs": _json_value(op.attrs),
                "regions": [[op_dict(nested) for nested in region]
                            for region in op.regions],
            }

        return {
            "format": "hgraph-v1",
            "name": self.name,
            "inputs": {key: value_dict(value)
                       for key, value in self.inputs.items()},
            "values": {key: value_dict(value)
                       for key, value in self.values.items()},
            "ops": [op_dict(op) for op in self.ops],
            "outputs": list(self.outputs),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HGraph":
        if payload.get("format") != "hgraph-v1":
            raise ValueError("unsupported HGraph format %r" % payload.get("format"))

        def value_from_dict(data):
            return HValue(data["name"], data["type"], tuple(data.get("shape", ())))

        def op_from_dict(data):
            return HOp(
                data["name"], list(data["inputs"]), list(data["outputs"]),
                dict(data.get("attrs", {})),
                [[op_from_dict(nested) for nested in region]
                 for region in data.get("regions", [])],
            )

        return cls(
            name=payload["name"],
            inputs={key: value_from_dict(value)
                    for key, value in payload["inputs"].items()},
            values={key: value_from_dict(value)
                    for key, value in payload["values"].items()},
            ops=[op_from_dict(op) for op in payload["ops"]],
            outputs=list(payload["outputs"]),
        )

    def verify(self) -> list[str]:
        errors: list[str] = []
        defined = set(self.inputs)
        def verify_ops(ops, scope):
            local_errors = []
            local_defined = set(scope)
            for op in ops:
                missing = [name for name in op.inputs if name not in local_defined]
                if missing:
                    local_errors.append(
                        "%s uses undefined values %s" % (op.name, missing)
                    )
                for name in op.outputs:
                    if name in local_defined:
                        local_errors.append("value %s is defined twice" % name)
                    if name not in self.values:
                        local_errors.append("missing type for value %s" % name)
                    local_defined.add(name)
                for region in op.regions:
                    region_args = op.attrs.get("region_args", [])
                    region_scope = local_defined | {
                        name
                        for names in region_args
                        for name in names
                    }
                    local_errors.extend(verify_ops(region, region_scope))
            return local_errors

        errors.extend(verify_ops(self.ops, defined))
        for op in self.ops:
            missing = [name for name in op.inputs if name not in defined]
            defined.update(op.outputs)
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
            result_info = []
            for output in op.outputs:
                value = self.values.get(output)
                if value is None:
                    continue
                shape = " " + str(value.shape) if value.shape else ""
                result_info.append("%s: %s%s" % (output, value.type, shape))
            if result_info:
                lines[-1] += " -> " + ", ".join(result_info)
            for index, region in enumerate(op.regions):
                region_args = op.attrs.get("region_args", [])
                args = region_args[index] if index < len(region_args) else []
                lines.append("    region%s {" % (
                    ("(" + ", ".join(args) + ")") if args else ""
                ))
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
