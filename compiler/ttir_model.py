"""Lightweight, target-independent data model for parsed Triton IR."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TType:
    kind: str                      # "scalar" | "ptr" | "tensor"
    elem: str                      # element type (pointee type for ptr)
    shape: Tuple[int, ...] = ()
    elem_is_ptr: bool = False      # tensor of pointers
    raw: str = ""

    def __str__(self):
        return self.raw


@dataclass
class TValue:
    id: int
    type: TType
    name: str
    is_block_arg: bool = False
    defining_op: Optional["TOp"] = None


@dataclass
class TBlock:
    id: int
    args: List[TValue] = field(default_factory=list)
    ops: List["TOp"] = field(default_factory=list)


@dataclass
class TOp:
    name: str
    operands: List[TValue] = field(default_factory=list)
    results: List[TValue] = field(default_factory=list)
    attrs: Dict[str, str] = field(default_factory=dict)
    regions: List[List[TBlock]] = field(default_factory=list)
    index: int = 0

    def attr_int(self, key: str) -> int:
        return int(self.attrs[key].split(":")[0].strip())


@dataclass
class TFunc:
    name: str
    args: List[TValue]
    body: TBlock

