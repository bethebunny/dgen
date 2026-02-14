"""Ch5: Affine dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #


@dataclass
class MemRefType:
    shape: list[int]

    def __str__(self) -> str:
        return f"memref<{'x'.join(str(d) for d in self.shape)}xf64>"


@dataclass
class IndexType:
    def __str__(self) -> str:
        return "index"


@dataclass
class F64Type:
    def __str__(self) -> str:
        return "f64"


AnyAffineType = Union[MemRefType, IndexType, F64Type]


# ===----------------------------------------------------------------------=== #
# Formatting helpers
# ===----------------------------------------------------------------------=== #


def format_float(v: float) -> str:
    iv = int(v)
    if float(iv) == v:
        return f"{iv}.0"
    return str(v)


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #


@dataclass
class AllocOp:
    result: str
    shape: list[int]


@dataclass
class DeallocOp:
    input: str


@dataclass
class AffineLoadOp:
    result: str
    memref: str
    indices: list[str]


@dataclass
class AffineStoreOp:
    value: str
    memref: str
    indices: list[str]


@dataclass
class ArithConstantOp:
    result: str
    value: float


@dataclass
class IndexConstantOp:
    result: str
    value: int


@dataclass
class ArithMulFOp:
    result: str
    lhs: str
    rhs: str


@dataclass
class ArithAddFOp:
    result: str
    lhs: str
    rhs: str


@dataclass
class AffinePrintOp:
    input: str


@dataclass
class AffineReturnOp:
    value: str | None


@dataclass
class AffineForOp:
    var_name: str
    lo: int
    hi: int
    body: list[AnyAffineOp]


AnyAffineOp = Union[
    AllocOp,
    DeallocOp,
    AffineLoadOp,
    AffineStoreOp,
    AffineForOp,
    ArithConstantOp,
    IndexConstantOp,
    ArithMulFOp,
    ArithAddFOp,
    AffinePrintOp,
    AffineReturnOp,
]


# ===----------------------------------------------------------------------=== #
# Structure
# ===----------------------------------------------------------------------=== #


@dataclass
class AffineValue:
    name: str
    type: AnyAffineType


@dataclass
class AffineBlock:
    args: list[AffineValue]
    ops: list[AnyAffineOp]


@dataclass
class AffineFuncOp:
    name: str
    body: AffineBlock


@dataclass
class AffineModule:
    functions: list[AffineFuncOp]
