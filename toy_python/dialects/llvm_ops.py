"""Ch6: LLVM-like dialect types and operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #


@dataclass
class LLVMPtrType:
    def __str__(self) -> str:
        return "ptr"


@dataclass
class LLVMIntType:
    bits: int

    def __str__(self) -> str:
        return f"i{self.bits}"


@dataclass
class LLVMFloatType:
    def __str__(self) -> str:
        return "f64"


@dataclass
class LLVMVoidType:
    def __str__(self) -> str:
        return "void"


AnyLLVMType = Union[LLVMPtrType, LLVMIntType, LLVMFloatType, LLVMVoidType]


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
class LLAllocaOp:
    result: str
    elem_count: int


@dataclass
class LLGepOp:
    result: str
    base: str
    index: str


@dataclass
class LLLoadOp:
    result: str
    ptr: str


@dataclass
class LLStoreOp:
    value: str
    ptr: str


@dataclass
class LLFAddOp:
    result: str
    lhs: str
    rhs: str


@dataclass
class LLFMulOp:
    result: str
    lhs: str
    rhs: str


@dataclass
class LLConstantOp:
    result: str
    value: float


@dataclass
class LLIndexConstOp:
    result: str
    value: int


@dataclass
class LLAddOp:
    result: str
    lhs: str
    rhs: str


@dataclass
class LLMulOp:
    result: str
    lhs: str
    rhs: str


@dataclass
class LLIcmpOp:
    result: str
    pred: str
    lhs: str
    rhs: str


@dataclass
class LLBrOp:
    dest: str


@dataclass
class LLCondBrOp:
    cond: str
    true_dest: str
    false_dest: str


@dataclass
class LLLabelOp:
    name: str


@dataclass
class PhiPair:
    value: str
    label: str


@dataclass
class LLPhiOp:
    result: str
    pairs: list[PhiPair]


@dataclass
class LLCallOp:
    result: str | None
    callee: str
    args: list[str]


@dataclass
class LLReturnOp:
    value: str | None


AnyLLVMOp = Union[
    LLAllocaOp,
    LLGepOp,
    LLLoadOp,
    LLStoreOp,
    LLFAddOp,
    LLFMulOp,
    LLConstantOp,
    LLIndexConstOp,
    LLAddOp,
    LLMulOp,
    LLIcmpOp,
    LLBrOp,
    LLCondBrOp,
    LLLabelOp,
    LLPhiOp,
    LLCallOp,
    LLReturnOp,
]


# ===----------------------------------------------------------------------=== #
# Structure
# ===----------------------------------------------------------------------=== #


@dataclass
class LLBlock:
    ops: list[AnyLLVMOp]


@dataclass
class LLFuncOp:
    name: str
    body: LLBlock


@dataclass
class LLModule:
    functions: list[LLFuncOp]
