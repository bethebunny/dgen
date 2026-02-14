"""Ch6: LLVM-like dialect types and operations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Union

from toy_python import asm


# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #


@dataclass
class PtrType:
    def __str__(self) -> str:
        return "ptr"


@dataclass
class IntType:
    bits: int

    def __str__(self) -> str:
        return f"i{self.bits}"


@dataclass
class FloatType:
    def __str__(self) -> str:
        return "f64"


@dataclass
class VoidType:
    def __str__(self) -> str:
        return "void"


AnyType = Union[PtrType, IntType, FloatType, VoidType]


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
class AllocaOp:
    result: str
    elem_count: int

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = alloca f64, {self.elem_count}"


@dataclass
class GepOp:
    result: str
    base: str
    index: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = gep %{self.base}, %{self.index}"


@dataclass
class LoadOp:
    result: str
    ptr: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = load %{self.ptr}"


@dataclass
class StoreOp:
    value: str
    ptr: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"store %{self.value}, %{self.ptr}"


@dataclass
class FAddOp:
    result: str
    lhs: str
    rhs: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = fadd %{self.lhs}, %{self.rhs}"


@dataclass
class FMulOp:
    result: str
    lhs: str
    rhs: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = fmul %{self.lhs}, %{self.rhs}"


@dataclass
class ConstantOp:
    result: str
    value: float

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = fconst {format_float(self.value)}"


@dataclass
class IndexConstOp:
    result: str
    value: int

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = iconst {self.value}"


@dataclass
class AddOp:
    result: str
    lhs: str
    rhs: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = add %{self.lhs}, %{self.rhs}"


@dataclass
class MulOp:
    result: str
    lhs: str
    rhs: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = mul %{self.lhs}, %{self.rhs}"


@dataclass
class IcmpOp:
    result: str
    pred: str
    lhs: str
    rhs: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = icmp {self.pred} %{self.lhs}, %{self.rhs}"


@dataclass
class BrOp:
    dest: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"br {self.dest}"


@dataclass
class CondBrOp:
    cond: str
    true_dest: str
    false_dest: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"cond_br %{self.cond}, {self.true_dest}, {self.false_dest}"


@dataclass
class LabelOp:
    name: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"{self.name}:"


@dataclass
class PhiPair:
    value: str
    label: str


@dataclass
class PhiOp:
    result: str
    pairs: list[PhiPair]

    @property
    def asm(self) -> Iterable[str]:
        pairs = " ".join(f"[%{p.value}, {p.label}]" for p in self.pairs)
        yield f"%{self.result} = phi {pairs}"


@dataclass
class CallOp:
    result: str | None
    callee: str
    args: list[str]

    @property
    def asm(self) -> Iterable[str]:
        args_str = ", ".join(f"%{a}" for a in self.args)
        if self.result is not None:
            yield f"%{self.result} = call @{self.callee}({args_str})"
        else:
            yield f"call @{self.callee}({args_str})"


@dataclass
class ReturnOp:
    value: str | None

    @property
    def asm(self) -> Iterable[str]:
        if self.value is not None:
            yield f"ret %{self.value}"
        else:
            yield "ret void"


AnyOp = Union[
    AllocaOp,
    GepOp,
    LoadOp,
    StoreOp,
    FAddOp,
    FMulOp,
    ConstantOp,
    IndexConstOp,
    AddOp,
    MulOp,
    IcmpOp,
    BrOp,
    CondBrOp,
    LabelOp,
    PhiOp,
    CallOp,
    ReturnOp,
]


# ===----------------------------------------------------------------------=== #
# Structure
# ===----------------------------------------------------------------------=== #


@dataclass
class Block:
    ops: list[AnyOp]

    @property
    def asm(self) -> Iterable[str]:
        for op in self.ops:
            if isinstance(op, LabelOp):
                yield from op.asm
            else:
                yield from asm.indent(op.asm)


@dataclass
class FuncOp:
    name: str
    body: Block

    @property
    def asm(self) -> Iterable[str]:
        yield f"define void @{self.name}():"
        yield from self.body.asm


@dataclass
class Module:
    functions: list[FuncOp]

    @property
    def asm(self) -> Iterable[str]:
        for i, function in enumerate(self.functions):
            if i > 0:
                yield ""
            yield from function.asm
