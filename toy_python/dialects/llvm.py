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

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = alloca f64, {self.elem_count}"


@dataclass
class LLGepOp:
    result: str
    base: str
    index: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = gep %{self.base}, %{self.index}"


@dataclass
class LLLoadOp:
    result: str
    ptr: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = load %{self.ptr}"


@dataclass
class LLStoreOp:
    value: str
    ptr: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"store %{self.value}, %{self.ptr}"


@dataclass
class LLFAddOp:
    result: str
    lhs: str
    rhs: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = fadd %{self.lhs}, %{self.rhs}"


@dataclass
class LLFMulOp:
    result: str
    lhs: str
    rhs: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = fmul %{self.lhs}, %{self.rhs}"


@dataclass
class LLConstantOp:
    result: str
    value: float

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = fconst {format_float(self.value)}"


@dataclass
class LLIndexConstOp:
    result: str
    value: int

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = iconst {self.value}"


@dataclass
class LLAddOp:
    result: str
    lhs: str
    rhs: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = add %{self.lhs}, %{self.rhs}"


@dataclass
class LLMulOp:
    result: str
    lhs: str
    rhs: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = mul %{self.lhs}, %{self.rhs}"


@dataclass
class LLIcmpOp:
    result: str
    pred: str
    lhs: str
    rhs: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"%{self.result} = icmp {self.pred} %{self.lhs}, %{self.rhs}"


@dataclass
class LLBrOp:
    dest: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"br {self.dest}"


@dataclass
class LLCondBrOp:
    cond: str
    true_dest: str
    false_dest: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"cond_br %{self.cond}, {self.true_dest}, {self.false_dest}"


@dataclass
class LLLabelOp:
    name: str

    @property
    def asm(self) -> Iterable[str]:
        yield f"{self.name}:"


@dataclass
class PhiPair:
    value: str
    label: str


@dataclass
class LLPhiOp:
    result: str
    pairs: list[PhiPair]

    @property
    def asm(self) -> Iterable[str]:
        pairs = " ".join(f"[%{p.value}, {p.label}]" for p in self.pairs)
        yield f"%{self.result} = phi {pairs}"


@dataclass
class LLCallOp:
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
class LLReturnOp:
    value: str | None

    @property
    def asm(self) -> Iterable[str]:
        if self.value is not None:
            yield f"ret %{self.value}"
        else:
            yield "ret void"


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

    @property
    def asm(self) -> Iterable[str]:
        for op in self.ops:
            if isinstance(op, LLLabelOp):
                yield from op.asm
            else:
                yield from asm.indent(op.asm)


@dataclass
class LLFuncOp:
    name: str
    body: LLBlock

    @property
    def asm(self) -> Iterable[str]:
        yield f"define void @{self.name}():"
        yield from self.body.asm


@dataclass
class LLModule:
    functions: list[LLFuncOp]

    @property
    def asm(self) -> Iterable[str]:
        for i, function in enumerate(self.functions):
            if i > 0:
                yield ""
            yield from function.asm
