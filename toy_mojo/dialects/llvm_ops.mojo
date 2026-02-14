"""Ch6: LLVM-like dialect types and operations."""

from utils import Variant
from collections import Optional
from toy.dialects import AsmWritable


# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct LLVMPtrType(Copyable, Movable, Stringable):
    fn __str__(self) -> String:
        return "ptr"


@fieldwise_init
struct LLVMIntType(Copyable, Movable, Stringable):
    var bits: Int

    fn __str__(self) -> String:
        return "i" + String(self.bits)


@fieldwise_init
struct LLVMFloatType(Copyable, Movable, Stringable):
    fn __str__(self) -> String:
        return "f64"


@fieldwise_init
struct LLVMVoidType(Copyable, Movable, Stringable):
    fn __str__(self) -> String:
        return "void"


comptime AnyLLVMType = Variant[
    LLVMPtrType, LLVMIntType, LLVMFloatType, LLVMVoidType
]


# ===----------------------------------------------------------------------=== #
# Formatting helpers
# ===----------------------------------------------------------------------=== #


fn _format_float(v: Float64) -> String:
    var iv = Int(v)
    if Float64(iv) == v:
        return String(iv) + ".0"
    return String(v)


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct LLAllocaOp(AsmWritable, Copyable, Movable):
    var result: String
    var elem_count: Int

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "%{} = alloca f64, {}".format(self.result, self.elem_count)
        )


@fieldwise_init
struct LLGepOp(AsmWritable, Copyable, Movable):
    var result: String
    var base: String
    var index: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "%{} = gep %{}, %{}".format(self.result, self.base, self.index)
        )


@fieldwise_init
struct LLLoadOp(AsmWritable, Copyable, Movable):
    var result: String
    var ptr: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("%{} = load %{}".format(self.result, self.ptr))


@fieldwise_init
struct LLStoreOp(AsmWritable, Copyable, Movable):
    var value: String
    var ptr: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("store %{}, %{}".format(self.value, self.ptr))


@fieldwise_init
struct LLFAddOp(AsmWritable, Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "%{} = fadd %{}, %{}".format(self.result, self.lhs, self.rhs)
        )


@fieldwise_init
struct LLFMulOp(AsmWritable, Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "%{} = fmul %{}, %{}".format(self.result, self.lhs, self.rhs)
        )


@fieldwise_init
struct LLConstantOp(AsmWritable, Copyable, Movable):
    var result: String
    var value: Float64

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "%{} = fconst {}".format(self.result, _format_float(self.value))
        )


@fieldwise_init
struct LLIndexConstOp(AsmWritable, Copyable, Movable):
    var result: String
    var value: Int

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("%{} = iconst {}".format(self.result, self.value))


@fieldwise_init
struct LLAddOp(AsmWritable, Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "%{} = add %{}, %{}".format(self.result, self.lhs, self.rhs)
        )


@fieldwise_init
struct LLMulOp(AsmWritable, Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "%{} = mul %{}, %{}".format(self.result, self.lhs, self.rhs)
        )


@fieldwise_init
struct LLIcmpOp(AsmWritable, Copyable, Movable):
    var result: String
    var pred: String
    var lhs: String
    var rhs: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "%{} = icmp {} %{}, %{}".format(
                self.result, self.pred, self.lhs, self.rhs
            )
        )


@fieldwise_init
struct LLBrOp(AsmWritable, Copyable, Movable):
    var dest: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("br {}".format(self.dest))


@fieldwise_init
struct LLCondBrOp(AsmWritable, Copyable, Movable):
    var cond: String
    var true_dest: String
    var false_dest: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "cond_br %{}, {}, {}".format(
                self.cond, self.true_dest, self.false_dest
            )
        )


@fieldwise_init
struct LLLabelOp(AsmWritable, Copyable, Movable):
    var name: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("{}:".format(self.name))


@fieldwise_init
struct LLPhiOp(AsmWritable, Copyable, Movable):
    var result: String
    var pairs: List[PhiPair]

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("%{} = phi".format(self.result))
        for ref pair in self.pairs:
            writer.write(" [%{}, {}]".format(pair.value, pair.label))


@fieldwise_init
struct PhiPair(Copyable, Movable):
    var value: String
    var label: String


@fieldwise_init
struct LLCallOp(AsmWritable, Copyable, Movable):
    var result: Optional[String]
    var callee: String
    var args: List[String]

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        if self.result:
            writer.write("%{} = ".format(self.result.value()))
        writer.write(
            "call @{}({})".format(
                self.callee,
                ", ".join(["%" + arg for arg in self.args]),
            )
        )


@fieldwise_init
struct LLReturnOp(AsmWritable, Copyable, Movable):
    var value: Optional[String]

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("ret")
        if self.value:
            writer.write(" %{}".format(self.value.value()))


comptime AnyLLVMOp = Variant[
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


@fieldwise_init
struct LLBlock(Copyable, Movable):
    var ops: List[AnyLLVMOp]


@fieldwise_init
struct LLFuncOp(Copyable, Movable):
    var name: String
    var body: LLBlock


@fieldwise_init
struct LLModule(Copyable, Movable):
    var functions: List[LLFuncOp]
