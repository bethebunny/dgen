"""Ch5: Affine dialect types and operations."""

from utils import Variant
from collections import Optional
from toy.dialects import AsmWritable


# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct MemRefType(AsmWritable, Copyable, Movable):
    var shape: List[Int]

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("memref<{}xf64>".format("x".join(self.shape)))


@fieldwise_init
struct IndexType(AsmWritable, Copyable, Movable):
    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("index")


@fieldwise_init
struct F64Type(AsmWritable, Copyable, Movable):
    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("f64")


comptime AnyAffineType = Variant[MemRefType, IndexType, F64Type]


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
struct AllocOp(AsmWritable, Copyable, Movable):
    var result: String
    var shape: List[Int]

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "%{} = Alloc<{}>()".format(self.result, "x".join(self.shape))
        )


@fieldwise_init
struct DeallocOp(AsmWritable, Copyable, Movable):
    var input: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("Dealloc(%{})".format(self.input))


@fieldwise_init
struct AffineLoadOp(AsmWritable, Copyable, Movable):
    var result: String
    var memref: String
    var indices: List[String]

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "%{} = AffineLoad %{}[{}]".format(
                self.result, self.memref, ", ".join(self.indices)
            )
        )


@fieldwise_init
struct AffineStoreOp(AsmWritable, Copyable, Movable):
    var value: String
    var memref: String
    var indices: List[String]

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "AffineStore %{}, %{}[{}]".format(
                self.value, self.memref, ", ".join(self.indices)
            )
        )


@fieldwise_init
struct ArithConstantOp(AsmWritable, Copyable, Movable):
    var result: String
    var value: Float64

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "%{} = ArithConstant({})".format(
                self.result, _format_float(self.value)
            )
        )


@fieldwise_init
struct IndexConstantOp(AsmWritable, Copyable, Movable):
    var result: String
    var value: Int

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("%{} = IndexConstant({})".format(self.result, self.value))


@fieldwise_init
struct ArithMulFOp(AsmWritable, Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "%{} = MulF(%{}, %{})".format(self.result, self.lhs, self.rhs)
        )


@fieldwise_init
struct ArithAddFOp(AsmWritable, Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "%{} = AddF(%{}, %{})".format(self.result, self.lhs, self.rhs)
        )


@fieldwise_init
struct AffinePrintOp(AsmWritable, Copyable, Movable):
    var input: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("PrintMemRef(%{}".format(self.input))


@fieldwise_init
struct AffineReturnOp(AsmWritable, Copyable, Movable):
    var value: Optional[String]

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("return")
        if self.value:
            writer.write(" %")
            writer.write(self.value.value())


# Forward-declare the variant with AffineForOp.
# AffineForOp contains List[AnyAffineOp] which heap-allocates,
# breaking the recursive size issue.


@fieldwise_init
struct AffineForOp(AsmWritable, Copyable, Movable):
    var var_name: String
    var lo: Int
    var hi: Int
    var body: List[AnyAffineOp]

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write(
            "AffineFor %{} = {} to {}:".format(self.var_name, self.lo, self.hi)
        )


comptime AnyAffineOp = Variant[
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


@fieldwise_init
struct AffineValue(Copyable, Movable):
    var name: String
    var type: AnyAffineType


@fieldwise_init
struct AffineBlock(Copyable, Movable):
    var args: List[AffineValue]
    var ops: List[AnyAffineOp]


@fieldwise_init
struct AffineFuncOp(Copyable, Movable):
    var name: String
    var body: AffineBlock


@fieldwise_init
struct AffineModule(Copyable, Movable):
    var functions: List[AffineFuncOp]
