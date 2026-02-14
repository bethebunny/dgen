"""Toy dialect IR types and operations.

This module defines the IR data structures for the Toy tutorial dialect,
representing what a DGEN code generator would produce from toy.dgen.
"""

from utils import Variant
from collections import Optional
from memory import OwnedPointer
from toy.dialects import AsmWritable


# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #

trait ToyType:
    pass


@fieldwise_init
struct UnrankedTensorType(ToyType, Copyable, Movable, Stringable):
    """tensor<*xf64>"""

    fn __str__(self) -> String:
        return "tensor<*xf64>"


@fieldwise_init
struct RankedTensorType(ToyType, Copyable, Movable, Stringable):
    """tensor<2x3xf64>"""

    var shape: List[Int]

    fn __str__(self) -> String:
        var s = String("tensor<")
        for i in range(len(self.shape)):
            s += String(self.shape[i])
            s += "x"
        s += "f64>"
        return s


comptime AnyToyType = Variant[UnrankedTensorType, RankedTensorType]


@fieldwise_init
struct FunctionType(ToyType, Copyable, Movable):
    """(tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>"""

    var inputs: List[AnyToyType]
    var result: Optional[AnyToyType]


fn type_to_string(t: AnyToyType) -> String:
    if t.isa[UnrankedTensorType]():
        return String(t[UnrankedTensorType])
    if t.isa[RankedTensorType]():
        return String(t[RankedTensorType])
    return "<function_type>"


# ===----------------------------------------------------------------------=== #
# Formatting helpers
# ===----------------------------------------------------------------------=== #

fn format_shape(shape: List[Int]) -> String:
    var s = String("<")
    for i in range(len(shape)):
        if i > 0:
            s += "x"
        s += String(shape[i])
    s += ">"
    return s


fn format_float_list(values: List[Float64]) -> String:
    var s = String("[")
    for i in range(len(values)):
        if i > 0:
            s += ", "
        # Format float: if it's a whole number, show as X.0
        var v = values[i]
        var iv = Int(v)
        if Float64(iv) == v:
            s += String(iv) + ".0"
        else:
            s += String(v)
    s += "]"
    return s


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #

@fieldwise_init
struct ConstantOp(AsmWritable, Copyable, Movable):
    var result: String
    var value: List[Float64]
    var shape: List[Int]
    var type: AnyToyType

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("%{} = Constant({} {}) : {}".format(
            self.result, format_shape(self.shape), format_float_list(self.value), type_to_string(self.type)
        ))


@fieldwise_init
struct TransposeOp(AsmWritable, Copyable, Movable):
    var result: String
    var input: String
    var type: AnyToyType

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("%{} = Transpose(%{}) : {}".format(self.result, self.input, type_to_string(self.type)))


@fieldwise_init
struct ReshapeOp(AsmWritable, Copyable, Movable):
    var result: String
    var input: String
    var type: AnyToyType

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("%{} = Reshape(%{}) : {}".format(self.result, self.input, type_to_string(self.type)))


@fieldwise_init
struct MulOp(AsmWritable, Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String
    var type: AnyToyType

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("%{} = Mul(%{}, %{}) : {}".format(self.result, self.lhs, self.rhs, type_to_string(self.type)))


@fieldwise_init
struct AddOp(AsmWritable, Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String
    var type: AnyToyType

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("%{} = Add(%{}, %{}) : {}".format(self.result, self.lhs, self.rhs, type_to_string(self.type)))


@fieldwise_init
struct GenericCallOp(AsmWritable, Copyable, Movable):
    var result: String
    var callee: String
    var args: List[String]
    var type: AnyToyType

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("%{} = GenericCall @{}(".format(self.result, self.callee))
        for i in range(len(self.args)):
            if i > 0:
                writer.write(", ")
            writer.write("%{}".format(self.args[i]))
        writer.write(") : {}".format(type_to_string(self.type)))


@fieldwise_init
struct PrintOp(AsmWritable, Copyable, Movable):
    var input: String

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        writer.write("Print(%{})".format(self.input))


@fieldwise_init
struct ReturnOp(AsmWritable, Copyable, Movable):
    var value: Optional[String]

    fn write_asm(self, mut writer: Some[Writer]) -> None:
        if self.value:
            writer.write("return %{}".format(self.value.value()))
        else:
            writer.write("return")


comptime AnyToyOp = Variant[
    ConstantOp, TransposeOp, ReshapeOp, MulOp, AddOp,
    GenericCallOp, PrintOp, ReturnOp,
]


# ===----------------------------------------------------------------------=== #
# Structure
# ===----------------------------------------------------------------------=== #

@fieldwise_init
struct ToyValue(Copyable, Movable):
    """A block argument (function parameter)."""

    var name: String
    var type: AnyToyType


@fieldwise_init
struct Block(Copyable, Movable):
    var args: List[ToyValue]
    var ops: List[AnyToyOp]


@fieldwise_init
struct FuncOp(Copyable, Movable):
    var name: String
    var func_type: FunctionType
    var body: Block


@fieldwise_init
struct Module(Copyable, Movable):
    var functions: List[FuncOp]
