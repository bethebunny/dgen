"""Toy dialect IR types and operations.

This module defines the IR data structures for the Toy tutorial dialect,
representing what a DGEN code generator would produce from toy.dgen.
"""

from utils import Variant
from collections import Optional
from memory import OwnedPointer


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
# Operations
# ===----------------------------------------------------------------------=== #

@fieldwise_init
struct ConstantOp(Copyable, Movable):
    var result: String
    var value: List[Float64]
    var shape: List[Int]
    var type: AnyToyType


@fieldwise_init
struct TransposeOp(Copyable, Movable):
    var result: String
    var input: String
    var type: AnyToyType


@fieldwise_init
struct ReshapeOp(Copyable, Movable):
    var result: String
    var input: String
    var type: AnyToyType


@fieldwise_init
struct MulOp(Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String
    var type: AnyToyType


@fieldwise_init
struct AddOp(Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String
    var type: AnyToyType


@fieldwise_init
struct GenericCallOp(Copyable, Movable):
    var result: String
    var callee: String
    var args: List[String]
    var type: AnyToyType


@fieldwise_init
struct PrintOp(Copyable, Movable):
    var input: String


@fieldwise_init
struct ReturnOp(Copyable, Movable):
    var value: Optional[String]


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
