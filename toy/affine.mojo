"""Ch5: Affine dialect types and operations."""

from utils import Variant
from collections import Optional


# ===----------------------------------------------------------------------=== #
# Types
# ===----------------------------------------------------------------------=== #

@fieldwise_init
struct MemRefType(Copyable, Movable, Stringable):
    var shape: List[Int]

    fn __str__(self) -> String:
        var s = String("memref<")
        for i in range(len(self.shape)):
            s += String(self.shape[i])
            s += "x"
        s += "f64>"
        return s


@fieldwise_init
struct IndexType(Copyable, Movable, Stringable):
    fn __str__(self) -> String:
        return "index"


@fieldwise_init
struct F64Type(Copyable, Movable, Stringable):
    fn __str__(self) -> String:
        return "f64"


comptime AnyAffineType = Variant[MemRefType, IndexType, F64Type]


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #

@fieldwise_init
struct AllocOp(Copyable, Movable):
    var result: String
    var shape: List[Int]


@fieldwise_init
struct DeallocOp(Copyable, Movable):
    var input: String


@fieldwise_init
struct AffineLoadOp(Copyable, Movable):
    var result: String
    var memref: String
    var indices: List[String]


@fieldwise_init
struct AffineStoreOp(Copyable, Movable):
    var value: String
    var memref: String
    var indices: List[String]


@fieldwise_init
struct ArithConstantOp(Copyable, Movable):
    var result: String
    var value: Float64


@fieldwise_init
struct IndexConstantOp(Copyable, Movable):
    var result: String
    var value: Int


@fieldwise_init
struct ArithMulFOp(Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String


@fieldwise_init
struct ArithAddFOp(Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String


@fieldwise_init
struct AffinePrintOp(Copyable, Movable):
    var input: String


@fieldwise_init
struct AffineReturnOp(Copyable, Movable):
    var value: Optional[String]


# Forward-declare the variant with AffineForOp.
# AffineForOp contains List[AnyAffineOp] which heap-allocates,
# breaking the recursive size issue.

@fieldwise_init
struct AffineForOp(Copyable, Movable):
    var var_name: String
    var lo: Int
    var hi: Int
    var body: List[AnyAffineOp]


comptime AnyAffineOp = Variant[
    AllocOp, DeallocOp, AffineLoadOp, AffineStoreOp,
    AffineForOp, ArithConstantOp, IndexConstantOp,
    ArithMulFOp, ArithAddFOp, AffinePrintOp, AffineReturnOp,
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
