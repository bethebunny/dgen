"""Ch6: LLVM-like dialect types and operations."""

from utils import Variant
from collections import Optional


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


comptime AnyLLVMType = Variant[LLVMPtrType, LLVMIntType, LLVMFloatType, LLVMVoidType]


# ===----------------------------------------------------------------------=== #
# Operations
# ===----------------------------------------------------------------------=== #

@fieldwise_init
struct LLAllocaOp(Copyable, Movable):
    var result: String
    var elem_count: Int


@fieldwise_init
struct LLGepOp(Copyable, Movable):
    var result: String
    var base: String
    var index: String


@fieldwise_init
struct LLLoadOp(Copyable, Movable):
    var result: String
    var ptr: String


@fieldwise_init
struct LLStoreOp(Copyable, Movable):
    var value: String
    var ptr: String


@fieldwise_init
struct LLFAddOp(Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String


@fieldwise_init
struct LLFMulOp(Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String


@fieldwise_init
struct LLConstantOp(Copyable, Movable):
    var result: String
    var value: Float64


@fieldwise_init
struct LLIndexConstOp(Copyable, Movable):
    var result: String
    var value: Int


@fieldwise_init
struct LLAddOp(Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String


@fieldwise_init
struct LLMulOp(Copyable, Movable):
    var result: String
    var lhs: String
    var rhs: String


@fieldwise_init
struct LLIcmpOp(Copyable, Movable):
    var result: String
    var pred: String
    var lhs: String
    var rhs: String


@fieldwise_init
struct LLBrOp(Copyable, Movable):
    var dest: String


@fieldwise_init
struct LLCondBrOp(Copyable, Movable):
    var cond: String
    var true_dest: String
    var false_dest: String


@fieldwise_init
struct LLLabelOp(Copyable, Movable):
    var name: String


@fieldwise_init
struct LLPhiOp(Copyable, Movable):
    var result: String
    var pairs: List[PhiPair]


@fieldwise_init
struct PhiPair(Copyable, Movable):
    var value: String
    var label: String


@fieldwise_init
struct LLCallOp(Copyable, Movable):
    var result: Optional[String]
    var callee: String
    var args: List[String]


@fieldwise_init
struct LLReturnOp(Copyable, Movable):
    var value: Optional[String]


comptime AnyLLVMOp = Variant[
    LLAllocaOp, LLGepOp, LLLoadOp, LLStoreOp,
    LLFAddOp, LLFMulOp, LLConstantOp, LLIndexConstOp,
    LLAddOp, LLMulOp, LLIcmpOp,
    LLBrOp, LLCondBrOp, LLLabelOp, LLPhiOp,
    LLCallOp, LLReturnOp,
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
