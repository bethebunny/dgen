"""Generic round-trip tests for Type subclasses.

Each type is tested for:
1. Type ASM round-trip (format → parse → equality)
2. Memory from_value round-trip (Python value → pack → unpack)
3. Memory from_asm round-trip (ASM literal → pack → unpack)
4. JIT identity round-trip (Memory → JIT function → result)
"""

import ctypes

import pytest

from dgen import Block, Dialect
from dgen.asm.formatting import type_asm
from dgen.asm.parser import IRParser, parse_expr
from dgen.block import BlockArgument
from dgen.codegen import compile as compile_module
from dgen.dialects import builtin, llvm
from dgen.type import Memory
from toy.dialects.affine import MemRefType, ShapeType
from toy.dialects.toy import InferredShapeTensor, TensorType


# ---------------------------------------------------------------------------
# Test data: (type, python_value, asm_literal, expected_unpack)
#
# Every type with a runtime representation should appear here.
# Types with genuinely void layouts (Nil, VoidType) are excluded.
# ---------------------------------------------------------------------------

BUILTIN_TYPES = [
    pytest.param(
        builtin.IndexType(), 42, "42", (42,),
        id="builtin.index",
    ),
    pytest.param(
        builtin.F64Type(), 3.14, "3.14", (3.14,),
        id="builtin.f64",
    ),
    pytest.param(
        builtin.String(), ("hello", 5), None, None,
        id="builtin.string",
    ),
    pytest.param(
        builtin.List(element_type=builtin.F64Type()), None, None, None,
        id="builtin.list",
    ),
]

LLVM_TYPES = [
    pytest.param(
        llvm.IntType(bits=64), 42, "42", (42,),
        id="llvm.i64",
    ),
    pytest.param(
        llvm.FloatType(), 3.14, "3.14", (3.14,),
        id="llvm.f64",
    ),
    pytest.param(
        llvm.PtrType(), (0,), None, None,
        id="llvm.ptr",
    ),
]

TOY_TYPES = [
    pytest.param(
        TensorType(shape=[3]),
        [1.0, 2.0, 3.0],
        "[1.0, 2.0, 3.0]",
        (1.0, 2.0, 3.0),
        id="toy.tensor_1d",
    ),
    pytest.param(
        TensorType(shape=[2, 3]),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]",
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        id="toy.tensor_2d",
    ),
]

AFFINE_TYPES = [
    pytest.param(
        MemRefType(shape=[2, 3]), (0,), None, None,
        id="affine.memref",
    ),
]

ALL_TYPES = BUILTIN_TYPES + LLVM_TYPES + TOY_TYPES + AFFINE_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_type(text: str) -> object:
    """Parse a type from ASM text, with all dialects registered."""
    parser = IRParser(text)
    for name in ("toy", "affine"):
        d = Dialect.get(name)
        for tname, tcls in d.types.items():
            parser._types[f"{name}.{tname}"] = tcls
    return parse_expr(parser)


def _identity_exe(ty):
    """Build and compile: main(x: ty) -> ty { return x }"""
    arg = BlockArgument(name="v0", type=ty)
    func = builtin.FuncOp(
        name="main",
        body=Block(ops=[builtin.ReturnOp(value=arg)], args=[arg]),
        type=builtin.Function(result=ty),
    )
    return compile_module(builtin.Module(functions=[func]))


# ---------------------------------------------------------------------------
# 0. Layout sanity: every non-void type has byte_size > 0
# ---------------------------------------------------------------------------

# Types that are genuinely void (no runtime representation):
# - Nil/VoidType: void return types
# - InferredShapeTensor: pre-inference placeholder, becomes TensorType
# - ShapeType: compile-time shape metadata
VOID_TYPES = [
    pytest.param(builtin.Nil(), id="builtin.nil"),
    pytest.param(llvm.VoidType(), id="llvm.void"),
    pytest.param(InferredShapeTensor(), id="toy.inferred_shape_tensor"),
    pytest.param(ShapeType(), id="affine.shape"),
]


@pytest.mark.parametrize("ty,_value,_asm,_expected", ALL_TYPES)
def test_layout_not_void(ty, _value, _asm, _expected):
    """Every non-void type must have a non-zero layout."""
    assert ty.__layout__.byte_size > 0


@pytest.mark.parametrize("ty", VOID_TYPES)
def test_layout_is_void(ty):
    """Genuinely void types must have zero-size layouts."""
    assert ty.__layout__.byte_size == 0


# ---------------------------------------------------------------------------
# 1. Type ASM round-trip (registered types only)
# ---------------------------------------------------------------------------

_ASM_TYPES = [
    pytest.param(builtin.IndexType(), id="builtin.index"),
    pytest.param(builtin.F64Type(), id="builtin.f64"),
    pytest.param(builtin.Nil(), id="builtin.nil"),
    pytest.param(builtin.String(), id="builtin.string"),
    pytest.param(builtin.List(element_type=builtin.F64Type()), id="builtin.list"),
    pytest.param(TensorType(shape=[3]), id="toy.tensor_1d"),
    pytest.param(TensorType(shape=[2, 3]), id="toy.tensor_2d"),
    pytest.param(InferredShapeTensor(), id="toy.inferred_shape_tensor"),
    pytest.param(ShapeType(), id="affine.shape"),
    pytest.param(MemRefType(shape=[2, 3]), id="affine.memref"),
]


@pytest.mark.parametrize("ty", _ASM_TYPES)
def test_type_asm_roundtrip(ty):
    text = type_asm(ty)
    parsed = _parse_type(text)
    assert parsed == ty


# ---------------------------------------------------------------------------
# 2. Memory from_value round-trip
# ---------------------------------------------------------------------------

# Types with Layout.parse() — can round-trip through from_value/from_asm
_PARSEABLE_TYPES = [
    pytest.param(builtin.IndexType(), 42, "42", (42,), id="builtin.index"),
    pytest.param(builtin.F64Type(), 3.14, "3.14", (3.14,), id="builtin.f64"),
    pytest.param(llvm.IntType(bits=64), 42, "42", (42,), id="llvm.i64"),
    pytest.param(llvm.FloatType(), 3.14, "3.14", (3.14,), id="llvm.f64"),
    pytest.param(
        TensorType(shape=[3]),
        [1.0, 2.0, 3.0], "[1.0, 2.0, 3.0]", (1.0, 2.0, 3.0),
        id="toy.tensor_1d",
    ),
    pytest.param(
        TensorType(shape=[2, 3]),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]",
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        id="toy.tensor_2d",
    ),
]


@pytest.mark.parametrize("ty,value,_asm,expected", _PARSEABLE_TYPES)
def test_from_value_roundtrip(ty, value, _asm, expected):
    mem = Memory.from_value(ty, value)
    assert mem.unpack() == expected


# ---------------------------------------------------------------------------
# 3. Memory from_asm round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ty,_value,asm_literal,expected", _PARSEABLE_TYPES)
def test_from_asm_roundtrip(ty, _value, asm_literal, expected):
    mem = Memory.from_asm(ty, asm_literal)
    assert mem.unpack() == expected


# ---------------------------------------------------------------------------
# 4. JIT identity round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ty,value,_asm,expected", _PARSEABLE_TYPES)
def test_jit_identity_roundtrip(ty, value, _asm, expected):
    exe = _identity_exe(ty)
    mem = Memory.from_value(ty, value)
    result = exe.run(mem)
    if exe.ctype._restype_ is ctypes.c_void_p:
        # Pointer type: verify the address survives the round-trip
        assert result == mem.address
    else:
        # Scalar type: verify the value survives
        assert result == expected[0]
