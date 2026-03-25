"""Generic round-trip tests for Type subclasses.

Each type is tested for:
1. Type ASM round-trip (format → parse → equality)
2. Memory from_value round-trip (Python value → pack → unpack)
3. Memory from_asm round-trip (ASM literal → pack → unpack)
4. JIT identity round-trip (Memory → JIT function → result)
"""

from copy import deepcopy

import pytest

from dgen import Block, asm
from dgen.asm.formatting import type_asm
from dgen.asm.parser import ASMParser, parse_module, value_expression
from dgen.testing import assert_ir_equivalent
from dgen.block import BlockArgument
from dgen.codegen import Executable, LLVMCodegen, compile as compile_module
from dgen.compiler import Compiler
from dgen.dialects import builtin, function, llvm
from dgen.dialects.builtin import String
from dgen.dialects.function import Function, FunctionOp
from dgen.module import ConstantOp, Module, string_value
from dgen.type import Memory
from toy.dialects import shape_constant
from toy.dialects.memory import MemRef, Shape
from toy.dialects.toy import InferredShapeTensor, Tensor
from dgen.testing import strip_prefix

# ---------------------------------------------------------------------------
# Test data: (type, python_value, asm_literal, expected_unpack)
#
# Every type with a runtime representation should appear here.
# Types with genuinely void layouts (Nil, Void) are excluded.
# ---------------------------------------------------------------------------

BUILTIN_TYPES = [
    pytest.param(
        builtin.Index(),
        42,
        "42",
        (42,),
        id="builtin.index",
    ),
    pytest.param(
        builtin.F64(),
        3.14,
        "3.14",
        (3.14,),
        id="builtin.f64",
    ),
    # String and List use Span — tested separately below (not in from_value tests)
]

LLVM_TYPES = [
    pytest.param(
        llvm.Int(bits=builtin.Index().constant(64)),
        42,
        "42",
        (42,),
        id="llvm.i64",
    ),
    pytest.param(
        llvm.Float(),
        3.14,
        "3.14",
        (3.14,),
        id="llvm.f64",
    ),
    pytest.param(
        llvm.Ptr(),
        (0,),
        None,
        None,
        id="llvm.ptr",
    ),
]

TOY_TYPES = [
    pytest.param(
        Tensor(shape=shape_constant([3])),
        [1.0, 2.0, 3.0],
        "[1.0, 2.0, 3.0]",
        (1.0, 2.0, 3.0),
        id="toy.tensor_1d",
    ),
    pytest.param(
        Tensor(shape=shape_constant([2, 3])),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]",
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        id="toy.tensor_2d",
    ),
]

MEMORY_TYPES = [
    pytest.param(
        Shape(rank=builtin.Index().constant(2)),
        [2, 3],
        "[2, 3]",
        (2, 3),
        id="memory.shape",
    ),
    pytest.param(
        MemRef(shape=shape_constant([2, 3])),
        (0,),
        None,
        None,
        id="memory.memref",
    ),
]

ALL_TYPES = BUILTIN_TYPES + LLVM_TYPES + TOY_TYPES + MEMORY_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_type(text: str) -> object:
    """Parse a type from ASM text, with all dialects registered."""
    parser = ASMParser(text)
    return value_expression(parser)


def _identity_exe(ty):
    """Build and compile: main(x: ty) -> ty { return x }"""
    arg = BlockArgument(name="v0", type=ty)
    func = FunctionOp(
        name="main",
        body=Block(result=arg, args=[arg]),
        result=ty,
        type=Function(result=ty),
    )
    return compile_module(Module(ops=[func]))


# ---------------------------------------------------------------------------
# 0. Layout sanity: every non-void type has byte_size > 0
# ---------------------------------------------------------------------------

# Types that are genuinely void (no runtime representation):
# - Nil/Void: void return types
# - InferredShapeTensor: pre-inference placeholder, becomes Tensor
# - Shape(rank=builtin.Index().constant(0)): empty shape (zero dimensions)
VOID_TYPES = [
    pytest.param(builtin.Nil(), id="builtin.nil"),
    pytest.param(llvm.Void(), id="llvm.void"),
    pytest.param(InferredShapeTensor(), id="toy.inferred_shape_tensor"),
    pytest.param(Shape(rank=builtin.Index().constant(0)), id="memory.shape_0"),
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
    pytest.param(builtin.Index(), id="builtin.index"),
    pytest.param(builtin.F64(), id="builtin.f64"),
    pytest.param(builtin.Nil(), id="builtin.nil"),
    pytest.param(builtin.String(), id="builtin.string"),
    pytest.param(builtin.List(element_type=builtin.Index()), id="builtin.list_index"),
    pytest.param(builtin.List(element_type=builtin.F64()), id="builtin.list_f64"),
    pytest.param(Tensor(shape=shape_constant([3])), id="toy.tensor_1d"),
    pytest.param(Tensor(shape=shape_constant([2, 3])), id="toy.tensor_2d"),
    pytest.param(InferredShapeTensor(), id="toy.inferred_shape_tensor"),
    pytest.param(Shape(rank=builtin.Index().constant(0)), id="memory.shape_0"),
    pytest.param(Shape(rank=builtin.Index().constant(2)), id="memory.shape_2"),
    pytest.param(MemRef(shape=shape_constant([2, 3])), id="memory.memref"),
]


@pytest.mark.parametrize("ty", _ASM_TYPES)
def test_type_asm_roundtrip(ty):
    text = type_asm(ty)
    parsed = _parse_type(text)
    assert type_asm(parsed) == text


# ---------------------------------------------------------------------------
# 2. Memory from_value round-trip
# ---------------------------------------------------------------------------

# Types that can round-trip through from_value/from_asm
_PARSEABLE_TYPES = [
    pytest.param(builtin.Index(), 42, "42", (42,), id="builtin.index"),
    pytest.param(builtin.F64(), 3.14, "3.14", (3.14,), id="builtin.f64"),
    # String uses Span — tested separately (not via from_value)
    # Tensor uses Span — tested separately below (not via unpack)
    pytest.param(
        llvm.Int(bits=builtin.Index().constant(64)),
        42,
        "42",
        (42,),
        id="llvm.i64",
    ),
    pytest.param(llvm.Float(), 3.14, "3.14", (3.14,), id="llvm.f64"),
    pytest.param(
        Shape(rank=builtin.Index().constant(2)),
        [2, 3],
        "[2, 3]",
        (2, 3),
        id="memory.shape",
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
    assert result.to_json() == mem.to_json()


# ---------------------------------------------------------------------------
# 5. String staging tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. Tensor memory round-trips
# ---------------------------------------------------------------------------


def test_tensor_1d_from_value_roundtrip():
    """1D Tensor round-trips through Memory — to_json returns flat list."""
    ty = Tensor(shape=shape_constant([3]))
    mem = Memory.from_value(ty, [1.0, 2.0, 3.0])
    assert mem.to_json() == [1.0, 2.0, 3.0]


def test_tensor_2d_from_value_roundtrip():
    """2D Tensor round-trips through Memory — to_json returns flat list."""
    ty = Tensor(shape=shape_constant([2, 3]))
    mem = Memory.from_value(ty, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert mem.to_json() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_tensor_from_asm_roundtrip():
    """Tensor ASM literal round-trips through Memory."""
    ty = Tensor(shape=shape_constant([3]))
    mem = Memory.from_asm(ty, "[1.0, 2.0, 3.0]")
    assert mem.to_json() == [1.0, 2.0, 3.0]


def test_tensor_layout_is_fatpointer():
    """Tensor layout is Span(F64) — 16 bytes."""
    from dgen.layout import Span

    ty = Tensor(shape=shape_constant([3]))
    assert isinstance(ty.__layout__, Span)
    assert ty.__layout__.byte_size == 16


# ---------------------------------------------------------------------------
# 5. List JIT round-trips
# ---------------------------------------------------------------------------


def test_list_jit_identity():
    """Pass List<index> through identity JIT function."""
    ty = builtin.List(element_type=builtin.Index())
    mem = Memory.from_value(ty, [10, 20, 30])
    exe = _identity_exe(ty)
    result = exe.run(mem)
    assert result.to_json() == mem.to_json()


def test_list_constant_jit_return():
    """JIT function returns a list constant: main() -> List<index>."""
    list_type = builtin.List(element_type=builtin.Index())
    const = ConstantOp(value=[3, 5, 7], type=list_type)
    func = FunctionOp(
        name="main",
        body=Block(result=const, args=[]),
        result=list_type,
        type=Function(result=list_type),
    )
    exe = compile_module(Module(ops=[func]))
    result = exe.run()
    assert result.to_json() == [3, 5, 7]


# ---------------------------------------------------------------------------
# 6. Nested type Memory round-trips
# ---------------------------------------------------------------------------


def test_list_of_lists_memory_roundtrip():
    """List<List<index>> round-trips through Memory."""
    inner = builtin.List(element_type=builtin.Index())
    outer = builtin.List(element_type=inner)
    mem = Memory.from_value(outer, [[1, 2], [3, 4, 5]])
    assert mem.to_json() == [[1, 2], [3, 4, 5]]


def test_list_of_strings_memory_roundtrip():
    """List<String> round-trips through Memory."""
    ty = builtin.List(element_type=builtin.String())
    mem = Memory.from_value(ty, ["hello", "world"])
    assert mem.to_json() == ["hello", "world"]


# ---------------------------------------------------------------------------
# 7. Mixed literals and SSA values in list ASM
# ---------------------------------------------------------------------------


def test_packop_mixed_constants_and_refs(ir_snapshot):
    """Parser handles [literal, %ref, literal] by creating ConstantOps."""
    ir_input = strip_prefix("""
        | import function
        |
        | %main : function.Function<()> = function.function<Nil>() body(%x: Index):
        |     %store : Nil = memory.store(%x, %x, [3, %x, 5])
    """)
    parsed = parse_module(ir_input)
    assert parsed == ir_snapshot


# ---------------------------------------------------------------------------
# 8. String tests
# ---------------------------------------------------------------------------


def test_string_constant_roundtrip():
    """String().constant creates a Constant[String], string_value extracts it."""
    c = String().constant("hello")
    assert string_value(c) == "hello"


def test_string_constant_different_lengths():
    """String type is the same regardless of value length (Span)."""
    s3 = builtin.String()
    s5 = builtin.String()
    assert type(s3) is type(s5)  # Same type — length is in the value, not the type
    assert s3.__layout__.byte_size == 16  # Span is always 16 bytes


def test_string_as_op_param():
    """String constants work as __params__ on ops — ASM round-trip."""
    ir = strip_prefix("""
        | import function
        | import llvm
        |
        | %main : function.Function<()> = function.function<Nil>() body():
        |     %0 : Index = 1
        |     %1 : Index = 2
        |     %cmp : Nil = llvm.icmp<"slt">(%0, %1)
    """)
    parsed = parse_module(ir)
    assert_ir_equivalent(parsed, asm.parse(asm.format(parsed)))


def test_string_jit_identity():
    """String value survives JIT identity function as a pointer."""
    ty = builtin.String()
    c = ty.constant("world")
    mem = c.__constant__
    exe = _identity_exe(ty)
    result = exe.run(mem)
    assert result.to_json() == mem.to_json()


def test_string_param_staging():
    """String function parameter used as IcmpOp predicate via staging.

    The staging system JIT-evaluates the `pred` parameter to resolve it
    from a BlockArgument to a Constant[String] before final codegen.
    The String value directly controls the generated comparison.
    """
    ir = strip_prefix("""
        | import function
        | import llvm
        |
        | %main : function.Function<Index> = function.function<Index>() body(%pred : String, %x : Index, %y : Index):
        |     %cmp : Nil = llvm.icmp<%pred>(%x, %y)
        |     %ext : Nil = llvm.zext(%cmp)
    """)
    module = parse_module(ir)

    identity_compiler: Compiler[Executable] = Compiler(passes=[], exit=LLVMCodegen())

    # "slt": 5 < 10 = true → 1
    exe = identity_compiler.compile(module)
    assert exe.run("slt", 5, 10).to_json() == 1

    # "sge": 5 >= 10 = false → 0
    exe = identity_compiler.compile(module)
    assert exe.run("sge", 5, 10).to_json() == 0


def test_compile_once_run_twice():
    """Compile once with runtime-dependent __params__, run with different args.

    The staging system builds a callback-based executable: the compiled code
    calls a host callback that JIT-compiles stage-2 with the resolved values.
    This enables compile-once, run-many with different runtime inputs.
    """
    ir = strip_prefix("""
        | import function
        | import llvm
        |
        | %main : function.Function<Index> = function.function<Index>() body(%pred : String, %x : Index, %y : Index):
        |     %cmp : Nil = llvm.icmp<%pred>(%x, %y)
        |     %ext : Nil = llvm.zext(%cmp)
    """)
    module = parse_module(ir)

    identity_compiler: Compiler[Executable] = Compiler(passes=[], exit=LLVMCodegen())

    exe = identity_compiler.compile(module)

    # Same executable, different arguments — each run JITs a specialized stage-2
    assert exe.run("slt", 5, 10).to_json() == 1  # 5 < 10 = true → 1
    assert exe.run("sge", 5, 10).to_json() == 0  # 5 >= 10 = false → 0
    assert exe.run("sgt", 10, 5).to_json() == 1  # 10 > 5 = true → 1
    assert exe.run("sle", 7, 7).to_json() == 1  # 7 <= 7 = true → 1
    assert exe.run("slt", 7, 7).to_json() == 0  # 7 < 7 = false → 0


# ---------------------------------------------------------------------------
# 10. Deepcopy correctness for pointer types
# ---------------------------------------------------------------------------


def test_deepcopy_string_constant():
    """Deepcopy of a String constant produces a working copy.

    Origins are shared (not deep-copied), so the copied buffer's pointers
    still reference valid backing data.
    """
    ty = builtin.String()
    mem = Memory.from_value(ty, "hello")
    copied = deepcopy(mem)
    assert copied.to_json() == "hello"
    # Origins are shared, not copied
    assert copied.origins is mem.origins


def test_deepcopy_list_of_strings():
    """Deepcopy of List<String> constant preserves nested pointer validity."""
    ty = builtin.List(element_type=builtin.String())
    mem = Memory.from_value(ty, ["hello", "world"])
    copied = deepcopy(mem)
    assert copied.to_json() == ["hello", "world"]


def test_deepcopy_list_of_lists():
    """Deepcopy of List<List<index>> constant preserves nested pointer validity."""
    inner = builtin.List(element_type=builtin.Index())
    outer = builtin.List(element_type=inner)
    mem = Memory.from_value(outer, [[1, 2], [3, 4, 5]])
    copied = deepcopy(mem)
    assert copied.to_json() == [[1, 2], [3, 4, 5]]


def test_deepcopy_module_with_list_constant():
    """Deepcopy an entire module containing a List constant.

    This is what staging does — deepcopy the module, then JIT subgraphs.
    The list constant's backing data must survive the copy.
    """
    list_type = builtin.List(element_type=builtin.Index())
    const = ConstantOp(value=[3, 5, 7], type=list_type)
    func = FunctionOp(
        name="main",
        body=Block(result=const, args=[]),
        result=list_type,
        type=Function(result=list_type),
    )
    module = Module(ops=[func])
    copied = deepcopy(module)

    # The copied module's constant should still be readable
    copied_const = copied.functions[0].body.ops[0]
    assert isinstance(copied_const, ConstantOp)
    assert copied_const.memory.to_json() == [3, 5, 7]


# ---------------------------------------------------------------------------
# 11. Value equality for pointer types
#
# TODO: Memory.__eq__ compares buffers byte-for-byte, which means two
# Span memories created from the same Python value compare as NOT
# equal (different heap pointers in the buffer). This breaks constant
# deduplication and makes equality checks unreliable for pointer types.
# Options: (a) define equality via to_json() for pointer layouts,
# (b) normalize pointer buffers, (c) content-hash the origins.
# ---------------------------------------------------------------------------


def test_memory_equality_scalars():
    """Scalar Memory equality works — same value, same buffer."""
    a = Memory.from_value(builtin.Index(), 42)
    b = Memory.from_value(builtin.Index(), 42)
    assert a == b


def test_memory_equality_strings():
    """Two String memories from the same Python value should be equal.

    Currently FAILS: different heap allocations → different pointer bytes
    in the buffer → not equal.
    """
    a = Memory.from_value(builtin.String(), "hello")
    b = Memory.from_value(builtin.String(), "hello")
    assert a == b


def test_memory_equality_lists():
    """Two List memories from the same Python value should be equal."""
    ty = builtin.List(element_type=builtin.Index())
    a = Memory.from_value(ty, [1, 2, 3])
    b = Memory.from_value(ty, [1, 2, 3])
    assert a == b


# ---------------------------------------------------------------------------
# 12. Edge cases
# ---------------------------------------------------------------------------


def test_empty_list():
    """Empty list round-trips through Memory."""
    ty = builtin.List(element_type=builtin.Index())
    mem = Memory.from_value(ty, [])
    assert mem.to_json() == []


def test_single_element_list():
    """Single-element list round-trips through Memory."""
    ty = builtin.List(element_type=builtin.Index())
    mem = Memory.from_value(ty, [42])
    assert mem.to_json() == [42]


def test_list_of_f64():
    """List<F64> round-trips through Memory (float variant of Span)."""
    ty = builtin.List(element_type=builtin.F64())
    mem = Memory.from_value(ty, [1.5, 2.5, 3.5])
    assert mem.to_json() == [1.5, 2.5, 3.5]


def test_three_level_nesting():
    """List<List<List<index>>> — 3 levels of Span nesting."""
    l1 = builtin.List(element_type=builtin.Index())
    l2 = builtin.List(element_type=l1)
    l3 = builtin.List(element_type=l2)
    mem = Memory.from_value(l3, [[[1, 2], [3]], [[4, 5, 6]]])
    assert mem.to_json() == [[[1, 2], [3]], [[4, 5, 6]]]


def test_empty_string():
    """Empty string round-trips through Memory."""
    mem = Memory.from_value(builtin.String(), "")
    assert mem.to_json() == ""


def test_list_of_empty_lists():
    """List of empty lists round-trips through Memory."""
    inner = builtin.List(element_type=builtin.Index())
    outer = builtin.List(element_type=inner)
    mem = Memory.from_value(outer, [[], [], []])
    assert mem.to_json() == [[], [], []]


# ---------------------------------------------------------------------------
# 13. Full pipeline: list value through JIT
# ---------------------------------------------------------------------------


def test_list_identity_jit_roundtrip_full():
    """Pass List<index>, read back via from_raw — full data integrity check."""
    list_type = builtin.List(element_type=builtin.Index())
    mem = Memory.from_value(list_type, [10, 20, 30])
    exe = _identity_exe(list_type)
    result = exe.run(mem)
    assert result.to_json() == [10, 20, 30]


def test_list_f64_jit_roundtrip():
    """Pass List<F64> through JIT identity and read back."""
    list_type = builtin.List(element_type=builtin.F64())
    mem = Memory.from_value(list_type, [1.1, 2.2, 3.3])
    exe = _identity_exe(list_type)
    result = exe.run(mem)
    assert result.to_json() == [1.1, 2.2, 3.3]
