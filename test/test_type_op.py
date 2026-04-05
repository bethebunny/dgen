"""Tests for the builtin type op: op type(value) -> Type."""

from dgen import Block, asm
from dgen.asm.parser import parse_module
from dgen.codegen import Executable, LLVMCodegen
from dgen.testing import llvm_compile as compile_module
from dgen.compiler import Compiler
from dgen.dialects.builtin import TypeOp
from dgen.dialects.index import Index
from dgen.module import ConstantOp
from dgen.passes.algebra_to_llvm import AlgebraToLLVM
from dgen.passes.builtin_to_llvm import BuiltinToLLVM
from dgen.testing import assert_ir_equivalent, strip_prefix
from dgen.type import TypeType


def test_type_op_construction():
    """TypeOp takes a value and defaults to TypeType() result type."""
    x = ConstantOp(name="x", value=42, type=Index())
    t = TypeOp(name="t", value=x)
    assert isinstance(t.type, TypeType)
    assert t.value is x


def test_type_op_in_block_ops():
    """TypeOp participates in use-def: reachable from block.result."""
    x = ConstantOp(name="x", value=42, type=Index())
    t = TypeOp(name="t", value=x)
    block = Block(result=t)
    assert x in block.ops
    assert t in block.ops


def test_type_op_asm_roundtrip():
    """TypeOp round-trips through ASM print/parse."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %x : index.Index = 42
        |     %t : Type = type(%x)
    """)
    module = parse_module(ir)
    func = module.functions[0]
    type_op = func.body.result
    assert isinstance(type_op, TypeOp)
    assert isinstance(type_op.type, TypeType)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_type_op_result_is_type_dependency():
    """TypeOp's result can be used as a type for another op."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], ()> = function.function<Nil>() body():
        |     %x : index.Index = 42
        |     %t : Type = type(%x)
        |     %y : %t = 7
    """)
    module = parse_module(ir)
    ops = list(module.functions[0].body.ops)
    type_op = ops[1]
    y_op = ops[2]
    assert isinstance(type_op, TypeOp)
    # %y's type is the SSA value %t (the TypeOp)
    assert y_op.type is type_op


# ============================================================================
# End-to-end: JIT compilation and execution
# ============================================================================


def test_type_op_jit_returns_type():
    """type(x) returns the type of x as a TypeType value through JIT."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[], Type> = function.function<Type>() body():
        |     %x : index.Index = 42
        |     %t : Type = type(%x)
    """)
    module = parse_module(ir)
    exe = compile_module(module)
    result = exe.run()
    assert result.to_json() == {"tag": "index.Index"}


def test_type_op_jit_with_argument():
    """type(x) works when x is a function argument (type still known at compile time)."""
    ir = strip_prefix("""
        | import function
        | import index
        |
        | %main : function.Function<[index.Index], Type> = function.function<Type>() body(%x: index.Index):
        |     %t : Type = type(%x)
    """)
    module = parse_module(ir)
    exe = compile_module(module)
    result = exe.run(99)
    assert result.to_json() == {"tag": "index.Index"}


def test_type_op_jit_used_as_annotation():
    """type(x) result used as a type annotation for another value, end-to-end."""
    ir = strip_prefix("""
        | import algebra
        | import function
        | import index
        |
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %t : Type = type(%x)
        |     %y : %t = algebra.add(%x, %x)
    """)
    module = parse_module(ir)
    exe = compile_module(module)
    assert exe.run(21).to_json() == 42


# ============================================================================
# Staging short-circuit: pass-resolved constants skip JIT
# ============================================================================


def test_type_op_staging_short_circuit():
    """TypeOp resolved by BuiltinToLLVM during staging doesn't need JIT.

    type(%x) is used as the type annotation for %y — a staging boundary.
    Staging extracts the TypeOp subgraph, runs the pipeline (which includes
    BuiltinToLLVM, lowering TypeOp → ConstantOp). Full end-to-end test.
    """
    ir = strip_prefix("""
        | import algebra
        | import function
        | import index
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %x : index.Index = 21
        |     %t : Type = type(%x)
        |     %y : %t = algebra.add(%x, %x)
    """)
    module = parse_module(ir)
    compiler: Compiler[Executable] = Compiler(
        passes=[BuiltinToLLVM(), AlgebraToLLVM()],
        exit=LLVMCodegen(),
    )
    exe = compiler.compile_module(module)
    assert exe.run().to_json() == 42
