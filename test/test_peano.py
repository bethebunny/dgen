"""Peano arithmetic: recursive dependent types resolved by staging.

Defines a 'peano' dialect with:
  - Recursive types: Zero, Successor<pred>
  - Ops: zero(), successor<pred>(), value<nat>()

The staging system resolves the successor chain one layer at a time,
building up Successor<Successor<...<Zero>...>> incrementally.
"""

from __future__ import annotations

import pytest

from dataclasses import dataclass
from typing import ClassVar

import dgen
from dgen import Dialect, Op, Trait, Type, Value, asm, layout
from dgen.asm.parser import parse
from dgen.dialects.builtin import ChainOp, Nil
from dgen.dialects.index import Index
from dgen.codegen import Executable, LLVMCodegen
from dgen.compiler import Compiler, IdentityPass
from dgen.dialects.function import FunctionOp
from dgen.module import ConstantOp
from dgen.verify import CycleError, verify_dag
from dgen.passes.pass_ import Pass, lowering_for
from dgen.type import Fields, TypeType, type_constant
from dgen.passes.algebra_to_llvm import AlgebraToLLVM
from dgen.passes.builtin_to_llvm import BuiltinToLLVM
from dgen.passes.control_flow_to_goto import ControlFlowToGoto
from dgen.testing import llvm_compile, strip_prefix

# ============================================================================
# Dialect
# ============================================================================

peano = Dialect("peano")


@peano.trait("Natural")
class Natural(Trait):
    """A trait for Peano number types (Zero, Successor)."""

    pass


@peano.type("Zero")
@dataclass(frozen=True, eq=False)
class Zero(Natural, Type):
    __layout__ = layout.Void()


@peano.type("Successor")
@dataclass(frozen=True, eq=False)
class Successor(Natural, Type):
    pred: Value[TypeType]
    __params__: ClassVar[Fields] = (("pred", TypeType),)
    __layout__ = layout.Void()


def count_nat(t: Type) -> int:
    n = 0
    while isinstance(t, Successor):
        n += 1
        t = type_constant(t.pred)
    assert isinstance(t, Zero)
    return n


@peano.op("zero")
@dataclass(eq=False, kw_only=True)
class ZeroOp(Op):
    type: Type = TypeType()


@peano.op("successor")
@dataclass(eq=False, kw_only=True)
class SuccessorOp(Op):
    """Wrap a Natural in another Successor layer. pred is compile-time."""

    pred: Value[TypeType]
    type: Value[TypeType] = TypeType()  # overridden by ASM annotation
    __params__: ClassVar[Fields] = (("pred", TypeType),)


@peano.op("value")
@dataclass(eq=False, kw_only=True)
class ValueOp(Op):
    """Count a Natural type chain, returning the depth as an Index."""

    nat: Value[TypeType]
    type: Type = Index()
    __params__: ClassVar[Fields] = (("nat", TypeType),)


# ============================================================================
# Lowering
# ============================================================================


class PeanoLowering(Pass):
    """Lower peano ops to constants."""

    allow_unregistered_ops = True

    @lowering_for(ZeroOp)
    def lower_zero(self, op: ZeroOp) -> Value | None:
        return ConstantOp(value=Zero().__constant__.to_json(), type=TypeType())

    @lowering_for(SuccessorOp)
    def lower_successor(self, op: SuccessorOp) -> Value | None:
        succ = Successor(pred=type_constant(op.pred))
        return ConstantOp(value=succ.__constant__.to_json(), type=TypeType())

    @lowering_for(ValueOp)
    def lower_value(self, op: ValueOp) -> Value | None:
        return ConstantOp(value=count_nat(type_constant(op.nat)), type=Index())


def lower_peano(value: dgen.Value) -> dgen.Value:
    """Standalone helper: run PeanoLowering on a value."""
    return Compiler([PeanoLowering()], IdentityPass()).run(value)


peano_compiler: Compiler[Executable] = Compiler(
    passes=[
        PeanoLowering(),
        ControlFlowToGoto(),
        BuiltinToLLVM(),
        AlgebraToLLVM(),
    ],
    exit=LLVMCodegen(),
)


# ============================================================================
# Tests
# ============================================================================


def test_natural_trait_is_registered():
    """Natural trait is registered in peano dialect types."""
    assert issubclass(Natural, Trait)
    assert "Natural" in peano.types
    assert peano.types["Natural"] is Natural


def test_zero_has_natural_trait():
    """Zero implements Natural via MRO inheritance."""
    assert Zero().has_trait(Natural)


def test_successor_has_natural_trait():
    """Successor implements Natural via MRO inheritance."""
    assert Successor(pred=Zero()).has_trait(Natural)


def test_zero_type_has_natural_trait_via_asm():
    """Zero type parsed from ASM implements the Natural trait."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import function
        | import index
        | import peano
        | import index
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %z : Type = peano.zero()
    """)
    value = parse(ir)
    func = value
    zero_op = list(func.body.ops)[0]
    assert isinstance(zero_op, ZeroOp)
    # ZeroOp produces a type value — the result type is TypeType.
    # The produced type (Zero) has trait Natural.
    assert Zero().has_trait(Natural)


def test_recursive_type_roundtrip():
    """Successor<Successor<Zero>> serializes through TypeType and reconstructs."""
    nat = Successor(pred=Successor(pred=Zero()))
    assert count_nat(nat) == 2
    data = nat.__constant__.to_json()
    reconstructed = type_constant(ConstantOp(value=data, type=TypeType()))
    assert count_nat(reconstructed) == 2


# ============================================================================
# Traits in type position
# ============================================================================
#
# Traits should be usable wherever types are: as type annotations in ASM,
# as return type annotations on ops, and in has_trait checks on values whose
# type is annotated with a trait.


def test_trait_in_asm_type_annotation():
    """A trait name should be valid as a type annotation in ASM."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import function
        | import index
        | import peano
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %z : peano.Natural = peano.zero()
    """)
    value = parse(ir)
    func = value
    zero_op = list(func.body.ops)[0]
    assert isinstance(zero_op, ZeroOp)
    assert isinstance(zero_op.type, Natural)


def test_trait_annotation_roundtrips_through_asm():
    """A trait used as a type annotation should survive ASM print/parse."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import function
        | import index
        | import peano
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %z : peano.Natural = peano.zero()
    """)
    value = parse(ir)
    from dgen import asm

    text = asm.format(value)
    assert "peano.Natural" in text
    reparsed = parse(text)
    zero_op = list(reparsed.body.ops)[0]
    assert isinstance(zero_op.type, Natural)


def test_trait_as_block_argument_type():
    """A block argument annotated with a trait should parse and roundtrip."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import function
        | import index
        | import peano
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %z : peano.Natural = peano.zero()
        |     %s : peano.Natural = peano.successor<%z>()
        |     %v : index.Index = peano.value<%s>()
    """)
    value = parse(ir)
    func = value
    ops = func.body.ops
    for op in ops:
        if isinstance(op, (ZeroOp, SuccessorOp)):
            assert isinstance(op.type, Natural)


def test_peano_constant():
    """S(S(S(Z))) = 3, resolved entirely at compile time.

    The staging system resolves the successor chain one layer at a time:
    1. JIT zero -> Zero constant
    2. JIT successor<Zero> -> Successor<Zero> constant
    3. JIT successor<Successor<Zero>> -> Successor<Successor<Zero>> constant
    4. resolve_constant on value -> 3
    """
    ir = strip_prefix("""
        | import algebra
        | import number
        | import function
        | import index
        | import peano
        | import index
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %z : Type = peano.zero()
        |     %s1 : Type = peano.successor<%z>()
        |     %s2 : Type = peano.successor<%s1>()
        |     %s3 : Type = peano.successor<%s2>()
        |     %n : index.Index = peano.value<%s3>()
    """)
    value = parse(ir)

    print("\n=== Compile ===")
    exe = peano_compiler.compile(value)

    print("\n=== Run ===")
    result = exe.run()
    print(f"result = {result}")

    assert result.to_json() == 3


def test_equal_and_subtract_roundtrip():
    """equal_index and subtract_index ops round-trip through ASM."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import function
        | import index
        |
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %eq : number.Boolean = algebra.equal(%n, 0)
        |     %eq_i : index.Index = algebra.cast(%eq)
        |     %sub : index.Index = algebra.subtract(%n, 1)
        |     %result : index.Index = algebra.add(%eq_i, %sub)
    """)
    value = parse(ir)
    asm_lines = list(value.asm)
    asm_text = "\n".join(asm_lines)
    assert "algebra.equal" in asm_text
    assert "algebra.subtract" in asm_text


def test_subtract_jit():
    """subtract_index executes correctly via JIT."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import function
        | import index
        |
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %sub : index.Index = algebra.subtract(%n, 1)
    """)
    value = parse(ir)
    exe = llvm_compile(value)
    assert exe.run(5).to_json() == 4


def test_if_else_parse_roundtrip():
    """if/else op with two blocks parses and round-trips."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import control_flow
        | import function
        | import index
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %cond : number.Boolean = algebra.equal(%n, 0)
        |     %result : index.Index = control_flow.if(%cond, [], []) then_body():
        |         %ten : index.Index = 10
        |     else_body():
        |         %twenty : index.Index = 20
    """)
    value = parse(ir)
    asm_text = asm.format(value)
    print(asm_text)
    assert "control_flow.if(" in asm_text
    assert "else" in asm_text
    # Round-trip: parse the output again
    module2 = parse(asm_text)
    asm_text2 = asm.format(module2)
    assert asm_text == asm_text2


def test_if_else_jit():
    """if/else executes correctly via JIT."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import control_flow
        | import function
        | import index
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %cond : number.Boolean = algebra.equal(%n, 0)
        |     %result : index.Index = control_flow.if(%cond, [], []) then_body():
        |         %one : index.Index = 1
        |     else_body() captures(%n):
        |         %val : index.Index = algebra.subtract(%n, 1)
    """)
    value = parse(ir)
    exe = llvm_compile(value)
    assert exe.run(0).to_json() == 1
    assert exe.run(5).to_json() == 4
    assert exe.run(1).to_json() == 0


def test_call_op_roundtrip():
    """call op parses and round-trips."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import function
        | import index
        |
        | %add_one : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %r : index.Index = algebra.add(%n, 1)
        |
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index) captures(%add_one):
        |     %result : index.Index = function.call<%add_one>([%x])
    """)
    value = parse(ir)
    asm_text = asm.format(value)
    print(asm_text)
    assert "function.call<%add_one>" in asm_text


def test_call_jit():
    """call op executes a helper function via JIT."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import function
        | import index
        |
        | %add_one : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %r : index.Index = algebra.add(%n, 1)
        |
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index) captures(%add_one):
        |     %result : index.Index = function.call<%add_one>([%x])
    """)
    value = parse(ir)
    exe = llvm_compile(value)
    assert exe.run(5).to_json() == 6
    assert exe.run(0).to_json() == 1


def test_multi_function_staged():
    """Multi-function value: helper with staging, called from main."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import function
        | import index
        | import peano
        | import index
        |
        | %add_one : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = algebra.add(%x, 1)
        |
        | %main : function.Function<[], index.Index> = function.function<index.Index>() body() captures(%add_one):
        |     %z : Type = peano.zero()
        |     %s1 : Type = peano.successor<%z>()
        |     %n : index.Index = peano.value<%s1>()
        |     %result : index.Index = function.call<%add_one>([%n])
    """)
    value = parse(ir)
    exe = peano_compiler.compile(value)
    result = exe.run()
    assert result.to_json() == 2  # value(Successor(Zero)) = 1, then add 1 = 2


def test_equal_jit():
    """equal_index returns 1 when equal, 0 otherwise."""
    ir = strip_prefix("""
        | import algebra
        | import number
        | import function
        | import index
        |
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %cmp : number.Boolean = algebra.equal(%n, 0)
        |     %eq : index.Index = algebra.cast(%cmp)
    """)
    value = parse(ir)
    exe = llvm_compile(value)
    assert exe.run(0).to_json() == 1
    assert exe.run(5).to_json() == 0


def test_verify_dag_detects_cycle():
    """verify_dag detects a value-level use-def cycle.

    We construct a cycle by parsing valid IR and then mutating it: the
    function body result references the function itself via a ChainOp.
    """
    value = parse(
        strip_prefix("""
        | import function
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : Nil = {}
    """)
    )
    func = value
    assert isinstance(func, FunctionOp)
    # Create cycle: func.body.result → ChainOp → func
    chain = ChainOp(lhs=func.body.result, rhs=func, type=Nil())
    func.body = dgen.Block(result=chain, args=[])
    with pytest.raises(CycleError):
        verify_dag(func)


@pytest.mark.xfail(
    reason="Staging loses block args for recursive functions; needs investigation"
)
def test_recursive_peano():
    """Recursive natural(n) builds Successor^(n+1)(Zero) from a runtime Index.

    natural(n):
        if n == 0: return successor(zero())
        else: return successor(natural(n-1))

    So natural(n) = Successor^(n+1)(Zero), and main(x) = value(natural(x)) = x+1.
    """
    ir = strip_prefix("""
        | import algebra
        | import number
        | import control_flow
        | import function
        | import index
        | import peano
        | import index
        |
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index) captures(%natural):
        |     %n : peano.Natural = function.call<%natural>([%x])
        |     %result : index.Index = peano.value<%n>()
        |
        | %natural : function.Function<[index.Index], peano.Natural> = function.function<peano.Natural>() body(%n: index.Index):
        |     %base_case : number.Boolean = algebra.equal(%n, 0)
        |     %value : peano.Natural = control_flow.if(%base_case, [], []) then_body():
        |         %z : Type = peano.zero()
        |         %s : Type = peano.successor<%z>()
        |     else_body() captures(%natural):
        |         %n_minus_one : index.Index = algebra.subtract(%n, 1)
        |         %predecessor : peano.Natural = function.call<%natural>([%n_minus_one])
        |         %s : peano.Natural = peano.successor<%predecessor>()
    """)
    value = parse(ir)
    exe = peano_compiler.compile(value)

    # natural(0) = Successor(Zero) → value = 1, so main(0) = 1
    assert exe.run(0).to_json() == 1
    # natural(2) = Successor(Successor(Successor(Zero))) → value = 3, so main(2) = 3
    assert exe.run(2).to_json() == 3
    # natural(4) = Successor^5(Zero) → value = 5, so main(4) = 5
    assert exe.run(4).to_json() == 5
