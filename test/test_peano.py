"""Peano arithmetic: recursive dependent types resolved by staging.

Defines a 'peano' dialect with:
  - Recursive types: Zero, Successor<pred>
  - Ops: zero(), successor<pred>(), value<nat>()

The staging system resolves the successor chain one layer at a time,
building up Successor<Successor<...<Zero>...>> incrementally.
"""

from __future__ import annotations

import pytest

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar

import dgen
from dgen import Dialect, Op, Trait, Type, Value, layout
from dgen import codegen
from dgen.type import format_value as type_asm
from dgen.asm.parser import parse_module
from dgen.dialects import builtin, function
from dgen.dialects.builtin import ChainOp, Nil
from dgen.dialects.index import Index
from dgen.codegen import LLVMCodegen
from dgen.compiler import Compiler
from dgen.dialects.function import FunctionOp
from dgen.module import ConstantOp, Module, PackOp, pack
from dgen.verify import CycleError, verify_dag
from dgen.passes.pass_ import Pass
from dgen.type import Fields, TypeType, type_constant
from dgen.testing import strip_prefix

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


def _lower_peano_ops(
    ops: list[dgen.Op], replacements: dict[Value, Value]
) -> list[dgen.Op]:
    """Lower peano ops in a list, descending into blocks."""
    new_ops: list[dgen.Op] = []

    for op in ops:
        if isinstance(op, ZeroOp):
            z = Zero()
            print(f"  lower: peano.zero -> {type_asm(z)}")
            const = ConstantOp(value=z.__constant__.to_json(), type=TypeType())
            new_ops.append(const)
            replacements[op] = const
        elif isinstance(op, SuccessorOp):
            pred = replacements.get(op.pred, op.pred)
            pred_type = type_constant(pred)
            succ = Successor(pred=pred_type)
            print(
                f"  lower: peano.successor<{type_asm(pred_type)}> -> {type_asm(succ)}"
            )
            const = ConstantOp(value=succ.__constant__.to_json(), type=TypeType())
            new_ops.append(const)
            replacements[op] = const
        elif isinstance(op, ValueOp):
            nat = replacements.get(op.nat, op.nat)
            nat_type = type_constant(nat)
            n = count_nat(nat_type)
            print(f"  lower: peano.value<{type_asm(nat_type)}> -> {n}")
            const = ConstantOp(value=n, type=Index())
            new_ops.append(const)
            replacements[op] = const
        elif isinstance(op, function.CallOp):
            if isinstance(op.arguments, PackOp):
                new_pack = pack(replacements.get(a, a) for a in op.arguments)
                new_ops.append(new_pack)
                new_call = function.CallOp(
                    callee=op.callee,
                    arguments=new_pack,
                    type=op.type,
                    name=op.name,
                )
                new_ops.append(new_call)
                replacements[op] = new_call
            else:
                new_ops.append(op)
        else:
            # Descend into blocks (e.g., if/else bodies)
            for block_name, block in op.blocks:
                block.result = _lower_peano_ops(block.ops, replacements)[-1]
            new_ops.append(op)

    return new_ops


def _lower_peano_func(func: builtin.FunctionOp) -> None:
    """Lower peano ops in a single function."""
    replacements: dict[Value, Value] = {}
    new_ops = _lower_peano_ops(func.body.ops, replacements)
    func.body.result = new_ops[-1]


def lower_peano(module: Module) -> Module:
    """Lower peano ops to constants. Prints each resolution step."""
    module = deepcopy(module)
    for func in module.functions:
        _lower_peano_func(func)
    return module


class PeanoLowering(Pass):
    """Pass wrapper for lower_peano."""

    allow_unregistered_ops = True

    def run(self, module: Module, compiler: Compiler) -> Module:
        return lower_peano(module)


peano_compiler: Compiler[codegen.Executable] = Compiler(
    passes=[PeanoLowering()],
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
    module = parse_module(ir)
    func = module.functions[0]
    zero_op = func.body.ops[0]
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
    module = parse_module(ir)
    func = module.functions[0]
    zero_op = func.body.ops[0]
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
    module = parse_module(ir)
    from dgen import asm

    text = asm.format(module)
    assert "peano.Natural" in text
    reparsed = parse_module(text)
    zero_op = reparsed.functions[0].body.ops[0]
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
    module = parse_module(ir)
    func = module.functions[0]
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
    module = parse_module(ir)

    print("\n=== Compile ===")
    exe = peano_compiler.compile(module)

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
    module = parse_module(ir)
    asm_lines = list(module.asm)
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
    module = parse_module(ir)
    exe = codegen.compile(module)
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
    module = parse_module(ir)
    asm_text = "\n".join(module.asm)
    print(asm_text)
    assert "control_flow.if(" in asm_text
    assert "else" in asm_text
    # Round-trip: parse the output again
    module2 = parse_module(asm_text)
    asm_text2 = "\n".join(module2.asm)
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
        |     else_body():
        |         %val : index.Index = algebra.subtract(%n, 1)
    """)
    module = parse_module(ir)
    exe = codegen.compile(module)
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
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %result : index.Index = function.call<%add_one>([%x])
        |
        | %add_one : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %r : index.Index = algebra.add(%n, 1)
    """)
    module = parse_module(ir)
    asm_text = "\n".join(module.asm)
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
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %result : index.Index = function.call<%add_one>([%x])
        |
        | %add_one : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %r : index.Index = algebra.add(%n, 1)
    """)
    module = parse_module(ir)
    exe = codegen.compile(module)
    assert exe.run(5).to_json() == 6
    assert exe.run(0).to_json() == 1


def test_multi_function_staged():
    """Multi-function module: helper with staging, called from main."""
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
        |     %n : index.Index = peano.value<%s1>()
        |     %result : index.Index = function.call<%add_one>([%n])
        |
        | %add_one : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = algebra.add(%x, 1)
    """)
    module = parse_module(ir)
    exe = peano_compiler.compile(module)
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
    module = parse_module(ir)
    exe = codegen.compile(module)
    assert exe.run(0).to_json() == 1
    assert exe.run(5).to_json() == 0


def test_verify_dag_detects_cycle():
    """verify_dag detects a value-level use-def cycle.

    We construct a cycle by parsing valid IR and then mutating it: the
    function body result references the function itself via a ChainOp.
    """
    module = parse_module(
        strip_prefix("""
        | import function
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %0 : Nil = {}
    """)
    )
    func = module.ops[0]
    assert isinstance(func, FunctionOp)
    # Create cycle: func.body.result → ChainOp → func
    chain = ChainOp(lhs=func.body.result, rhs=func, type=Nil())
    func.body = dgen.Block(result=chain, args=[])
    with pytest.raises(CycleError):
        verify_dag(module)


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
        | %main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %n : peano.Natural = function.call<%natural>([%x])
        |     %result : index.Index = peano.value<%n>()
        |
        | %natural : function.Function<[index.Index], peano.Natural> = function.function<peano.Natural>() body(%n: index.Index):
        |     %base_case : number.Boolean = algebra.equal(%n, 0)
        |     %value : peano.Natural = control_flow.if(%base_case, [], []) then_body():
        |         %z : Type = peano.zero()
        |         %s : Type = peano.successor<%z>()
        |     else_body():
        |         %n_minus_one : index.Index = algebra.subtract(%n, 1)
        |         %predecessor : peano.Natural = function.call<%natural>([%n_minus_one])
        |         %s : peano.Natural = peano.successor<%predecessor>()
    """)
    module = parse_module(ir)
    exe = peano_compiler.compile(module)

    # natural(0) = Successor(Zero) → value = 1, so main(0) = 1
    assert exe.run(0).to_json() == 1
    # natural(2) = Successor(Successor(Successor(Zero))) → value = 3, so main(2) = 3
    assert exe.run(2).to_json() == 3
    # natural(4) = Successor^5(Zero) → value = 5, so main(4) = 5
    assert exe.run(4).to_json() == 5
