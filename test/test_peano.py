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
from dgen import Dialect, Op, Type, Value, layout
from dgen import codegen
from dgen.asm.formatting import type_asm
from dgen.asm.parser import parse_module
from dgen.dialects import builtin, function
from dgen.dialects.builtin import ChainOp, Index, Nil
from dgen.codegen import LLVMCodegen
from dgen.compiler import Compiler
from dgen.dialects.function import DefineOp
from dgen.module import ConstantOp, Module, PackOp
from dgen.verify import CycleError, verify_dag
from dgen.passes.pass_ import Pass
from dgen.type import Fields, TypeType, type_constant
from dgen.testing import strip_prefix

# ============================================================================
# Dialect
# ============================================================================

peano = Dialect("peano")


@peano.trait("Natural")
@dataclass(frozen=True, eq=False)
class Natural(TypeType):
    """A TypeType constrained to Zero or Successor."""

    pass


@peano.type("Zero")
@dataclass(frozen=True, eq=False)
class Zero(Type):
    __traits__: ClassVar[tuple[type, ...]] = (Natural,)
    __layout__ = layout.Void()


@peano.type("Successor")
@dataclass(frozen=True, eq=False)
class Successor(Type):
    pred: Value[TypeType]
    __params__: ClassVar[Fields] = (("pred", TypeType),)
    __traits__: ClassVar[tuple[type, ...]] = (Natural,)
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
                new_values = [replacements.get(a, a) for a in op.arguments.values]
                new_pack = PackOp(values=new_values, type=op.arguments.type)
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


def _lower_peano_func(func: builtin.DefineOp) -> None:
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
    """Natural trait is a TypeType subclass registered in peano dialect."""
    assert issubclass(Natural, TypeType)
    assert "Natural" in peano.types
    assert peano.types["Natural"] is Natural


def test_natural_is_typetype():
    """Natural() is a TypeType subclass."""
    nat = Natural()
    assert isinstance(nat, TypeType)


def test_zero_has_natural_trait():
    """Zero declares Natural in its __traits__."""
    assert Natural in Zero.__traits__


def test_successor_has_natural_trait():
    """Successor declares Natural in its __traits__."""
    assert Natural in Successor.__traits__


def test_natural_in_asm():
    """peano.Natural parses as a type annotation."""
    ir = strip_prefix("""
        | import function
        | import peano
        |
        | %main : Nil = function.define<Index>() body():
        |     %z : peano.Natural = peano.zero()
    """)
    module = parse_module(ir)
    func = module.functions[0]
    zero_op = func.body.ops[0]
    assert isinstance(zero_op.type, Natural)


def test_recursive_type_roundtrip():
    """Successor<Successor<Zero>> serializes through TypeType and reconstructs."""
    nat = Successor(pred=Successor(pred=Zero()))
    assert count_nat(nat) == 2
    data = nat.__constant__.to_json()
    reconstructed = type_constant(ConstantOp(value=data, type=TypeType()))
    assert count_nat(reconstructed) == 2


def test_peano_constant():
    """S(S(S(Z))) = 3, resolved entirely at compile time.

    The staging system resolves the successor chain one layer at a time:
    1. JIT zero -> Zero constant
    2. JIT successor<Zero> -> Successor<Zero> constant
    3. JIT successor<Successor<Zero>> -> Successor<Successor<Zero>> constant
    4. resolve_constant on value -> 3
    """
    ir = strip_prefix("""
        | import function
        | import peano
        |
        | %main : Nil = function.define<Index>() body():
        |     %z : Type = peano.zero()
        |     %s1 : Type = peano.successor<%z>()
        |     %s2 : Type = peano.successor<%s1>()
        |     %s3 : Type = peano.successor<%s2>()
        |     %n : Index = peano.value<%s3>()
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
        | import function
        |
        | %main : Nil = function.define<Index>() body(%n: Index):
        |     %eq : Index = equal_index(%n, 0)
        |     %sub : Index = subtract_index(%n, 1)
        |     %result : Index = add_index(%eq, %sub)
    """)
    module = parse_module(ir)
    asm_lines = list(module.asm)
    asm_text = "\n".join(asm_lines)
    assert "equal_index" in asm_text
    assert "subtract_index" in asm_text


def test_subtract_jit():
    """subtract_index executes correctly via JIT."""
    ir = strip_prefix("""
        | import function
        |
        | %main : Nil = function.define<Index>() body(%n: Index):
        |     %sub : Index = subtract_index(%n, 1)
    """)
    module = parse_module(ir)
    exe = codegen.compile(module)
    assert exe.run(5).to_json() == 4


def test_if_else_parse_roundtrip():
    """if/else op with two blocks parses and round-trips."""
    ir = strip_prefix("""
        | import control_flow
        | import function
        | %main : Nil = function.define<Index>() body(%n: Index):
        |     %cond : Index = equal_index(%n, 0)
        |     %result : Index = control_flow.if(%cond, [], []) then_body():
        |         %ten : Index = 10
        |     else_body():
        |         %twenty : Index = 20
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
        | import control_flow
        | import function
        | %main : Nil = function.define<Index>() body(%n: Index):
        |     %cond : Index = equal_index(%n, 0)
        |     %result : Index = control_flow.if(%cond, [], []) then_body():
        |         %one : Index = 1
        |     else_body():
        |         %val : Index = subtract_index(%n, 1)
    """)
    module = parse_module(ir)
    exe = codegen.compile(module)
    assert exe.run(0).to_json() == 1
    assert exe.run(5).to_json() == 4
    assert exe.run(1).to_json() == 0


def test_call_op_roundtrip():
    """call op parses and round-trips."""
    ir = strip_prefix("""
        | import function
        |
        | %main : Nil = function.define<Index>() body(%x: Index):
        |     %result : Index = function.call<%add_one>([%x])
        |
        | %add_one : Nil = function.define<Index>() body(%n: Index):
        |     %r : Index = add_index(%n, 1)
    """)
    module = parse_module(ir)
    asm_text = "\n".join(module.asm)
    print(asm_text)
    assert "function.call<%add_one>" in asm_text


def test_call_jit():
    """call op executes a helper function via JIT."""
    ir = strip_prefix("""
        | import function
        |
        | %main : Nil = function.define<Index>() body(%x: Index):
        |     %result : Index = function.call<%add_one>([%x])
        |
        | %add_one : Nil = function.define<Index>() body(%n: Index):
        |     %r : Index = add_index(%n, 1)
    """)
    module = parse_module(ir)
    exe = codegen.compile(module)
    assert exe.run(5).to_json() == 6
    assert exe.run(0).to_json() == 1


def test_multi_function_staged():
    """Multi-function module: helper with staging, called from main."""
    ir = strip_prefix("""
        | import function
        | import peano
        |
        | %main : Nil = function.define<Index>() body():
        |     %z : Type = peano.zero()
        |     %s1 : Type = peano.successor<%z>()
        |     %n : Index = peano.value<%s1>()
        |     %result : Index = function.call<%add_one>([%n])
        |
        | %add_one : Nil = function.define<Index>() body(%x: Index):
        |     %r : Index = add_index(%x, 1)
    """)
    module = parse_module(ir)
    exe = peano_compiler.compile(module)
    result = exe.run()
    assert result.to_json() == 2  # value(Successor(Zero)) = 1, then add 1 = 2


def test_equal_jit():
    """equal_index returns 1 when equal, 0 otherwise."""
    ir = strip_prefix("""
        | import function
        |
        | %main : Nil = function.define<Index>() body(%n: Index):
        |     %eq : Index = equal_index(%n, 0)
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
        |
        | %f : Nil = function.define<Nil>() body():
        |     %0 : Nil = {}
    """)
    )
    func = module.ops[0]
    assert isinstance(func, DefineOp)
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
        | import control_flow
        | import function
        | import peano
        |
        | %main : Nil = function.define<Index>() body(%x: Index):
        |     %n : peano.Natural = function.call<%natural>([%x])
        |     %result : Index = peano.value<%n>()
        |
        | %natural : Nil = function.define<peano.Natural>() body(%n: Index):
        |     %base_case : Index = equal_index(%n, 0)
        |     %value : peano.Natural = control_flow.if(%base_case, [], []) then_body():
        |         %z : Type = peano.zero()
        |         %s : Type = peano.successor<%z>()
        |     else_body():
        |         %n_minus_one : Index = subtract_index(%n, 1)
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
