"""Peano arithmetic: recursive dependent types resolved by staging.

Defines a 'peano' dialect with:
  - Recursive types: Zero, Successor<pred>
  - Ops: zero(), successor<pred>(), value<nat>()

The staging system resolves the successor chain one layer at a time,
building up Successor<Successor<...<Zero>...>> incrementally.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar

import dgen
from dgen import Dialect, Op, Type, Value, layout
from dgen.asm.formatting import type_asm
from dgen.asm.parser import parse_module
from dgen.dialects import builtin
from dgen.dialects.builtin import Index
from dgen.module import ConstantOp, Module
from dgen.staging import compile_staged
from dgen.type import Fields, TypeType, type_constant
from toy.test.helpers import strip_prefix

# ============================================================================
# Dialect
# ============================================================================

peano = Dialect("peano")


@peano.type("Zero")
@dataclass(frozen=True)
class Zero(Type):
    __layout__ = layout.Void()


@peano.type("Successor")
@dataclass(frozen=True)
class Successor(Type):
    pred: Value[TypeType]
    __params__: ClassVar[Fields] = (("pred", Type),)
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
    type: Type = TypeType(concrete=Zero())


@peano.op("successor")
@dataclass(eq=False, kw_only=True)
class SuccessorOp(Op):
    """Wrap a Natural in another Successor layer. pred is compile-time."""

    pred: Value[TypeType]
    type: Value[TypeType] = TypeType(concrete=Zero())  # overridden by ASM annotation
    __params__: ClassVar[Fields] = (("pred", Type),)


@peano.op("value")
@dataclass(eq=False, kw_only=True)
class ValueOp(Op):
    """Count a Natural type chain, returning the depth as an Index."""

    nat: Value[TypeType]
    type: Type = Index()
    __params__: ClassVar[Fields] = (("nat", Type),)


# ============================================================================
# Lowering
# ============================================================================


def lower_peano(module: Module) -> Module:
    """Lower peano ops to constants. Prints each resolution step."""
    module = deepcopy(module)
    func = module.functions[0]
    replacements: dict[int, Value] = {}
    new_ops: list[dgen.Op] = []

    for op in func.body.ops:
        if isinstance(op, ZeroOp):
            z = Zero()
            print(f"  lower: peano.zero -> {type_asm(z)}")
            const = ConstantOp(
                value=z.__constant__.to_json(), type=TypeType(concrete=z)
            )
            new_ops.append(const)
            replacements[id(op)] = const
        elif isinstance(op, SuccessorOp):
            pred = replacements.get(id(op.pred), op.pred)
            pred_type = type_constant(pred)
            succ = Successor(pred=pred_type)
            print(
                f"  lower: peano.successor<{type_asm(pred_type)}> -> {type_asm(succ)}"
            )
            const = ConstantOp(
                value=succ.__constant__.to_json(), type=TypeType(concrete=succ)
            )
            new_ops.append(const)
            replacements[id(op)] = const
        elif isinstance(op, ValueOp):
            nat = replacements.get(id(op.nat), op.nat)
            nat_type = type_constant(nat)
            n = count_nat(nat_type)
            print(f"  lower: peano.value<{type_asm(nat_type)}> -> {n}")
            const = ConstantOp(value=n, type=Index())
            new_ops.append(const)
            replacements[id(op)] = const
        elif isinstance(op, builtin.ReturnOp):
            val = replacements.get(id(op.value), op.value)
            new_ops.append(builtin.ReturnOp(value=val, type=op.type))
        else:
            new_ops.append(op)

    func.body.ops = new_ops
    return module


# ============================================================================
# Tests
# ============================================================================


def test_recursive_type_roundtrip():
    """Successor<Successor<Zero>> serializes through TypeType and reconstructs."""
    nat = Successor(pred=Successor(pred=Zero()))
    assert count_nat(nat) == 2
    data = nat.__constant__.to_json()
    reconstructed = type_constant(ConstantOp(value=data, type=TypeType(concrete=nat)))
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
        | import peano
        |
        | %main : Nil = function<Index>() ():
        |     %z : TypeType<peano.Zero> = peano.zero()
        |     %s1 : TypeType<peano.Successor<peano.Zero>> = peano.successor<%z>()
        |     %s2 : TypeType<peano.Successor<peano.Successor<peano.Zero>>> = peano.successor<%s1>()
        |     %s3 : TypeType<peano.Successor<peano.Successor<peano.Successor<peano.Zero>>>> = peano.successor<%s2>()
        |     %n : Index = peano.value<%s3>()
        |     %_ : Nil = return(%n)
    """)
    module = parse_module(ir)

    print("\n=== Compile ===")
    exe = compile_staged(module, infer=lambda m: m, lower=lower_peano)

    print("\n=== Run ===")
    result = exe.run()
    print(f"result = {result}")

    assert result == 3


def test_equal_and_subtract_roundtrip():
    """equal_index and subtract_index ops round-trip through ASM."""
    ir = strip_prefix("""
        | %main : Nil = function<Index>() (%n: Index):
        |     %eq : Index = equal_index(%n, 0)
        |     %sub : Index = subtract_index(%n, 1)
        |     %result : Index = add_index(%eq, %sub)
        |     %_ : Nil = return(%result)
    """)
    module = parse_module(ir)
    asm_lines = list(module.asm)
    asm_text = "\n".join(asm_lines)
    assert "equal_index" in asm_text
    assert "subtract_index" in asm_text


def test_subtract_jit():
    """subtract_index executes correctly via JIT."""
    ir = strip_prefix("""
        | %main : Nil = function<Index>() (%n: Index):
        |     %sub : Index = subtract_index(%n, 1)
        |     %_ : Nil = return(%sub)
    """)
    module = parse_module(ir)
    from dgen import codegen

    exe = codegen.compile(module)
    assert exe.run(5) == 4


def test_if_else_parse_roundtrip():
    """if/else op with two blocks parses and round-trips."""
    ir = strip_prefix("""
        | %main : Nil = function<Index>() (%n: Index):
        |     %cond : Index = equal_index(%n, 0)
        |     %ten : Index = 10
        |     %twenty : Index = 20
        |     %result : Index = if(%cond) ():
        |         %_ : Nil = return(%ten)
        |     else ():
        |         %_ : Nil = return(%twenty)
        |     %_ : Nil = return(%result)
    """)
    module = parse_module(ir)
    asm_text = "\n".join(module.asm)
    print(asm_text)
    assert "if(" in asm_text or "if (" in asm_text
    assert "else" in asm_text
    # Round-trip: parse the output again
    module2 = parse_module(asm_text)
    asm_text2 = "\n".join(module2.asm)
    assert asm_text == asm_text2


def test_if_else_jit():
    """if/else executes correctly via JIT."""
    ir = strip_prefix("""
        | %main : Nil = function<Index>() (%n: Index):
        |     %cond : Index = equal_index(%n, 0)
        |     %one : Index = 1
        |     %result : Index = if(%cond) ():
        |         %_ : Nil = return(%one)
        |     else ():
        |         %val : Index = subtract_index(%n, 1)
        |         %_ : Nil = return(%val)
        |     %_ : Nil = return(%result)
    """)
    module = parse_module(ir)
    from dgen import codegen

    exe = codegen.compile(module)
    assert exe.run(0) == 1
    assert exe.run(5) == 4
    assert exe.run(1) == 0


def test_call_op_roundtrip():
    """call op parses and round-trips."""
    ir = strip_prefix("""
        | %main : Nil = function<Index>() (%x: Index):
        |     %result : Index = call<%add_one>([%x])
        |     %_ : Nil = return(%result)
        |
        | %add_one : Nil = function<Index>() (%n: Index):
        |     %r : Index = add_index(%n, 1)
        |     %_ : Nil = return(%r)
    """)
    module = parse_module(ir)
    asm_text = "\n".join(module.asm)
    print(asm_text)
    assert "call<%add_one>" in asm_text


def test_call_jit():
    """call op executes a helper function via JIT."""
    ir = strip_prefix("""
        | %main : Nil = function<Index>() (%x: Index):
        |     %result : Index = call<%add_one>([%x])
        |     %_ : Nil = return(%result)
        |
        | %add_one : Nil = function<Index>() (%n: Index):
        |     %r : Index = add_index(%n, 1)
        |     %_ : Nil = return(%r)
    """)
    module = parse_module(ir)
    from dgen import codegen

    exe = codegen.compile(module)
    assert exe.run(5) == 6
    assert exe.run(0) == 1


def test_equal_jit():
    """equal_index returns 1 when equal, 0 otherwise."""
    ir = strip_prefix("""
        | %main : Nil = function<Index>() (%n: Index):
        |     %eq : Index = equal_index(%n, 0)
        |     %_ : Nil = return(%eq)
    """)
    module = parse_module(ir)
    from dgen import codegen

    exe = codegen.compile(module)
    assert exe.run(0) == 1
    assert exe.run(5) == 0
