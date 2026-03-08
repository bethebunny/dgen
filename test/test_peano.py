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
from dgen import Block, Dialect, Op, Type, Value, layout
from dgen.asm.formatting import type_asm
from dgen.asm.parser import parse_module
from dgen.dialects import builtin
from dgen.dialects.builtin import FunctionOp, Index, Nil
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
            print(f"  lower: peano.successor<{type_asm(pred_type)}> -> {type_asm(succ)}")
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
