"""Tests for ``error.try`` and ``error.raise``: IR construction, ASM
round-trip, lowering to goto, and end-to-end JIT execution.

v1 semantics:

- ``try<T>() body<%h: RaiseHandler<T>>(): ... except(%err: T): ...`` runs
  ``body`` with a fresh handler of type ``RaiseHandler<T>`` bound to the
  block's compile-time ``handler`` parameter. If the body completes
  normally its result is the try's value. If any ``raise<%h>(err)`` fires,
  control transfers to ``except`` with ``err`` as its block argument and
  ``except``'s result is the try's value.
- ``raise<T>(%h, err) -> Never`` is the effect-performing primitive.
- The lowering mirrors ``if``'s: both paths terminate at a merge inside a
  synthesized ``goto.region``. A ``raise`` with no enclosing ``try``
  (undischarged effect) is rejected by the pass — no silent UB.

The ASM block name is ``except``; the Python attribute is ``except_``
(keyword-safe mapping from the spec layer).
"""

from __future__ import annotations

import pytest

from dgen import asm
from dgen.asm.parser import parse
from dgen.block import Block, BlockArgument, BlockParameter
from dgen.dialects import algebra, builtin, error, goto
from dgen.dialects.builtin import Nil
from dgen.dialects.index import Index
from dgen.ir.traversal import all_values
from dgen.llvm import lower_to_llvm
from dgen.passes import lower_builtin_dialects
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.passes.raise_catch_to_goto import (
    RaiseCatchToGoto,
    UndischargedEffectError,
)
from dgen.testing import assert_ir_equivalent, strip_prefix


# -- Op / trait shape --------------------------------------------------------


def test_try_op_shape():
    """TryOp has body + except blocks (ASM names), error_type param."""
    assert error.TryOp.__blocks__ == ("body", "except")
    # Python-side dataclass field is the keyword-safe ``except_``.
    field_names = {f.name for f in error.TryOp.__dataclass_fields__.values()}
    assert {"body", "except_"}.issubset(field_names)
    assert [name for name, _ in error.TryOp.__params__] == ["error_type"]


def test_raise_handler_declares_parametric_handler_trait():
    """RaiseHandler<E> declares Handler<Raise<E>>."""
    h = error.RaiseHandler(error_type=Index())
    assert h.has_trait(builtin.Handler(effect_type=error.Raise(error_type=Index())))


def test_raise_declares_effect_trait():
    """Raise<E> declares the Effect trait."""
    assert error.Raise(error_type=Index()).has_trait(builtin.Effect())


# -- ASM round-trip (bare IR — no function wrapper) -------------------------


TRY_THEN_RAISE = strip_prefix("""
    | import algebra
    | import error
    | import index
    | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
    |     %val : index.Index = 7
    |     %raised : Never = error.raise<index.Index>(%h, %val)
    | except(%err: index.Index):
    |     %one : index.Index = 1
    |     %rec : index.Index = algebra.add(%err, %one)
""")


def test_try_asm_roundtrip():
    """try + raise survive ASM format → parse."""
    value = parse(TRY_THEN_RAISE)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_try_ir_shape():
    """Parsed try has the expected structure: handler param, error arg."""
    try_op = parse(TRY_THEN_RAISE)
    assert isinstance(try_op, error.TryOp)
    assert isinstance(try_op.type, Index)
    # body binds the handler as a block parameter (compile-time).
    assert len(try_op.body.parameters) == 1
    assert isinstance(try_op.body.parameters[0].type, error.RaiseHandler)
    # except receives the error value as a block argument (runtime).
    assert len(try_op.except_.args) == 1
    assert isinstance(try_op.except_.args[0].type, Index)


# -- Lowering: try/raise → goto --------------------------------------------


def _lower_try_only(value):
    """Run only RaiseCatchToGoto (no surrounding lowerings)."""
    return Compiler([RaiseCatchToGoto()], IdentityPass()).compile(value)


def test_try_lowered_removes_try_and_raise():
    """After the pass, no TryOp or RaiseOp remains in the IR."""
    lowered = _lower_try_only(parse(TRY_THEN_RAISE))
    for v in all_values(lowered):
        assert not isinstance(v, error.TryOp), "TryOp survived lowering"
        assert not isinstance(v, error.RaiseOp), "RaiseOp survived lowering"


def test_try_lowered_uses_goto_primitives():
    """Lowered IR contains goto.region + goto.label + goto.branch."""
    lowered = _lower_try_only(parse(TRY_THEN_RAISE))
    kinds = {type(v) for v in all_values(lowered)}
    assert goto.RegionOp in kinds
    assert goto.LabelOp in kinds
    assert goto.BranchOp in kinds


def test_try_with_raise_lowered_snapshot(ir_snapshot):
    """Full structure of a try whose body unconditionally raises:
    region wrapping a chain that branches body→merge and except→merge,
    with the raise rewritten to a branch targeting the except label."""
    assert _lower_try_only(parse(TRY_THEN_RAISE)) == ir_snapshot


def test_try_no_raise_lowered_snapshot(ir_snapshot):
    """When the body never raises, the except label is orphaned and
    ``replace_uses_of`` leaves it unreachable — it doesn't appear in the
    lowered IR. The structure collapses to a region whose body branches
    straight to the merge."""
    no_raise = strip_prefix("""
        | import algebra
        | import error
        | import index
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |     %one : index.Index = 1
        |     %two : index.Index = 2
        |     %sum : index.Index = algebra.add(%one, %two)
        | except(%err: index.Index):
        |     %z : index.Index = 0
        |     %rec : index.Index = algebra.add(%err, %z)
    """)
    assert _lower_try_only(parse(no_raise)) == ir_snapshot


def test_try_with_nested_if_raise_lowered_snapshot(ir_snapshot):
    """A raise inside a nested ``if`` exercises the cascade through nested
    block captures: the substitution flows handler→except_label through
    the if's then_body capture list."""
    nested = strip_prefix("""
        | import algebra
        | import control_flow
        | import error
        | import function
        | import index
        | import number
        | %f : function.Function<[number.Boolean], index.Index> = function.function<index.Index>() body(%cond: number.Boolean):
        |     %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>() captures(%cond):
        |         %ten : index.Index = 10
        |         %out : index.Index = control_flow.if(%cond, [], []) then_body() captures(%h):
        |             %val : index.Index = 99
        |             %raised : Never = error.raise<index.Index>(%h, %val)
        |         else_body():
        |             %five : index.Index = 5
        |         %chained : index.Index = algebra.add(%out, %ten)
        |     except(%err: index.Index):
        |         %one : index.Index = 1
        |         %rec : index.Index = algebra.add(%err, %one)
    """)
    assert _lower_try_only(parse(nested)) == ir_snapshot


# -- Undischarged-effect verification ---------------------------------------


def test_undischarged_raise_is_rejected():
    """A RaiseOp whose handler doesn't resolve to any try fails at
    ``verify_postconditions`` — codegen never sees undefined behavior.

    Construct a bare raise with a dangling handler (a ``BlockParameter``
    that no try produced) and verify the lowering pass rejects it. No
    enclosing function needed — the verifier walks the value reachable
    from the root.
    """
    dangling_handler = BlockParameter(
        name="h", type=error.RaiseHandler(error_type=Index())
    )
    raise_op = error.RaiseOp(
        error_type=Index(),
        handler=dangling_handler,
        error=Index().constant(0),
    )
    # Wrap the raise in a block that captures the dangling handler; the
    # closed-block invariant requires every BlockParameter referenced by a
    # block's values to be a parameter of that block.
    root = Block(result=raise_op, parameters=[dangling_handler])
    with pytest.raises(UndischargedEffectError, match="not discharged"):
        _lower_try_only(root.result)


# -- Branch type compatibility ----------------------------------------------


def test_body_type_mismatch_rejected():
    """A body whose result type doesn't match the try's declared type fails
    at ``verify_preconditions`` — caught before LLVM sees a phi-type error."""
    bad = parse(
        strip_prefix("""
        | import error
        | import index
        | import number
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |     %wrong : number.Float64 = 1.5
        | except(%err: index.Index):
        |     %z : index.Index = 0
    """)
    )
    with pytest.raises(TypeError, match="body block result type.*does not match"):
        _lower_try_only(bad)


def test_except_type_mismatch_rejected():
    """An except whose result type doesn't match the try's declared type
    fails at ``verify_preconditions``."""
    bad = parse(
        strip_prefix("""
        | import error
        | import index
        | import number
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |     %ok : index.Index = 1
        | except(%err: index.Index):
        |     %wrong : number.Float64 = 2.5
    """)
    )
    with pytest.raises(TypeError, match="except block result type.*does not match"):
        _lower_try_only(bad)


def test_diverging_body_is_compatible_with_any_type():
    """When the body diverges (Never), only except's type needs to match the
    try. Never participates in no merge phi, so it's always compatible."""
    # Body raises unconditionally → result type is Never. except returns
    # an Index that matches the try's declared Index type.
    value = parse(
        strip_prefix("""
        | import algebra
        | import error
        | import index
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |     %v : index.Index = 7
        |     %r : Never = error.raise<index.Index>(%h, %v)
        | except(%err: index.Index):
        |     %z : index.Index = 0
        |     %rec : index.Index = algebra.add(%err, %z)
    """)
    )
    # Should lower without raising — Never is universally compatible.
    _lower_try_only(value)


def test_diverging_except_is_compatible_with_any_type():
    """When except diverges (Never — typically a re-raise to outer), only
    body's type needs to match the try."""
    value = parse(
        strip_prefix("""
        | import error
        | import index
        | %outer : index.Index = error.try<index.Index>() body<%ho: error.RaiseHandler<index.Index>>():
        |     %inner : index.Index = error.try<index.Index>() body<%hi: error.RaiseHandler<index.Index>>():
        |         %ok : index.Index = 5
        |     except(%err: index.Index) captures(%ho):
        |         %r : Never = error.raise<index.Index>(%ho, %err)
        | except(%err: index.Index):
        |     %z : index.Index = 0
    """)
    )
    _lower_try_only(value)


# -- End-to-end: compile + JIT with recovery value ------------------------


def _jit(value):
    compiler = Compiler(passes=[lower_builtin_dialects()], exit=lower_to_llvm())
    return compiler.compile(value)


def test_e2e_raise_fires_except_runs_recovery_value():
    """Body unconditionally raises; except recovers to err+1. JIT-compiled
    and executed — the function actually returns the recovery value.

    This is the acid test: if the test were checking emitted IR only, a
    broken lowering could still pass. Here we run the program and assert
    the numerical result. ``lower_to_llvm`` auto-wraps a bare value in a
    no-arg ``main`` function, so we don't need an explicit function.
    """
    value = parse(
        strip_prefix("""
        | import algebra
        | import error
        | import index
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |     %val : index.Index = 42
        |     %raised : Never = error.raise<index.Index>(%h, %val)
        | except(%err: index.Index):
        |     %one : index.Index = 1
        |     %rec : index.Index = algebra.add(%err, %one)
    """)
    )
    exe = _jit(value)
    assert exe.run().to_json() == 43  # 42 was raised, except returned err+1


def test_e2e_no_raise_runs_body_result():
    """Body completes normally; except is dead code. JIT returns body's value."""
    value = parse(
        strip_prefix("""
        | import algebra
        | import error
        | import index
        | %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |     %one : index.Index = 1
        |     %two : index.Index = 2
        |     %sum : index.Index = algebra.add(%one, %two)
        | except(%err: index.Index):
        |     %z : index.Index = 0
        |     %rec : index.Index = algebra.add(%err, %z)
    """)
    )
    exe = _jit(value)
    assert exe.run().to_json() == 3


def test_e2e_conditional_raise_exercises_both_paths():
    """Use a runtime condition to choose between raising or not.

    Parameterizing the function by a boolean lets us JIT once and call
    twice, confirming both the body-normal and the raise-recovery paths
    produce the expected values.
    """
    value = parse(
        strip_prefix("""
        | import algebra
        | import control_flow
        | import error
        | import function
        | import index
        | import number
        | %f : function.Function<[number.Boolean], index.Index> = function.function<index.Index>() body(%cond: number.Boolean):
        |     %t : index.Index = error.try<index.Index>() body<%h: error.RaiseHandler<index.Index>>() captures(%cond):
        |         %ten : index.Index = 10
        |         %out : index.Index = control_flow.if(%cond, [], []) then_body() captures(%h):
        |             %val : index.Index = 99
        |             %raised : Never = error.raise<index.Index>(%h, %val)
        |         else_body():
        |             %five : index.Index = 5
        |         %chained : index.Index = algebra.add(%out, %ten)
        |     except(%err: index.Index):
        |         %one : index.Index = 1
        |         %rec : index.Index = algebra.add(%err, %one)
    """)
    )
    exe = _jit(value)
    # cond=False: no raise → 5 + 10 = 15
    assert exe.run(False).to_json() == 15
    # cond=True: raise 99 → except returns 99 + 1 = 100
    assert exe.run(True).to_json() == 100


def test_e2e_nested_trys_disambiguate_block_names():
    """Two trys in the same function — inner nested in outer's body —
    must produce distinct LLVM block names. Without per-try ``cid``,
    both would emit ``try0``/``try_exit0``/``except0`` and LLVM would
    reject the duplicate labels.

    Inner raises and its except recovers; outer's except stays dead.
    """
    value = parse(
        strip_prefix("""
        | import algebra
        | import error
        | import index
        | %outer : index.Index = error.try<index.Index>() body<%ho: error.RaiseHandler<index.Index>>():
        |     %inner : index.Index = error.try<index.Index>() body<%hi: error.RaiseHandler<index.Index>>():
        |         %v : index.Index = 5
        |         %raised : Never = error.raise<index.Index>(%hi, %v)
        |     except(%err: index.Index):
        |         %one : index.Index = 1
        |         %rec : index.Index = algebra.add(%err, %one)
        |     %ten : index.Index = 10
        |     %sum : index.Index = algebra.add(%inner, %ten)
        | except(%err: index.Index):
        |     %z : index.Index = 0
        |     %dead : index.Index = algebra.add(%err, %z)
    """)
    )
    exe = _jit(value)
    # Inner raises 5 → inner-except returns 5+1=6 → outer body adds 10 → 16
    assert exe.run().to_json() == 16


# -- Convenience helper (catch() constructor) --------------------------------


def _trivial_except(error_type):
    err = BlockArgument(name="err", type=error_type)
    return Block(args=[err], result=err)


def test_tryop_constructable_from_python():
    """TryOp can be constructed in Python with ``except_`` as the kwarg."""
    handler = BlockParameter(name="h", type=error.RaiseHandler(error_type=Index()))
    body = Block(parameters=[handler], result=Index().constant(7))
    op = error.TryOp(
        error_type=Index(),
        body=body,
        except_=_trivial_except(Index()),
        type=Index(),
    )
    # op.blocks yields ASM names — 'except' not 'except_'.
    names = [asm_name for asm_name, _ in op.blocks]
    assert names == ["body", "except"]
    # But the dot-access attribute is `except_`.
    assert op.except_ is not None
    # Type-check: silence unused-import linters.
    assert algebra is not None and Nil is not None
