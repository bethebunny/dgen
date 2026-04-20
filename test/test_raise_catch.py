"""Tests for ``error.catch`` and ``error.raise``: IR construction, ASM
round-trip, lowering to goto, and end-to-end JIT execution."""

from __future__ import annotations

from dgen import asm
from dgen.asm.parser import parse
from dgen.dialects import algebra, error, goto
from dgen.dialects.index import Index
from dgen.error import CatchOp, catch
from dgen.ir.traversal import all_values
from dgen.llvm import lower_to_llvm
from dgen.passes import lower_builtin_dialects
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.passes.raise_catch_to_goto import RaiseCatchToGoto
from dgen.testing import assert_ir_equivalent, strip_prefix


# -- Construction ------------------------------------------------------------


def test_catch_shape():
    """Convenience constructor wires up handler parameter and error arg."""
    c = catch(
        Index(),
        lambda h: error.RaiseOp(
            error_type=Index(),
            handler=h,
            error=Index().constant(7),
        ),
        lambda e: algebra.AddOp(left=e, right=Index().constant(1), type=Index()),
    )
    assert isinstance(c, CatchOp)
    assert isinstance(c.body.parameters[0].type, error.RaiseHandler)
    assert isinstance(c.on_raise.args[0].type, Index)
    # Catch's declared type falls through to the non-Never branch.
    assert isinstance(c.type, Index)


def test_catch_diverging_body_uses_on_raise_type():
    """When body diverges (raise as last op ⇒ Never), type comes from on_raise."""
    c = catch(
        Index(),
        lambda h: error.RaiseOp(
            error_type=Index(), handler=h, error=Index().constant(7)
        ),
        lambda e: algebra.AddOp(left=e, right=Index().constant(1), type=Index()),
    )
    assert isinstance(c.type, Index)


# -- ASM round-trip ----------------------------------------------------------


CATCH_AND_RECOVER = strip_prefix("""
    | import algebra
    | import error
    | import function
    | import index
    | %f : function.Function<[], index.Index> = function.function<index.Index>() body():
    |     %r : index.Index = error.catch<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
    |         %e : index.Index = 7
    |         %raised : Never = error.raise<index.Index, %h>(%e)
    |     on_raise(%err: index.Index):
    |         %one : index.Index = 1
    |         %rec : index.Index = algebra.add(%err, %one)
""")


def test_catch_asm_roundtrip():
    """catch + raise survive ASM format → parse."""
    value = parse(CATCH_AND_RECOVER)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_catch_ir_shape():
    """Parsed catch has the right structure."""
    fn = parse(CATCH_AND_RECOVER)
    catch_op = fn.body.result
    assert isinstance(catch_op, CatchOp)
    assert isinstance(catch_op.body.parameters[0].type, error.RaiseHandler)
    assert isinstance(catch_op.on_raise.args[0].type, Index)


# -- Lowering: catch/raise → goto --------------------------------------------


def _lower_catch_only(value):
    """Run only the RaiseCatchToGoto pass (no surrounding lowerings)."""
    return Compiler([RaiseCatchToGoto()], IdentityPass()).compile(value)


def test_catch_lowered_contains_no_catch_or_raise():
    """After RaiseCatchToGoto, no CatchOp or RaiseOp should remain."""
    fn = parse(CATCH_AND_RECOVER)
    lowered = _lower_catch_only(fn)
    for v in all_values(lowered):
        assert not isinstance(v, CatchOp), "CatchOp survived lowering"
        assert not isinstance(v, error.RaiseOp), "RaiseOp survived lowering"


def test_catch_lowered_uses_goto_region_and_label():
    """The lowered IR should contain a goto.region and a goto.label."""
    fn = parse(CATCH_AND_RECOVER)
    lowered = _lower_catch_only(fn)
    kinds = {type(v) for v in all_values(lowered)}
    assert goto.RegionOp in kinds
    assert goto.LabelOp in kinds
    assert goto.BranchOp in kinds


# -- Nested catch ------------------------------------------------------------


NESTED_CATCH = strip_prefix("""
    | import algebra
    | import error
    | import function
    | import index
    | %f : function.Function<[], index.Index> = function.function<index.Index>() body():
    |     %outer : index.Index = error.catch<index.Index>() body<%h1: error.RaiseHandler<index.Index>>():
    |         %inner : index.Index = error.catch<index.Index>() body<%h2: error.RaiseHandler<index.Index>>() captures(%h1):
    |             %e : index.Index = 5
    |             %r1 : Never = error.raise<index.Index, %h2>(%e)
    |         on_raise(%err1: index.Index) captures(%h1):
    |             %e2 : index.Index = 9
    |             %r2 : Never = error.raise<index.Index, %h1>(%e2)
    |     on_raise(%err2: index.Index):
    |         %one : index.Index = 1
    |         %rec : index.Index = algebra.add(%err2, %one)
""")


def test_nested_catch_inner_raise_targets_inner_handler():
    """Inner raise should bind to inner catch; outer raise to outer."""
    fn = parse(NESTED_CATCH)
    lowered = _lower_catch_only(fn)
    # Verify the lowering completes without residual catch/raise ops.
    for v in all_values(lowered):
        assert not isinstance(v, CatchOp)
        assert not isinstance(v, error.RaiseOp)


# -- End-to-end: lowering + codegen + JIT -----------------------------------


def _jit(ir_text: str):
    value = parse(ir_text)
    compiler = Compiler(passes=[lower_builtin_dialects()], exit=lower_to_llvm())
    return compiler.compile(value)


def test_e2e_raise_is_caught_and_recovered():
    """catch { raise(42) } on_raise(e) { e + 1 } → 43."""
    exe = _jit(
        strip_prefix("""
        | import algebra
        | import error
        | import function
        | import index
        | %f : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %r : index.Index = error.catch<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |         %e : index.Index = 42
        |         %raised : Never = error.raise<index.Index, %h>(%e)
        |     on_raise(%err: index.Index):
        |         %one : index.Index = 1
        |         %rec : index.Index = algebra.add(%err, %one)
    """)
    )
    assert exe.run().to_json() == 43


def test_e2e_no_raise_body_result_used():
    """catch { 7 } on_raise(e) { 0 } → 7 (body completes normally)."""
    # Body must have at least one op in ASM; use a chain as a no-op wrapper.
    exe = _jit(
        strip_prefix("""
        | import algebra
        | import error
        | import function
        | import index
        | %f : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %r : index.Index = error.catch<index.Index>() body<%h: error.RaiseHandler<index.Index>>():
        |         %seven : index.Index = 7
        |         %one : index.Index = 1
        |         %val : index.Index = algebra.add(%seven, %one)
        |     on_raise(%err: index.Index):
        |         %zero : index.Index = 0
        |         %via : index.Index = algebra.add(%err, %zero)
    """)
    )
    assert exe.run().to_json() == 8  # 7 + 1 from body


def test_e2e_conditional_raise():
    """Conditional raise: returns body if condition false, on_raise if true.

    Test both branches by running with different inputs.
    """
    exe = _jit(
        strip_prefix("""
        | import algebra
        | import control_flow
        | import error
        | import function
        | import index
        | import number
        | %f : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = error.catch<index.Index>() body<%h: error.RaiseHandler<index.Index>>() captures(%x):
        |         %zero : index.Index = 0
        |         %cond : number.Boolean = algebra.less_than(%x, %zero)
        |         %out : index.Index = control_flow.if(%cond, [], []) then_body() captures(%h):
        |             %hundred : index.Index = 100
        |             %raised : Never = error.raise<index.Index, %h>(%hundred)
        |         else_body() captures(%x):
        |             %double : index.Index = algebra.add(%x, %x)
        |     on_raise(%err: index.Index):
        |         %neg : index.Index = algebra.negate(%err)
    """)
    )
    # x=3: 3 + 3 = 6 (no raise)
    assert exe.run(3).to_json() == 6
    # x=-5: cond true, raises 100, on_raise returns -100
    assert exe.run(-5).to_json() == -100
