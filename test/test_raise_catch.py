"""Tests for ``error.catch`` and ``error.raise``: IR construction, ASM
round-trip, lowering to goto, and end-to-end JIT execution.

In the v1 design:

- ``catch<T>() on_raise(%err: T): <body>`` produces a ``RaiseHandler<T>``
  value. The handler is a compile-time marker (``RaiseHandler`` has
  ``layout Void``) that flows through the enclosing dataflow.
- ``raise<T, handler>(err)`` transfers control to the ``on_raise`` block of
  the catch that produced ``handler``. It has result type ``Never``.
- ``on_raise``'s body should diverge (e.g. re-raise to an outer handler).
  v1 does not provide a built-in merge back into the enclosing scope;
  recovery patterns require explicit control flow. When ``on_raise`` doesn't
  end with a terminator, codegen inserts ``unreachable`` as a well-formed UB
  sink so the compiled IR is still valid.
"""

from __future__ import annotations

from dgen import asm
from dgen.asm.parser import parse
from dgen.dialects import builtin, error, goto
from dgen.dialects.index import Index
from dgen.ir.traversal import all_values
from dgen.llvm import lower_to_llvm
from dgen.passes import lower_builtin_dialects
from dgen.passes.compiler import Compiler, IdentityPass
from dgen.passes.raise_catch_to_goto import RaiseCatchToGoto
from dgen.testing import assert_ir_equivalent, strip_prefix


# -- Construction ------------------------------------------------------------


def test_catch_op_shape():
    """CatchOp has no body block; on_raise is its only block."""
    assert error.CatchOp.__blocks__ == ("on_raise",)
    assert [name for name, _ in error.CatchOp.__params__] == ["error_type"]


def test_raise_handler_declares_parametric_handler_trait():
    """RaiseHandler<E> declares Handler<Raise<E>>."""
    h = error.RaiseHandler(error_type=Index())
    assert h.has_trait(builtin.Handler(effect_type=error.Raise(error_type=Index())))


def test_raise_is_effect():
    """Raise<E> inherits Effect."""
    assert error.Raise(error_type=Index()).has_trait(builtin.Effect)


# -- ASM round-trip ----------------------------------------------------------


CATCH_THEN_RAISE = strip_prefix("""
    | import algebra
    | import error
    | import function
    | import index
    | %f : function.Function<[], index.Index> = function.function<index.Index>() body():
    |     %h : error.RaiseHandler<index.Index> = error.catch<index.Index>() on_raise(%err: index.Index):
    |         %nop : index.Index = algebra.add(%err, %err)
    |     %val : index.Index = 7
    |     %raised : Never = error.raise<index.Index>(%h, %val)
""")


def test_catch_asm_roundtrip():
    """catch + raise survive ASM format → parse."""
    value = parse(CATCH_THEN_RAISE)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_catch_ir_shape():
    """Parsed catch has the expected structure: handler result, one block."""
    fn = parse(CATCH_THEN_RAISE)
    final_raise = fn.body.result
    assert isinstance(final_raise, error.RaiseOp)
    catch_op = final_raise.handler
    assert isinstance(catch_op, error.CatchOp)
    # catch_op.type is the handler's declared type: RaiseHandler<Index>
    assert isinstance(catch_op.type, error.RaiseHandler)
    # on_raise block has a single Index-typed error argument.
    assert len(catch_op.on_raise.args) == 1
    assert isinstance(catch_op.on_raise.args[0].type, Index)


# -- Lowering: catch/raise → goto --------------------------------------------


def _lower_catch_only(value):
    """Run only the RaiseCatchToGoto pass (no surrounding lowerings)."""
    return Compiler([RaiseCatchToGoto()], IdentityPass()).compile(value)


def test_catch_lowered_removes_catch_and_raise():
    """After the pass, no CatchOp or RaiseOp remains in the IR."""
    fn = parse(CATCH_THEN_RAISE)
    lowered = _lower_catch_only(fn)
    for v in all_values(lowered):
        assert not isinstance(v, error.CatchOp), "CatchOp survived lowering"
        assert not isinstance(v, error.RaiseOp), "RaiseOp survived lowering"


def test_catch_lowered_uses_goto_primitives():
    """Lowered IR contains a goto.label (from on_raise) and goto.branch
    (from each raise). No region wraps the body — v1 doesn't introduce one."""
    fn = parse(CATCH_THEN_RAISE)
    lowered = _lower_catch_only(fn)
    kinds = {type(v) for v in all_values(lowered)}
    assert goto.LabelOp in kinds
    assert goto.BranchOp in kinds
    assert goto.RegionOp not in kinds


# -- End-to-end: compile + JIT ---------------------------------------------


def _jit(ir_text: str):
    value = parse(ir_text)
    compiler = Compiler(passes=[lower_builtin_dialects()], exit=lower_to_llvm())
    return compiler.compile(value)


def test_e2e_catch_without_raise_runs_body():
    """Function body doesn't raise: on_raise is dead code, body path runs.

    The catch erases to a no-op at runtime (RaiseHandler has Void layout) so
    the function simply computes and returns the body's normal result.
    """
    exe = _jit(
        strip_prefix("""
        | import algebra
        | import error
        | import function
        | import index
        | %f : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %h : error.RaiseHandler<index.Index> = error.catch<index.Index>() on_raise(%err: index.Index):
        |         %nop : index.Index = algebra.add(%err, %err)
        |     %one : index.Index = 1
        |     %two : index.Index = 2
        |     %result : index.Index = algebra.add(%one, %two)
    """)
    )
    assert exe.run().to_json() == 3


def test_e2e_lowered_llvm_shape():
    """Lowered LLVM contains the on_raise label and an ``unreachable``
    fallback for the diverging block body. We don't execute this program —
    running it would transfer control to ``on_raise`` via raise and then
    hit the unreachable (LLVM UB). The emission being well-formed is
    sufficient: llvmlite parses it during JIT compile, which verifies
    basic-block terminator rules.
    """
    exe = _jit(
        strip_prefix("""
        | import algebra
        | import error
        | import function
        | import index
        | %f : function.Function<[], index.Index> = function.function<index.Index>() body():
        |     %h : error.RaiseHandler<index.Index> = error.catch<index.Index>() on_raise(%err: index.Index):
        |         %nop : index.Index = algebra.add(%err, %err)
        |     %val : index.Index = 7
        |     %raised : Never = error.raise<index.Index>(%h, %val)
    """)
    )
    ir_text = exe.ir.decode() if isinstance(exe.ir, bytes) else exe.ir
    # on_raise was lowered into a goto.label, emitted as %on_raise0.
    assert "on_raise0:" in ir_text
    # The on_raise body has no terminator (the nop add doesn't branch), so
    # codegen inserts ``unreachable`` to make the block well-formed.
    assert "unreachable" in ir_text
    # The raise became a direct branch to on_raise.
    assert "br label %on_raise0" in ir_text
