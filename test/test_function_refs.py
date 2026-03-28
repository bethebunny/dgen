"""Tests for function references under the closed-block invariant.

Function references must be in scope like any other value. Currently,
CallOp.callee is a bare Value that bypasses verification. These tests
define the correct behavior we're working toward.

The parser also silently allows forward references by creating placeholder
Values (parser.resolve creates a bare Value on first use if the name isn't
defined yet). This violates the topo-order requirement.
"""

import pytest

from dgen import asm
from dgen.asm.parser import ParseError, parse_module
from dgen.dialects import function
from dgen.testing import strip_prefix
from dgen.verify import ClosedBlockError, verify_closed_blocks

_SINGLE_CALL = strip_prefix("""
    | import algebra
    | import function
    | import index
    | %main : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
    |     %r : index.Index = function.call<%helper>([%x])
    | %helper : function.Function<index.Index> = function.function<index.Index>() body(%n: index.Index):
    |     %r : index.Index = algebra.add(%n, 1)
""")


def test_callee_is_bare_value_not_function_op():
    """CallOp.callee is a bare Value, not a FunctionOp reference."""
    module = parse_module(_SINGLE_CALL)
    main = module.functions[0]
    call_op = main.body.ops[-1]
    assert isinstance(call_op, function.CallOp)
    assert not isinstance(call_op.callee, function.FunctionOp)
    assert call_op.callee.name == "helper"


def test_function_ref_roundtrip():
    """Function references survive ASM round-trip."""
    module = parse_module(_SINGLE_CALL)
    roundtripped = asm.parse(asm.format(module))
    main = roundtripped.functions[0]
    call_op = main.body.ops[-1]
    assert call_op.callee.name == "helper"


@pytest.mark.xfail(reason="bare Value callee bypasses verifier")
def test_single_caller_requires_capture():
    """A function body references a module-level callee without capturing it."""
    module = parse_module(_SINGLE_CALL)
    with pytest.raises(ClosedBlockError):
        verify_closed_blocks(module)


@pytest.mark.xfail(reason="bare Value callee bypasses verifier")
def test_two_callers_require_capture():
    """Two functions both reference the same module-level callee."""
    ir = strip_prefix("""
        | import algebra
        | import function
        | import index
        | %main : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = function.call<%helper>([%x])
        | %other : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = function.call<%helper>([%x])
        | %helper : function.Function<index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %r : index.Index = algebra.add(%n, 1)
    """)
    module = parse_module(ir)
    with pytest.raises(ClosedBlockError):
        verify_closed_blocks(module)


@pytest.mark.xfail(reason="bare Value callee bypasses verifier")
def test_nested_call_requires_capture():
    """A call inside a loop body references a module-level callee without capturing it."""
    ir = strip_prefix("""
        | import algebra
        | import control_flow
        | import function
        | import index
        | import number
        | %main : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %loop : Nil = control_flow.for<0, 3>([]) body(%i: index.Index):
        |         %r : index.Index = function.call<%helper>([%i])
        | %helper : function.Function<index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %r : index.Index = algebra.add(%n, 1)
    """)
    module = parse_module(ir)
    with pytest.raises(ClosedBlockError):
        verify_closed_blocks(module)


# ---------------------------------------------------------------------------
# Parser: forward references should be rejected
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="parser silently creates placeholder Values for undefined names")
def test_parser_rejects_forward_reference():
    """Referencing %helper before it's defined should be a parse error.

    The parser requires topo-order: definitions before uses. Currently
    parser.resolve() silently creates a bare Value(name=..., type=Nil())
    for undefined names instead of raising.
    """
    ir = strip_prefix("""
        | import algebra
        | import function
        | import index
        | %main : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = function.call<%helper>([%x])
        | %helper : function.Function<index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %r : index.Index = algebra.add(%n, 1)
    """)
    with pytest.raises(ParseError):
        parse_module(ir)


@pytest.mark.xfail(reason="parser silently creates placeholder Values for undefined names")
def test_parser_rejects_undefined_reference():
    """Referencing a name that's never defined should be a parse error."""
    ir = strip_prefix("""
        | import function
        | import index
        | %main : function.Function<index.Index> = function.function<index.Index>() body(%x: index.Index):
        |     %r : index.Index = function.call<%nonexistent>([%x])
    """
    )
    with pytest.raises(ParseError):
        parse_module(ir)


# ---------------------------------------------------------------------------
# SSA name uniqueness
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="parser allows duplicate SSA names in the same scope")
def test_parser_rejects_duplicate_ssa_names_in_scope():
    """Two ops with the same SSA name in the same scope should be a parse error."""
    ir = strip_prefix("""
        | import algebra
        | import function
        | import index
        | %helper : function.Function<index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %r : index.Index = algebra.add(%n, 1)
        | %helper : function.Function<index.Index> = function.function<index.Index>() body(%n: index.Index):
        |     %r : index.Index = algebra.multiply(%n, 2)
    """)
    with pytest.raises(ParseError):
        parse_module(ir)


@pytest.mark.xfail(
    reason="codegen emits both as @helper, producing invalid LLVM IR. "
    "Needs name mangling or scoped symbol resolution."
)
def test_same_function_name_in_different_scopes():
    """Two FunctionOps named %helper in different scopes is legal IR but
    crashes codegen — both emit as `define @helper`, conflicting in LLVM.

    This is valid from the IR's perspective (different scopes, different
    values) but codegen uses the SSA name directly as the LLVM symbol,
    with no scope mangling.
    """
    import dgen
    from dgen import codegen
    from dgen.block import BlockArgument
    from dgen.dialects.index import Index
    from dgen.module import Module, pack

    # Build IR programmatically: module-level %helper and a nested %helper
    # inside %main's body (different scopes).
    inner_helper = function.FunctionOp(
        name="helper",
        body=dgen.Block(
            result=dgen.module.ConstantOp(value=99, type=Index()),
            args=[],
        ),
        result=Index(),
        type=function.Function(result=Index()),
    )
    x = BlockArgument(name="x", type=Index())
    call_inner = function.CallOp(
        callee=inner_helper, arguments=pack([x]), type=Index()
    )
    main = function.FunctionOp(
        name="main",
        body=dgen.Block(result=call_inner, args=[x]),
        result=Index(),
        type=function.Function(result=Index()),
    )
    outer_helper = function.FunctionOp(
        name="helper",
        body=dgen.Block(
            result=dgen.module.ConstantOp(value=42, type=Index()),
            args=[],
        ),
        result=Index(),
        type=function.Function(result=Index()),
    )
    module = Module(ops=[main, outer_helper])
    exe = codegen.compile(module)
    # Should work: inner helper returns 99
    assert exe.run(0).to_json() == 99
