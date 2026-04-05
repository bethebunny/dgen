"""Tests for Block.parameters: the block-parameter binding form.

Block parameters are bound once by an op's lowering pass (e.g. ``%self``
for loop-header labels) and are never passed by callers.  They are leaves
in the use-def graph, identical to block args in that regard, but occupy a
distinct list and are emitted with ``block_name<%param: T>`` syntax.
"""

from dgen import asm
from dgen.asm.parser import parse
from dgen.dialects import goto
from dgen.testing import assert_ir_equivalent, strip_prefix


def test_roundtrip_label_with_self_param():
    ir = strip_prefix("""
        | import goto
        | import index
        |
        | %loop : goto.Label = goto.label([]) body<%self: goto.Label>(%i: index.Index):
        |     %zero : index.Index = 0
    """)
    label = parse(ir)
    assert isinstance(label, goto.LabelOp)
    assert len(label.body.parameters) == 1
    assert label.body.parameters[0].name == "self"
    assert isinstance(label.body.parameters[0].type, goto.Label)
    assert len(label.body.args) == 1
    assert label.body.args[0].name == "i"
    assert_ir_equivalent(label, asm.parse(asm.format(label)))


def test_roundtrip_label_no_params():
    """Labels without block parameters must still parse/format correctly."""
    ir = strip_prefix("""
        | import goto
        | import index
        |
        | %loop : goto.Label = goto.label([]) body(%i: index.Index):
        |     %zero : index.Index = 0
    """)
    label = parse(ir)
    assert isinstance(label, goto.LabelOp)
    assert label.body.parameters == []
    assert len(label.body.args) == 1
    assert_ir_equivalent(label, asm.parse(asm.format(label)))


def test_roundtrip_label_self_param_and_args():
    """Full round-trip: parse → format → parse, result structurally equal."""
    ir = strip_prefix("""
        | import goto
        | import index
        |
        | %loop : goto.Label = goto.label([]) body<%self: goto.Label>(%i: index.Index):
        |     %zero : index.Index = 0
    """)
    value = parse(ir)
    assert_ir_equivalent(value, asm.parse(asm.format(value)))


def test_roundtrip_label_params_no_args():
    ir = strip_prefix("""
        | import goto
        | import index
        |
        | %exit : goto.Label = goto.label([]) body<%self: goto.Label>():
        |     %zero : index.Index = 0
    """)
    label = parse(ir)
    assert isinstance(label, goto.LabelOp)
    assert len(label.body.parameters) == 1
    assert label.body.args == []
    assert_ir_equivalent(label, asm.parse(asm.format(label)))


def test_verify_block_param_in_scope():
    """An op that references a block parameter must pass verify_closed_blocks."""
    ir = strip_prefix("""
        | import function
        | import index
        | import goto
        | import index
        |
        | %f : function.Function<[], ()> = function.function<Nil>() body():
        |     %exit : goto.Label = goto.label([]) body<%self: goto.Label>():
        |         %zero : index.Index = 0
    """)
    from dgen.verify import verify_closed_blocks

    value = parse(ir)
    verify_closed_blocks(value)  # Should not raise
