"""Tests for Block.parameters: the block-parameter binding form.

Block parameters are bound once by an op's lowering pass (e.g. ``%self``
for loop-header labels) and are never passed by callers.  They are leaves
in the use-def graph, identical to block args in that regard, but occupy a
distinct list and are emitted with ``block_name<%param: T>`` syntax.
"""

from dgen import asm
from dgen.asm.parser import parse_module
from dgen.dialects import goto
from dgen.testing import assert_ir_equivalent, strip_prefix


def test_roundtrip_label_with_self_param():
    ir = strip_prefix("""
        | import goto
        |
        | %loop : goto.Label = goto.label() body<%self: goto.Label>(%i: Index):
        |     %zero : Index = 0
    """)
    module = parse_module(ir)
    (label,) = module.ops
    assert isinstance(label, goto.LabelOp)
    assert len(label.body.parameters) == 1
    assert label.body.parameters[0].name == "self"
    assert isinstance(label.body.parameters[0].type, goto.Label)
    assert len(label.body.args) == 1
    assert label.body.args[0].name == "i"
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_label_no_params():
    """Labels without block parameters must still parse/format correctly."""
    ir = strip_prefix("""
        | import goto
        |
        | %loop : goto.Label = goto.label() body(%i: Index):
        |     %zero : Index = 0
    """)
    module = parse_module(ir)
    (label,) = module.ops
    assert isinstance(label, goto.LabelOp)
    assert label.body.parameters == []
    assert len(label.body.args) == 1
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_label_self_param_and_args():
    """Full round-trip: parse → format → parse, result structurally equal."""
    ir = strip_prefix("""
        | import goto
        |
        | %loop : goto.Label = goto.label() body<%self: goto.Label>(%i: Index):
        |     %zero : Index = 0
    """)
    module = parse_module(ir)
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_roundtrip_label_params_no_args():
    ir = strip_prefix("""
        | import goto
        |
        | %exit : goto.Label = goto.label() body<%self: goto.Label>():
        |     %zero : Index = 0
    """)
    module = parse_module(ir)
    (label,) = module.ops
    assert isinstance(label, goto.LabelOp)
    assert len(label.body.parameters) == 1
    assert label.body.args == []
    assert_ir_equivalent(module, asm.parse(asm.format(module)))


def test_verify_block_param_in_scope():
    """An op that references a block parameter must pass verify_closed_blocks."""
    ir = strip_prefix("""
        | import function
        | import goto
        |
        | %f : function.Function<()> = function.function<Nil>() body():
        |     %exit : goto.Label = goto.label() body<%self: goto.Label>():
        |         %zero : Index = 0
    """)
    from dgen.verify import verify_closed_blocks

    module = parse_module(ir)
    verify_closed_blocks(module)  # Should not raise
