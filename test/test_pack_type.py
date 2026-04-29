"""Tests for ``pack()`` and ``PackOp.type`` honesty.

The type a PackOp carries should reflect what's inside it. The previous
heuristic (``Span<first.type>``) silently lied about heterogeneous mixes
and would crash ``__constant__``. These tests pin down the new contract:
homogeneous → ``Span<T>``, heterogeneous → ``Tuple<types>``.
"""

from __future__ import annotations

from dgen import asm
from dgen.asm.parser import parse
from dgen.builtins import PackOp, pack
from dgen.dialects.builtin import Nil, Span, String, Tuple
from dgen.dialects.function import FunctionOp
from dgen.dialects.index import Index
from dgen.dialects.number import Float64
from dgen.dialects.record import PackOp as RecordPackOp
from dgen.testing import assert_ir_equivalent, strip_prefix
from dgen.type import constant


# -- pack() helper: type honesty -------------------------------------------


def test_pack_empty_is_span_of_nil():
    p = pack([])
    assert isinstance(p.type, Span)
    assert isinstance(p.type.pointee, Nil)


def test_pack_homogeneous_is_span():
    p = pack([Index().constant(1), Index().constant(2)])
    assert isinstance(p.type, Span)
    assert isinstance(p.type.pointee, Index)


def test_pack_homogeneous_with_shared_type_instance_is_span():
    """Two values sharing the *same* Type instance — the cheap identity path."""
    t = Index()
    p = pack([t.constant(1), t.constant(2), t.constant(3)])
    assert isinstance(p.type, Span)


def test_pack_heterogeneous_is_tuple():
    p = pack([Index().constant(1), String().constant("hi")])
    assert isinstance(p.type, Tuple)
    assert isinstance(p.type.types, PackOp)
    types = list(p.type.types)
    assert len(types) == 2
    assert isinstance(types[0], Index)
    assert isinstance(types[1], String)


def test_pack_heterogeneous_three_kinds():
    p = pack([Index().constant(1), String().constant("x"), Float64().constant(2.5)])
    assert isinstance(p.type, Tuple)
    assert isinstance(p.type.types, PackOp)
    types = list(p.type.types)
    assert isinstance(types[0], Index)
    assert isinstance(types[1], String)
    assert isinstance(types[2], Float64)


def test_pack_inner_types_pack_is_span_of_typetype():
    """The Tuple<types>'s ``types`` field is itself a homogeneous Span<TypeType>."""
    from dgen.type import TypeType

    p = pack([Index().constant(1), String().constant("x")])
    assert isinstance(p.type, Tuple)
    inner = p.type.types
    assert isinstance(inner, PackOp)
    assert isinstance(inner.type, Span)
    assert isinstance(inner.type.pointee, TypeType)


# -- PackOp.__constant__ no longer crashes on heterogeneous packs ---------


def test_pack_constant_homogeneous_roundtrip():
    p = pack([Index().constant(7), Index().constant(11)])
    assert constant(p) == [7, 11]


def test_pack_constant_heterogeneous_does_not_crash():
    """The previous Span<Index> lie crashed here; Tuple typing serializes
    each field by its own layout."""
    p = pack([Index().constant(7), String().constant("hello")])
    mem = p.__constant__
    # Tuple's layout is Record; native value is a dict keyed by field name.
    rebuilt = mem.to_native_value()
    assert rebuilt == {"0": 7, "1": "hello"}


# -- Parser produces honest PackOp types ---------------------------------


def test_parser_homogeneous_pack_type_is_span():
    """Pack literal in an op's operand position with homogeneous values."""
    ir = strip_prefix("""
        | import index
        | import record
        | import function
        |
        | %main : function.Function<[index.Index, index.Index], Nil> = function.function<Nil>() body(%a: index.Index, %b: index.Index):
        |     %t : Tuple<[index.Index, index.Index]> = record.pack([%a, %b])
    """)
    fn = parse(ir)
    assert isinstance(fn, FunctionOp)
    record_pack = fn.body.result
    assert isinstance(record_pack, RecordPackOp)
    pack_op = record_pack.values
    assert isinstance(pack_op, PackOp)
    assert isinstance(pack_op.type, Span)
    assert isinstance(pack_op.type.pointee, Index)


def test_parser_heterogeneous_pack_type_is_tuple():
    """Pack literal in an op's operand position with mixed-type values."""
    ir = strip_prefix("""
        | import index
        | import record
        | import function
        |
        | %main : function.Function<[index.Index, String], Nil> = function.function<Nil>() body(%a: index.Index, %s: String):
        |     %t : Tuple<[index.Index, String]> = record.pack([%a, %s])
    """)
    fn = parse(ir)
    assert isinstance(fn, FunctionOp)
    record_pack = fn.body.result
    assert isinstance(record_pack, RecordPackOp)
    pack_op = record_pack.values
    assert isinstance(pack_op, PackOp)
    assert isinstance(pack_op.type, Tuple)
    assert isinstance(pack_op.type.types, PackOp)
    types = list(pack_op.type.types)
    assert isinstance(types[0], Index)
    assert isinstance(types[1], String)


# -- Round-trip preserved (formatter inlines [...] sugar) ----------------


def test_heterogeneous_pack_roundtrips():
    ir = strip_prefix("""
        | import index
        | import record
        | import function
        |
        | %main : function.Function<[index.Index, String], Nil> = function.function<Nil>() body(%a: index.Index, %s: String):
        |     %t : Tuple<[index.Index, String]> = record.pack([%a, %s])
    """)
    fn = parse(ir)
    assert_ir_equivalent(fn, asm.parse(asm.format(fn)))
