"""Tests for the resolved dialect model."""

from __future__ import annotations

import dataclasses
import importlib

import pytest

from dgen.gen import model
from dgen.gen.importer import model_of
from dgen.gen.parser import parse

SHIPPED = [
    "dgen.dialects.index",
    "dgen.dialects.builtin",
    "dgen.dialects.number",
    "dgen.dialects.algebra",
    "dgen.dialects.function",
    "dgen.dialects.control_flow",
    "dgen.dialects.goto",
    "dgen.dialects.memory",
    "dgen.dialects.ndbuffer",
    "dgen.dialects.llvm",
    "toy.dialects.toy",
    "actor.dialects.actor",
    "dcc.dialects.c",
    "dcc2.dialects.c",
]


def dialect(module_name: str) -> model.Dialect:
    return model_of(importlib.import_module(module_name))


def op(module_name: str, name: str) -> model.Op:
    return next(o for o in dialect(module_name).ops if o.name == name)


def names(cls: type, attribute: str) -> list[str]:
    return [name for name, _ in vars(cls).get(attribute, ())]


def has_type_default(cls: type) -> bool:
    assert dataclasses.is_dataclass(cls)
    field = next(f for f in dataclasses.fields(cls) if f.name == "type")
    return field.default is not dataclasses.MISSING


@pytest.mark.parametrize("module_name", SHIPPED)
def test_model_matches_built_classes(module_name):
    module = importlib.import_module(module_name)
    d = model_of(module)
    built = vars(module)[d.name]
    # Python-registered ops and types, eg. builtin.constant, come after the file's.
    declared = [t.name for t in (*d.traits, *d.types)]
    assert declared == list(built.types)[: len(declared)]
    assert [o.name for o in d.ops] == list(built.ops)[: len(d.ops)]
    for t in d.types:
        assert [p.name for p in t.params] == names(built.types[t.name], "__params__")
    for o in d.ops:
        cls = built.ops[o.name]
        assert [p.name for p in o.params] == names(cls, "__params__")
        assert [x.name for x in o.operands] == names(cls, "__operands__")
        assert list(o.blocks) == list(vars(cls).get("__blocks__", ()))
        assert (o.fixed_type is not None) == has_type_default(cls), o.name


def test_reexported_name_resolves_to_defining_dialect():
    index = dialect("dgen.dialects.builtin").lookup("Index")
    assert isinstance(index, model.Type)
    assert index.module == "dgen.dialects.index"
    assert index.dialect == "index"


def test_alias_lookup():
    shape = dialect("toy.dialects.toy").lookup("ndbuffer.Shape")
    assert shape == dialect("dgen.dialects.ndbuffer").lookup("Shape")


def test_type_arguments_resolve():
    shape = dialect("dgen.dialects.ndbuffer").lookup("Shape")
    assert isinstance(shape, model.Type)
    (dims,) = shape.data
    index = dialect("dgen.dialects.index").lookup("Index")
    assert dims.type == model.Layout(
        "Array", (model.TypeExpr(index), model.ParamRef("rank"))
    )
    gep = op("dgen.dialects.llvm", "gep")
    assert gep.operands[1].type is not None
    assert gep.operands[1].type.args == (model.Literal(64),)


def test_holds_type():
    array = dialect("dgen.dialects.builtin").lookup("Array")
    assert isinstance(array, model.Type)
    assert [p.holds_type for p in array.params] == [True, False]


def test_fixed_type():
    assert op("dgen.dialects.builtin", "type").fixed_type == model.TypeExpr(None)
    add = op("dgen.dialects.llvm", "add").fixed_type
    assert add is not None
    assert add.args == (model.Literal(64),)
    assert op("toy.dialects.toy", "transpose").fixed_type is None
    assert op("dgen.dialects.memory", "stack_allocate").fixed_type is None
    assert op("dgen.dialects.builtin", "chain").fixed_type is None


def test_expression_constraint_keeps_text():
    concat = op("toy.dialects.toy", "concat")
    assert concat.constraints == (model.Expression("axis < lhs.shape.rank"),)


SOURCE = """\
from builtin import Index
trait Numeric
type Num:
    layout Int
    has trait Numeric
op negate(input) -> Num:
    requires input has trait Numeric
    requires input has type Num
"""


def from_source(source: str) -> model.Dialect:
    return model.Dialect.from_ast(
        parse(source), name="t", module="t", imported=dialect_named
    )


def dialect_named(name: str) -> model.Dialect:
    return dialect(f"dgen.dialects.{name}")


def test_constraints_resolve():
    d = from_source(SOURCE)
    (numeric,) = d.traits
    (num,) = d.types
    assert num.traits == (numeric,)
    (negate,) = d.ops
    assert negate.constraints == (
        model.HasTrait("input", numeric),
        model.HasType("input", model.TypeExpr(num)),
    )
    assert negate.fixed_type == model.TypeExpr(num)


def test_unknown_name_raises():
    with pytest.raises(NameError, match="Missing"):
        from_source("op f(x: Missing)\n")


def test_type_may_name_a_type_declared_below():
    d = from_source(
        "type Pair<items: Span<Item>>:\n    data: Nil\ntype Item:\n    data: Nil\nfrom builtin import Nil, Span\n"
    )
    assert [t.name for t in d.types] == ["Pair", "Item"]
    (items,) = d.types[0].params
    assert items.type.args == (model.TypeExpr(d.types[1]),)


def test_type_cycle_raises():
    with pytest.raises(NameError, match="B"):
        from_source("type A<b: B>:\n    layout Int\ntype B<a: A>:\n    layout Int\n")
