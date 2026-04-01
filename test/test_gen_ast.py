"""Tests for .dgen AST types."""

from dgen.gen.ast import (
    DataField,
    DgenFile,
    EqConstraint,
    ExpressionConstraint,
    MatchConstraint,
    OpDecl,
    TraitConstraint,
    TypeDecl,
    TypeRef,
)


def test_ast_construction():
    f = DgenFile(
        types=[
            TypeDecl(name="index", data=[DataField(name="data", type=TypeRef("Index"))])
        ],
        ops=[OpDecl(name="return", return_type=TypeRef("Nil"))],
    )
    assert len(f.types) == 1
    assert f.types[0].name == "index"
    assert len(f.ops) == 1
    assert f.ops[0].return_type is not None
    assert f.ops[0].return_type.name == "Nil"


def test_type_ref_with_args():
    ref = TypeRef("list", [TypeRef("Type")])
    assert ref.name == "list"
    assert len(ref.args) == 1
    assert ref.args[0].name == "Type"


def test_data_field_with_compound_type():
    data = DataField(
        name="dims", type=TypeRef("Array", [TypeRef("Index"), TypeRef("rank")])
    )
    assert data.name == "dims"
    assert data.type.name == "Array"
    assert len(data.type.args) == 2
    assert data.type.args[0].name == "Index"
    assert data.type.args[1].name == "rank"


def test_constraint_match():
    c = MatchConstraint(lhs="X", pattern="Tensor")
    assert c.lhs == "X"
    assert c.pattern == "Tensor"


def test_constraint_eq():
    c = EqConstraint(lhs="X", rhs="Result")
    assert c.lhs == "X"
    assert c.rhs == "Result"


def test_constraint_expr():
    c = ExpressionConstraint(expr="axis < X.rank")
    assert c.expr == "axis < X.rank"


def test_constraint_trait():
    c = TraitConstraint(lhs="lhs", trait="AddMagma")
    assert c.lhs == "lhs"
    assert c.trait == "AddMagma"


def test_op_with_constraints():
    op = OpDecl(
        name="tile",
        return_type=TypeRef("Type"),
        constraints=[
            MatchConstraint(lhs="X", pattern="Tensor"),
        ],
    )
    assert len(op.constraints) == 1
    assert isinstance(op.constraints[0], MatchConstraint)
