"""Tests for .dgen AST types."""

from dgen.gen.ast import (
    Assignment,
    AttrExpr,
    BinOpExpr,
    CallExpr,
    Constraint,
    DataField,
    DgenFile,
    ForStmt,
    IfStmt,
    LiteralExpr,
    MethodDecl,
    NameExpr,
    OpDecl,
    ReturnStmt,
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
    c = Constraint(kind="match", lhs="$X", pattern="Tensor")
    assert c.kind == "match"
    assert c.lhs == "$X"
    assert c.pattern == "Tensor"


def test_constraint_eq():
    c = Constraint(kind="eq", lhs="$X", rhs="$Result")
    assert c.kind == "eq"


def test_constraint_expr():
    c = Constraint(kind="expr", expr="axis < $X.rank")
    assert c.kind == "expr"


def test_op_with_constraints():
    op = OpDecl(
        name="tile",
        return_type=TypeRef("Type"),
        constraints=[
            Constraint(kind="match", lhs="$X", pattern="Tensor"),
        ],
    )
    assert len(op.constraints) == 1


def test_expr_name():
    e = NameExpr(name="x")
    assert e.name == "x"


def test_expr_attr():
    e = AttrExpr(value=NameExpr(name="self"), attr="dims")
    assert e.attr == "dims"


def test_expr_binop():
    e = BinOpExpr(op="*", left=NameExpr(name="a"), right=NameExpr(name="b"))
    assert e.op == "*"


def test_expr_call():
    e = CallExpr(func=NameExpr(name="foo"), args=[NameExpr(name="x")])
    assert len(e.args) == 1


def test_expr_literal():
    e = LiteralExpr(value=42)
    assert e.value == 42


def test_assignment():
    s = Assignment(
        name="count",
        type=TypeRef("Index"),
        value=LiteralExpr(value=1),
    )
    assert s.name == "count"


def test_return_stmt():
    s = ReturnStmt(value=NameExpr(name="count"))
    assert isinstance(s.value, NameExpr)


def test_for_stmt():
    s = ForStmt(
        var="dim",
        iter=AttrExpr(value=NameExpr(name="self"), attr="dims"),
        body=[],
    )
    assert s.var == "dim"


def test_if_stmt():
    s = IfStmt(
        condition=BinOpExpr(
            op="==", left=NameExpr(name="x"), right=LiteralExpr(value=0)
        ),
        then_body=[ReturnStmt(value=LiteralExpr(value=0))],
        else_body=[],
    )
    assert isinstance(s.condition, BinOpExpr)
    assert s.condition.op == "=="


def test_method_decl():
    m = MethodDecl(
        name="num_elements",
        params=[],
        return_type=TypeRef("Index"),
        body=[ReturnStmt(value=LiteralExpr(value=1))],
    )
    assert m.name == "num_elements"


def test_type_with_methods():
    t = TypeDecl(
        name="Shape",
        methods=[
            MethodDecl(
                name="num_elements",
                params=[],
                return_type=TypeRef("Index"),
                body=[ReturnStmt(value=LiteralExpr(value=1))],
            )
        ],
    )
    assert len(t.methods) == 1
