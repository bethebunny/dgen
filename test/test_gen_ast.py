"""Tests for .dgen AST types."""

from dgen.gen.ast import DgenFile, LayoutExpr, OpDecl, TypeDecl, TypeRef


def test_ast_construction():
    f = DgenFile(
        types=[TypeDecl(name="index", layout=LayoutExpr("INT"))],
        ops=[OpDecl(name="return", return_type=TypeRef("Nil"))],
    )
    assert len(f.types) == 1
    assert f.types[0].name == "index"
    assert len(f.ops) == 1
    assert f.ops[0].return_type.name == "Nil"


def test_type_ref_with_args():
    ref = TypeRef("list", [TypeRef("Type")])
    assert ref.name == "list"
    assert len(ref.args) == 1
    assert ref.args[0].name == "Type"


def test_layout_expr_with_args():
    layout = LayoutExpr("Array", ["INT", "rank"])
    assert layout.name == "Array"
    assert layout.args == ["INT", "rank"]
