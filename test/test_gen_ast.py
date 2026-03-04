"""Tests for .dgen AST types."""

from dgen.gen.ast import DataField, DgenFile, OpDecl, TypeDecl, TypeRef


def test_ast_construction():
    f = DgenFile(
        types=[
            TypeDecl(name="index", data=DataField(name="data", type=TypeRef("Index")))
        ],
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


def test_data_field_with_compound_type():
    data = DataField(
        name="dims", type=TypeRef("Array", [TypeRef("Index"), TypeRef("rank")])
    )
    assert data.name == "dims"
    assert data.type.name == "Array"
    assert len(data.type.args) == 2
    assert data.type.args[0].name == "Index"
    assert data.type.args[1].name == "rank"
