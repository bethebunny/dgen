"""Tests for Python code generator from .dgen AST."""

from dgen.gen.ast import (
    DataField,
    DgenFile,
    ImportDecl,
    OpDecl,
    OperandDecl,
    ParamDecl,
    TraitDecl,
    TypeDecl,
    TypeRef,
)
from dgen.gen.python import generate


def test_generate_header():
    code = generate(DgenFile(), dialect_name="test")
    assert "# GENERATED" in code
    assert "from dgen import" in code
    assert 'Dialect("test")' in code


def test_generate_simple_type():
    f = DgenFile(
        types=[
            TypeDecl(name="Index", data=DataField(name="data", type=TypeRef("Index")))
        ]
    )
    code = generate(f, dialect_name="test")
    assert '@test.type("Index")' in code
    assert "class Index(Type):" in code
    assert "@dataclass(frozen=True)" in code
    assert "__layout__ = layout.Int()" in code


def test_generate_type_no_data():
    """Type with no data field at all (layout provided externally)."""
    f = DgenFile(
        types=[
            TypeDecl(
                name="Tensor",
                params=[
                    ParamDecl(name="shape", type=TypeRef("Shape")),
                ],
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "class Tensor(Type):" in code
    assert "__layout__" not in code


def test_generate_parameterized_type():
    f = DgenFile(
        types=[
            TypeDecl(
                name="Shape",
                params=[ParamDecl(name="rank", type=TypeRef("Index"))],
                data=DataField(
                    name="dims",
                    type=TypeRef("Array", [TypeRef("Index"), TypeRef("rank")]),
                ),
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "class Shape(Type):" in code
    assert "rank: Value[Index]" in code
    assert '__params__ = (("rank", Index),)' in code
    # Parametric layout becomes a property
    assert "def __layout__(self)" in code
    assert "layout.Array(layout.Int()," in code


def test_generate_type_fatpointer_data():
    f = DgenFile(
        types=[
            TypeDecl(
                name="String",
                data=DataField(
                    name="storage",
                    type=TypeRef("FatPointer", [TypeRef("Byte")]),
                ),
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "class String(Type):" in code
    assert "__layout__ = layout.FatPointer(layout.Byte())" in code


def test_generate_type_fatpointer_param():
    """FatPointer<element_type> where element_type is a Type param."""
    f = DgenFile(
        types=[
            TypeDecl(
                name="List",
                params=[ParamDecl(name="element_type", type=TypeRef("Type"))],
                data=DataField(
                    name="storage",
                    type=TypeRef("FatPointer", [TypeRef("element_type")]),
                ),
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "class List(Type):" in code
    assert "def __layout__(self)" in code
    assert "self.element_type.__layout__" in code


def test_generate_type_default_param():
    f = DgenFile(
        types=[
            TypeDecl(
                name="Tensor",
                params=[
                    ParamDecl(name="shape", type=TypeRef("Shape")),
                    ParamDecl(name="dtype", type=TypeRef("Type"), default="F64"),
                ],
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "shape: Value[Shape]" in code
    assert "dtype: Type" in code
    # Default should reference the type class
    assert "F64()" in code


def test_generate_trait():
    f = DgenFile(traits=[TraitDecl(name="HasSingleBlock")])
    code = generate(f, dialect_name="test")
    assert "class HasSingleBlock:" in code
    assert "pass" in code


def test_generate_simple_op():
    f = DgenFile(
        ops=[
            OpDecl(
                name="transpose",
                operands=[OperandDecl(name="input", type=TypeRef("Type"))],
                return_type=TypeRef("Type"),
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert '@test.op("transpose")' in code
    assert "@dataclass(eq=False, kw_only=True)" in code
    assert "class TransposeOp(Op):" in code
    assert "input: Value" in code
    assert "type: Type" in code
    assert '__operands__ = (("input", Type),)' in code


def test_generate_op_with_params():
    f = DgenFile(
        ops=[
            OpDecl(
                name="concat",
                params=[ParamDecl(name="axis", type=TypeRef("Index"))],
                operands=[
                    OperandDecl(name="lhs", type=TypeRef("Type")),
                    OperandDecl(name="rhs", type=TypeRef("Type")),
                ],
                return_type=TypeRef("Type"),
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "axis: Value[Index]" in code
    assert "lhs: Value" in code
    assert "rhs: Value" in code
    assert '__params__ = (("axis", Index),)' in code
    assert '__operands__ = (("lhs", Type), ("rhs", Type),)' in code


def test_generate_op_with_block():
    f = DgenFile(
        traits=[TraitDecl(name="HasSingleBlock")],
        ops=[
            OpDecl(
                name="for",
                params=[
                    ParamDecl(name="lo", type=TypeRef("Index")),
                    ParamDecl(name="hi", type=TypeRef("Index")),
                ],
                return_type=TypeRef("Nil"),
                blocks=["body"],
            )
        ],
    )
    code = generate(f, dialect_name="test")
    assert "class ForOp(HasSingleBlock, Op):" in code
    assert "body: Block" in code
    assert '__blocks__ = ("body",)' in code


def test_generate_op_return_default():
    """Concrete return type generates a default."""
    f = DgenFile(
        types=[TypeDecl(name="Nil", layout="Void")],
        ops=[
            OpDecl(
                name="store",
                operands=[
                    OperandDecl(name="value", type=TypeRef("Type")),
                    OperandDecl(name="ptr", type=TypeRef("Type")),
                ],
                return_type=TypeRef("Nil"),
            )
        ],
    )
    code = generate(f, dialect_name="test")
    assert "type: Type = Nil()" in code


def test_generate_op_return_generic():
    """Return type 'Type' means no default."""
    f = DgenFile(
        ops=[
            OpDecl(
                name="transpose",
                operands=[OperandDecl(name="input", type=TypeRef("Type"))],
                return_type=TypeRef("Type"),
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "type: Type\n" in code or "type: Type" in code
    # Should NOT have a default
    assert "type: Type =" not in code


def test_generate_op_default_operand():
    """Operand with default value."""
    f = DgenFile(
        ops=[
            OpDecl(
                name="return",
                operands=[
                    OperandDecl(name="value", type=TypeRef("Type"), default="Nil")
                ],
                return_type=TypeRef("Nil"),
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "value: Value | Nil = Nil()" in code


def test_generate_list_operand():
    f = DgenFile(
        ops=[
            OpDecl(
                name="pack",
                operands=[
                    OperandDecl(name="values", type=TypeRef("list", [TypeRef("Type")]))
                ],
                return_type=TypeRef("List"),
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "values: list[Value]" in code


def test_generate_list_param():
    f = DgenFile(
        ops=[
            OpDecl(
                name="phi",
                params=[
                    ParamDecl(name="labels", type=TypeRef("list", [TypeRef("String")]))
                ],
                operands=[
                    OperandDecl(name="values", type=TypeRef("list", [TypeRef("Type")]))
                ],
                return_type=TypeRef("Type"),
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "labels: list[Value[String]]" in code
    assert "values: list[Value]" in code
    assert '__params__ = (("labels", String),)' in code


def test_generate_op_typed_operand():
    """Operand with specific type (not generic Type)."""
    f = DgenFile(
        ops=[
            OpDecl(
                name="alloc",
                operands=[OperandDecl(name="shape", type=TypeRef("Shape"))],
                return_type=TypeRef("Type"),
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "shape: Value" in code
    assert '__operands__ = (("shape", Shape),)' in code


def test_generate_imports():
    f = DgenFile(imports=[ImportDecl(module="builtin", names=["Index", "Nil"])])
    code = generate(
        f,
        dialect_name="test",
        import_map={"builtin": "dgen.dialects.builtin"},
    )
    assert "from dgen.dialects.builtin import Index, Nil" in code


def test_generate_imported_trait():
    """Ops with blocks should inherit HasSingleBlock even when it's imported, not local."""
    f = DgenFile(
        imports=[
            ImportDecl(module="builtin", names=["Index", "Nil", "HasSingleBlock"])
        ],
        ops=[
            OpDecl(
                name="for",
                params=[
                    ParamDecl(name="lo", type=TypeRef("Index")),
                    ParamDecl(name="hi", type=TypeRef("Index")),
                ],
                return_type=TypeRef("Nil"),
                blocks=["body"],
            )
        ],
    )
    code = generate(
        f,
        dialect_name="test",
        import_map={"builtin": "dgen.dialects.builtin"},
    )
    assert "HasSingleBlock" in code
    assert "HasSingleBlockType" not in code
    assert "class ForOp(HasSingleBlock, Op):" in code


def test_generate_valid_python():
    """The generated code should be valid Python that can be exec'd."""
    f = DgenFile(
        types=[
            TypeDecl(name="Index", data=DataField(name="data", type=TypeRef("Index")))
        ],
        ops=[
            OpDecl(
                name="nop",
                return_type=TypeRef("Nil"),
            )
        ],
    )
    code = generate(f, dialect_name="test")
    # Should be parseable as Python
    compile(code, "<test>", "exec")


def test_generate_nil_data_field():
    """A type with data: Nil should get __layout__ = layout.Void()."""
    f = DgenFile(
        types=[
            TypeDecl(
                name="InferredShapeTensor",
                params=[
                    ParamDecl(name="dtype", type=TypeRef("Type"), default="F64"),
                ],
                data=DataField(name="data", type=TypeRef("Nil")),
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "class InferredShapeTensor(Type):" in code
    assert "__layout__ = layout.Void()" in code


def test_parse_layout_keyword():
    from dgen.gen.parser import parse

    f = parse("type Index:\n    layout Int\n")
    assert len(f.types) == 1
    assert f.types[0].name == "Index"
    assert f.types[0].layout == "Int"
    assert f.types[0].data is None


def test_generate_layout_keyword():
    """layout keyword generates static __layout__."""
    f = DgenFile(
        types=[TypeDecl(name="Index", layout="Int")]
    )
    code = generate(f, dialect_name="test")
    assert "class Index(Type):" in code
    assert "__layout__ = layout.Int()" in code


def test_generate_layout_keyword_pointer():
    """layout Pointer generates __layout__ = layout.Pointer(layout.Void())."""
    f = DgenFile(
        types=[TypeDecl(name="Ptr", layout="Pointer")]
    )
    code = generate(f, dialect_name="test")
    assert "class Ptr(Type):" in code
    assert "__layout__ = layout.Pointer(layout.Void())" in code


def test_generate_pointer_data():
    """Pointer<Nil> should generate __layout__ = layout.Pointer(layout.Void())."""
    f = DgenFile(
        types=[
            TypeDecl(
                name="MemRef",
                params=[
                    ParamDecl(name="shape", type=TypeRef("Shape")),
                    ParamDecl(name="dtype", type=TypeRef("Type"), default="F64"),
                ],
                data=DataField(name="data", type=TypeRef("Pointer", [TypeRef("Nil")])),
            )
        ]
    )
    code = generate(f, dialect_name="test")
    assert "class MemRef(Type):" in code
    assert "__layout__ = layout.Pointer(layout.Void())" in code
