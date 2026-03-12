"""Tests for .dgen file parser."""

from dgen.gen.parser import parse


def test_parse_import():
    result = parse("from builtin import Index, Nil\n")
    assert len(result.imports) == 1
    assert result.imports[0].module == "builtin"
    assert result.imports[0].names == ["Index", "Nil"]


def test_parse_multiple_imports():
    result = parse("from builtin import Index, Nil\nfrom affine import Shape\n")
    assert len(result.imports) == 2
    assert result.imports[1].module == "affine"
    assert result.imports[1].names == ["Shape"]


def test_parse_trait():
    result = parse("trait HasSingleBlock\n")
    assert len(result.traits) == 1
    assert result.traits[0].name == "HasSingleBlock"


def test_parse_simple_type():
    result = parse("type index:\n    data: Index\n")
    assert len(result.types) == 1
    t = result.types[0]
    assert t.name == "index"
    assert len(t.data) == 1
    assert t.data[0].name == "data"
    assert t.data[0].type.name == "Index"


def test_parse_type_no_body():
    result = parse("type Nil\n")
    assert len(result.types) == 1
    t = result.types[0]
    assert t.name == "Nil"
    assert t.data == []


def test_parse_parameterized_type():
    result = parse("type Shape<rank: Index>:\n    dims: Array<Index, rank>\n")
    t = result.types[0]
    assert t.name == "Shape"
    assert len(t.params) == 1
    assert t.params[0].name == "rank"
    assert t.params[0].type.name == "Index"
    assert len(t.data) == 1
    assert t.data[0].name == "dims"
    assert t.data[0].type.name == "Array"
    assert len(t.data[0].type.args) == 2
    assert t.data[0].type.args[0].name == "Index"
    assert t.data[0].type.args[1].name == "rank"


def test_parse_type_with_default_param():
    result = parse("type Tensor<shape: Shape, dtype: Type = F64>:\n    data: Nil\n")
    t = result.types[0]
    assert len(t.params) == 2
    assert t.params[0].name == "shape"
    assert t.params[0].default is None
    assert t.params[1].name == "dtype"
    assert t.params[1].type.name == "Type"
    assert t.params[1].default == "F64"


def test_parse_type_fatpointer_field():
    result = parse("type String:\n    storage: FatPointer<Byte>\n")
    t = result.types[0]
    assert len(t.data) == 1
    assert t.data[0].name == "storage"
    assert t.data[0].type.name == "FatPointer"
    assert len(t.data[0].type.args) == 1
    assert t.data[0].type.args[0].name == "Byte"


def test_parse_simple_op():
    result = parse("op transpose(input: Type) -> Type\n")
    op = result.ops[0]
    assert op.name == "transpose"
    assert len(op.operands) == 1
    assert op.operands[0].name == "input"
    assert op.operands[0].type is not None
    assert op.operands[0].type.name == "Type"
    assert op.return_type is not None
    assert op.return_type.name == "Type"


def test_parse_op_with_params():
    result = parse("op concat<axis: Index>(lhs: Type, rhs: Type) -> Type\n")
    op = result.ops[0]
    assert len(op.params) == 1
    assert op.params[0].name == "axis"
    assert op.params[0].type.name == "Index"
    assert len(op.operands) == 2
    assert op.operands[0].name == "lhs"
    assert op.operands[1].name == "rhs"


def test_parse_op_with_block():
    src = "op for<lo: Index, hi: Index>() -> Nil:\n    block body\n"
    op = parse(src).ops[0]
    assert op.blocks == ["body"]
    assert op.return_type is not None
    assert op.return_type.name == "Nil"
    assert len(op.params) == 2


def test_parse_op_with_default_operand():
    result = parse("op return(value: Type = Nil) -> Nil\n")
    op = result.ops[0]
    assert op.operands[0].name == "value"
    assert op.operands[0].type is not None
    assert op.operands[0].type.name == "Type"
    assert op.operands[0].default == "Nil"


def test_parse_list_operand():
    result = parse("op pack(values: list<Type>) -> List\n")
    op = result.ops[0]
    assert op.operands[0].type is not None
    assert op.operands[0].type.name == "list"
    assert op.operands[0].type.args[0].name == "Type"


def test_parse_op_with_list_param():
    result = parse("op phi<labels: list<String>>(values: list<Type>) -> Type\n")
    op = result.ops[0]
    assert op.params[0].type.name == "list"
    assert op.params[0].type.args[0].name == "String"
    assert op.operands[0].type is not None
    assert op.operands[0].type.name == "list"
    assert op.operands[0].type.args[0].name == "Type"


def test_parse_no_operands():
    result = parse("op function() -> Function:\n    block body\n")
    op = result.ops[0]
    assert op.operands == []
    assert op.blocks == ["body"]


def test_parse_op_no_params_no_operands_no_body():
    result = parse("op nop() -> Nil\n")
    op = result.ops[0]
    assert op.params == []
    assert op.operands == []
    assert op.blocks == []
    assert op.return_type is not None
    assert op.return_type.name == "Nil"


def test_parse_comments_and_blank_lines():
    src = """\
# This is a comment
from builtin import Index

# Another comment

type index:
    data: Index
"""
    result = parse(src)
    assert len(result.imports) == 1
    assert len(result.types) == 1


def test_parse_full_file():
    src = """\
from builtin import Index, Nil

trait HasSingleBlock

type Shape<rank: Index>:
    dims: Array<Index, rank>

op alloc(shape: Shape) -> Type
op for<lo: Index, hi: Index>() -> Nil:
    block body
"""
    result = parse(src)
    assert len(result.imports) == 1
    assert len(result.traits) == 1
    assert len(result.types) == 1
    assert len(result.ops) == 2
    assert result.ops[1].blocks == ["body"]


def test_parse_type_comment_in_body():
    src = """\
type Tensor<shape: Shape>:
    # Layout computed externally
"""
    result = parse(src)
    t = result.types[0]
    assert t.name == "Tensor"
    assert t.data == []


def test_parse_untyped_operand():
    """Operands without type annotation default to no constraint."""
    result = parse("op transpose(input) -> Type\n")
    op = result.ops[0]
    assert op.operands[0].name == "input"
    assert op.operands[0].type is None


def test_parse_mixed_typed_untyped_operands():
    """Mix of typed and untyped operands."""
    result = parse("op add(lhs, rhs: Type) -> Type\n")
    op = result.ops[0]
    assert op.operands[0].name == "lhs"
    assert op.operands[0].type is None
    assert op.operands[1].name == "rhs"
    assert op.operands[1].type is not None
    assert op.operands[1].type.name == "Type"


def test_parse_op_no_return_type():
    """Op with no -> clause has None return type."""
    result = parse("op nop()\n")
    op = result.ops[0]
    assert op.return_type is None


def test_parse_multiple_block_lines():
    src = """\
op multi() -> Nil:
    block first
    block second
"""
    result = parse(src)
    op = result.ops[0]
    assert op.blocks == ["first", "second"]


def test_parse_multiple_data_fields():
    src = "type Foo:\n    x: Index\n    y: F64\n"
    result = parse(src)
    t = result.types[0]
    assert len(t.data) == 2
    assert t.data[0].name == "x"
    assert t.data[0].type.name == "Index"
    assert t.data[1].name == "y"
    assert t.data[1].type.name == "F64"


def test_parse_has_trait_on_type():
    src = "type Float64:\n    has trait FloatingPoint\n"
    result = parse(src)
    assert result.types[0].traits == ["FloatingPoint"]


def test_parse_bare_list_operand():
    """Bare list (no inner type) in operand position parses as a type named 'list'."""
    result = parse("op pack(values: list)\n")
    op = result.ops[0]
    assert op.operands[0].type is not None
    assert op.operands[0].type.name == "list"


def test_parse_has_trait_on_op():
    src = "op for() -> Nil:\n    block body\n    has trait HasSingleBlock\n"
    result = parse(src)
    assert result.ops[0].traits == ["HasSingleBlock"]
    assert result.ops[0].blocks == ["body"]


def test_parse_namespace_import():
    result = parse("import affine\n")
    assert len(result.imports) == 1
    assert result.imports[0].module == "affine"
    assert result.imports[0].names == []


def test_parse_qualified_type_ref():
    result = parse("type Tensor<shape: affine.Shape>:\n    data: Nil\n")
    t = result.types[0]
    assert t.params[0].type.name == "affine.Shape"


def test_parse_trait_with_static_fields():
    src = "trait DType:\n    static signed: Boolean\n    static bitwidth: Index\n"
    result = parse(src)
    t = result.traits[0]
    assert t.name == "DType"
    assert len(t.statics) == 2
    assert t.statics[0].name == "signed"
    assert t.statics[0].type.name == "Boolean"
    assert t.statics[1].name == "bitwidth"
    assert t.statics[1].type.name == "Index"


def test_parse_trait_with_static_default():
    src = "trait DType:\n    static bitwidth: Index = 64\n"
    result = parse(src)
    t = result.traits[0]
    assert len(t.statics) == 1
    assert t.statics[0].name == "bitwidth"
    assert t.statics[0].type.name == "Index"
    assert t.statics[0].default == "64"


def test_parse_bare_trait_still_works():
    result = parse("trait HasSingleBlock\n")
    assert len(result.traits) == 1
    assert result.traits[0].name == "HasSingleBlock"
    assert result.traits[0].statics == []


def test_parse_type_with_static_fields():
    src = "type F64:\n    has trait FloatingPoint\n    static bitwidth: Index = 64\n"
    result = parse(src)
    t = result.types[0]
    assert t.traits == ["FloatingPoint"]
    assert len(t.statics) == 1
    assert t.statics[0].name == "bitwidth"
    assert t.statics[0].default == "64"


def test_parse_type_with_static_no_default():
    src = "type F64:\n    static signed: Boolean\n"
    result = parse(src)
    t = result.types[0]
    assert len(t.statics) == 1
    assert t.statics[0].name == "signed"
    assert t.statics[0].type.name == "Boolean"
    assert t.statics[0].default is None


def test_parse_metavar_operand():
    """$X in operand position is a metavariable."""
    result = parse("op tile(x: $X) -> $Result\n")
    op = result.ops[0]
    assert op.operands[0].name == "x"
    assert op.operands[0].type is not None
    assert op.operands[0].type.name == "$X"
    assert op.return_type is not None
    assert op.return_type.name == "$Result"


def test_parse_requires_match():
    src = "op tile(x: $X) -> $Result:\n    requires $X ~= Tensor\n"
    op = parse(src).ops[0]
    assert len(op.constraints) == 1
    c = op.constraints[0]
    assert c.kind == "match"
    assert c.lhs == "$X"
    assert c.pattern == "Tensor"


def test_parse_requires_eq():
    src = "op sqrt(x: $X) -> $Result:\n    requires $X == $Result\n"
    op = parse(src).ops[0]
    assert len(op.constraints) == 1
    c = op.constraints[0]
    assert c.kind == "eq"
    assert c.lhs == "$X"
    assert c.rhs == "$Result"


def test_parse_requires_expr():
    src = "op tile<axis: Index>(x: $X) -> $Result:\n    requires axis < $X.rank\n"
    op = parse(src).ops[0]
    assert len(op.constraints) == 1
    c = op.constraints[0]
    assert c.kind == "expr"
    assert c.expr == "axis < $X.rank"


def test_parse_multiple_requires():
    src = """\
op tile<axis: Index>(x: $X) -> $Result:
    requires $X ~= Tensor
    requires $Result ~= Tensor
    requires $X.dtype == $Result.dtype
"""
    op = parse(src).ops[0]
    assert len(op.constraints) == 3
    assert op.constraints[0].kind == "match"
    assert op.constraints[1].kind == "match"
    assert op.constraints[2].kind == "expr"


def test_parse_requires_with_block():
    src = "op foo() -> Nil:\n    block body\n    requires $X ~= Tensor\n"
    op = parse(src).ops[0]
    assert op.blocks == ["body"]
    assert len(op.constraints) == 1
