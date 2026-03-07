"""Tests for type values as first-class SSA citizens."""

from dgen import Block, asm
from dgen.asm.formatting import format_expr
from dgen.asm.parser import IRParser, parse_expr, parse_module
from dgen.block import BlockArgument
from dgen.codegen import compile as compile_module
from dgen.dialects import builtin
from dgen.dialects.builtin import FunctionOp, Index
from dgen.module import ConstantOp, Function, Module
from dgen.type import Memory
from toy.test.helpers import strip_prefix


def test_parse_dict_literal():
    """parse_expr handles {key: value, ...} and returns a Python dict."""
    parser = IRParser('{"tag": "builtin.Index"}')
    result = parse_expr(parser)
    assert result == {"tag": "builtin.Index"}


def test_format_dict_literal():
    """format_expr handles dicts."""
    result = format_expr({"tag": "builtin.Index"})
    assert result == '{"tag": "builtin.Index"}'


def test_typetype_constant_asm_roundtrip():
    """TypeType constant with dict literal round-trips through ASM."""
    ir = strip_prefix("""
        | %main = function () -> ():
        |     %t : TypeType<Index> = {"tag": "builtin.Index"}
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_ssa_ref_as_op_type():
    """SSA ref in type position: %x's type is unresolved Value, op not ready."""
    ir = strip_prefix("""
        | %main = function () -> ():
        |     %t : TypeType<Index> = {"tag": "builtin.Index"}
        |     %x : %t = 42
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    ops = module.functions[0].body.ops
    t_op = ops[0]  # %t = TypeType constant
    x_op = ops[1]  # %x : %t = 42
    # %x's type is the SSA value %t, not a resolved Type
    assert x_op.type is t_op
    assert not x_op.ready


def test_ssa_ref_as_op_type_roundtrip():
    """SSA ref in type position round-trips through ASM."""
    ir = strip_prefix("""
        | %main = function () -> ():
        |     %t : TypeType<Index> = {"tag": "builtin.Index"}
        |     %x : %t = 42
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_typetype_memory_roundtrip():
    """TypeType constant serializes to dict and round-trips through Memory."""
    idx = Index()
    mem = idx.__constant__
    data = mem.to_json()
    assert data == {"tag": "builtin.Index"}
    # Round-trip: dict → Memory → dict
    mem2 = Memory.from_json(idx.type, data)
    assert mem2.to_json() == data


def test_parameterized_typetype_constant_roundtrip():
    """TypeType for Array<Index, 4> round-trips with nested params in dict."""
    arr_ty = builtin.Array(
        element_type=Index(),
        n=Index().constant(4),
    )
    mem = arr_ty.__constant__
    data = mem.to_json()
    assert data == {"tag": "builtin.Array", "element_type": {"tag": "builtin.Index"}, "n": 4}

    # ASM round-trip with the parameterized TypeType
    ir = strip_prefix("""
        | %main = function () -> ():
        |     %t : TypeType<Array<Index, 4>> = {"tag": "builtin.Array", "element_type": {"tag": "builtin.Index"}, "n": 4}
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir


def test_array_with_ssa_dimension():
    """Array<Index, %n> — SSA value as type parameter, round-trips through ASM."""
    ir = strip_prefix("""
        | %main = function () -> ():
        |     %n : Index = 4
        |     %arr : Array<Index, %n> = [1, 2, 3, 4]
        |     %_ : () = return(())
    """)
    module = parse_module(ir)
    assert asm.format(module) == ir

    # Verify the Array type's `n` param is the SSA value %n
    ops = module.functions[0].body.ops
    n_op = ops[0]
    arr_op = ops[1]
    assert isinstance(arr_op.type, builtin.Array)
    assert arr_op.type.n is n_op


def test_type_value_jit_identity():
    """TypeType value survives JIT identity function (passed as pointer)."""
    idx = Index()
    mem = idx.__constant__

    # Build: main(t: TypeType<Index>) -> TypeType<Index> { return t }
    arg = BlockArgument(name="t", type=idx.type)
    func = FunctionOp(
        name="main",
        body=Block(ops=[builtin.ReturnOp(value=arg)], args=[arg]),
        type=Function(result=idx.type),
    )
    exe = compile_module(Module(functions=[func]))
    result = exe.run(mem)
    # TypeType is pointer-passed (Record layout), verify address survives
    assert result == mem.address


def test_type_constant_jit_return():
    """JIT function returns a TypeType constant, read back via Memory.from_raw."""
    idx = Index()
    const = ConstantOp(value=idx.__constant__.to_json(), type=idx.type)
    ret = builtin.ReturnOp(value=const)
    func = FunctionOp(
        name="main",
        body=Block(ops=[const, ret], args=[]),
        type=Function(result=idx.type),
    )
    exe = compile_module(Module(functions=[func]))
    raw = exe.run()
    assert isinstance(raw, int)
    result = Memory.from_raw(idx.type, raw).to_json()
    assert result == {"tag": "builtin.Index"}
