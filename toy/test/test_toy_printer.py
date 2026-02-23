"""Phase 1 tests: construct IR manually, print, verify output."""

import dgen
from dgen import asm
from dgen.block import BlockArgument
from dgen.dialects import builtin
from toy.dialects import toy
from toy.test.helpers import strip_prefix


def inferred() -> dgen.Type:
    return toy.InferredShapeTensor()


def ranked(shape: list[int]) -> dgen.Type:
    return toy.TensorType(shape=shape)


def test_constant_op():
    op = builtin.ConstantOp(
        name="0",
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        type=ranked([2, 3]),
    )
    assert (
        asm.format(op)
        == "%0 : toy.Tensor([2, 3], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]"
    )


def test_transpose_op():
    a = dgen.Value(name="a", type=builtin.Nil())
    op = toy.TransposeOp(name="0", input=a, type=inferred())
    assert asm.format(op) == "%0 : toy.InferredShapeTensor(f64) = toy.transpose(%a)"


def test_reshape_op():
    v0 = dgen.Value(name="0", type=builtin.Nil())
    op = toy.ReshapeOp(name="1", input=v0, type=ranked([2, 3]))
    assert asm.format(op) == "%1 : toy.Tensor([2, 3], f64) = toy.reshape(%0)"


def test_mul_op():
    v0 = dgen.Value(name="0", type=builtin.Nil())
    v1 = dgen.Value(name="1", type=builtin.Nil())
    op = toy.MulOp(name="2", lhs=v0, rhs=v1, type=inferred())
    assert asm.format(op) == "%2 : toy.InferredShapeTensor(f64) = toy.mul(%0, %1)"


def test_add_op():
    v0 = dgen.Value(name="0", type=builtin.Nil())
    v1 = dgen.Value(name="1", type=builtin.Nil())
    op = toy.AddOp(name="2", lhs=v0, rhs=v1, type=inferred())
    assert asm.format(op) == "%2 : toy.InferredShapeTensor(f64) = toy.add(%0, %1)"


def test_generic_call_op():
    v1 = dgen.Value(name="1", type=builtin.Nil())
    v3 = dgen.Value(name="3", type=builtin.Nil())
    op = toy.GenericCallOp(
        name="4",
        callee="multiply_transpose",
        args=[v1, v3],
        type=inferred(),
    )
    assert (
        asm.format(op)
        == '%4 : toy.InferredShapeTensor(f64) = toy.generic_call("multiply_transpose", [%1, %3])'
    )


def test_print_op():
    v5 = dgen.Value(name="5", type=builtin.Nil())
    op = toy.PrintOp(input=v5)
    # PrintOp has no name -> auto-numbered as %0
    assert asm.format(op) == "%0 : () = toy.print(%5)"


def test_return_op_with_value():
    v2 = dgen.Value(name="2", type=builtin.Nil())
    op = builtin.ReturnOp(value=v2)
    assert asm.format(op) == "%0 : () = return(%2)"


def test_return_op_void():
    op = builtin.ReturnOp()
    assert asm.format(op) == "%0 : () = return()"


def test_concat_op():
    v0 = dgen.Value(name="0", type=builtin.Nil())
    v1 = dgen.Value(name="1", type=builtin.Nil())
    op = toy.ConcatOp(name="2", lhs=v0, rhs=v1, axis=0, type=inferred())
    assert (
        asm.format(op)
        == "%2 : toy.InferredShapeTensor(f64) = toy.concat(%0, %1, 0)"
    )


def test_tile_op():
    v0 = dgen.Value(name="0", type=builtin.Nil())
    n = dgen.Value(name="n", type=builtin.IndexType())
    op = toy.TileOp(name="1", input=v0, count=n, type=inferred())
    assert (
        asm.format(op)
        == "%1 : toy.InferredShapeTensor(f64) = toy.tile(%0, %n)"
    )


def test_add_index_op():
    x = dgen.Value(name="x", type=builtin.IndexType())
    y = dgen.Value(name="y", type=builtin.IndexType())
    op = builtin.AddIndexOp(name="0", lhs=x, rhs=y)
    assert asm.format(op) == "%0 : index = add_index(%x, %y)"


def test_full_module():
    # Build multiply_transpose function
    mt_arg_a = BlockArgument(name="a", type=inferred())
    mt_arg_b = BlockArgument(name="b", type=inferred())

    t0 = toy.TransposeOp(input=mt_arg_a, type=inferred())
    t1 = toy.TransposeOp(input=mt_arg_b, type=inferred())
    m0 = toy.MulOp(lhs=t0, rhs=t1, type=inferred())
    ret_mt = builtin.ReturnOp(value=m0)

    mt_func_type = toy.FunctionType(inputs=[inferred(), inferred()], result=inferred())
    mt_func = builtin.FuncOp(
        name="multiply_transpose",
        type=mt_func_type,
        body=dgen.Block(ops=[t0, t1, m0, ret_mt], args=[mt_arg_a, mt_arg_b]),
    )

    # Build main function
    c0 = builtin.ConstantOp(
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        type=ranked([2, 3]),
    )
    r1 = toy.ReshapeOp(input=c0, type=ranked([2, 3]))
    c2 = builtin.ConstantOp(
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        type=ranked([6]),
    )
    r3 = toy.ReshapeOp(input=c2, type=ranked([2, 3]))
    call4 = toy.GenericCallOp(
        callee="multiply_transpose",
        args=[r1, r3],
        type=inferred(),
    )
    call5 = toy.GenericCallOp(
        callee="multiply_transpose",
        args=[r3, r1],
        type=inferred(),
    )
    print_op = toy.PrintOp(input=call5)
    ret_main = builtin.ReturnOp()

    main_func_type = toy.FunctionType(inputs=[], result=builtin.Nil())
    main_func = builtin.FuncOp(
        name="main",
        type=main_func_type,
        body=dgen.Block(
            ops=[c0, r1, c2, r3, call4, call5, print_op, ret_main], args=[]
        ),
    )

    module = builtin.Module(functions=[mt_func, main_func])

    expected = strip_prefix("""
        | import toy
        |
        | %multiply_transpose = function (%a: toy.InferredShapeTensor(f64), %b: toy.InferredShapeTensor(f64)) -> toy.InferredShapeTensor(f64):
        |     %0 : toy.InferredShapeTensor(f64) = toy.transpose(%a)
        |     %1 : toy.InferredShapeTensor(f64) = toy.transpose(%b)
        |     %2 : toy.InferredShapeTensor(f64) = toy.mul(%0, %1)
        |     %3 : () = return(%2)
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor([2, 3], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor([2, 3], f64) = toy.reshape(%0)
        |     %2 : toy.Tensor([6], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %3 : toy.Tensor([2, 3], f64) = toy.reshape(%2)
        |     %4 : toy.InferredShapeTensor(f64) = toy.generic_call("multiply_transpose", [%1, %3])
        |     %5 : toy.InferredShapeTensor(f64) = toy.generic_call("multiply_transpose", [%3, %1])
        |     %6 : () = toy.print(%5)
        |     %7 : () = return()
    """)
    assert asm.format(module) == expected
