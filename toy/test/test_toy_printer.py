"""Phase 1 tests: construct IR manually, print, verify output."""

from collections.abc import Sequence

import dgen
from dgen.block import BlockArgument
from dgen.dialects import algebra, builtin, function, index
from dgen.dialects.function import Function
from dgen.module import ConstantOp, PackOp, pack
from toy.dialects import shape_constant, toy


def inferred() -> dgen.Type:
    return toy.InferredShapeTensor()


def ranked(shape: Sequence[int]) -> dgen.Type:
    return toy.Tensor(shape=shape_constant(shape))


def test_constant_op():
    op = ConstantOp(
        name="0",
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        type=ranked([2, 3]),
    )
    assert (
        "\n".join(op.asm)
        == "%0 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]"
    )


def test_transpose_op():
    a = dgen.Value(name="a", type=builtin.Nil())
    op = toy.TransposeOp(name="0", input=a, type=inferred())
    assert (
        "\n".join(op.asm)
        == "%0 : toy.InferredShapeTensor<number.Float64> = toy.transpose(%a)"
    )


def test_reshape_op():
    v0 = dgen.Value(name="0", type=builtin.Nil())
    op = toy.ReshapeOp(name="1", input=v0, type=ranked([2, 3]))
    assert (
        "\n".join(op.asm)
        == "%1 : toy.Tensor<ndbuffer.Shape<2>([2, 3]), number.Float64> = toy.reshape(%0)"
    )


def test_mul_op():
    v0 = dgen.Value(name="0", type=builtin.Nil())
    v1 = dgen.Value(name="1", type=builtin.Nil())
    op = toy.MulOp(name="2", lhs=v0, rhs=v1, type=inferred())
    assert (
        "\n".join(op.asm)
        == "%2 : toy.InferredShapeTensor<number.Float64> = toy.mul(%0, %1)"
    )


def test_add_op():
    v0 = dgen.Value(name="0", type=builtin.Nil())
    v1 = dgen.Value(name="1", type=builtin.Nil())
    op = toy.AddOp(name="2", lhs=v0, rhs=v1, type=inferred())
    assert (
        "\n".join(op.asm)
        == "%2 : toy.InferredShapeTensor<number.Float64> = toy.add(%0, %1)"
    )


def test_call_op():
    v1 = dgen.Value(name="1", type=builtin.Nil())
    v3 = dgen.Value(name="3", type=builtin.Nil())
    callee = dgen.Value(name="multiply_transpose", type=builtin.Nil())
    pack = PackOp(
        name="p",
        values=[v1, v3],
        type=builtin.Span(pointee=inferred()),
    )
    op = function.CallOp(
        name="4",
        callee=callee,
        arguments=pack,
        type=inferred(),
    )
    assert (
        "\n".join(op.asm)
        == "%4 : toy.InferredShapeTensor<number.Float64> = function.call<%multiply_transpose>([%1, %3])"
    )


def test_print_op():
    v5 = dgen.Value(name="5", type=builtin.Nil())
    op = toy.PrintOp(input=v5)
    # PrintOp has no name -> auto-numbered as %0
    assert "\n".join(op.asm) == "%0 : Nil = toy.print(%5)"


def test_concat_op():
    v0 = dgen.Value(name="0", type=builtin.Nil())
    v1 = dgen.Value(name="1", type=builtin.Nil())
    op = toy.ConcatOp(
        name="2",
        axis=index.Index().constant(0),
        lhs=v0,
        rhs=v1,
        type=inferred(),
    )
    assert (
        "\n".join(op.asm)
        == "%2 : toy.InferredShapeTensor<number.Float64> = toy.concat<0>(%0, %1)"
    )


def test_tile_op():
    v0 = dgen.Value(name="0", type=builtin.Nil())
    n = dgen.Value(name="n", type=index.Index())
    op = toy.TileOp(name="1", input=v0, count=n, type=inferred())
    assert (
        "\n".join(op.asm)
        == "%1 : toy.InferredShapeTensor<number.Float64> = toy.tile<%n>(%0)"
    )


def test_add_index_op():
    x = dgen.Value(name="x", type=index.Index())
    y = dgen.Value(name="y", type=index.Index())
    op = algebra.AddOp(name="0", left=x, right=y, type=index.Index())
    assert "\n".join(op.asm) == "%0 : index.Index = algebra.add(%x, %y)"


def test_full_module(ir_snapshot):
    # Build multiply_transpose function
    mt_arg_a = BlockArgument(name="a", type=inferred())
    mt_arg_b = BlockArgument(name="b", type=inferred())

    t0 = toy.TransposeOp(input=mt_arg_a, type=inferred())
    t1 = toy.TransposeOp(input=mt_arg_b, type=inferred())
    m0 = toy.MulOp(lhs=t0, rhs=t1, type=inferred())

    mt_func = function.FunctionOp(
        name="multiply_transpose",
        result_type=inferred(),
        body=dgen.Block(result=m0, args=[mt_arg_a, mt_arg_b]),
        type=Function(
            arguments=pack([mt_arg_a.type, mt_arg_b.type]), result_type=inferred()
        ),
    )

    # Build main function
    c0 = ConstantOp(
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        type=ranked([2, 3]),
    )
    r1 = toy.ReshapeOp(input=c0, type=ranked([2, 3]))
    c2 = ConstantOp(
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        type=ranked([6]),
    )
    r3 = toy.ReshapeOp(input=c2, type=ranked([2, 3]))
    pack4 = pack([r1, r3])
    function.CallOp(
        callee=mt_func,
        arguments=pack4,
        type=inferred(),
    )
    pack5 = pack([r3, r1])
    call5 = function.CallOp(
        callee=mt_func,
        arguments=pack5,
        type=inferred(),
    )
    print_op = toy.PrintOp(input=call5)

    main_func = function.FunctionOp(
        name="main",
        result_type=builtin.Nil(),
        body=dgen.Block(result=print_op, args=[], captures=[mt_func]),
        type=Function(arguments=pack(), result_type=builtin.Nil()),
    )

    # main_func captures mt_func, so it is the single root.
    assert main_func == ir_snapshot
