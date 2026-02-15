"""Phase 1 tests: construct IR manually, print, verify output."""

from toy_python.dialects import builtin, toy
from toy_python import asm
from toy_python.tests.helpers import strip_prefix


def unranked() -> builtin.Type:
    return toy.UnrankedTensorType()


def ranked(shape: list[int]) -> builtin.Type:
    return toy.RankedTensorType(shape=shape)


def test_constant_op():
    op = toy.ConstantOp(
        result="0",
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        shape=[2, 3],
        type=ranked([2, 3]),
    )
    assert (
        asm.format(op)
        == "%0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>"
    )


def test_transpose_op():
    op = toy.TransposeOp(result="0", input="a", type=unranked())
    assert asm.format(op) == "%0 = toy.transpose(%a) : tensor<*xf64>"


def test_reshape_op():
    op = toy.ReshapeOp(result="1", input="0", type=ranked([2, 3]))
    assert asm.format(op) == "%1 = toy.reshape(%0) : tensor<2x3xf64>"


def test_mul_op():
    op = toy.MulOp(result="2", lhs="0", rhs="1", type=unranked())
    assert asm.format(op) == "%2 = toy.mul(%0, %1) : tensor<*xf64>"


def test_add_op():
    op = toy.AddOp(result="2", lhs="0", rhs="1", type=unranked())
    assert asm.format(op) == "%2 = toy.add(%0, %1) : tensor<*xf64>"


def test_generic_call_op():
    op = toy.GenericCallOp(
        result="4",
        callee="multiply_transpose",
        args=["1", "3"],
        type=unranked(),
    )
    assert (
        asm.format(op)
        == "%4 = toy.generic_call(@multiply_transpose, [%1, %3]) : tensor<*xf64>"
    )


def test_print_op():
    op = toy.PrintOp(result="_", input="5")
    assert asm.format(op) == "%_ = toy.print(%5)"


def test_return_op_with_value():
    op = builtin.ReturnOp(result="_", value="2")
    assert asm.format(op) == "%_ = return(%2)"


def test_return_op_void():
    op = builtin.ReturnOp(result="_", value=None)
    assert asm.format(op) == "%_ = return()"


def test_full_module():
    # Build multiply_transpose function
    mt_args = [
        builtin.Value(name="a", type=unranked()),
        builtin.Value(name="b", type=unranked()),
    ]

    mt_ops = [
        toy.TransposeOp(result="0", input="a", type=unranked()),
        toy.TransposeOp(result="1", input="b", type=unranked()),
        toy.MulOp(result="2", lhs="0", rhs="1", type=unranked()),
        builtin.ReturnOp(result="_", value="2"),
    ]

    mt_func_type = toy.FunctionType(
        inputs=[unranked(), unranked()], result=unranked()
    )
    mt_func = builtin.FuncOp(
        name="multiply_transpose",
        func_type=mt_func_type,
        body=builtin.Block(ops=mt_ops, args=mt_args),
    )

    # Build main function
    main_ops = [
        toy.ConstantOp(
            result="0",
            value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape=[2, 3],
            type=ranked([2, 3]),
        ),
        toy.ReshapeOp(result="1", input="0", type=ranked([2, 3])),
        toy.ConstantOp(
            result="2",
            value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape=[6],
            type=ranked([6]),
        ),
        toy.ReshapeOp(result="3", input="2", type=ranked([2, 3])),
        toy.GenericCallOp(
            result="4",
            callee="multiply_transpose",
            args=["1", "3"],
            type=unranked(),
        ),
        toy.GenericCallOp(
            result="5",
            callee="multiply_transpose",
            args=["3", "1"],
            type=unranked(),
        ),
        toy.PrintOp(result="_", input="5"),
        builtin.ReturnOp(result="_", value=None),
    ]

    main_func_type = toy.FunctionType(inputs=[], result=builtin.Nil())
    main_func = builtin.FuncOp(
        name="main",
        func_type=main_func_type,
        body=builtin.Block(ops=main_ops),
    )

    module = builtin.Module(functions=[mt_func, main_func])

    expected = strip_prefix("""
        | from builtin import function, return
        | import toy
        |
        | %multiply_transpose = function (%a: tensor<*xf64>, %b: tensor<*xf64>) -> tensor<*xf64>:
        |     %0 = toy.transpose(%a) : tensor<*xf64>
        |     %1 = toy.transpose(%b) : tensor<*xf64>
        |     %2 = toy.mul(%0, %1) : tensor<*xf64>
        |     %_ = return(%2)
        |
        | %main = function () -> ():
        |     %0 = toy.constant(<2x3>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>
        |     %1 = toy.reshape(%0) : tensor<2x3xf64>
        |     %2 = toy.constant(<6>, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<6xf64>
        |     %3 = toy.reshape(%2) : tensor<2x3xf64>
        |     %4 = toy.generic_call(@multiply_transpose, [%1, %3]) : tensor<*xf64>
        |     %5 = toy.generic_call(@multiply_transpose, [%3, %1]) : tensor<*xf64>
        |     %_ = toy.print(%5)
        |     %_ = return()
    """)
    assert asm.format(module) == expected
