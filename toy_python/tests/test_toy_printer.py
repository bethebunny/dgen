"""Phase 1 tests: construct IR manually, print, verify output."""

from toy_python.dialects import toy
from toy_python import asm


def unranked() -> toy.AnyType:
    return toy.UnrankedTensorType()


def ranked(shape: list[int]) -> toy.AnyType:
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
        == "%0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>"
    )


def test_transpose_op():
    op = toy.TransposeOp(result="0", input="a", type=unranked())
    assert asm.format(op) == "%0 = Transpose(%a) : tensor<*xf64>"


def test_reshape_op():
    op = toy.ReshapeOp(result="1", input="0", type=ranked([2, 3]))
    assert asm.format(op) == "%1 = Reshape(%0) : tensor<2x3xf64>"


def test_mul_op():
    op = toy.MulOp(result="2", lhs="0", rhs="1", type=unranked())
    assert asm.format(op) == "%2 = Mul(%0, %1) : tensor<*xf64>"


def test_add_op():
    op = toy.AddOp(result="2", lhs="0", rhs="1", type=unranked())
    assert asm.format(op) == "%2 = Add(%0, %1) : tensor<*xf64>"


def test_generic_call_op():
    op = toy.GenericCallOp(
        result="4",
        callee="multiply_transpose",
        args=["1", "3"],
        type=unranked(),
    )
    assert (
        asm.format(op)
        == "%4 = GenericCall @multiply_transpose(%1, %3) : tensor<*xf64>"
    )


def test_print_op():
    op = toy.PrintOp(input="5")
    assert asm.format(op) == "Print(%5)"


def test_return_op_with_value():
    op = toy.ReturnOp(value="2")
    assert asm.format(op) == "return %2"


def test_return_op_void():
    op = toy.ReturnOp(value=None)
    assert asm.format(op) == "return"


def test_full_module():
    # Build multiply_transpose function
    mt_args = [
        toy.Value(name="a", type=unranked()),
        toy.Value(name="b", type=unranked()),
    ]

    mt_ops = [
        toy.TransposeOp(result="0", input="a", type=unranked()),
        toy.TransposeOp(result="1", input="b", type=unranked()),
        toy.MulOp(result="2", lhs="0", rhs="1", type=unranked()),
        toy.ReturnOp(value="2"),
    ]

    mt_func_type = toy.FunctionType(
        inputs=[unranked(), unranked()], result=unranked()
    )
    mt_func = toy.FuncOp(
        name="multiply_transpose",
        func_type=mt_func_type,
        body=toy.Block(args=mt_args, ops=mt_ops),
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
        toy.PrintOp(input="5"),
        toy.ReturnOp(value=None),
    ]

    main_func_type = toy.FunctionType(inputs=[], result=None)
    main_func = toy.FuncOp(
        name="main",
        func_type=main_func_type,
        body=toy.Block(args=[], ops=main_ops),
    )

    module = toy.Module(functions=[mt_func, main_func])

    expected = (
        "from toy use *\n"
        "\n"
        "%multiply_transpose = function (%a: tensor<*xf64>, %b: tensor<*xf64>) -> tensor<*xf64>:\n"
        "    %0 = Transpose(%a) : tensor<*xf64>\n"
        "    %1 = Transpose(%b) : tensor<*xf64>\n"
        "    %2 = Mul(%0, %1) : tensor<*xf64>\n"
        "    return %2\n"
        "\n"
        "%main = function ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Reshape(%0) : tensor<2x3xf64>\n"
        "    %2 = Constant(<6> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<6xf64>\n"
        "    %3 = Reshape(%2) : tensor<2x3xf64>\n"
        "    %4 = GenericCall @multiply_transpose(%1, %3) : tensor<*xf64>\n"
        "    %5 = GenericCall @multiply_transpose(%3, %1) : tensor<*xf64>\n"
        "    Print(%5)\n"
        "    return"
    )
    assert asm.format(module) == expected
