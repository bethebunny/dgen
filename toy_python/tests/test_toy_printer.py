"""Phase 1 tests: construct IR manually, print, verify output."""

from toy_python.dialects.toy_ops import (
    Module,
    FuncOp,
    Block,
    ToyValue,
    AnyToyType,
    ConstantOp,
    TransposeOp,
    ReshapeOp,
    MulOp,
    AddOp,
    GenericCallOp,
    PrintOp,
    ReturnOp,
    UnrankedTensorType,
    RankedTensorType,
    FunctionType,
)
from toy_python.dialects.toy_printer import print_module, print_op


def unranked() -> AnyToyType:
    return UnrankedTensorType()


def ranked(shape: list[int]) -> AnyToyType:
    return RankedTensorType(shape=shape)


def test_constant_op():
    op = ConstantOp(
        result="0",
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        shape=[2, 3],
        type=ranked([2, 3]),
    )
    assert (
        print_op(op)
        == "%0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>"
    )


def test_transpose_op():
    op = TransposeOp(result="0", input="a", type=unranked())
    assert print_op(op) == "%0 = Transpose(%a) : tensor<*xf64>"


def test_reshape_op():
    op = ReshapeOp(result="1", input="0", type=ranked([2, 3]))
    assert print_op(op) == "%1 = Reshape(%0) : tensor<2x3xf64>"


def test_mul_op():
    op = MulOp(result="2", lhs="0", rhs="1", type=unranked())
    assert print_op(op) == "%2 = Mul(%0, %1) : tensor<*xf64>"


def test_add_op():
    op = AddOp(result="2", lhs="0", rhs="1", type=unranked())
    assert print_op(op) == "%2 = Add(%0, %1) : tensor<*xf64>"


def test_generic_call_op():
    op = GenericCallOp(
        result="4",
        callee="multiply_transpose",
        args=["1", "3"],
        type=unranked(),
    )
    assert (
        print_op(op)
        == "%4 = GenericCall @multiply_transpose(%1, %3) : tensor<*xf64>"
    )


def test_print_op():
    op = PrintOp(input="5")
    assert print_op(op) == "Print(%5)"


def test_return_op_with_value():
    op = ReturnOp(value="2")
    assert print_op(op) == "return %2"


def test_return_op_void():
    op = ReturnOp(value=None)
    assert print_op(op) == "return"


def test_full_module():
    # Build multiply_transpose function
    mt_args = [
        ToyValue(name="a", type=unranked()),
        ToyValue(name="b", type=unranked()),
    ]

    mt_ops = [
        TransposeOp(result="0", input="a", type=unranked()),
        TransposeOp(result="1", input="b", type=unranked()),
        MulOp(result="2", lhs="0", rhs="1", type=unranked()),
        ReturnOp(value="2"),
    ]

    mt_func_type = FunctionType(
        inputs=[unranked(), unranked()], result=unranked()
    )
    mt_func = FuncOp(
        name="multiply_transpose",
        func_type=mt_func_type,
        body=Block(args=mt_args, ops=mt_ops),
    )

    # Build main function
    main_ops = [
        ConstantOp(
            result="0",
            value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape=[2, 3],
            type=ranked([2, 3]),
        ),
        ReshapeOp(result="1", input="0", type=ranked([2, 3])),
        ConstantOp(
            result="2",
            value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape=[6],
            type=ranked([6]),
        ),
        ReshapeOp(result="3", input="2", type=ranked([2, 3])),
        GenericCallOp(
            result="4",
            callee="multiply_transpose",
            args=["1", "3"],
            type=unranked(),
        ),
        GenericCallOp(
            result="5",
            callee="multiply_transpose",
            args=["3", "1"],
            type=unranked(),
        ),
        PrintOp(input="5"),
        ReturnOp(value=None),
    ]

    main_func_type = FunctionType(inputs=[], result=None)
    main_func = FuncOp(
        name="main",
        func_type=main_func_type,
        body=Block(args=[], ops=main_ops),
    )

    module = Module(functions=[mt_func, main_func])

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
        "    return\n"
    )
    assert print_module(module) == expected
