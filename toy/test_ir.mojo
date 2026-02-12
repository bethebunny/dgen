"""Phase 1 tests: construct IR manually, print, verify output."""

from testing import assert_equal, TestSuite
from collections import Optional

from toy.toy import (
    Module, FuncOp, Block, ToyValue, AnyToyOp, AnyToyType,
    ConstantOp, TransposeOp, ReshapeOp, MulOp, AddOp,
    GenericCallOp, PrintOp, ReturnOp,
    UnrankedTensorType, RankedTensorType, FunctionType,
)
from toy.printer import print_module, print_op


fn unranked() -> AnyToyType:
    return AnyToyType(UnrankedTensorType())


fn ranked(var shape: List[Int]) -> AnyToyType:
    return AnyToyType(RankedTensorType(shape=shape^))


def test_constant_op():
    var op = AnyToyOp(ConstantOp(
        result="0",
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        shape=[2, 3],
        type=ranked([2, 3]),
    ))
    assert_equal(
        print_op(op),
        "%0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>",
    )


def test_transpose_op():
    var op = AnyToyOp(TransposeOp(result="0", input="a", type=unranked()))
    assert_equal(print_op(op), "%0 = Transpose(%a) : tensor<*xf64>")


def test_reshape_op():
    var op = AnyToyOp(ReshapeOp(result="1", input="0", type=ranked([2, 3])))
    assert_equal(print_op(op), "%1 = Reshape(%0) : tensor<2x3xf64>")


def test_mul_op():
    var op = AnyToyOp(MulOp(result="2", lhs="0", rhs="1", type=unranked()))
    assert_equal(print_op(op), "%2 = Mul(%0, %1) : tensor<*xf64>")


def test_add_op():
    var op = AnyToyOp(AddOp(result="2", lhs="0", rhs="1", type=unranked()))
    assert_equal(print_op(op), "%2 = Add(%0, %1) : tensor<*xf64>")


def test_generic_call_op():
    var op = AnyToyOp(GenericCallOp(
        result="4",
        callee="multiply_transpose",
        args=["1", "3"],
        type=unranked(),
    ))
    assert_equal(
        print_op(op),
        "%4 = GenericCall @multiply_transpose(%1, %3) : tensor<*xf64>",
    )


def test_print_op():
    var op = AnyToyOp(PrintOp(input="5"))
    assert_equal(print_op(op), "Print(%5)")


def test_return_op_with_value():
    var op = AnyToyOp(ReturnOp(value=String("2")))
    assert_equal(print_op(op), "return %2")


def test_return_op_void():
    var op = AnyToyOp(ReturnOp(value=Optional[String]()))
    assert_equal(print_op(op), "return")


def test_full_module():
    # Build multiply_transpose function
    var mt_args = List[ToyValue]()
    mt_args.append(ToyValue(name="a", type=unranked()))
    mt_args.append(ToyValue(name="b", type=unranked()))

    var mt_ops: List[AnyToyOp] = [
        AnyToyOp(TransposeOp(result="0", input="a", type=unranked())),
        AnyToyOp(TransposeOp(result="1", input="b", type=unranked())),
        AnyToyOp(MulOp(result="2", lhs="0", rhs="1", type=unranked())),
        AnyToyOp(ReturnOp(value=String("2"))),
    ]

    var mt_inputs: List[AnyToyType] = [unranked(), unranked()]
    var mt_func_type = FunctionType(inputs=mt_inputs^, result=unranked())

    var mt_func = FuncOp(
        name="multiply_transpose",
        func_type=mt_func_type^,
        body=Block(args=mt_args^, ops=mt_ops^),
    )

    # Build main function
    var main_ops = List[AnyToyOp]()
    main_ops.append(AnyToyOp(ConstantOp(
        result="0",
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        shape=[2, 3],
        type=ranked([2, 3]),
    )))
    main_ops.append(AnyToyOp(ReshapeOp(result="1", input="0", type=ranked([2, 3]))))
    main_ops.append(AnyToyOp(ConstantOp(
        result="2",
        value=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        shape=[6],
        type=ranked([6]),
    )))
    main_ops.append(AnyToyOp(ReshapeOp(result="3", input="2", type=ranked([2, 3]))))
    main_ops.append(AnyToyOp(GenericCallOp(
        result="4", callee="multiply_transpose",
        args=["1", "3"], type=unranked(),
    )))
    main_ops.append(AnyToyOp(GenericCallOp(
        result="5", callee="multiply_transpose",
        args=["3", "1"], type=unranked(),
    )))
    main_ops.append(AnyToyOp(PrintOp(input="5")))
    main_ops.append(AnyToyOp(ReturnOp(value=Optional[String]())))

    var main_func_type = FunctionType(
        inputs=List[AnyToyType](), result=Optional[AnyToyType](),
    )
    var main_func = FuncOp(
        name="main",
        func_type=main_func_type^,
        body=Block(args=List[ToyValue](), ops=main_ops^),
    )

    var module = Module(functions=List[FuncOp]())
    module.functions.append(mt_func^)
    module.functions.append(main_func^)

    var expected = String(
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
    assert_equal(print_module(module), expected)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
