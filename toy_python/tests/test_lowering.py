"""Phase 3 tests: Toy source -> IR -> text, compare against expected."""

from toy_python.parser.toy_parser import parse_toy
from toy_python.parser.lowering import lower
from toy_python import asm


def compile_toy(source: str) -> str:
    ast = parse_toy(source)
    ir = lower(ast)
    return asm.format(ir)


def test_simple_constant():
    source = (
        "def main() {\n"
        "  var x = [[1, 2, 3], [4, 5, 6]];\n"
        "  print(x);\n"
        "  return;\n"
        "}\n"
    )
    result = compile_toy(source)
    expected = (
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    Print(%0)\n"
        "    return\n"
    )
    assert result == expected


def test_explicit_shape_with_reshape():
    source = (
        "def main() {\n"
        "  var x = [[1, 2, 3], [4, 5, 6]];\n"
        "  var y<2, 3> = x;\n"
        "  print(y);\n"
        "  return;\n"
        "}\n"
    )
    result = compile_toy(source)
    expected = (
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Reshape(%0) : tensor<2x3xf64>\n"
        "    Print(%1)\n"
        "    return\n"
    )
    assert result == expected


def test_binary_operations():
    source = (
        "def main() {\n"
        "  var a = [[1, 2], [3, 4]];\n"
        "  var b = [[5, 6], [7, 8]];\n"
        "  var c = a * b;\n"
        "  print(c);\n"
        "  return;\n"
        "}\n"
    )
    result = compile_toy(source)
    expected = (
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x2> [1.0, 2.0, 3.0, 4.0]) : tensor<2x2xf64>\n"
        "    %1 = Constant(<2x2> [5.0, 6.0, 7.0, 8.0]) : tensor<2x2xf64>\n"
        "    %2 = Mul(%0, %1) : tensor<*xf64>\n"
        "    Print(%2)\n"
        "    return\n"
    )
    assert result == expected


def test_transpose_builtin():
    source = (
        "def main() {\n"
        "  var a = [[1, 2, 3], [4, 5, 6]];\n"
        "  var b = transpose(a);\n"
        "  print(b);\n"
        "  return;\n"
        "}\n"
    )
    result = compile_toy(source)
    expected = (
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Transpose(%0) : tensor<*xf64>\n"
        "    Print(%1)\n"
        "    return\n"
    )
    assert result == expected


def test_generic_call():
    source = (
        "def multiply_transpose(a, b) {\n"
        "  return transpose(a) * transpose(b);\n"
        "}\n"
        "\n"
        "def main() {\n"
        "  var a = [[1, 2, 3], [4, 5, 6]];\n"
        "  var b = [[1, 2, 3], [4, 5, 6]];\n"
        "  var c = multiply_transpose(a, b);\n"
        "  print(c);\n"
        "  return;\n"
        "}\n"
    )
    result = compile_toy(source)
    expected = (
        "%multiply_transpose = function (%a: tensor<*xf64>, %b: tensor<*xf64>) -> tensor<*xf64>:\n"
        "    %0 = Transpose(%a) : tensor<*xf64>\n"
        "    %1 = Transpose(%b) : tensor<*xf64>\n"
        "    %2 = Mul(%0, %1) : tensor<*xf64>\n"
        "    return %2\n"
        "\n"
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %2 = GenericCall @multiply_transpose(%0, %1) : tensor<*xf64>\n"
        "    Print(%2)\n"
        "    return\n"
    )
    assert result == expected


def test_full_tutorial_example():
    """The complete multiply_transpose example from the MLIR Toy tutorial."""
    source = (
        "def multiply_transpose(a, b) {\n"
        "  return transpose(a) * transpose(b);\n"
        "}\n"
        "\n"
        "def main() {\n"
        "  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];\n"
        "  var b<2, 3> = [1, 2, 3, 4, 5, 6];\n"
        "  var c = multiply_transpose(a, b);\n"
        "  var d = multiply_transpose(b, a);\n"
        "  print(d);\n"
        "  return;\n"
        "}\n"
    )
    result = compile_toy(source)
    expected = (
        "%multiply_transpose = function (%a: tensor<*xf64>, %b: tensor<*xf64>) -> tensor<*xf64>:\n"
        "    %0 = Transpose(%a) : tensor<*xf64>\n"
        "    %1 = Transpose(%b) : tensor<*xf64>\n"
        "    %2 = Mul(%0, %1) : tensor<*xf64>\n"
        "    return %2\n"
        "\n"
        "%main = function () -> ():\n"
        "    %0 = Constant(<2x3> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<2x3xf64>\n"
        "    %1 = Reshape(%0) : tensor<2x3xf64>\n"
        "    %2 = Constant(<6> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) : tensor<6xf64>\n"
        "    %3 = Reshape(%2) : tensor<2x3xf64>\n"
        "    %4 = GenericCall @multiply_transpose(%1, %3) : tensor<*xf64>\n"
        "    %5 = GenericCall @multiply_transpose(%3, %1) : tensor<*xf64>\n"
        "    Print(%5)\n"
        "    return\n"
    )
    assert result == expected
