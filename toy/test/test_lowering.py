"""Phase 3 tests: Toy source -> IR -> text, compare against expected."""

from dgen import asm
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.test.helpers import strip_prefix


def compile_toy(source: str) -> str:
    ast = parse_toy(source)
    ir = lower(ast)
    return asm.format(ir)


def test_simple_constant():
    source = strip_prefix("""
        | def main() {
        |   var x = [[1, 2, 3], [4, 5, 6]];
        |   print(x);
        |   return;
        | }
    """)
    result = compile_toy(source)
    expected = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor([2, 3], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : () = toy.print(%0)
        |     %2 : () = return()
    """)
    assert result == expected


def test_explicit_shape_with_reshape():
    source = strip_prefix("""
        | def main() {
        |   var x = [[1, 2, 3], [4, 5, 6]];
        |   var y<2, 3> = x;
        |   print(y);
        |   return;
        | }
    """)
    result = compile_toy(source)
    expected = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor([2, 3], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.Tensor([2, 3], f64) = toy.reshape(%0)
        |     %2 : () = toy.print(%1)
        |     %3 : () = return()
    """)
    assert result == expected


def test_binary_operations():
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2], [3, 4]];
        |   var b = [[5, 6], [7, 8]];
        |   var c = a * b;
        |   print(c);
        |   return;
        | }
    """)
    result = compile_toy(source)
    expected = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor([2, 2], f64) = [1.0, 2.0, 3.0, 4.0]
        |     %1 : toy.Tensor([2, 2], f64) = [5.0, 6.0, 7.0, 8.0]
        |     %2 : toy.InferredShapeTensor(f64) = toy.mul(%0, %1)
        |     %3 : () = toy.print(%2)
        |     %4 : () = return()
    """)
    assert result == expected


def test_transpose_builtin():
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = transpose(a);
        |   print(b);
        |   return;
        | }
    """)
    result = compile_toy(source)
    expected = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor([2, 3], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %1 : toy.InferredShapeTensor(f64) = toy.transpose(%0)
        |     %2 : () = toy.print(%1)
        |     %3 : () = return()
    """)
    assert result == expected


def test_generic_call():
    source = strip_prefix("""
        | def multiply_transpose(a, b) {
        |   return transpose(a) * transpose(b);
        | }
        |
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = [[1, 2, 3], [4, 5, 6]];
        |   var c = multiply_transpose(a, b);
        |   print(c);
        |   return;
        | }
    """)
    result = compile_toy(source)
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
        |     %1 : toy.Tensor([2, 3], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        |     %2 : toy.InferredShapeTensor(f64) = toy.generic_call("multiply_transpose", [%0, %1])
        |     %3 : () = toy.print(%2)
        |     %4 : () = return()
    """)
    assert result == expected


def test_3d_constant():
    source = strip_prefix("""
        | def main() {
        |   var x = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        |   print(x);
        |   return;
        | }
    """)
    result = compile_toy(source)
    expected = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor([2, 2, 2], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : () = toy.print(%0)
        |     %2 : () = return()
    """)
    assert result == expected


def test_3d_binary_operations():
    source = strip_prefix("""
        | def main() {
        |   var a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        |   var b = [[[2, 3], [4, 5]], [[6, 7], [8, 9]]];
        |   var c = a + b;
        |   print(c);
        |   return;
        | }
    """)
    result = compile_toy(source)
    expected = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor([2, 2, 2], f64) = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        |     %1 : toy.Tensor([2, 2, 2], f64) = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        |     %2 : toy.InferredShapeTensor(f64) = toy.add(%0, %1)
        |     %3 : () = toy.print(%2)
        |     %4 : () = return()
    """)
    assert result == expected


def test_full_tutorial_example():
    """The complete multiply_transpose example from the MLIR Toy tutorial."""
    source = strip_prefix("""
        | def multiply_transpose(a, b) {
        |   return transpose(a) * transpose(b);
        | }
        |
        | def main() {
        |   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
        |   var b<2, 3> = [1, 2, 3, 4, 5, 6];
        |   var c = multiply_transpose(a, b);
        |   var d = multiply_transpose(b, a);
        |   print(d);
        |   return;
        | }
    """)
    result = compile_toy(source)
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
    assert result == expected
