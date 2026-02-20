"""Ch4 tests: Shape inference pass."""

from dgen import asm
from toy.parser.lowering import lower
from toy.parser.toy_parser import parse_toy
from toy.passes.shape_inference import infer_shapes
from toy.test.helpers import strip_prefix


def compile_and_infer(source: str) -> str:
    ast = parse_toy(source)
    ir = lower(ast)
    ir = infer_shapes(ir)
    return asm.format(ir)


def test_transpose():
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2, 3], [4, 5, 6]];
        |   var b = transpose(a);
        |   print(b);
        |   return;
        | }
    """)
    result = compile_and_infer(source)
    expected = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor[(2, 3), f64] = constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        |     %1 : toy.Tensor[(3, 2), f64] = toy.transpose(%0)
        |     %2 : () = toy.print(%1)
        |     %3 : () = return()
    """)
    assert result == expected


def test_mul():
    source = strip_prefix("""
        | def main() {
        |   var a = [[1, 2], [3, 4]];
        |   var b = [[5, 6], [7, 8]];
        |   var c = a * b;
        |   print(c);
        |   return;
        | }
    """)
    result = compile_and_infer(source)
    expected = strip_prefix("""
        | import toy
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor[(2, 2), f64] = constant([1.0, 2.0, 3.0, 4.0])
        |     %1 : toy.Tensor[(2, 2), f64] = constant([5.0, 6.0, 7.0, 8.0])
        |     %2 : toy.Tensor[(2, 2), f64] = toy.mul(%0, %1)
        |     %3 : () = toy.print(%2)
        |     %4 : () = return()
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
    result = compile_and_infer(source)
    expected = strip_prefix("""
        | import toy
        |
        | %multiply_transpose = function (%a: toy.Tensor[(2, 3), f64], %b: toy.Tensor[(2, 3), f64]) -> toy.Tensor[(3, 2), f64]:
        |     %0 : toy.Tensor[(3, 2), f64] = toy.transpose(%a)
        |     %1 : toy.Tensor[(3, 2), f64] = toy.transpose(%b)
        |     %2 : toy.Tensor[(3, 2), f64] = toy.mul(%0, %1)
        |     %3 : () = return(%2)
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor[(2, 3), f64] = constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        |     %1 : toy.Tensor[(2, 3), f64] = constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        |     %2 : toy.Tensor[(3, 2), f64] = toy.generic_call(@multiply_transpose, [%0, %1])
        |     %3 : () = toy.print(%2)
        |     %4 : () = return()
    """)
    assert result == expected


def test_full_tutorial_example():
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
    result = compile_and_infer(source)
    expected = strip_prefix("""
        | import toy
        |
        | %multiply_transpose = function (%a: toy.Tensor[(2, 3), f64], %b: toy.Tensor[(2, 3), f64]) -> toy.Tensor[(3, 2), f64]:
        |     %0 : toy.Tensor[(3, 2), f64] = toy.transpose(%a)
        |     %1 : toy.Tensor[(3, 2), f64] = toy.transpose(%b)
        |     %2 : toy.Tensor[(3, 2), f64] = toy.mul(%0, %1)
        |     %3 : () = return(%2)
        |
        | %main = function () -> ():
        |     %0 : toy.Tensor[(2, 3), f64] = constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        |     %1 : toy.Tensor[(2, 3), f64] = toy.reshape(%0)
        |     %2 : toy.Tensor[(6), f64] = constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        |     %3 : toy.Tensor[(2, 3), f64] = toy.reshape(%2)
        |     %4 : toy.Tensor[(3, 2), f64] = toy.generic_call(@multiply_transpose, [%1, %3])
        |     %5 : toy.Tensor[(3, 2), f64] = toy.generic_call(@multiply_transpose, [%3, %1])
        |     %6 : () = toy.print(%5)
        |     %7 : () = return()
    """)
    assert result == expected
